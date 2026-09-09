"""Capability-tiered provider router with quota-aware degradation.

Routing policy
--------------
Each ``Purpose`` maps to an ordered tier list of provider names. Within a tier,
providers are ranked by remaining quota headroom, so a run spreads load across
free tiers instead of draining the first one and then stalling.

Order of operations per call, and why:

1. **Cache first, before provider selection.** The fingerprint covers the model
   id, so a cache probe happens per candidate model — the first candidate whose
   fingerprint is on disk wins without touching the network. This is what makes
   seven ablation arms affordable.
2. **Skip providers that are unavailable, exhausted or cooling down**, rather
   than learning it from a 429. Discovering a limit by hitting it costs both
   latency and a quota slot.
3. **Degrade, do not retry, on rate limits.** Provider-internal retries handle
   transient 5xx/timeouts; the router's job is to move on.
4. **Paid providers are unreachable unless explicitly allowed.** The
   ``strong-naked`` ablation arm is the only caller that opts in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from tally.models.base import (
    Completion,
    LLMProvider,
    Message,
    NotConfigured,
    PaidCallBlocked,
    ProviderError,
    ProviderUnavailable,
    Purpose,
    RateLimited,
    request_fingerprint,
)
from tally.models.cache import CompletionCache
from tally.models.quota import QuotaLimit, QuotaTracker
from tally.observability.accounting import Accountant
from tally.observability.trace import SpanKind, Tracer

# Purpose -> ordered provider preference. Names must exist in the registry.
# ``ollama`` appears last on the reasoning purposes rather than not at all. It is
# not a good planner at 8B, and the ordering says so — but a machine with no
# cloud credentials and a local daemon is the *actual* zero-cost configuration,
# and a policy that omits it there fails every case with NoProviderAvailable
# instead of running slowly. Degraded and running beats correct and unusable; the
# arm's conditions record which provider served the work either way.
DEFAULT_POLICY: dict[Purpose, tuple[str, ...]] = {
    Purpose.PLAN: ("gemini", "glm", "cerebras", "groq", "siliconflow", "ollama"),
    Purpose.DECIDE: ("gemini", "glm", "cerebras", "groq", "siliconflow", "ollama"),
    Purpose.CODE: ("gemini", "glm", "cerebras", "groq", "siliconflow", "ollama"),
    Purpose.REFLECT: ("gemini", "glm", "groq", "siliconflow", "ollama"),
    Purpose.SUMMARIZE: ("glm", "gemini", "ollama", "groq"),
    # Local first: these are high-volume, low-difficulty calls. Sending them to a
    # free cloud tier would burn the daily request budget that PLAN needs.
    Purpose.EXTRACT: ("ollama", "glm", "gemini"),
    Purpose.CLASSIFY: ("ollama", "glm", "gemini"),
    Purpose.REWRITE: ("ollama", "glm", "gemini"),
    # The judge must not be the actor, otherwise the eval grades its own work — so
    # a local-only setup deliberately has no judge. An eval that cannot grade
    # independently should fail loudly rather than let one model score itself.
    Purpose.JUDGE: ("gemini", "glm", "cerebras"),
}

# Conservative free-tier ceilings. Wrong-but-low is safe: the router degrades
# early instead of getting 429s. Tune per account.
DEFAULT_LIMITS: dict[str, QuotaLimit] = {
    "gemini": QuotaLimit(per_minute=10, per_day=200),
    "glm": QuotaLimit(per_minute=30, per_day=1000),
    "groq": QuotaLimit(per_minute=25, per_day=900),
    "cerebras": QuotaLimit(per_minute=25, per_day=800),
    "siliconflow": QuotaLimit(per_minute=20, per_day=500),
    "ollama": QuotaLimit(),  # local: unlimited requests, bounded by wall clock
    "fake": QuotaLimit(),
}


class NoProviderAvailable(ProviderError):
    """Every candidate for a purpose was unavailable, exhausted or failed."""


@dataclass
class RouteAttempt:
    provider: str
    outcome: str
    detail: str = ""


@dataclass
class Registry:
    """Named providers plus the routing policy over them."""

    providers: dict[str, LLMProvider] = field(default_factory=dict)
    policy: dict[Purpose, tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_POLICY)
    )
    limits: dict[str, QuotaLimit] = field(default_factory=lambda: dict(DEFAULT_LIMITS))

    def add(self, provider: LLMProvider, *, limit: QuotaLimit | None = None) -> "Registry":
        self.providers[provider.name] = provider
        if limit is not None:
            self.limits[provider.name] = limit
        return self

    def candidates(self, purpose: Purpose) -> tuple[str, ...]:
        return self.policy.get(purpose, tuple(self.providers))

    def override(self, purpose: Purpose, names: Sequence[str]) -> "Registry":
        self.policy[purpose] = tuple(names)
        return self


class Router:
    def __init__(
        self,
        registry: Registry,
        *,
        cache: CompletionCache | None = None,
        quota: QuotaTracker | None = None,
        accountant: Accountant | None = None,
        tracer: Tracer | None = None,
        agent: str = "root",
        scenario: str = "-",
    ) -> None:
        self.registry = registry
        self.cache = cache or CompletionCache(Path("data/cache/completions"), enabled=False)
        self.quota = quota or QuotaTracker(registry.limits)
        self.accountant = accountant or Accountant()
        self.tracer = tracer
        self.agent = agent
        self.scenario = scenario
        self.attempts: list[RouteAttempt] = []

    # -- public API --------------------------------------------------------
    def complete(
        self,
        purpose: Purpose,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        task: str = "-",
        allow_paid: bool = False,
    ) -> Completion:
        ranked = self._rank(purpose, allow_paid=allow_paid)
        if not ranked:
            raise NoProviderAvailable(
                f"no provider available for purpose={purpose.value}; "
                f"configured={sorted(self.registry.providers)}"
            )

        # Phase 1: a cache hit on any candidate short-circuits the whole call.
        for name in ranked:
            provider = self.registry.providers[name]
            fp = request_fingerprint(
                provider.model,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )
            hit = self.cache.get(fp)
            if hit is not None:
                self.attempts.append(RouteAttempt(name, "cache_hit"))
                self._account(hit, purpose, task, cached=True)
                self._trace(purpose, name, hit, cached=True, task=task)
                return hit

        # Phase 2: live calls, degrading down the ranked list.
        errors: list[str] = []
        for name in ranked:
            provider = self.registry.providers[name]
            fp = request_fingerprint(
                provider.model,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )
            try:
                self.quota.note_call(name)
                completion = provider.generate(
                    messages, max_tokens=max_tokens, temperature=temperature, stop=stop
                )
            except RateLimited as exc:
                self.quota.note_rate_limited(name, exc.retry_after_s)
                self.attempts.append(RouteAttempt(name, "rate_limited", str(exc)))
                errors.append(f"{name}: rate_limited")
                continue
            except NotConfigured as exc:
                self.attempts.append(RouteAttempt(name, "not_configured", str(exc)))
                errors.append(f"{name}: not_configured")
                continue
            except PaidCallBlocked as exc:
                self.attempts.append(RouteAttempt(name, "paid_blocked", str(exc)))
                errors.append(f"{name}: paid_blocked")
                continue
            except ProviderUnavailable as exc:
                self.attempts.append(RouteAttempt(name, "unavailable", str(exc)))
                errors.append(f"{name}: unavailable")
                continue

            self.attempts.append(RouteAttempt(name, "ok"))
            self.cache.put(fp, completion)
            self._account(completion, purpose, task, cached=False)
            self._trace(purpose, name, completion, cached=False, task=task)
            return completion

        raise NoProviderAvailable(
            f"all candidates failed for purpose={purpose.value}: {'; '.join(errors)}"
        )

    # -- selection ---------------------------------------------------------
    def _rank(self, purpose: Purpose, *, allow_paid: bool) -> list[str]:
        usable: list[tuple[int, float, str]] = []
        for tier, name in enumerate(self.registry.candidates(purpose)):
            provider = self.registry.providers.get(name)
            if provider is None or not provider.available():
                continue
            if getattr(provider, "is_paid", False) and not allow_paid:
                continue
            if self.quota.in_cooldown(name) or self.quota.is_exhausted(name):
                continue
            headroom = self.quota.headroom(name)
            # Soft-limited providers keep their tier but lose the headroom tie-break,
            # so an equal-tier alternative with room wins.
            usable.append((tier, -headroom, name))
        usable.sort()
        return [name for _, _, name in usable]

    # -- instrumentation ---------------------------------------------------
    def _account(self, c: Completion, purpose: Purpose, task: str, *, cached: bool) -> None:
        self.accountant.record(
            model=c.model,
            provider=c.provider,
            tokens_in=c.tokens_in,
            tokens_out=c.tokens_out,
            agent=self.agent,
            task=task,
            scenario=self.scenario,
            purpose=purpose.value,
            cached=cached,
        )

    def _trace(
        self, purpose: Purpose, name: str, c: Completion, *, cached: bool, task: str
    ) -> None:
        if self.tracer is None:
            return
        with self.tracer.span(SpanKind.LLM_CALL, f"{purpose.value}:{name}") as sp:
            sp.set(
                purpose=purpose.value,
                provider=c.provider,
                model=c.model,
                tokens_in=c.tokens_in,
                tokens_out=c.tokens_out,
                cached=cached,
                agent=self.agent,
                task=task,
                finish_reason=c.finish_reason,
            )

    def child(self, *, agent: str) -> "Router":
        """A sub-agent router sharing cache/quota/accounting but attributed separately."""
        return Router(
            self.registry,
            cache=self.cache,
            quota=self.quota,
            accountant=self.accountant,
            tracer=self.tracer,
            agent=agent,
            scenario=self.scenario,
        )

    def report(self) -> dict[str, object]:
        return {
            "attempts": [a.__dict__ for a in self.attempts],
            "cache": self.cache.stats(),
            "quota": self.quota.report(),
            "usage": self.accountant.report(),
        }
