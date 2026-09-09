"""Build the provider registry from settings.

Nothing here raises on missing credentials: an unconfigured provider is simply
absent from the registry, and the router degrades to whatever is present. That
keeps `teamclaw` runnable on a machine with zero keys (fake provider only),
which is the state every test runs in.
"""

from __future__ import annotations

from teamclaw.config import Settings, settings as load_settings
from teamclaw.models.providers.fake import FakeProvider
from teamclaw.models.providers.gemini import GeminiProvider
from teamclaw.models.providers.ollama_local import OllamaProvider
from teamclaw.models.providers.openai_compat import OpenAICompatProvider
from teamclaw.models.base import Purpose
from teamclaw.models.quota import QuotaLimit
from teamclaw.models.router import DEFAULT_LIMITS, Registry

# Free-tier OpenAI-compatible endpoints. Model ids are the free/flash variants.
OPENAI_COMPAT_SPECS = (
    ("glm", "https://open.bigmodel.cn/api/paas/v4", "glm-4-flash", "glm_key"),
    ("groq", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "groq_key"),
    ("cerebras", "https://api.cerebras.ai/v1", "llama-3.3-70b", "cerebras_key"),
    ("siliconflow", "https://api.siliconflow.cn/v1", "Qwen/Qwen3-8B", "siliconflow_key"),
)


def build_registry(
    cfg: Settings | None = None,
    *,
    include_fake: bool = False,
    include_local: bool = True,
    include_paid: bool = False,
) -> Registry:
    cfg = cfg or load_settings()
    registry = Registry(limits=dict(DEFAULT_LIMITS))

    # A relay is one endpoint serving many models, so it is registered as
    # several named providers rather than one. The router selects by purpose, and
    # collapsing them into a single entry would make "the judge must not be the
    # actor" unenforceable — both would resolve to the same provider name.
    if cfg.relay_key and cfg.relay_base_url:
        registry.add(
            OpenAICompatProvider(name="relay", base_url=cfg.relay_base_url,
                                 api_key=cfg.relay_key, model=cfg.relay_model),
            limit=QuotaLimit(per_minute=60, per_day=20_000),
        )
        registry.add(
            OpenAICompatProvider(name="relay-cheap", base_url=cfg.relay_base_url,
                                 api_key=cfg.relay_key, model=cfg.relay_cheap_model),
            limit=QuotaLimit(per_minute=90, per_day=40_000),
        )
        registry.add(
            OpenAICompatProvider(name="relay-judge", base_url=cfg.relay_base_url,
                                 api_key=cfg.relay_key, model=cfg.relay_judge_model),
            limit=QuotaLimit(per_minute=30, per_day=5_000),
        )
        _prefer_relay(registry)

    if cfg.gemini_key:
        registry.add(GeminiProvider(api_key=cfg.gemini_key, model="gemini-2.5-flash"))

    for name, base_url, model, attr in OPENAI_COMPAT_SPECS:
        key = getattr(cfg, attr, "")
        if key:
            registry.add(
                OpenAICompatProvider(
                    name=name, base_url=base_url, api_key=key, model=model
                )
            )

    if include_local:
        # Registered unconditionally: available() probes the daemon, and an absent
        # daemon just means the router skips it.
        registry.add(OllamaProvider(base_url=cfg.ollama_base_url, model="qwen3:8b"))

    if include_paid and cfg.strong_key and cfg.strong_base_url and cfg.strong_model:
        registry.add(
            OpenAICompatProvider(
                name="strong",
                base_url=cfg.strong_base_url,
                api_key=cfg.strong_key,
                model=cfg.strong_model,
                is_paid=True,
                allow_paid=cfg.allow_paid,
            )
        )

    if include_fake or not registry.providers:
        registry.add(FakeProvider())
        # Route every purpose to the fake provider as a last resort, so a
        # credential-less machine still exercises the full loop.
        for purpose, names in list(registry.policy.items()):
            if "fake" not in names:
                registry.policy[purpose] = (*names, "fake")

    return registry


def _prefer_relay(registry: Registry) -> None:
    """Put the relay in front for each purpose, keeping the tier's intent.

    The reasoning purposes get the workhorse model, the high-volume extraction
    purposes get the cheap one, and judging gets a third — because a judge that
    resolves to the same provider as the actor is not an independent grader, and
    with one relay serving every model that is easy to end up with by accident.
    """
    workhorse, cheap, judge = "relay", "relay-cheap", "relay-judge"
    front = {
        Purpose.PLAN: workhorse, Purpose.DECIDE: workhorse, Purpose.CODE: workhorse,
        Purpose.REFLECT: workhorse, Purpose.SUMMARIZE: cheap,
        Purpose.EXTRACT: cheap, Purpose.CLASSIFY: cheap, Purpose.REWRITE: cheap,
        Purpose.JUDGE: judge,
    }
    for purpose, preferred in front.items():
        existing = tuple(n for n in registry.policy.get(purpose, ()) if n != preferred)
        registry.policy[purpose] = (preferred, *existing)


def describe_registry(registry: Registry) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "model": p.model,
            "paid": bool(getattr(p, "is_paid", False)),
            "available": p.available(),
        }
        for name, p in sorted(registry.providers.items())
    ]
