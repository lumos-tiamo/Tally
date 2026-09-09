"""Router policy, quota-aware degradation, and the completion cache."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tally.models.base import (
    Message,
    NotConfigured,
    PaidCallBlocked,
    ProviderUnavailable,
    Purpose,
    RateLimited,
    request_fingerprint,
)
from tally.models.cache import CompletionCache
from tally.models.providers.fake import FakeProvider
from tally.models.quota import QuotaLimit, QuotaTracker
from tally.models.router import NoProviderAvailable, Registry, Router


def registry_with(*providers, policy=None):  # noqa: ANN001
    registry = Registry(providers={p.name: p for p in providers})
    for purpose in list(registry.policy):
        registry.policy[purpose] = policy or tuple(p.name for p in providers)
    return registry


class PaidProvider(FakeProvider):
    is_paid = True


# --- routing --------------------------------------------------------------
def test_a_rate_limited_provider_is_skipped_not_retried():
    limited = FakeProvider(name="limited", fail_times=1,
                           failure=RateLimited("429", retry_after_s=30))
    backup = FakeProvider(name="backup", default="from backup")
    router = Router(registry_with(limited, backup))
    completion = router.complete(Purpose.PLAN, [Message.user("hi")])
    assert completion.text == "from backup"
    assert [a.outcome for a in router.attempts] == ["rate_limited", "ok"]
    assert router.quota.in_cooldown("limited")


def test_an_unconfigured_provider_is_skipped_silently():
    broken = FakeProvider(name="broken", fail_times=1, failure=NotConfigured("no key"))
    good = FakeProvider(name="good", default="ok")
    router = Router(registry_with(broken, good))
    assert router.complete(Purpose.PLAN, [Message.user("hi")]).text == "ok"


def test_all_candidates_failing_raises_rather_than_returning_nothing():
    a = FakeProvider(name="a", fail_times=1, failure=ProviderUnavailable("down"))
    b = FakeProvider(name="b", fail_times=1, failure=ProviderUnavailable("down"))
    router = Router(registry_with(a, b))
    with pytest.raises(NoProviderAvailable):
        router.complete(Purpose.PLAN, [Message.user("hi")])


def test_a_paid_provider_is_unreachable_without_explicit_opt_in():
    """Only the strong-naked ablation arm may spend money."""
    paid = PaidProvider(name="paid", default="expensive")
    router = Router(registry_with(paid))
    with pytest.raises(NoProviderAvailable):
        router.complete(Purpose.PLAN, [Message.user("hi")])
    assert router.complete(Purpose.PLAN, [Message.user("hi")],
                           allow_paid=True).text == "expensive"


def test_paid_call_blocked_is_raised_by_the_provider_itself():
    from tally.models.providers.openai_compat import OpenAICompatProvider

    provider = OpenAICompatProvider(
        name="strong", base_url="https://example.invalid", api_key="k",
        model="m", is_paid=True, allow_paid=False,
    )
    with pytest.raises(PaidCallBlocked):
        provider.generate([Message.user("hi")])


def test_high_volume_purposes_route_to_local_before_the_scarce_free_tier():
    """Extraction must not burn the daily request budget planning needs."""
    from tally.models.router import DEFAULT_POLICY

    assert DEFAULT_POLICY[Purpose.EXTRACT][0] == "ollama"
    assert DEFAULT_POLICY[Purpose.PLAN][0] != "ollama"


def test_a_local_only_machine_can_serve_every_purpose_except_judging():
    """The actual zero-cost configuration has to work, not just be describable.

    A machine with no cloud credentials and a running Ollama is the real
    zero-budget setup. A policy that omitted the local provider from the
    reasoning purposes would fail every case with NoProviderAvailable rather
    than run slowly, so the local tier is last on those lists rather than absent.

    Judging is the deliberate exception: it stays cloud-only, because a local-only
    setup would otherwise have the actor grade its own output, and an eval that
    cannot grade independently should fail loudly instead.
    """
    local = FakeProvider(name="ollama", model="qwen3:8b")
    router = Router(Registry(providers={"ollama": local}))
    for purpose in Purpose:
        ranked = router._rank(purpose, allow_paid=False)
        if purpose is Purpose.JUDGE:
            assert ranked == [], "a local model must not be allowed to judge"
        else:
            assert ranked == ["ollama"], f"{purpose.value} is unreachable locally"


def test_the_local_tier_is_last_resort_for_reasoning_and_first_for_extraction():
    """Ordering encodes that an 8B model is a fallback planner, not a good one."""
    from tally.models.router import DEFAULT_POLICY

    assert DEFAULT_POLICY[Purpose.PLAN][-1] == "ollama"
    assert DEFAULT_POLICY[Purpose.CODE][-1] == "ollama"
    assert DEFAULT_POLICY[Purpose.EXTRACT][0] == "ollama"
    assert "ollama" not in DEFAULT_POLICY[Purpose.JUDGE]


# --- quota ----------------------------------------------------------------
def test_an_exhausted_provider_is_not_called_again():
    tracker = QuotaTracker({"a": QuotaLimit(per_minute=2)})
    a = FakeProvider(name="a", default="a")
    b = FakeProvider(name="b", default="b")
    router = Router(registry_with(a, b), quota=tracker)
    assert router.complete(Purpose.PLAN, [Message.user("1")]).text == "a"
    assert router.complete(Purpose.PLAN, [Message.user("2")]).text == "a"
    assert router.complete(Purpose.PLAN, [Message.user("3")]).text == "b"


def test_headroom_ranks_equal_tier_providers():
    tracker = QuotaTracker({"a": QuotaLimit(per_day=10), "b": QuotaLimit(per_day=10)})
    for _ in range(8):
        tracker.note_call("a")
    assert tracker.headroom("b") > tracker.headroom("a")


def test_quota_state_survives_a_restart(tmp_path: Path):
    path = tmp_path / "quota.json"
    first = QuotaTracker({"a": QuotaLimit(per_day=5)}, state_path=path)
    for _ in range(3):
        first.note_call("a")
    first.save()
    second = QuotaTracker({"a": QuotaLimit(per_day=5)}, state_path=path)
    assert second.usage("a")["day"] == 3


def test_a_cooldown_expires():
    tracker = QuotaTracker({})
    tracker.note_rate_limited("a", retry_after_s=0.01)
    assert tracker.in_cooldown("a")
    time.sleep(0.05)
    assert not tracker.in_cooldown("a")


# --- cache ----------------------------------------------------------------
def test_a_cache_hit_avoids_the_provider_entirely():
    """What makes seven ablation arms affordable on a free tier."""
    provider = FakeProvider(name="p", default="answer")
    cache = CompletionCache(Path("/tmp/does-not-matter"), enabled=False)
    router = Router(registry_with(provider), cache=cache)
    router.complete(Purpose.PLAN, [Message.user("q")])
    assert len(provider.calls) == 1


def test_cache_round_trip(tmp_path: Path):
    provider = FakeProvider(name="p", default="answer")
    cache = CompletionCache(tmp_path / "cache")
    router = Router(registry_with(provider), cache=cache)
    first = router.complete(Purpose.PLAN, [Message.user("q")])
    second = router.complete(Purpose.PLAN, [Message.user("q")])
    assert not first.cached and second.cached
    assert len(provider.calls) == 1, "the second call must not reach the provider"
    assert cache.stats()["hits"] == 1


def test_a_corrupt_cache_entry_is_treated_as_a_miss(tmp_path: Path):
    cache = CompletionCache(tmp_path / "cache")
    fingerprint = "a" * 64
    path = cache._path(fingerprint)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json", encoding="utf-8")
    assert cache.get(fingerprint) is None


def test_temperature_is_part_of_the_fingerprint():
    """A sampled run must never masquerade as a deterministic one."""
    messages = [Message.user("q")]
    cold = request_fingerprint("m", messages, max_tokens=10, temperature=0.0, stop=None)
    warm = request_fingerprint("m", messages, max_tokens=10, temperature=0.7, stop=None)
    assert cold != warm


# --- accounting -----------------------------------------------------------
def test_cost_is_attributed_per_agent_so_a_subagent_is_visible():
    provider = FakeProvider(name="p")
    router = Router(registry_with(provider), agent="parent")
    router.complete(Purpose.PLAN, [Message.user("a")])
    child = router.child(agent="extractor")
    child.complete(Purpose.PLAN, [Message.user("b")])
    by_agent = router.accountant.by("agent")
    assert set(by_agent) == {"parent", "extractor"}


def test_a_cache_hit_is_billed_at_zero():
    from tally.observability.accounting import Accountant

    accountant = Accountant()
    entry = accountant.record(model="strong-paid", provider="p", tokens_in=1_000_000,
                              tokens_out=1_000_000, cached=True)
    assert entry.cost_usd == 0.0
    assert accountant.total().tokens_total == 2_000_000
