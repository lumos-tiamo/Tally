"""Session state: the fallback, the dedupe, and the rate limit.

The fallback being *labelled* is the point of most of these. A memory backend
that silently pretends to be shared is the failure that only appears under a
second worker, in production.
"""

from __future__ import annotations

import time

from teamclaw.session import MemoryBackend, SessionStore


def store() -> SessionStore:
    return SessionStore(MemoryBackend())


def test_no_redis_falls_back_and_says_why():
    session = SessionStore.connect(None)
    assert session.backend.name == "memory"
    assert "no TEAMCLAW_REDIS_URL" in session.degraded_reason


def test_an_unreachable_redis_falls_back_rather_than_raising():
    """A dead cache must not stop the platform from starting."""
    session = SessionStore.connect("redis://127.0.0.1:6390/0")
    assert session.backend.name == "memory"
    assert "redis" in session.degraded_reason.lower()


def test_the_memory_backend_warns_that_it_is_not_shared():
    warnings = store().warnings()
    assert warnings and "not be shared" in warnings[0]


def test_a_redelivered_event_is_recognised():
    """Channels retry on a slow 200, and an agent run is slow by nature."""
    session = store()
    assert session.seen_before("feishu", "evt_1") is False
    assert session.seen_before("feishu", "evt_1") is True
    assert session.seen_before("feishu", "evt_2") is False


def test_an_event_without_an_id_is_never_deduplicated():
    """Better to run twice than to drop every message from a channel with no ids."""
    session = store()
    assert session.seen_before("feishu", "") is False
    assert session.seen_before("feishu", "") is False


def test_dedupe_is_per_channel():
    session = store()
    session.seen_before("feishu", "shared_id")
    assert session.seen_before("slack", "shared_id") is False


def test_the_rate_limit_closes_after_its_budget():
    session = store()
    outcomes = [session.rate_limit("t", limit=3, window_s=60)[0] for _ in range(5)]
    assert outcomes == [True, True, True, False, False]


def test_rate_limits_are_per_scope():
    session = store()
    for _ in range(3):
        session.rate_limit("tenant:a", limit=3, window_s=60)
    allowed, _ = session.rate_limit("tenant:b", limit=3, window_s=60)
    assert allowed


def test_ttls_expire():
    session = store()
    session.set_json("k", value={"v": 1}, ttl_s=1)
    assert session.get_json("k") == {"v": 1}
    time.sleep(1.1)
    assert session.get_json("k") is None


def test_the_hitl_queue_round_trips():
    session = store()
    session.push_hitl("run_1", {"reason": "repeated_failure", "question": "retry?"})
    pending = session.pending_hitl()
    assert len(pending) == 1 and pending[0]["run_id"] == "run_1"

    session.resolve_hitl("run_1", "guidance", "call help() first")
    assert session.pending_hitl() == []
    answer = session.hitl_answer("run_1")
    assert answer["decision"] == "guidance" and "help()" in answer["message"]


def test_active_run_tracking_round_trips():
    session = store()
    assert session.active_run("conv_1") is None
    session.set_active_run("conv_1", "run_9")
    assert session.active_run("conv_1") == "run_9"
    session.clear_active_run("conv_1")
    assert session.active_run("conv_1") is None


def test_a_corrupt_value_reads_as_absent_rather_than_raising():
    session = store()
    session.backend.set(session._key("k"), "{not json")
    assert session.get_json("k") is None
