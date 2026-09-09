"""Memory-layer behaviour: conflict, decay, expiry, and recall quality."""

from __future__ import annotations

import time

from tally.context.memory import (
    ConflictPolicy,
    MemoryKind,
    MemoryStore,
    recall_metrics,
)


def test_a_low_confidence_write_cannot_destroy_a_verified_fact():
    """The failure mode that makes memory claims untrustworthy."""
    store = MemoryStore(agent="t")
    good = store.write("Apple's fiscal year ends in late September.",
                       key="aapl:fy_end", confidence=0.95)
    store.write("Apple's fiscal year ends in December.", key="aapl:fy_end", confidence=0.3)
    active = [r for r in store.active() if r.key == "aapl:fy_end"]
    assert len(active) == 1
    assert active[0].record_id == good.record_id
    assert store.conflicts and store.conflicts[0]["winner"] == good.record_id


def test_a_better_verified_correction_still_wins():
    store = MemoryStore(agent="t")
    store.write("FY ends in September.", key="k", confidence=0.9)
    better = store.write("FY2024 ended 2024-09-28 per the 10-K cover.", key="k", confidence=0.99)
    assert [r.record_id for r in store.active() if r.key == "k"] == [better.record_id]


def test_last_write_wins_is_available_but_not_the_default():
    store = MemoryStore(agent="t", conflict_policy=ConflictPolicy.LAST_WRITE_WINS)
    store.write("old", key="k", confidence=0.99)
    new = store.write("new", key="k", confidence=0.1)
    assert [r.text for r in store.active() if r.key == "k"] == [new.text]


def test_focus_is_always_the_latest_write():
    """The current objective is a statement about now, not a claim to weigh."""
    store = MemoryStore(agent="t")
    store.set_focus("Analyse FY2024.")
    store.set_focus("Switch to FY2023.")
    assert store.focus() == "Switch to FY2023."


def test_expired_records_leave_recall():
    store = MemoryStore(agent="t")
    fresh = store.write("still true", ttl_days=30)
    stale = store.write("was true last week", ttl_days=1)
    stale.created_at = time.time() - 5 * 86_400
    assert store.prune_expired() == 1
    ids = {r.record_id for r in store.active()}
    assert fresh.record_id in ids and stale.record_id not in ids


def test_decay_ranks_recent_facts_above_old_ones():
    store = MemoryStore(agent="t")
    old = store.write("revenue grew in the quarter", kind=MemoryKind.EPISODE)
    old.created_at = time.time() - 200 * 86_400
    store.write("revenue grew in the quarter", kind=MemoryKind.EPISODE)
    recalls = store.recall("revenue grew", k=2, mark_used=False)
    assert len(recalls) == 2
    assert recalls[0].score > recalls[1].score
    assert recalls[1].record.record_id == old.record_id


def test_persona_never_decays():
    store = MemoryStore(agent="t")
    p = store.write("You are an analyst.", kind=MemoryKind.PERSONA, key="p")
    p.created_at = time.time() - 3_000 * 86_400
    assert p.decay() == 1.0
    assert store.persona() == "You are an analyst."


def test_recall_metrics_give_the_memory_layer_its_own_number():
    store = MemoryStore(agent="t")
    target = store.write("Deferred revenue is recognised over the contract term.")
    store.write("The registrant's headquarters are in Cupertino.")
    store.write("Segment reporting follows management's internal view.")
    metrics = recall_metrics(store, [("how is deferred revenue recognised",
                                      {target.record_id})], k=2)
    assert metrics["probes"] == 1
    assert metrics["recall_at_k"] == 1.0


def test_store_survives_a_corrupt_line_on_disk(tmp_path):
    path = tmp_path / "mem.jsonl"
    store = MemoryStore(path=path, agent="t")
    store.write("good record")
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    reloaded = MemoryStore(path=path, agent="t")
    assert len(reloaded.records) == 1
