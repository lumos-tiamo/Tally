"""The ledger's guarantees, stated as tests.

Each test names the failure mode it prevents, because a budget allocator's
correctness is entirely about what it refuses to do under pressure.
"""

from __future__ import annotations

import pytest

from teamclaw.context.ledger import Budget, BudgetTooSmall, ContextLedger, make_bid
from teamclaw.context.slots import DEFAULT_POLICIES, Item, SlotBid, SlotName, SlotPolicy


def build(window: int = 8_000, max_output: int = 1_000) -> ContextLedger:
    return ContextLedger(budget=Budget(window=window, max_output=max_output, safety_margin=200))


def test_retrieved_memory_cannot_evict_the_system_prompt():
    """The core guarantee: instructions survive any amount of retrieved memory."""
    led = build(window=3_000, max_output=500)
    system_text = "OPERATING CONTRACT. " * 40
    bids = {
        SlotName.SYSTEM: make_bid(led, SlotName.SYSTEM, [system_text], pinned=True),
        SlotName.MEMORY: SlotBid(
            led.policy_for(SlotName.MEMORY),
            [led.item("a memory " * 200, score=1.0, label=f"m{i}") for i in range(40)],
        ),
    }
    built = led.build(bids)
    system_fill = built.record.slot(SlotName.SYSTEM)
    assert system_fill is not None
    assert system_text.strip() in built.messages[0].content
    assert system_fill.used > 0
    # Memory is the elastic slot, so it is what gets cut.
    memory_fill = built.record.slot(SlotName.MEMORY)
    assert memory_fill is not None
    assert memory_fill.evicted, "memory should have been sacrificed, not the system slot"


def test_hard_floors_that_do_not_fit_are_a_configuration_error():
    """Better to fail loudly than to silently truncate the operating contract."""
    tiny = ContextLedger(budget=Budget(window=600, max_output=500, safety_margin=50))
    bids = {n: make_bid(tiny, n, ["x"]) for n in (SlotName.SYSTEM, SlotName.TOOLS,
                                                  SlotName.WORKSPACE, SlotName.HISTORY)}
    with pytest.raises(BudgetTooSmall):
        tiny.build(bids)


def test_surplus_flows_to_inflexible_slots_first():
    """A slot that wants less gives its allowance to the least elastic claimant."""
    led = build(window=6_000, max_output=500)
    # TOOLS (elasticity .55) and MEMORY (elasticity .90) both want more than their
    # share; WORKSPACE wants almost nothing and donates.
    bids = {
        SlotName.SYSTEM: make_bid(led, SlotName.SYSTEM, ["sys"], pinned=True),
        SlotName.WORKSPACE: make_bid(led, SlotName.WORKSPACE, ["tiny"]),
        SlotName.TOOLS: SlotBid(
            led.policy_for(SlotName.TOOLS),
            [led.item("def tool(x): ...", score=1.0, label=f"t{i}") for i in range(200)],
        ),
        SlotName.MEMORY: SlotBid(
            led.policy_for(SlotName.MEMORY),
            [led.item("memory text", score=1.0, label=f"m{i}") for i in range(200)],
        ),
    }
    alloc = led.allocate(bids)
    tools_policy = led.policy_for(SlotName.TOOLS)
    memory_policy = led.policy_for(SlotName.MEMORY)
    # Tools has the larger share AND lower elasticity, so it must not lose to memory.
    assert alloc[SlotName.TOOLS] > alloc[SlotName.MEMORY]
    assert tools_policy.elasticity < memory_policy.elasticity


def test_empty_slot_donates_its_share_rather_than_reserving_dead_space():
    led = build()
    with_memory = led.allocate({
        SlotName.SYSTEM: make_bid(led, SlotName.SYSTEM, ["sys"], pinned=True),
        SlotName.HISTORY: SlotBid(
            led.policy_for(SlotName.HISTORY),
            [led.item("turn " * 100, score=1.0, label=f"h{i}") for i in range(50)],
        ),
        SlotName.MEMORY: SlotBid(led.policy_for(SlotName.MEMORY),
                                 [led.item("m " * 100, score=1.0)]),
    })
    without_memory = led.allocate({
        SlotName.SYSTEM: make_bid(led, SlotName.SYSTEM, ["sys"], pinned=True),
        SlotName.HISTORY: SlotBid(
            led.policy_for(SlotName.HISTORY),
            [led.item("turn " * 100, score=1.0, label=f"h{i}") for i in range(50)],
        ),
        SlotName.MEMORY: SlotBid(led.policy_for(SlotName.MEMORY), []),
    })
    assert without_memory[SlotName.HISTORY] > with_memory[SlotName.HISTORY]


def test_history_keeps_the_recent_tail_and_drops_the_oldest():
    led = build(window=2_400, max_output=300)
    items = [
        Item(text=f"turn {i} " * 30, tokens=led.count(f"turn {i} " * 30),
             score=float(i), kind="user", label=f"turn-{i}")
        for i in range(30)
    ]
    fill = SlotBid(led.policy_for(SlotName.HISTORY), items).fit(600, led.count)
    kept = {i.label for i in fill.kept}
    assert "turn-29" in kept, "the most recent turn must survive"
    assert "turn-0" not in kept, "the oldest turn should be the first to go"


def test_degrade_hook_substitutes_a_compact_stand_in():
    led = build(window=2_000, max_output=300)
    index = Item(text="[skill index: a, b, c]", tokens=0, label="skill-index")
    bid = SlotBid(
        led.policy_for(SlotName.SKILL),
        [led.item("SKILL BODY " * 100, score=1.0, label=f"s{i}") for i in range(5)],
        degrade=lambda evicted, remaining: index,
    )
    fill = bid.fit(60, led.count)
    assert fill.stand_in is not None
    assert "skill index" in fill.text


def test_ledger_record_is_auditable():
    """Every prompt must be explainable after the fact.

    The budget is deliberately tight and the memory slot deliberately oversized:
    a record only proves it is auditable if there is an eviction in it to explain.
    """
    led = build(window=2_000, max_output=400)
    bids = {
        SlotName.SYSTEM: make_bid(led, SlotName.SYSTEM, ["sys"], pinned=True),
        SlotName.MEMORY: SlotBid(
            led.policy_for(SlotName.MEMORY),
            [led.item("mem " * 120, score=0.5, label=f"m{i}") for i in range(40)],
        ),
    }
    record = led.build(bids).record
    payload = record.to_json()
    assert payload["budget"] == led.budget.available
    assert {s["slot"] for s in payload["slots"]} == {"system", "memory"}
    memory = record.slot(SlotName.MEMORY)
    assert memory is not None and memory.evicted
    assert memory.to_json()["evicted_labels"], "eviction must name what was dropped"
    assert "tok" in record.stacked_bar()


def test_utilisation_reflects_actual_usage():
    led = build(window=4_000, max_output=500)
    small = led.build({SlotName.SYSTEM: make_bid(led, SlotName.SYSTEM, ["hi"], pinned=True)})
    big = led.build({
        SlotName.SYSTEM: make_bid(led, SlotName.SYSTEM, ["hi"], pinned=True),
        SlotName.HISTORY: SlotBid(
            led.policy_for(SlotName.HISTORY),
            [led.item("turn " * 80, score=float(i), label=f"h{i}") for i in range(40)],
        ),
    })
    assert big.record.utilisation > small.record.utilisation
    assert 0.0 <= big.record.utilisation <= 1.05
