"""The Context Ledger: slot-bidding allocation of a token budget.

Allocation algorithm
--------------------
Given ``B = window - max_output - safety``:

1. **Reserve every hard floor.** If the floors alone exceed ``B`` the
   configuration is wrong, not the run — raise rather than silently truncate
   the system prompt.
2. **Split the discretionary remainder by share**, renormalised over slots that
   actually have content. An empty memory slot donates its share instead of
   reserving dead space.
3. **Return surplus and redistribute it.** A slot that wants less than its
   allocation gives the difference back; the surplus goes to slots that want
   more, ordered by *ascending elasticity* — inflexible slots are topped up
   first. This is the concrete meaning of "sacrifice the elastic slots first".
4. **Each slot fits its own content** via its eviction strategy and may emit a
   compact stand-in (a skill index, a history summary) for what it dropped.
5. **Emit a ``LedgerRecord``** listing per-slot allowance, usage, eviction count
   and reason, so the composition of every prompt is auditable after the fact.

Step 3 is where the guarantee lives: because floors are reserved in step 1 and
surplus flows to low-elasticity slots in step 3, retrieved memory can never
squeeze out the operating contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from tally.context.slots import (
    DEFAULT_POLICIES,
    Item,
    SlotBid,
    SlotFill,
    SlotName,
    SlotPolicy,
)
from tally.context.tokenizer import TokenCounter, count_tokens, default_counter
from tally.models.base import Message


class BudgetTooSmall(RuntimeError):
    """Hard floors do not fit in the budget: a configuration error."""


@dataclass(frozen=True)
class Budget:
    window: int
    max_output: int = 2048
    safety_margin: int = 512

    @property
    def available(self) -> int:
        return max(0, self.window - self.max_output - self.safety_margin)


@dataclass
class LedgerRecord:
    """Auditable account of one prompt construction."""

    budget: int
    window: int
    allocated: int
    used: int
    fills: list[SlotFill] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def utilisation(self) -> float:
        return self.used / self.budget if self.budget else 0.0

    @property
    def evicted_tokens(self) -> int:
        return sum(sum(i.tokens for i in f.evicted) for f in self.fills)

    def slot(self, name: SlotName) -> SlotFill | None:
        for f in self.fills:
            if f.policy.name is name:
                return f
        return None

    def to_json(self) -> dict[str, object]:
        return {
            "window": self.window,
            "budget": self.budget,
            "allocated": self.allocated,
            "used": self.used,
            "utilisation": round(self.utilisation, 4),
            "evicted_tokens": self.evicted_tokens,
            "slots": [f.to_json() for f in self.fills],
            "notes": self.notes,
        }

    def stacked_bar(self, width: int = 60) -> str:
        """Text rendering of the token split — the figure the spec promises."""
        if not self.used:
            return "(empty context)"
        glyphs = {
            SlotName.SYSTEM: "S",
            SlotName.SKILL: "K",
            SlotName.TOOLS: "T",
            SlotName.WORKSPACE: "W",
            SlotName.MEMORY: "M",
            SlotName.HISTORY: "H",
        }
        bar = ""
        for f in self.fills:
            cells = round(width * f.used / self.budget) if self.budget else 0
            bar += glyphs[f.policy.name] * cells
        bar = bar[:width].ljust(width, ".")
        legend = "  ".join(
            f"{glyphs[f.policy.name]}={f.policy.name.value}:{f.used}"
            for f in self.fills
            if f.used
        )
        return f"[{bar}] {self.used}/{self.budget} tok\n  {legend}"


@dataclass
class BuiltContext:
    messages: list[Message]
    record: LedgerRecord

    @property
    def tokens(self) -> int:
        return self.record.used


class ContextLedger:
    def __init__(
        self,
        *,
        budget: Budget,
        policies: Sequence[SlotPolicy] = DEFAULT_POLICIES,
        counter: TokenCounter | None = None,
    ) -> None:
        self.budget = budget
        self.policies = list(policies)
        self.counter = counter or default_counter()
        self._order = [p.name for p in self.policies]

    # -- helpers -----------------------------------------------------------
    def count(self, text: str) -> int:
        return count_tokens(text, self.counter)

    def item(
        self,
        text: str,
        *,
        score: float = 0.0,
        pinned: bool = False,
        label: str = "",
        kind: str = "",
    ) -> Item:
        return Item(
            text=text,
            tokens=self.count(text),
            score=score,
            pinned=pinned,
            label=label,
            kind=kind,
        )

    # -- allocation --------------------------------------------------------
    def allocate(self, bids: dict[SlotName, SlotBid]) -> dict[SlotName, int]:
        total = self.budget.available
        active = [self.policy_for(n) for n in self._order if n in bids]

        floors = {p.name: p.hard_floor_tokens for p in active}
        floor_sum = sum(floors.values())
        if floor_sum > total:
            raise BudgetTooSmall(
                f"hard floors ({floor_sum} tok) exceed available budget ({total} tok); "
                f"raise the window or lower the floors"
            )

        discretionary = total - floor_sum
        with_content = [p for p in active if bids[p.name].has_content]
        share_sum = sum(p.share for p in with_content) or 1.0

        alloc: dict[SlotName, int] = {p.name: floors[p.name] for p in active}
        for p in with_content:
            alloc[p.name] += int(discretionary * (p.share / share_sum))

        # -- surplus redistribution ----------------------------------------
        # Slots wanting less than granted return the difference; the pool flows
        # to needy slots in ascending elasticity order (inflexible first).
        surplus = 0
        needy: list[SlotPolicy] = []
        for p in active:
            want = bids[p.name].requested_tokens
            if want < alloc[p.name]:
                surplus += alloc[p.name] - want
                alloc[p.name] = want
            elif want > alloc[p.name]:
                needy.append(p)

        for p in sorted(needy, key=lambda q: q.elasticity):
            if surplus <= 0:
                break
            deficit = bids[p.name].requested_tokens - alloc[p.name]
            grant = min(deficit, surplus)
            alloc[p.name] += grant
            surplus -= grant

        return alloc

    def policy_for(self, name: SlotName) -> SlotPolicy:
        for p in self.policies:
            if p.name is name:
                return p
        raise KeyError(f"no policy for slot {name}")

    # -- build -------------------------------------------------------------
    def build(
        self,
        bids: dict[SlotName, SlotBid],
        *,
        user_turn: str | None = None,
    ) -> BuiltContext:
        alloc = self.allocate(bids)
        fills: list[SlotFill] = []
        notes: list[str] = []

        for name in self._order:
            bid = bids.get(name)
            if bid is None:
                continue
            fill = bid.fit(alloc[name], self.count)
            if fill.overflowed:
                notes.append(
                    f"{name.value}: pinned content {fill.used} tok exceeded "
                    f"allowance {fill.allowance} tok"
                )
            fills.append(fill)

        used = sum(f.used for f in fills)
        record = LedgerRecord(
            budget=self.budget.available,
            window=self.budget.window,
            allocated=sum(alloc.values()),
            used=used,
            fills=fills,
            notes=notes,
        )

        messages = self._assemble(fills, user_turn)
        if user_turn:
            record.used += self.count(user_turn)
        return BuiltContext(messages=messages, record=record)

    def _assemble(self, fills: list[SlotFill], user_turn: str | None) -> list[Message]:
        """System-ish slots collapse into one system message; history stays as turns.

        Keeping history as real turns (rather than a flattened blob) matters for
        providers that apply different handling to the system instruction, and it
        keeps the assistant's own prior output attributable.
        """
        by_name = {f.policy.name: f for f in fills}
        system_order = (SlotName.SYSTEM, SlotName.SKILL, SlotName.TOOLS, SlotName.WORKSPACE, SlotName.MEMORY)
        blocks: list[str] = []
        headings = {
            SlotName.SYSTEM: None,
            SlotName.SKILL: "## Active skills",
            SlotName.TOOLS: "## Available tools",
            SlotName.WORKSPACE: "## Workspace state",
            SlotName.MEMORY: "## Long-term memory",
        }
        for name in system_order:
            fill = by_name.get(name)
            if fill is None or not fill.text.strip():
                continue
            heading = headings[name]
            blocks.append(f"{heading}\n{fill.text}" if heading else fill.text)

        messages: list[Message] = []
        if blocks:
            messages.append(Message.system("\n\n".join(blocks)))

        history = by_name.get(SlotName.HISTORY)
        if history is not None:
            for item in history.kept:
                role = item.kind or "user"
                if role == "assistant":
                    messages.append(Message.assistant(item.text))
                else:
                    messages.append(Message.user(item.text))

        if user_turn:
            messages.append(Message.user(user_turn))
        return messages


def make_bid(
    ledger: ContextLedger,
    name: SlotName,
    texts: Iterable[str] | None = None,
    *,
    items: Sequence[Item] | None = None,
    pinned: bool = False,
    degrade: Callable[[Sequence[Item], int], Item | None] | None = None,
) -> SlotBid:
    """Convenience constructor used by the agent loop and tests."""
    built = list(items or [])
    for text in texts or []:
        built.append(ledger.item(text, pinned=pinned))
    return SlotBid(policy=ledger.policy_for(name), items=built, degrade=degrade)


