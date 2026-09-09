"""Context slots: the units that bid for the token budget.

A slot owns two things: a *policy* (how much it may take, how readily it gives
ground) and an *eviction strategy* (what to drop when it gets less than it
asked for). Separating these is what keeps the ledger generic — adding a new
slot never touches the allocation algorithm.

Policy fields
-------------
``hard_floor_tokens``
    The slot is never allocated less than this. ``0`` means the slot may be
    dropped entirely. This is the mechanism that stops retrieved memory from
    squeezing out the system instructions — the failure mode the spec calls out.
``elasticity``
    0.0 = inflexible (gets surplus first, sacrificed last).
    1.0 = highly elastic (sacrificed first).
``share``
    Target fraction of the *discretionary* budget (what remains after every
    hard floor is reserved). Shares are renormalised over slots that actually
    have content, so an empty memory slot donates its share rather than
    reserving dead space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence


class SlotName(str, Enum):
    SYSTEM = "system"        # persona + operating contract
    SKILL = "skill"          # instructions for the skill(s) active this step
    TOOLS = "tools"          # tool stub signatures, retrieval-filtered
    WORKSPACE = "workspace"  # file tree + artifact digests, never file bodies
    MEMORY = "memory"        # long-term memory, scored
    HISTORY = "history"      # session history: tail verbatim, middle summarised


@dataclass(frozen=True)
class SlotPolicy:
    name: SlotName
    hard_floor_tokens: int
    elasticity: float
    share: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.elasticity <= 1.0:
            raise ValueError(f"{self.name}: elasticity must be in [0,1]")
        if self.share < 0:
            raise ValueError(f"{self.name}: share must be >= 0")
        if self.hard_floor_tokens < 0:
            raise ValueError(f"{self.name}: hard_floor_tokens must be >= 0")


@dataclass
class Item:
    """One evictable unit of context: a memory, a tool stub, a turn of history."""

    text: str
    tokens: int
    score: float = 0.0          # higher = keep
    pinned: bool = False        # survives eviction regardless of score
    label: str = ""             # shows up in the ledger record
    kind: str = ""

    def to_json(self) -> dict[str, object]:
        return {
            "label": self.label,
            "kind": self.kind,
            "tokens": self.tokens,
            "score": round(self.score, 4),
            "pinned": self.pinned,
        }


@dataclass
class SlotBid:
    """What a slot wants, and how it degrades if it gets less."""

    policy: SlotPolicy
    items: list[Item] = field(default_factory=list)
    # Optional degradation hook: called with the items that were evicted and
    # must return a compact stand-in (e.g. a skill index, a history summary).
    # Returning None means "drop silently".
    degrade: Callable[[Sequence[Item], int], Item | None] | None = None
    separator: str = "\n\n"

    @property
    def requested_tokens(self) -> int:
        return sum(i.tokens for i in self.items)

    @property
    def pinned_tokens(self) -> int:
        return sum(i.tokens for i in self.items if i.pinned)

    @property
    def has_content(self) -> bool:
        return bool(self.items)

    def fit(self, allowance: int, counter: Callable[[str], int]) -> "SlotFill":
        """Select items that fit in ``allowance``, then optionally degrade the rest.

        Pinned items are taken first and may exceed the allowance — that is the
        point of pinning, and the ledger reports the overrun rather than
        silently dropping instructions.
        """
        kept: list[Item] = [i for i in self.items if i.pinned]
        used = sum(i.tokens for i in kept)
        evicted: list[Item] = []

        # Highest score first; ties broken by smaller size so a budget fits more
        # useful items rather than one large one.
        for item in sorted(
            (i for i in self.items if not i.pinned),
            key=lambda i: (-i.score, i.tokens),
        ):
            if used + item.tokens <= allowance:
                kept.append(item)
                used += item.tokens
            else:
                evicted.append(item)

        stand_in: Item | None = None
        if evicted and self.degrade is not None:
            remaining = max(0, allowance - used)
            stand_in = self.degrade(evicted, remaining)
            if stand_in is not None:
                stand_in.tokens = stand_in.tokens or counter(stand_in.text)
                if used + stand_in.tokens <= allowance or not kept:
                    kept.append(stand_in)
                    used += stand_in.tokens
                else:
                    stand_in = None

        # Preserve the caller's original ordering for readability of the prompt,
        # with any stand-in appended last.
        order = {id(i): n for n, i in enumerate(self.items)}
        kept.sort(key=lambda i: order.get(id(i), len(order)))

        return SlotFill(
            policy=self.policy,
            kept=kept,
            evicted=evicted,
            stand_in=stand_in,
            allowance=allowance,
            used=used,
            text=self.separator.join(i.text for i in kept if i.text),
        )


@dataclass
class SlotFill:
    policy: SlotPolicy
    kept: list[Item]
    evicted: list[Item]
    stand_in: Item | None
    allowance: int
    used: int
    text: str

    @property
    def overflowed(self) -> bool:
        """True when pinned content alone exceeded the allowance."""
        return self.used > self.allowance

    def to_json(self) -> dict[str, object]:
        return {
            "slot": self.policy.name.value,
            "allowance": self.allowance,
            "used": self.used,
            "kept": len(self.kept),
            "evicted": len(self.evicted),
            "evicted_tokens": sum(i.tokens for i in self.evicted),
            "degraded": self.stand_in is not None,
            "overflowed": self.overflowed,
            "evicted_labels": [i.label for i in self.evicted[:8] if i.label],
        }


# Default policy set. Numbers chosen so that on a 32k window the discretionary
# pool splits roughly: tools 20%, workspace 15%, memory 15%, history 45%, and
# skills take what they need up to 5%.
DEFAULT_POLICIES: tuple[SlotPolicy, ...] = (
    SlotPolicy(SlotName.SYSTEM, hard_floor_tokens=256, elasticity=0.0, share=0.05),
    SlotPolicy(SlotName.SKILL, hard_floor_tokens=0, elasticity=0.35, share=0.05),
    SlotPolicy(SlotName.TOOLS, hard_floor_tokens=200, elasticity=0.55, share=0.20),
    SlotPolicy(SlotName.WORKSPACE, hard_floor_tokens=120, elasticity=0.45, share=0.15),
    SlotPolicy(SlotName.MEMORY, hard_floor_tokens=0, elasticity=0.90, share=0.15),
    SlotPolicy(SlotName.HISTORY, hard_floor_tokens=400, elasticity=0.75, share=0.40),
)

POLICY_BY_NAME: dict[SlotName, SlotPolicy] = {p.name: p for p in DEFAULT_POLICIES}
