"""Token and cost accounting, attributed along three axes.

The three axes are deliberate: the spec promises attribution "by agent / by task
/ by scenario", and the ablation study needs per-arm cost. A flat token counter
cannot answer "which sub-agent burned the free quota".

Prices are per *million* tokens, USD. Free-tier models are priced 0.0 but still
tracked in ``tokens_*`` so quota pressure stays visible: on a zero-budget run the
binding constraint is requests-per-day, not dollars.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# model id -> (usd per 1M input, usd per 1M output)
PRICE_TABLE: dict[str, tuple[float, float]] = {
    # free tiers
    "gemini-2.5-flash": (0.0, 0.0),
    "gemini-2.5-flash-lite": (0.0, 0.0),
    "glm-4-flash": (0.0, 0.0),
    "glm-4.5-flash": (0.0, 0.0),
    "llama-3.3-70b-versatile": (0.0, 0.0),
    "qwen3-8b-local": (0.0, 0.0),
    "bge-m3-local": (0.0, 0.0),
    "fake-model": (0.0, 0.0),
    # paid, used only by the strong-naked ablation arm
    "strong-paid": (5.0, 25.0),
}
DEFAULT_PRICE = (0.0, 0.0)


def price_for(model: str) -> tuple[float, float]:
    if model in PRICE_TABLE:
        return PRICE_TABLE[model]
    for known, price in PRICE_TABLE.items():
        if model.startswith(known):
            return price
    return DEFAULT_PRICE


def cost_of(model: str, tokens_in: int, tokens_out: int) -> float:
    pin, pout = price_for(model)
    return (tokens_in / 1_000_000) * pin + (tokens_out / 1_000_000) * pout


@dataclass(frozen=True)
class CostEntry:
    model: str
    provider: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    agent: str = "root"
    task: str = "-"
    scenario: str = "-"
    purpose: str = "-"
    cached: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": self.cost_usd,
            "agent": self.agent,
            "task": self.task,
            "scenario": self.scenario,
            "purpose": self.purpose,
            "cached": self.cached,
        }


@dataclass
class UsageTotals:
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    cache_hits: int = 0

    def add(self, e: CostEntry) -> None:
        self.tokens_in += e.tokens_in
        self.tokens_out += e.tokens_out
        self.cost_usd += e.cost_usd
        self.calls += 1
        if e.cached:
            self.cache_hits += 1

    @property
    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / self.calls if self.calls else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_total": self.tokens_total,
            "cost_usd": round(self.cost_usd, 6),
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
        }


class Accountant:
    """Collects CostEntry records and aggregates them on demand."""

    def __init__(self, sink: Path | None = None) -> None:
        self.entries: list[CostEntry] = []
        self._sink = sink
        if sink is not None:
            sink.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        model: str,
        provider: str,
        tokens_in: int,
        tokens_out: int,
        agent: str = "root",
        task: str = "-",
        scenario: str = "-",
        purpose: str = "-",
        cached: bool = False,
        cost_usd: float | None = None,
    ) -> CostEntry:
        # A cache hit costs nothing even though the tokens were logically consumed.
        billed = 0.0 if cached else (cost_usd if cost_usd is not None else cost_of(model, tokens_in, tokens_out))
        entry = CostEntry(
            model=model,
            provider=provider,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=billed,
            agent=agent,
            task=task,
            scenario=scenario,
            purpose=purpose,
            cached=cached,
        )
        self.entries.append(entry)
        if self._sink is not None:
            with self._sink.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry.to_json(), ensure_ascii=False) + "\n")
        return entry

    # -- aggregation -------------------------------------------------------
    def total(self) -> UsageTotals:
        return self._fold(self.entries)

    def by(self, axis: str) -> dict[str, UsageTotals]:
        if axis not in {"agent", "task", "scenario", "model", "provider", "purpose"}:
            raise ValueError(f"unsupported axis: {axis!r}")
        buckets: dict[str, list[CostEntry]] = defaultdict(list)
        for e in self.entries:
            buckets[getattr(e, axis)].append(e)
        return {k: self._fold(v) for k, v in sorted(buckets.items())}

    @staticmethod
    def _fold(entries: Iterable[CostEntry]) -> UsageTotals:
        t = UsageTotals()
        for e in entries:
            t.add(e)
        return t

    def report(self) -> dict[str, Any]:
        return {
            "total": self.total().to_json(),
            "by_purpose": {k: v.to_json() for k, v in self.by("purpose").items()},
            "by_model": {k: v.to_json() for k, v in self.by("model").items()},
            "by_agent": {k: v.to_json() for k, v in self.by("agent").items()},
        }
