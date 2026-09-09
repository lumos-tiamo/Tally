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

# Two separate facts, kept separate.
#
# A model has a list price. A *route* decides whether it is charged. Mixing the
# two into one table meant ``gemini-2.5-flash`` was stored at 0.0 because it has
# a free tier, and a paid gateway serving the same model then reported $0.00 for
# a run that actually spent — which is worse than reporting nothing, because it
# reads as a measurement.
#
# Prices are published list prices in USD per million tokens, and therefore
# *estimates*: a gateway marks up or discounts and this code cannot know which.
# Token counts come from the provider's own usage block and are exact. Anywhere
# a figure has to be defensible, quote tokens.
LIST_PRICES: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "glm-4-flash": (0.10, 0.10),
    "glm-4.5-flash": (0.10, 0.10),
    "glm-5": (0.60, 2.00),
    "deepseek-v4-flash": (0.10, 0.30),
    "deepseek-v4-pro": (0.55, 2.20),
    "qwen3.6-35b-a3b": (0.20, 0.60),
    "Kimi-K2.6": (0.55, 2.20),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5.1": (1.25, 10.00),
    "claude-opus-4-6": (5.00, 25.00),
    "llama-3.3-70b-versatile": (0.60, 0.80),
    "llama-3.3-70b": (0.60, 0.80),
    "qwen3:8b": (0.0, 0.0),
    "bge-m3-local": (0.0, 0.0),
    "fake-model": (0.0, 0.0),
    "strong-paid": (5.00, 25.00),
}

# Providers that bill. Everything else is a free tier or local, and costs zero
# however the model is priced elsewhere.
BILLING_PROVIDERS = ("relay", "strong")

# Used when a billing route serves a model with no listed price. Deliberately
# mid-range rather than zero: an unpriced model should be over-reported, because
# a cost estimate that silently rounds to zero is the failure worth avoiding.
UNKNOWN_PAID_PRICE = (1.0, 4.0)

DEFAULT_PRICE = (0.0, 0.0)
PRICES_ARE_ESTIMATES = True

# Back-compatible alias; LIST_PRICES is the definition.
PRICE_TABLE = LIST_PRICES


def is_billed(provider: str) -> bool:
    return any(provider.startswith(prefix) for prefix in BILLING_PROVIDERS)


def price_for(model: str, provider: str = "") -> tuple[float, float]:
    """List price for a model on a route, in USD per million tokens.

    Zero for a free-tier or local route whatever the model, and a positive
    estimate for a billing route even when the model is unrecognised.
    """
    if not is_billed(provider):
        return DEFAULT_PRICE
    if model in LIST_PRICES:
        return LIST_PRICES[model]
    # Longest prefix wins, so `gemini-2.5-flash-lite` is not priced as
    # `gemini-2.5-flash`.
    for known in sorted(LIST_PRICES, key=len, reverse=True):
        if model.startswith(known):
            return LIST_PRICES[known]
    return UNKNOWN_PAID_PRICE


def cost_of(model: str, tokens_in: int, tokens_out: int, provider: str = "") -> float:
    pin, pout = price_for(model, provider)
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
        billed = 0.0 if cached else (
            cost_usd if cost_usd is not None
            else cost_of(model, tokens_in, tokens_out, provider)
        )
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
        billed = [e for e in self.entries if is_billed(e.provider) and not e.cached]
        return {
            "total": self.total().to_json(),
            "cost_note": (
                "Dollar figures are list-price estimates for billed routes, not "
                "invoiced amounts — a gateway marks up or discounts and this code "
                "cannot know which. Token counts come from the provider's own "
                "usage block and are exact."
                if billed else
                "No billed calls in this run; every dollar figure is zero because "
                "nothing was charged, not because pricing is unknown."
            ),
            "billed_calls": len(billed),
            "by_purpose": {k: v.to_json() for k, v in self.by("purpose").items()},
            "by_model": {k: v.to_json() for k, v in self.by("model").items()},
            "by_agent": {k: v.to_json() for k, v in self.by("agent").items()},
        }
