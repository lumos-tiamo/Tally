"""US-GAAP concept mapping: XBRL company facts -> the locked L1 field set.

This layer is the reason the ground-truth set can be built semi-automatically
instead of hand-labelled, and it is almost entirely made of corrections for ways
the raw data misleads you.

The four that matter
--------------------
**1. One field, many tags.** ASC 606 split ``Revenues`` into
``RevenueFromContractWithCustomerExcludingAssessedTax`` and its *Including*
sibling; filers use one, occasionally neither. Resolution walks a preference
list per field (see :mod:`.fields`) and records which tag actually answered, so
a coverage report can show the real distribution rather than assuming.

**2. ``fy`` is the filing's fiscal year, not the fact's.** A FY2024 10-K carries
FY2023 and FY2022 comparatives, and every one of those facts is tagged
``fy: 2024``. Selecting on ``fy`` therefore silently mixes three years. Facts are
matched on their **period end date** against the fiscal-year-end taken from the
filing index instead.

**3. Duration facts are not all annual.** A quarterly revenue fact has the same
shape as an annual one. Only the span distinguishes them, so duration facts are
filtered to a 340–400 day window before anything else happens. This is the
easiest way to build a ground-truth set that is confidently wrong.

**4. Restatements mean the same period appears twice.** When two facts cover the
same period with different values, the later ``filed`` date wins — that is the
company's current belief about its own history. Both are kept in
:attr:`Resolution.superseded` so a restatement is visible rather than silently
resolved.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from tally.scenarios.dd_finance.fields import (
    L1_BY_KEY,
    L1_FIELDS,
    FieldDef,
    Period,
    Sign,
)

ANNUAL_MIN_DAYS = 340
ANNUAL_MAX_DAYS = 400
# How far a fact's period end may sit from the fiscal-year end and still count.
# 52/53-week filers drift by a few days year to year; 7 covers that without
# reaching into an adjacent quarter.
FY_END_TOLERANCE_DAYS = 7
ACCEPTED_FORMS = ("10-K", "10-K/A", "20-F", "40-F")


def _date(raw: str | None) -> dt.date | None:
    if not raw:
        return None
    try:
        return dt.date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


@dataclass(frozen=True)
class Fact:
    concept: str
    value: float
    unit: str
    end: dt.date
    start: dt.date | None = None
    accn: str = ""
    form: str = ""
    filed: dt.date | None = None
    fy: int | None = None
    fp: str = ""
    frame: str = ""

    @property
    def span_days(self) -> int | None:
        if self.start is None:
            return None
        return (self.end - self.start).days

    @property
    def is_annual_duration(self) -> bool:
        span = self.span_days
        return span is not None and ANNUAL_MIN_DAYS <= span <= ANNUAL_MAX_DAYS

    @property
    def is_instant(self) -> bool:
        return self.start is None

    def to_json(self) -> dict[str, Any]:
        return {
            "concept": self.concept,
            "value": self.value,
            "unit": self.unit,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat(),
            "span_days": self.span_days,
            "accn": self.accn,
            "form": self.form,
            "filed": self.filed.isoformat() if self.filed else None,
            "fy": self.fy,
            "fp": self.fp,
        }


@dataclass
class Resolution:
    key: str
    value: float | None
    concept: str = ""
    fact: Fact | None = None
    reason: str = ""
    candidates_tried: tuple[str, ...] = ()
    superseded: list[Fact] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return self.value is not None

    @property
    def restated(self) -> bool:
        return bool(self.superseded)

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "concept": self.concept,
            "reason": self.reason,
            "restated": self.restated,
            "candidates_tried": list(self.candidates_tried),
            "fact": self.fact.to_json() if self.fact else None,
            "superseded": [f.to_json() for f in self.superseded[:3]],
        }


def parse_company_facts(payload: dict[str, Any]) -> dict[str, list[Fact]]:
    """Flatten a ``companyfacts`` document into ``concept -> [Fact]``.

    Only ``us-gaap`` and USD units are kept. ``dei`` facts carry entity metadata
    rather than financials, and non-USD units on a US filer are almost always a
    per-share or ratio concept we do not use.
    """
    out: dict[str, list[Fact]] = {}
    facts = (payload or {}).get("facts") or {}
    for taxonomy in ("us-gaap",):
        for concept, body in (facts.get(taxonomy) or {}).items():
            for unit, entries in ((body or {}).get("units") or {}).items():
                if unit != "USD":
                    continue
                bucket = out.setdefault(concept, [])
                for e in entries or []:
                    end = _date(e.get("end"))
                    if end is None or e.get("val") is None:
                        continue
                    try:
                        value = float(e["val"])
                    except (TypeError, ValueError):
                        continue
                    bucket.append(
                        Fact(
                            concept=concept, value=value, unit=unit, end=end,
                            start=_date(e.get("start")), accn=str(e.get("accn", "")),
                            form=str(e.get("form", "")), filed=_date(e.get("filed")),
                            fy=int(e["fy"]) if e.get("fy") is not None else None,
                            fp=str(e.get("fp", "")), frame=str(e.get("frame", "")),
                        )
                    )
    return out


class ConceptResolver:
    """Resolves L1 fields for one company-year from parsed company facts."""

    def __init__(
        self,
        facts: dict[str, list[Fact]],
        *,
        accepted_forms: Sequence[str] = ACCEPTED_FORMS,
        fy_end_tolerance_days: int = FY_END_TOLERANCE_DAYS,
    ) -> None:
        self.facts = facts
        self.accepted_forms = tuple(accepted_forms)
        self.tolerance = fy_end_tolerance_days

    # -- candidate filtering -----------------------------------------------
    def _candidates(self, concept: str, fdef: FieldDef, fy_end: dt.date) -> list[Fact]:
        pool = self.facts.get(concept) or []
        kept: list[Fact] = []
        for f in pool:
            if self.accepted_forms and f.form not in self.accepted_forms:
                continue
            if abs((f.end - fy_end).days) > self.tolerance:
                continue
            if fdef.period is Period.INSTANT:
                if not f.is_instant:
                    continue
            else:
                if not f.is_annual_duration:
                    continue
            kept.append(f)
        return kept

    @staticmethod
    def _pick(candidates: Sequence[Fact]) -> tuple[Fact | None, list[Fact]]:
        """Latest filing wins; the rest are recorded as superseded."""
        if not candidates:
            return None, []
        ordered = sorted(
            candidates,
            key=lambda f: (f.filed or dt.date.min, f.accn),
            reverse=True,
        )
        winner = ordered[0]
        # Only count a differing value as a restatement; identical repeats of the
        # same number across filings are noise, not news.
        superseded = [
            f for f in ordered[1:] if abs(f.value - winner.value) > 1e-6
        ]
        return winner, superseded

    # -- public API --------------------------------------------------------
    def resolve_field(self, key: str, fy_end: dt.date) -> Resolution:
        fdef = L1_BY_KEY[key]
        tried: list[str] = []
        for concept in fdef.concepts:
            tried.append(concept)
            winner, superseded = self._pick(self._candidates(concept, fdef, fy_end))
            if winner is None:
                continue
            value = winner.value
            if fdef.sign is Sign.POSITIVE:
                value = abs(value)
            return Resolution(
                key=key, value=value, concept=concept, fact=winner,
                candidates_tried=tuple(tried), superseded=superseded,
            )
        return Resolution(
            key=key, value=None, reason="not_disclosed", candidates_tried=tuple(tried)
        )

    def resolve_all(
        self, fy_end: dt.date, *, keys: Iterable[str] | None = None
    ) -> dict[str, Resolution]:
        wanted = list(keys) if keys is not None else [f.key for f in L1_FIELDS]
        return {k: self.resolve_field(k, fy_end) for k in wanted}

    def values(self, fy_end: dt.date) -> dict[str, float | None]:
        return {k: r.value for k, r in self.resolve_all(fy_end).items()}

    # -- diagnostics -------------------------------------------------------
    def coverage(self, fy_end: dt.date) -> dict[str, Any]:
        res = self.resolve_all(fy_end)
        resolved = [k for k, r in res.items() if r.resolved]
        return {
            "fy_end": fy_end.isoformat(),
            "resolved": len(resolved),
            "total": len(res),
            "coverage": round(len(resolved) / len(res), 4) if res else 0.0,
            "missing": [k for k, r in res.items() if not r.resolved],
            "restated": [k for k, r in res.items() if r.restated],
            "concept_used": {k: r.concept for k, r in res.items() if r.resolved},
        }


# Concepts that complete the equity section. These are *not* L1 fields: L1's
# ``total_equity`` is deliberately parent-only so that ROE stays consistent with
# ``NetIncomeLoss``, which also excludes noncontrolling interests. But the
# balance sheet does not balance without the rest of the section, so the identity
# check resolves these separately.
IDENTITY_CONCEPTS: dict[str, tuple[str, ...]] = {
    "minority_interest": (
        "MinorityInterest",
        # Some filers (Uber among them) tag the non-redeemable portion explicitly
        # instead of using the umbrella MinorityInterest concept.
        "NonredeemableNoncontrollingInterest",
    ),
    "redeemable_nci": (
        "RedeemableNoncontrollingInterestEquityCarryingAmount",
        "RedeemableNoncontrollingInterestEquityFairValue",
    ),
    "temporary_equity": (
        "TemporaryEquityCarryingAmountAttributableToParent",
        "TemporaryEquityCarryingAmountIncludingPortionAttributableToNoncontrollingInterests",
    ),
    "equity_including_nci": (
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    # The filer's own total of the right-hand side. When present this is the
    # strongest available cross-check: it must equal Assets by construction.
    "liabilities_and_equity": ("LiabilitiesAndStockholdersEquity",),
}


def resolve_identity_components(
    facts: dict[str, list[Fact]],
    fy_end: dt.date,
    *,
    accepted_forms: Sequence[str] = ACCEPTED_FORMS,
    tolerance_days: int = FY_END_TOLERANCE_DAYS,
) -> dict[str, float | None]:
    """Resolve the equity-section components used by the identity check.

    All of these are instant concepts, so the duration filtering in
    :class:`ConceptResolver` does not apply and a small dedicated resolver is
    clearer than bending the field-driven one.
    """
    out: dict[str, float | None] = {}
    for name, candidates in IDENTITY_CONCEPTS.items():
        value: float | None = None
        for concept in candidates:
            pool = [
                f
                for f in (facts.get(concept) or [])
                if f.is_instant
                and f.form in accepted_forms
                and abs((f.end - fy_end).days) <= tolerance_days
            ]
            if not pool:
                continue
            winner = sorted(pool, key=lambda f: (f.filed or dt.date.min, f.accn))[-1]
            value = winner.value
            break
        out[name] = value
    return out


def concept_usage_histogram(resolutions: Iterable[dict[str, Resolution]]) -> dict[str, dict[str, int]]:
    """Which tag actually answered, per field, across a corpus.

    This is the evidence that the mapping layer is doing work: if every filer
    used the same tag it would be a lookup table, and the distribution shows they
    do not.
    """
    hist: dict[str, dict[str, int]] = {}
    for res in resolutions:
        for key, r in res.items():
            if not r.resolved:
                continue
            hist.setdefault(key, {})
            hist[key][r.concept] = hist[key].get(r.concept, 0) + 1
    return hist
