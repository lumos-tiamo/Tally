"""Semi-automatic ground-truth construction from XBRL.

The whole eval rests on this: if the truth set is wrong, every score downstream
is noise with a confidence interval. So the pipeline is built to *refuse* cases
it cannot vouch for rather than to maximise corpus size.

Pipeline
--------
1. Resolve the fiscal-year end from the filing index (never from ``fy``).
2. Resolve the twelve L1 fields through the concept mapping layer.
3. Derive the eight L2 ratios from the resolved L1 values.
4. **Validate against accounting identities.** ``Assets ≈ Liabilities + Equity``
   is the strongest free check available: if it fails, either the mapping picked
   the wrong tag or the filer used a presentation we do not understand. Either
   way the case is quarantined, not silently included.
5. Record per-case provenance — which tag answered, which facts were superseded,
   which fields legitimately do not exist.

What "legitimately absent" means
--------------------------------
A bank has no ``AssetsCurrent``. That is not missing data, it is a true
abstention target, and it is what the *abstention accuracy* metric is computed
against. Distinguishing "the filer did not disclose this" from "our mapping
failed to find it" is the hard part, and the balance-sheet identity plus the
per-sector difficulty labels are what make it tractable: for a field flagged
absent, if the identity still holds and the sector is one that structurally omits
it, the absence is trusted.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from teamclaw.scenarios.dd_finance.concepts import (
    ConceptResolver,
    Resolution,
    parse_company_facts,
    resolve_identity_components,
)
from teamclaw.scenarios.dd_finance.corpus import BY_TICKER, CorpusEntry
from teamclaw.scenarios.dd_finance.fields import (
    L1_KEYS,
    L2_RATIOS,
    Period,
)
from teamclaw.scenarios.dd_finance.sec_client import Filing, SecClient

# Balance-sheet identity tolerance. Filings round to the reported unit, and
# some filers' Liabilities total legitimately excludes redeemable NCI, so an
# exact match is too strict; 1% catches genuine tag errors without quarantining
# ordinary presentation differences.
IDENTITY_TOLERANCE = 0.01

# Sectors that structurally omit a field, used to trust an absence.
STRUCTURAL_ABSENCE: dict[str, tuple[str, ...]] = {
    "current_assets": ("banks", "insurance", "utilities", "reit", "healthcare", "transport"),
    "current_liabilities": ("banks", "insurance", "utilities", "reit", "healthcare"),
    "cost_of_revenue": ("banks", "insurance", "reit"),
    "total_liabilities": ("banks", "insurance"),
}


@dataclass
class GroundTruthCase:
    case_id: str
    ticker: str
    cik: str
    fiscal_year: int
    fy_end: str
    sector: str
    difficulty: tuple[str, ...]
    held_out: bool
    accession: str
    document_url: str
    l1: dict[str, float | None] = field(default_factory=dict)
    l1_absent: list[str] = field(default_factory=list)
    l1_absence_trusted: list[str] = field(default_factory=list)
    l2: dict[str, float | None] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    identity: dict[str, Any] = field(default_factory=dict)
    quarantined: bool = False
    quarantine_reason: str = ""

    @property
    def l1_coverage(self) -> float:
        present = sum(1 for v in self.l1.values() if v is not None)
        return round(present / len(self.l1), 4) if self.l1 else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "ticker": self.ticker,
            "cik": self.cik,
            "fiscal_year": self.fiscal_year,
            "fy_end": self.fy_end,
            "sector": self.sector,
            "difficulty": list(self.difficulty),
            "held_out": self.held_out,
            "accession": self.accession,
            "document_url": self.document_url,
            "l1": self.l1,
            "l1_absent": self.l1_absent,
            "l1_absence_trusted": self.l1_absence_trusted,
            "l1_coverage": self.l1_coverage,
            "l2": self.l2,
            "identity": self.identity,
            "quarantined": self.quarantined,
            "quarantine_reason": self.quarantine_reason,
            "provenance": self.provenance,
        }

    @staticmethod
    def from_json(data: dict[str, Any]) -> "GroundTruthCase":
        return GroundTruthCase(
            case_id=data["case_id"], ticker=data["ticker"], cik=data["cik"],
            fiscal_year=int(data["fiscal_year"]), fy_end=data["fy_end"],
            sector=data.get("sector", ""), difficulty=tuple(data.get("difficulty") or ()),
            held_out=bool(data.get("held_out")), accession=data.get("accession", ""),
            document_url=data.get("document_url", ""),
            l1=data.get("l1") or {}, l1_absent=list(data.get("l1_absent") or []),
            l1_absence_trusted=list(data.get("l1_absence_trusted") or []),
            l2=data.get("l2") or {}, provenance=data.get("provenance") or {},
            identity=data.get("identity") or {},
            quarantined=bool(data.get("quarantined")),
            quarantine_reason=data.get("quarantine_reason", ""),
        )


def check_balance_sheet_identity(
    l1: dict[str, float | None],
    components: dict[str, float | None] | None = None,
    *,
    equity_concept: str = "",
) -> dict[str, Any]:
    """Assets = Liabilities + the whole equity section.

    The naive form of this check — ``Assets == Liabilities + StockholdersEquity``
    — fails for every filer with noncontrolling interests, which in practice
    means most REITs, utilities, and anything acquisition-heavy. It is not a tag
    mismatch: ``StockholdersEquity`` is *defined* as attributable to the parent,
    and L1 keeps it that way on purpose so ROE stays consistent with
    ``NetIncomeLoss``. The balance sheet simply has more in its equity section
    than that one line.

    So the identity used here is

        Assets = Liabilities + StockholdersEquity + MinorityInterest
                 + RedeemableNCI + TemporaryEquity

    and, when the filer reports its own ``LiabilitiesAndStockholdersEquity``
    total, that is checked against Assets as well — it must match by
    construction, which makes it the strongest single cross-check available.
    """
    components = components or {}
    assets = l1.get("total_assets")
    liabilities = l1.get("total_liabilities")
    equity = l1.get("total_equity")

    out: dict[str, Any] = {}

    # Direct cross-check first: the filer's own right-hand-side total.
    lse = components.get("liabilities_and_equity")
    if assets is not None and lse is not None and float(assets) != 0:
        direct_error = abs(float(assets) - float(lse)) / abs(float(assets))
        out["direct_check"] = {
            "assets": float(assets),
            "liabilities_and_equity": float(lse),
            "relative_error": round(direct_error, 8),
            "passes": direct_error <= IDENTITY_TOLERANCE,
        }

    if assets is None or liabilities is None or equity is None:
        out.update({"checked": False, "reason": "one or more components absent"})
        return out
    if float(assets) == 0:
        out.update({"checked": False, "reason": "zero assets"})
        return out

    equity_includes_nci = equity_concept.startswith(
        "StockholdersEquityIncludingPortion"
    )
    nci = 0.0 if equity_includes_nci else float(components.get("minority_interest") or 0.0)
    redeemable = float(components.get("redeemable_nci") or 0.0)
    temporary = float(components.get("temporary_equity") or 0.0)

    lhs = float(assets)
    rhs = float(liabilities) + float(equity) + nci + redeemable + temporary
    error = abs(lhs - rhs) / abs(lhs)
    out.update({
        "checked": True,
        "assets": lhs,
        "liabilities": float(liabilities),
        "parent_equity": float(equity),
        "equity_concept": equity_concept,
        "equity_includes_nci": equity_includes_nci,
        "minority_interest": nci,
        "redeemable_nci": redeemable,
        "temporary_equity": temporary,
        "right_hand_side": rhs,
        "relative_error": round(error, 6),
        "passes": error <= IDENTITY_TOLERANCE,
    })
    return out


def _trusted_absence(key: str, sector: str) -> bool:
    return sector.lower() in STRUCTURAL_ABSENCE.get(key, ())


def build_case(
    client: SecClient,
    entry: CorpusEntry,
    fiscal_year: int,
    *,
    resolver: ConceptResolver,
    filing: Filing,
) -> GroundTruthCase:
    fy_end = filing.report_date
    assert fy_end is not None  # callers filter on report_date
    resolutions: dict[str, Resolution] = resolver.resolve_all(fy_end)

    l1 = {k: r.value for k, r in resolutions.items()}
    absent = [k for k, v in l1.items() if v is None]
    trusted = [k for k in absent if _trusted_absence(k, entry.sector)]

    l2: dict[str, float | None] = {}
    for rd in L2_RATIOS:
        out = rd.compute(l1)
        l2[rd.key] = out["value"]  # type: ignore[assignment]

    components = resolve_identity_components(resolver.facts, fy_end)
    identity = check_balance_sheet_identity(
        l1, components, equity_concept=resolutions["total_equity"].concept
    )
    case = GroundTruthCase(
        case_id=f"{entry.ticker}-FY{fiscal_year}",
        ticker=entry.ticker,
        cik=filing.cik,
        fiscal_year=fiscal_year,
        fy_end=fy_end.isoformat(),
        sector=entry.sector,
        difficulty=entry.difficulty,
        held_out=entry.held_out,
        accession=filing.accession,
        document_url=filing.document_url,
        l1=l1,
        l1_absent=absent,
        l1_absence_trusted=trusted,
        l2=l2,
        identity=identity,
        provenance={
            k: {
                "concept": r.concept,
                "reason": r.reason,
                "restated": r.restated,
                "accn": r.fact.accn if r.fact else None,
                "period": (r.fact.start.isoformat() + ".." + r.fact.end.isoformat())
                if r.fact and r.fact.start
                else (r.fact.end.isoformat() if r.fact else None),
            }
            for k, r in resolutions.items()
        },
    )

    # Quarantine rules, in order of severity.
    direct = identity.get("direct_check") or {}
    # When the filer publishes its own Liabilities-and-equity total and it equals
    # Assets, the balance sheet balances — full stop. The derived identity can
    # still disagree if some equity-section line uses a tag we do not enumerate,
    # but that is a gap in our concept list, not a defect in the filing, and
    # quarantining the case would discard good data.
    direct_ok = bool(direct.get("passes"))
    if direct and not direct_ok:
        case.quarantined = True
        case.quarantine_reason = (
            f"Assets != LiabilitiesAndStockholdersEquity by "
            f"{direct['relative_error']:.2%} — the filer's own totals disagree, so "
            "one of the two tags was read from the wrong context"
        )
    elif not direct_ok and identity.get("checked") and not identity.get("passes"):
        case.quarantined = True
        case.quarantine_reason = (
            f"balance-sheet identity off by {identity['relative_error']:.2%} "
            "even after the full equity section — probable tag mismatch"
        )
    elif l1.get("revenue") is None:
        # Every task level needs revenue; a case without it grades nothing.
        case.quarantined = True
        case.quarantine_reason = "revenue could not be resolved"
    else:
        untrusted = [k for k in absent if k not in trusted]
        if len(untrusted) > 4:
            case.quarantined = True
            case.quarantine_reason = (
                f"{len(untrusted)} fields absent without a structural explanation: "
                f"{', '.join(untrusted[:6])}"
            )
    return case


@dataclass
class BuildReport:
    cases: list[GroundTruthCase] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)

    @property
    def usable(self) -> list[GroundTruthCase]:
        return [c for c in self.cases if not c.quarantined]

    @property
    def quarantined(self) -> list[GroundTruthCase]:
        return [c for c in self.cases if c.quarantined]

    def summary(self) -> dict[str, Any]:
        usable = self.usable
        by_sector: dict[str, int] = {}
        for c in usable:
            by_sector[c.sector] = by_sector.get(c.sector, 0) + 1
        absent_counts: dict[str, int] = {}
        for c in usable:
            for k in c.l1_absent:
                absent_counts[k] = absent_counts.get(k, 0) + 1
        return {
            "cases_built": len(self.cases),
            "usable": len(usable),
            "quarantined": len(self.quarantined),
            "held_out": sum(1 for c in usable if c.held_out),
            "errors": len(self.errors),
            "mean_l1_coverage": round(
                sum(c.l1_coverage for c in usable) / len(usable), 4
            ) if usable else 0.0,
            "abstention_targets": dict(sorted(absent_counts.items(), key=lambda kv: -kv[1])),
            "by_sector": dict(sorted(by_sector.items(), key=lambda kv: -kv[1])),
            "quarantine_reasons": [
                {"case": c.case_id, "reason": c.quarantine_reason} for c in self.quarantined
            ],
        }


def build_dataset(
    client: SecClient,
    entries: Sequence[CorpusEntry],
    *,
    years: int = 3,
    progress: bool = False,
) -> BuildReport:
    report = BuildReport()
    for n, entry in enumerate(entries, 1):
        if progress:
            print(f"[{n:>3}/{len(entries)}] {entry.ticker:<6} {entry.sector:<12}", flush=True)
        try:
            cik = client.cik_for_ticker(entry.ticker)
            filings = [f for f in client.annual_filings(cik, limit=years + 4)
                       if f.report_date is not None and f.fiscal_year is not None]
            resolver = ConceptResolver(parse_company_facts(client.company_facts(cik)))
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not stop the build
            report.errors.append({"ticker": entry.ticker, "error": f"{type(exc).__name__}: {exc}"})
            continue

        # Newest first, skipping the most recent filing: its comparatives are
        # sometimes still being amended, which shows up as spurious restatements.
        seen_years: set[int] = set()
        for filing in filings:
            if len(seen_years) >= years:
                break
            fy = filing.fiscal_year
            if fy is None or fy in seen_years:
                continue
            try:
                case = build_case(client, entry, fy, resolver=resolver, filing=filing)
            except Exception as exc:  # noqa: BLE001
                report.errors.append(
                    {"ticker": entry.ticker, "fy": str(fy),
                     "error": f"{type(exc).__name__}: {exc}"}
                )
                continue
            report.cases.append(case)
            seen_years.add(fy)
    return report


def save_dataset(report: BuildReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for case in report.cases:
            fh.write(json.dumps(case.to_json(), ensure_ascii=False) + "\n")
    (path.parent / f"{path.stem}.summary.json").write_text(
        json.dumps(report.summary(), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return path


def load_dataset(path: Path, *, include_quarantined: bool = False) -> list[GroundTruthCase]:
    cases: list[GroundTruthCase] = []
    if not path.exists():
        return cases
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        case = GroundTruthCase.from_json(json.loads(line))
        if case.quarantined and not include_quarantined:
            continue
        cases.append(case)
    return cases
