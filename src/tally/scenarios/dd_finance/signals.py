"""Cross-year financial signals: checkable ground truth for the L3 task.

L3 is the judgement level — trend attribution and risk identification — and the
obvious way to score it is an LLM judge. That alone would be weak: a judge is a
measuring instrument whose own reliability has to be established, and a purely
subjective score cannot separate "wrote persuasively" from "noticed the right
thing".

But a large part of what makes a diligence finding *correct* is derivable. If
revenue rose while operating cash flow fell, that divergence is a fact about the
numbers, computable from the L1 values of consecutive fiscal years. So L3 gets
an objective backbone: a signal set per company, generated from the truth data,
against which the agent's findings are scored for **recall** (did it surface what
is there) and **false-signal rate** (did it assert what is not). The judge then
grades only what genuinely needs judgement — the quality of the attribution and
the calibration of the language.

Every signal is defined to be *material*: each carries a threshold chosen so
that ordinary year-to-year noise does not trigger it. A signal set full of
marginal triggers would make recall meaningless, since an agent could score well
by listing every possible concern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Sequence


class Severity(str, Enum):
    WATCH = "watch"        # worth a sentence
    CONCERN = "concern"    # worth investigating
    RED_FLAG = "red_flag"  # a diligence blocker until explained


@dataclass(frozen=True)
class SignalDef:
    key: str
    label: str
    severity: Severity
    question: str          # what a reader should learn from it
    requires: tuple[str, ...]
    detect: Callable[[dict[str, float], dict[str, float]], tuple[bool, dict[str, float]]]
    keywords: tuple[str, ...] = ()   # used to match an agent's prose to the signal

    def evaluate(
        self, current: dict[str, float | None], prior: dict[str, float | None]
    ) -> tuple[bool, dict[str, float]]:
        """Return (present, evidence). Absent inputs mean 'cannot tell', not 'no'."""
        for key in self.requires:
            if current.get(key) is None or prior.get(key) is None:
                return False, {}
        cur = {k: float(current[k]) for k in self.requires}  # type: ignore[arg-type]
        pri = {k: float(prior[k]) for k in self.requires}  # type: ignore[arg-type]
        return self.detect(cur, pri)


def _growth(current: float, prior: float) -> float | None:
    return None if prior == 0 else (current - prior) / abs(prior)


# --- detectors ------------------------------------------------------------
def _revenue_up_cashflow_down(cur, pri):  # noqa: ANN001
    rev = _growth(cur["revenue"], pri["revenue"])
    cfo = _growth(cur["cash_from_operations"], pri["cash_from_operations"])
    if rev is None or cfo is None:
        return False, {}
    # Revenue must be up meaningfully and cash flow down meaningfully: a flat
    # year on either side is not a divergence.
    present = rev > 0.03 and cfo < -0.05
    return present, {"revenue_growth": round(rev, 4), "cfo_growth": round(cfo, 4),
                     "divergence_pp": round(100 * (rev - cfo), 1)}


def _margin_compression(cur, pri):  # noqa: ANN001
    def margin(v):  # noqa: ANN001
        return None if v["revenue"] == 0 else (v["revenue"] - v["cost_of_revenue"]) / v["revenue"]

    now, before = margin(cur), margin(pri)
    if now is None or before is None:
        return False, {}
    delta = now - before
    # 150bp of gross-margin movement is a real change at these scales.
    return delta < -0.015, {"gross_margin": round(now, 4),
                            "prior_gross_margin": round(before, 4),
                            "change_bp": round(10_000 * delta, 0)}


def _receivables_outpacing_revenue(cur, pri):  # noqa: ANN001
    rev = _growth(cur["revenue"], pri["revenue"])
    ar = _growth(cur["accounts_receivable"], pri["accounts_receivable"])
    if rev is None or ar is None:
        return False, {}
    # Receivables growing 10pp faster than revenue suggests collection or
    # revenue-recognition pressure rather than ordinary scaling.
    present = ar - rev > 0.10 and ar > 0.05
    return present, {"revenue_growth": round(rev, 4), "receivables_growth": round(ar, 4),
                     "gap_pp": round(100 * (ar - rev), 1)}


def _negative_free_cash_flow(cur, pri):  # noqa: ANN001
    fcf = cur["cash_from_operations"] - cur["capex"]
    prior_fcf = pri["cash_from_operations"] - pri["capex"]
    return fcf < 0, {"free_cash_flow": fcf, "prior_free_cash_flow": prior_fcf,
                     "turned_negative": bool(prior_fcf >= 0 and fcf < 0)}


def _equity_decline(cur, pri):  # noqa: ANN001
    change = _growth(cur["total_equity"], pri["total_equity"])
    if change is None:
        return False, {}
    # Equity can fall for benign reasons (buybacks); 10% is where it warrants a
    # sentence explaining which.
    return change < -0.10, {"equity_change": round(change, 4),
                            "total_equity": cur["total_equity"],
                            "prior_total_equity": pri["total_equity"]}


def _revenue_decline(cur, pri):  # noqa: ANN001
    change = _growth(cur["revenue"], pri["revenue"])
    if change is None:
        return False, {}
    return change < -0.05, {"revenue_growth": round(change, 4)}


def _operating_loss(cur, pri):  # noqa: ANN001
    return cur["operating_income"] < 0, {
        "operating_income": cur["operating_income"],
        "prior_operating_income": pri["operating_income"],
        "turned_negative": bool(pri["operating_income"] >= 0 > cur["operating_income"]),
    }


def _liquidity_pressure(cur, pri):  # noqa: ANN001
    if cur["current_liabilities"] == 0:
        return False, {}
    ratio = cur["current_assets"] / cur["current_liabilities"]
    prior = (pri["current_assets"] / pri["current_liabilities"]
             if pri["current_liabilities"] else None)
    return ratio < 1.0, {"current_ratio": round(ratio, 4),
                         "prior_current_ratio": round(prior, 4) if prior else None}


def _leverage_increase(cur, pri):  # noqa: ANN001
    if cur["total_assets"] == 0 or pri["total_assets"] == 0:
        return False, {}
    now = cur["total_liabilities"] / cur["total_assets"]
    before = pri["total_liabilities"] / pri["total_assets"]
    return (now - before) > 0.05, {"liabilities_to_assets": round(now, 4),
                                   "prior": round(before, 4),
                                   "change_pp": round(100 * (now - before), 1)}


SIGNALS: tuple[SignalDef, ...] = (
    SignalDef(
        "revenue_up_cashflow_down", "Revenue grew while operating cash flow fell",
        Severity.RED_FLAG,
        "Is growth converting into cash, or into receivables and inventory?",
        ("revenue", "cash_from_operations"), _revenue_up_cashflow_down,
        keywords=("cash flow", "operating cash", "cash conversion", "diverge",
                  "revenue grew", "cash declined"),
    ),
    SignalDef(
        "receivables_outpacing_revenue", "Receivables grew much faster than revenue",
        Severity.CONCERN,
        "Are collections slowing, or is revenue being recognised earlier?",
        ("revenue", "accounts_receivable"), _receivables_outpacing_revenue,
        keywords=("receivable", "collection", "days sales", "dso", "credit terms"),
    ),
    SignalDef(
        "margin_compression", "Gross margin compressed year on year",
        Severity.CONCERN,
        "Is pricing power eroding, or input cost rising?",
        ("revenue", "cost_of_revenue"), _margin_compression,
        keywords=("gross margin", "margin compress", "margin decline", "cost of sales",
                  "pricing"),
    ),
    SignalDef(
        "negative_free_cash_flow", "Free cash flow is negative",
        Severity.CONCERN,
        "Is the business funding itself, or drawing on the balance sheet?",
        ("cash_from_operations", "capex"), _negative_free_cash_flow,
        keywords=("free cash flow", "fcf", "cash burn", "capital expenditure",
                  "outspending"),
    ),
    SignalDef(
        "revenue_decline", "Revenue declined year on year",
        Severity.CONCERN, "Is the decline volume, price, or perimeter?",
        ("revenue",), _revenue_decline,
        keywords=("revenue declined", "revenue fell", "sales decreased", "top line"),
    ),
    SignalDef(
        "operating_loss", "Operating income is negative",
        Severity.RED_FLAG, "Is the loss structural or one-off?",
        ("operating_income",), _operating_loss,
        keywords=("operating loss", "operating income was negative", "unprofitable"),
    ),
    SignalDef(
        "liquidity_pressure", "Current ratio is below 1.0",
        Severity.WATCH,
        "Can near-term obligations be met without new financing?",
        ("current_assets", "current_liabilities"), _liquidity_pressure,
        keywords=("current ratio", "liquidity", "working capital", "short-term"),
    ),
    SignalDef(
        "leverage_increase", "Liabilities grew as a share of assets",
        Severity.WATCH, "What funded the increase, and on what terms?",
        ("total_assets", "total_liabilities"), _leverage_increase,
        keywords=("leverage", "debt", "liabilities", "gearing", "borrow"),
    ),
    SignalDef(
        "equity_decline", "Shareholders' equity fell materially",
        Severity.WATCH, "Buybacks, losses, or other comprehensive income?",
        ("total_equity",), _equity_decline,
        keywords=("equity declined", "shareholders' equity", "buyback", "repurchase",
                  "accumulated deficit"),
    ),
)
SIGNALS_BY_KEY: dict[str, SignalDef] = {s.key: s for s in SIGNALS}


@dataclass
class DetectedSignal:
    key: str
    label: str
    severity: str
    question: str
    evidence: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {"key": self.key, "label": self.label, "severity": self.severity,
                "question": self.question, "evidence": self.evidence}


def detect(
    current: dict[str, float | None], prior: dict[str, float | None]
) -> list[DetectedSignal]:
    """Signals present between two consecutive fiscal years."""
    found: list[DetectedSignal] = []
    for definition in SIGNALS:
        present, evidence = definition.evaluate(current, prior)
        if present:
            found.append(DetectedSignal(
                key=definition.key, label=definition.label,
                severity=definition.severity.value, question=definition.question,
                evidence=evidence,
            ))
    return found


def detectable(
    current: dict[str, float | None], prior: dict[str, float | None]
) -> list[str]:
    """Signals whose inputs are all present, i.e. those we could have detected.

    Scoring must distinguish "the agent missed a signal" from "the data could not
    support that signal in the first place". Without this, a bank case penalises
    an agent for not reporting a current ratio it has no way to compute.
    """
    keys: list[str] = []
    for definition in SIGNALS:
        if all(current.get(k) is not None and prior.get(k) is not None
               for k in definition.requires):
            keys.append(definition.key)
    return keys
