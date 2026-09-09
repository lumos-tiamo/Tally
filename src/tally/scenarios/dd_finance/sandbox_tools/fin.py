"""In-sandbox financial arithmetic with mandatory provenance.

Every function returns the value *together with* the numerator, the denominator
and the field keys they came from. That shape is not decoration — the eval's
calculation-consistency metric re-derives the value from those two numbers, so a
result that cannot show its inputs is scored as unverifiable.

The abstention discipline is enforced here rather than left to the agent:
:func:`ratio` returns ``None`` with a reason when an input is missing or a
denominator is zero. A tool that silently returns 0.0 for "revenue not found"
teaches the model to report zeros, and a zero gross margin is a much worse answer
than an admitted gap.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

RATIO_DEFS = {
    "gross_margin": (("revenue", "cost_of_revenue"), ("revenue",), "ratio"),
    "operating_margin": (("operating_income",), ("revenue",), "ratio"),
    "net_margin": (("net_income",), ("revenue",), "ratio"),
    "roe": (("net_income",), ("total_equity",), "ratio"),
    "roa": (("net_income",), ("total_assets",), "ratio"),
    "current_ratio": (("current_assets",), ("current_liabilities",), "ratio"),
    "receivables_turnover": (("revenue",), ("accounts_receivable",), "ratio"),
    "free_cash_flow": (("cash_from_operations",), ("capex",), "difference"),
}


def _combine(values: dict, keys: tuple) -> float | None:
    parts = [values.get(k) for k in keys]
    if any(p is None for p in parts):
        return None
    nums = [float(p) for p in parts]
    return nums[0] - sum(nums[1:]) if len(nums) > 1 else nums[0]


def ratio(name: str, values: dict) -> dict:
    """Compute one of the eight defined L2 metrics from L1 values.

    ``values`` maps L1 field keys to numbers (or None). Returns the standard
    provenance envelope; abstains rather than guessing.
    """
    spec = RATIO_DEFS.get(name)
    if spec is None:
        return {"key": name, "value": None, "reason": "unknown_ratio",
                "known": sorted(RATIO_DEFS)}
    num_keys, den_keys, op = spec
    num = _combine(values, num_keys)
    den = _combine(values, den_keys)
    envelope = {
        "key": name, "op": op,
        "numerator": num, "denominator": den,
        "numerator_keys": list(num_keys), "denominator_keys": list(den_keys),
    }
    if num is None or den is None:
        missing = [k for k in (*num_keys, *den_keys) if values.get(k) is None]
        return {**envelope, "value": None, "reason": "missing_inputs", "missing": missing}
    if op == "ratio":
        if den == 0:
            return {**envelope, "value": None, "reason": "zero_denominator"}
        return {**envelope, "value": num / den, "reason": ""}
    return {**envelope, "value": num - den, "reason": ""}


def all_ratios(values: dict) -> dict:
    """Compute all eight, abstaining individually where inputs are absent."""
    return {name: ratio(name, values) for name in RATIO_DEFS}


def rescale(value: float, scale: str) -> dict:
    """Convert a figure stated 'in thousands/millions/billions' to units.

    Use this instead of multiplying by hand: a mis-scaled figure is off by three
    orders of magnitude and passes every sanity check that only looks at digits.
    """
    factors = {"units": 1.0, "thousands": 1e3, "millions": 1e6, "billions": 1e9}
    key = str(scale or "units").strip().lower()
    if key not in factors:
        return {"value": None, "reason": f"unknown scale {scale!r}",
                "known": sorted(factors)}
    return {"value": float(value) * factors[key], "input": float(value),
            "scale": key, "factor": factors[key]}


def growth(current: float | None, prior: float | None) -> dict:
    """Year-over-year growth with provenance, abstaining on a zero base."""
    if current is None or prior is None:
        return {"value": None, "reason": "missing_inputs",
                "numerator": None, "denominator": prior}
    if float(prior) == 0:
        return {"value": None, "reason": "zero_denominator",
                "numerator": float(current), "denominator": 0.0}
    return {
        "value": (float(current) - float(prior)) / abs(float(prior)),
        "numerator": float(current) - float(prior),
        "denominator": abs(float(prior)),
        "reason": "",
    }


def check_balance_sheet(values: dict, *, tolerance: float = 0.01) -> dict:
    """Assets vs Liabilities + Equity, as a self-check before reporting.

    Worth running on your own extraction: if it does not balance, at least one of
    the three figures came from the wrong column or the wrong context, and it is
    cheaper to find that here than to report it.
    """
    a, l, e = values.get("total_assets"), values.get("total_liabilities"), values.get("total_equity")
    if a is None or l is None or e is None:
        return {"checked": False, "reason": "one or more components absent"}
    a, l, e = float(a), float(l), float(e)
    if a == 0:
        return {"checked": False, "reason": "zero assets"}
    err = abs(a - (l + e)) / abs(a)
    return {
        "checked": True, "assets": a, "liabilities_plus_equity": l + e,
        "relative_error": err, "balances": err <= tolerance,
        "note": ("a residual close to a noncontrolling-interest line is expected: "
                 "StockholdersEquity is parent-only") if err > tolerance else "",
    }


def summarise(values: dict) -> dict:
    """Digest of an extraction: what is present, what is missing. Print this."""
    present = {k: v for k, v in values.items() if v is not None}
    return {
        "fields_present": len(present),
        "fields_missing": sorted(k for k, v in values.items() if v is None),
        "magnitudes": {k: f"{v:,.0f}" for k, v in sorted(present.items())},
    }


def load_values(path: str) -> dict:
    """Read a previously saved L1 extraction from the workspace."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and all(
        isinstance(v, dict) and "value" in v for v in data.values() if v is not None
    ):
        # Accept the full {value, page, quote} contract as well as bare numbers.
        return {k: (v or {}).get("value") for k, v in data.items()}
    return data
