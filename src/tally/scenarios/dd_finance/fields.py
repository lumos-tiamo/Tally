"""The L1 field set and L2 ratio set, locked.

These are frozen at phase 0 for a reason stated in the spec: if the field list
moves, the ground-truth set has to be rebuilt and no metric compares across
phases. Adding a field is a deliberate act, not a convenience.

Each field carries three things the eval needs:

``period``
    ``INSTANT`` for balance-sheet items, ``DURATION`` for income-statement and
    cash-flow items. This is not cosmetic. XBRL facts for instant concepts have
    only an ``end`` date, while duration concepts have ``start`` *and* ``end`` —
    and a duration fact whose window is a quarter looks exactly like an annual
    one unless you check the span. Reading a Q4 revenue as FY revenue is the
    single easiest way to build a ground-truth set that is quietly wrong.

``sign``
    Whether the reported value is expected positive. Capex is reported as a
    positive outflow in ``PaymentsToAcquire*`` concepts but appears negative in
    some presentations; normalising the sign at extraction time keeps every
    downstream formula honest.

``concepts``
    Candidate us-gaap tags in preference order — see
    :mod:`tally.scenarios.dd_finance.concepts`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Period(str, Enum):
    INSTANT = "instant"      # balance sheet: a point in time
    DURATION = "duration"    # income / cash flow: a span


class Sign(str, Enum):
    POSITIVE = "positive"    # normalise to >= 0
    NATURAL = "natural"      # keep whatever the filing reports (may be a loss)


@dataclass(frozen=True)
class FieldDef:
    key: str
    label: str
    period: Period
    concepts: tuple[str, ...]
    sign: Sign = Sign.NATURAL
    statement: str = ""
    unit: str = "USD"
    notes: str = ""

    def to_json(self) -> dict[str, object]:
        return {
            "key": self.key,
            "label": self.label,
            "period": self.period.value,
            "statement": self.statement,
            "unit": self.unit,
            "concepts": list(self.concepts),
            "sign": self.sign.value,
        }


# --- L1: the twelve extraction targets -----------------------------------
L1_FIELDS: tuple[FieldDef, ...] = (
    FieldDef(
        key="revenue", label="Total revenue", period=Period.DURATION,
        statement="income", sign=Sign.POSITIVE,
        concepts=(
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
            "SalesRevenueGoodsNet",
            "RevenuesNetOfInterestExpense",
        ),
        notes="ASC 606 split the old Revenues tag into two 'assessed tax' variants; "
              "filers use one or the other, rarely both.",
    ),
    FieldDef(
        key="cost_of_revenue", label="Cost of revenue", period=Period.DURATION,
        statement="income", sign=Sign.POSITIVE,
        concepts=(
            "CostOfRevenue",
            "CostOfGoodsAndServicesSold",
            "CostOfGoodsSold",
            "CostOfServices",
        ),
    ),
    FieldDef(
        key="operating_income", label="Operating income (loss)", period=Period.DURATION,
        statement="income",
        concepts=("OperatingIncomeLoss",),
    ),
    FieldDef(
        key="net_income", label="Net income (loss)", period=Period.DURATION,
        statement="income",
        concepts=(
            "NetIncomeLoss",
            "ProfitLoss",
            "NetIncomeLossAvailableToCommonStockholdersBasic",
        ),
        notes="NetIncomeLoss is attributable to the parent; ProfitLoss includes "
              "noncontrolling interests. Preferring the former keeps ROE consistent "
              "with StockholdersEquity, which also excludes NCI.",
    ),
    FieldDef(
        key="total_assets", label="Total assets", period=Period.INSTANT,
        statement="balance", sign=Sign.POSITIVE,
        concepts=("Assets",),
    ),
    FieldDef(
        key="total_liabilities", label="Total liabilities", period=Period.INSTANT,
        statement="balance", sign=Sign.POSITIVE,
        concepts=("Liabilities",),
        notes="Many filers omit a Liabilities total and present only the "
              "liabilities-and-equity subtotal; absence here is common and must be "
              "scored as a legitimate abstention, not an error.",
    ),
    FieldDef(
        key="total_equity", label="Total stockholders' equity", period=Period.INSTANT,
        statement="balance",
        concepts=(
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ),
    ),
    FieldDef(
        key="current_assets", label="Total current assets", period=Period.INSTANT,
        statement="balance", sign=Sign.POSITIVE,
        concepts=("AssetsCurrent",),
        notes="Absent for filers using an unclassified balance sheet (most banks "
              "and insurers).",
    ),
    FieldDef(
        key="current_liabilities", label="Total current liabilities", period=Period.INSTANT,
        statement="balance", sign=Sign.POSITIVE,
        concepts=("LiabilitiesCurrent",),
    ),
    FieldDef(
        key="accounts_receivable", label="Accounts receivable, net", period=Period.INSTANT,
        statement="balance", sign=Sign.POSITIVE,
        concepts=(
            "AccountsReceivableNetCurrent",
            "ReceivablesNetCurrent",
            "AccountsAndOtherReceivablesNetCurrent",
        ),
    ),
    FieldDef(
        key="cash_from_operations", label="Net cash from operating activities",
        period=Period.DURATION, statement="cashflow",
        concepts=(
            "NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ),
    ),
    FieldDef(
        key="capex", label="Capital expenditures", period=Period.DURATION,
        statement="cashflow", sign=Sign.POSITIVE,
        concepts=(
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsToAcquireProductiveAssets",
            "PaymentsToAcquireOtherPropertyPlantAndEquipment",
        ),
        notes="Reported as a positive outflow in these concepts. Normalised positive "
              "so free cash flow is always CFO minus capex.",
    ),
)

L1_BY_KEY: dict[str, FieldDef] = {f.key: f for f in L1_FIELDS}
L1_KEYS: tuple[str, ...] = tuple(f.key for f in L1_FIELDS)


# --- L2: the eight derived metrics ---------------------------------------
class Op(str, Enum):
    RATIO = "ratio"          # numerator / denominator
    DIFFERENCE = "difference"  # numerator - denominator


@dataclass(frozen=True)
class RatioDef:
    key: str
    label: str
    op: Op
    numerator: tuple[str, ...]        # L1 keys combined by `numerator_op`
    denominator: tuple[str, ...]
    numerator_op: Op = Op.DIFFERENCE  # only used when numerator has 2 keys
    unit: str = "ratio"
    notes: str = ""

    def inputs(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys([*self.numerator, *self.denominator]))

    def compute(self, values: dict[str, float | None]) -> dict[str, object]:
        """Evaluate from L1 values, abstaining rather than guessing.

        Returns the value *and* its provenance: the resolved numerator,
        denominator and their source keys. The eval's calculation-consistency
        metric re-derives ``value`` from those two numbers, which catches the
        failure mode where the inputs are right and the arithmetic is not — and
        the opposite one, where the arithmetic checks out against numbers that
        were invented.
        """
        def combine(keys: tuple[str, ...], op: Op) -> float | None:
            parts = [values.get(k) for k in keys]
            if any(p is None for p in parts):
                return None
            nums = [float(p) for p in parts]  # type: ignore[arg-type]
            if len(nums) == 1:
                return nums[0]
            return nums[0] - sum(nums[1:]) if op is Op.DIFFERENCE else nums[0]

        num = combine(self.numerator, self.numerator_op)
        den = combine(self.denominator, Op.RATIO)

        missing = [k for k in self.inputs() if values.get(k) is None]
        if num is None or den is None:
            return {
                "key": self.key, "value": None, "reason": "missing_inputs",
                "missing": missing, "numerator": num, "denominator": den,
                "numerator_keys": list(self.numerator),
                "denominator_keys": list(self.denominator),
            }
        if self.op is Op.RATIO and den == 0:
            return {
                "key": self.key, "value": None, "reason": "zero_denominator",
                "numerator": num, "denominator": den,
                "numerator_keys": list(self.numerator),
                "denominator_keys": list(self.denominator),
            }
        value = (num / den) if self.op is Op.RATIO else (num - den)
        return {
            "key": self.key, "value": value, "reason": "",
            "numerator": num, "denominator": den,
            "numerator_keys": list(self.numerator),
            "denominator_keys": list(self.denominator),
            "op": self.op.value,
        }

    def to_json(self) -> dict[str, object]:
        return {
            "key": self.key, "label": self.label, "op": self.op.value,
            "numerator": list(self.numerator), "denominator": list(self.denominator),
            "unit": self.unit,
        }


L2_RATIOS: tuple[RatioDef, ...] = (
    RatioDef("gross_margin", "Gross margin", Op.RATIO,
             numerator=("revenue", "cost_of_revenue"), denominator=("revenue",)),
    RatioDef("operating_margin", "Operating margin", Op.RATIO,
             numerator=("operating_income",), denominator=("revenue",)),
    RatioDef("net_margin", "Net margin", Op.RATIO,
             numerator=("net_income",), denominator=("revenue",)),
    RatioDef("roe", "Return on equity", Op.RATIO,
             numerator=("net_income",), denominator=("total_equity",),
             notes="Period-end equity, not average: the filing reports one balance "
                   "sheet date per year, and averaging would need the prior year, "
                   "which the single-filing task does not provide."),
    RatioDef("roa", "Return on assets", Op.RATIO,
             numerator=("net_income",), denominator=("total_assets",)),
    RatioDef("current_ratio", "Current ratio", Op.RATIO,
             numerator=("current_assets",), denominator=("current_liabilities",)),
    RatioDef("receivables_turnover", "Accounts receivable turnover", Op.RATIO,
             numerator=("revenue",), denominator=("accounts_receivable",)),
    RatioDef("free_cash_flow", "Free cash flow", Op.DIFFERENCE,
             numerator=("cash_from_operations",), denominator=("capex",), unit="USD",
             notes="An absolute amount, not a ratio, but it keeps the same "
                   "numerator/denominator provenance contract so one consistency "
                   "check covers all eight."),
)

L2_BY_KEY: dict[str, RatioDef] = {r.key: r for r in L2_RATIOS}
L2_KEYS: tuple[str, ...] = tuple(r.key for r in L2_RATIOS)


# --- tolerance bands ------------------------------------------------------
@dataclass(frozen=True)
class Tolerance:
    name: str
    relative: float

    def accepts(self, got: float, truth: float) -> bool:
        if truth == 0:
            return abs(got) <= self.relative
        return abs(got - truth) / abs(truth) <= self.relative


TOLERANCES: tuple[Tolerance, ...] = (
    Tolerance("exact", 0.0),
    Tolerance("strict", 0.005),
    Tolerance("loose", 0.02),
)
# L1 is a copy task, so it is graded strictly. L2 accumulates rounding across a
# multi-step derivation, so its headline band is the loose one; all three are
# always reported.
L1_HEADLINE_TOLERANCE = "strict"
L2_HEADLINE_TOLERANCE = "loose"


def field_schema() -> dict[str, object]:
    """The contract handed to the agent and used to validate its output."""
    return {
        "l1": [f.to_json() for f in L1_FIELDS],
        "l2": [r.to_json() for r in L2_RATIOS],
        "l1_output_contract": {
            "value": "number | null",
            "unit": "string, e.g. USD",
            "fiscal_year": "integer",
            "page": "integer | string section id where the figure appears",
            "quote": "10-20 words quoted verbatim around the figure",
            "reason": "'not_disclosed' when value is null",
        },
        "l2_output_contract": {
            "value": "number | null",
            "numerator": "number",
            "denominator": "number",
            "numerator_keys": "list of L1 field keys",
            "denominator_keys": "list of L1 field keys",
        },
    }
