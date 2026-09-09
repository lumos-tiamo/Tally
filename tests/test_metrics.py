"""The four metrics, each tested against the failure it exists to catch."""

from __future__ import annotations

from tally.evaluation.metrics import (
    abstention_accuracy,
    aggregate,
    calculation_consistency,
    citation_verifiability,
    numbers_in,
    numeric_accuracy,
    parse_number,
)

SOURCE = (
    "CONSOLIDATED STATEMENTS OF OPERATIONS (In millions) "
    "Years ended September 28, 2024 September 30, 2023 "
    "Total net sales 391,035 383,285 394,328 "
    "Total cost of sales 210,352 214,137 223,546"
)


# --- number parsing -------------------------------------------------------
def test_a_statement_row_parses_as_separate_figures():
    """The bug that made every citation against a table row look fabricated."""
    assert numbers_in("Total net sales 391,035 383,285 394,328") == [391035.0, 383285.0, 394328.0]


def test_a_bare_four_digit_year_is_one_number():
    assert numbers_in("Years ended September 28, 2024") == [28.0, 2024.0]


def test_accounting_parentheses_are_negative():
    assert parse_number("(9,447)") == -9447.0
    assert numbers_in("Capital expenditures (9,447) (10,959)") == [-9447.0, -10959.0]


# --- numeric accuracy -----------------------------------------------------
def test_a_figure_read_from_a_millions_table_is_accepted():
    truth = {"revenue": 391_035_000_000.0}
    predicted = {"revenue": {"value": "391,035", "unit": "USD millions"}}
    assert numeric_accuracy(predicted, truth)["strict"].rate == 1.0


def test_tolerance_bands_separate_rounding_from_nonsense():
    truth = {"a": 1_000_000.0, "b": 1_000_000.0}
    predicted = {"a": 1_002_000.0, "b": 1_400_000.0}   # +0.2% and +40%
    bands = numeric_accuracy(predicted, truth, allow_scale_variants=False)
    assert bands["exact"].rate == 0.0
    assert bands["strict"].rate == 0.5
    assert bands["loose"].rate == 0.5


def test_abstaining_cannot_inflate_numeric_accuracy():
    """A field with no truth value is not gradable as a number."""
    truth = {"revenue": 100.0, "current_assets": None}
    predicted = {"revenue": 100.0, "current_assets": {"value": None}}
    assert numeric_accuracy(predicted, truth)["strict"].total == 1


# --- citation verifiability -----------------------------------------------
def test_a_locatable_quote_with_its_number_verifies():
    predicted = {"revenue": {"value": "391,035", "quote": "Total net sales 391,035 383,285"}}
    assert citation_verifiability(predicted, SOURCE).rate == 1.0


def test_a_fabricated_quote_is_caught():
    predicted = {"revenue": {"value": 391_035, "quote": "Revenue for the year was excellent"}}
    result = citation_verifiability(predicted, SOURCE)
    assert result.rate == 0.0
    assert result.detail["quote_not_found"] == 1


def test_a_real_quote_with_the_wrong_number_is_caught_separately():
    """The subtler failure: real text, misattributed figure."""
    predicted = {"revenue": {"value": 999_999, "quote": "Total net sales 391,035 383,285"}}
    result = citation_verifiability(predicted, SOURCE)
    assert result.rate == 0.0
    assert result.detail["number_far_from_quote"] == 1


def test_unicode_differences_do_not_read_as_hallucination():
    source = "Total shareholders’ equity 56,950"
    predicted = {"total_equity": {"value": "56,950", "quote": "Total shareholders' equity 56,950"}}
    assert citation_verifiability(predicted, source).rate == 1.0


# --- calculation consistency ---------------------------------------------
def test_broken_arithmetic_over_correct_inputs_is_caught():
    ratios = {"net_margin": {"value": 0.30, "numerator": 93_736, "denominator": 391_035}}
    result = calculation_consistency(ratios)
    assert result.rate == 0.0
    assert result.detail["arithmetic_mismatch"] == 1


def test_a_result_without_provenance_is_unverifiable():
    result = calculation_consistency({"roe": {"value": 1.6459}})
    assert result.detail["missing_provenance"] == 1
    assert result.rate == 0.0


def test_consistent_arithmetic_passes():
    ratios = {"gross_margin": {"value": 180_683 / 391_035,
                               "numerator": 180_683, "denominator": 391_035}}
    assert calculation_consistency(ratios).rate == 1.0


def test_free_cash_flow_uses_subtraction_not_division():
    ratios = {"free_cash_flow": {"value": 108_807, "numerator": 118_254, "denominator": 9_447}}
    assert calculation_consistency(ratios).rate == 1.0


# --- abstention accuracy --------------------------------------------------
def test_inventing_an_undisclosed_field_is_penalised():
    truth = {"current_assets": None}
    predicted = {"current_assets": {"value": 152_987_000_000}}
    result = abstention_accuracy(predicted, truth, absent_keys=["current_assets"])
    assert result.rate == 0.0
    assert result.detail["fabricated"] == 1


def test_declining_to_answer_an_undisclosed_field_is_correct():
    truth = {"current_assets": None}
    predicted = {"current_assets": {"value": None, "reason": "not_disclosed"}}
    assert abstention_accuracy(predicted, truth, absent_keys=["current_assets"]).rate == 1.0


def test_abstention_without_a_stated_reason_is_flagged_but_accepted():
    predicted = {"current_assets": {"value": None}}
    result = abstention_accuracy(predicted, {"current_assets": None},
                                 absent_keys=["current_assets"])
    assert result.rate == 1.0
    assert result.detail.get("abstained_without_reason") == 1


# --- aggregation ----------------------------------------------------------
def test_pooling_is_micro_averaged():
    """A case with twelve gradable fields must outweigh one with three."""
    big = numeric_accuracy({f"f{i}": 1.0 for i in range(12)},
                           {f"f{i}": 1.0 for i in range(12)},
                           allow_scale_variants=False)["strict"]
    small = numeric_accuracy({f"g{i}": 2.0 for i in range(3)},
                             {f"g{i}": 1.0 for i in range(3)},
                             allow_scale_variants=False)["strict"]
    pooled = aggregate([big, small])
    assert pooled.total == 15
    assert pooled.rate == 12 / 15
