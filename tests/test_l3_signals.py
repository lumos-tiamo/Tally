"""Cross-year signals and the L3 scoring they back.

The point of a computable backbone for a judgement task is that the two
pathological agents are both caught: the one that lists every possible worry
(perfect recall, poor precision) and the one that says nothing (the reverse).
Most of these tests are about that pair.
"""

from __future__ import annotations

import pytest

from tally.evaluation.harness import score_case
from tally.evaluation.metrics import aggregate, signal_findings
from tally.scenarios.dd_finance.signals import (
    SIGNALS,
    SIGNALS_BY_KEY,
    Severity,
    detect,
    detectable,
)

FULL = {
    "revenue": 100.0, "cost_of_revenue": 60.0, "operating_income": 20.0,
    "net_income": 15.0, "total_assets": 200.0, "total_liabilities": 120.0,
    "total_equity": 80.0, "current_assets": 60.0, "current_liabilities": 40.0,
    "accounts_receivable": 20.0, "cash_from_operations": 25.0, "capex": 10.0,
}


def year(**overrides: float) -> dict[str, float]:
    return {**FULL, **overrides}


def keys(found) -> set[str]:  # noqa: ANN001
    return {s.key for s in found}


# --- detection ------------------------------------------------------------
def test_revenue_up_while_cash_flow_falls_is_flagged_as_a_red_flag():
    found = detect(year(revenue=110.0, cash_from_operations=20.0), year())
    assert "revenue_up_cashflow_down" in keys(found)
    signal = next(s for s in found if s.key == "revenue_up_cashflow_down")
    assert signal.severity == Severity.RED_FLAG.value
    assert signal.evidence["revenue_growth"] == pytest.approx(0.10)
    assert signal.evidence["cfo_growth"] == pytest.approx(-0.20)


def test_ordinary_year_to_year_noise_does_not_trigger_anything():
    """Thresholds exist so that recall means something."""
    quiet = detect(year(revenue=101.0, cash_from_operations=24.8, capex=10.2), year())
    assert keys(quiet) == set()


def test_margin_compression_needs_a_material_move():
    # 50bp: below the threshold.
    assert "margin_compression" not in keys(detect(year(cost_of_revenue=60.5), year()))
    # 300bp: above it.
    assert "margin_compression" in keys(detect(year(cost_of_revenue=63.0), year()))


def test_receivables_outpacing_revenue_is_flagged():
    found = detect(year(revenue=105.0, accounts_receivable=25.0), year())
    assert "receivables_outpacing_revenue" in keys(found)


def test_receivables_growing_with_revenue_is_not_flagged():
    found = detect(year(revenue=120.0, accounts_receivable=24.0), year())
    assert "receivables_outpacing_revenue" not in keys(found)


def test_negative_free_cash_flow_records_whether_it_just_turned():
    found = detect(year(cash_from_operations=5.0, capex=20.0), year())
    signal = next(s for s in found if s.key == "negative_free_cash_flow")
    assert signal.evidence["turned_negative"] is True


def test_a_signal_whose_inputs_are_absent_is_not_reported_as_false():
    """Absent data means 'cannot tell', never 'no'."""
    bank = year(current_assets=None, current_liabilities=None)  # type: ignore[arg-type]
    found = detect(bank, bank)
    assert "liquidity_pressure" not in keys(found)
    assert "liquidity_pressure" not in detectable(bank, bank)


def test_detectable_shrinks_for_an_unclassified_balance_sheet():
    bank = year(current_assets=None, current_liabilities=None,  # type: ignore[arg-type]
                cost_of_revenue=None)  # type: ignore[arg-type]
    able = set(detectable(bank, bank))
    assert "liquidity_pressure" not in able
    assert "margin_compression" not in able
    assert "revenue_decline" in able


def test_every_signal_declares_its_inputs_and_a_reader_question():
    for definition in SIGNALS:
        assert definition.requires, f"{definition.key} declares no inputs"
        assert definition.question.endswith("?"), f"{definition.key} asks nothing"
        assert definition.keywords, f"{definition.key} cannot be matched in prose"


# --- scoring --------------------------------------------------------------
def test_an_agent_that_lists_everything_gets_recall_but_loses_precision():
    present = ["margin_compression"]
    able = [d.key for d in SIGNALS]
    everything = {"findings": [{"signal": k} for k in able]}
    recall, precision = signal_findings(everything, present, able)
    assert recall.rate == 1.0
    assert precision.rate < 0.2


def test_an_agent_that_reports_nothing_misses_what_is_there():
    present = ["margin_compression", "operating_loss"]
    able = [d.key for d in SIGNALS]
    recall, precision = signal_findings({"summary": "All fine."}, present, able)
    assert recall.rate == 0.0
    assert precision.total == 0


def test_a_quiet_case_answered_quietly_has_no_rate_and_is_credited():
    """Undefined, not zero — reading 0.00 as failure inverts the best outcome."""
    able = [d.key for d in SIGNALS]
    recall, precision = signal_findings({"summary": "No material concerns."}, [], able)
    assert recall.applicable is False
    assert recall.to_json()["rate"] is None
    assert recall.detail["correctly_silent"] == 1


def test_fabricating_on_a_quiet_case_is_recorded():
    able = [d.key for d in SIGNALS]
    _, precision = signal_findings({"findings": [{"signal": "operating_loss"}]}, [], able)
    assert precision.rate == 0.0
    assert precision.detail["fabricated_on_quiet_case"] == 1


def test_a_signal_the_data_could_not_support_is_excluded_from_recall():
    """Otherwise a bank is penalised for not computing a current ratio."""
    present = ["liquidity_pressure", "margin_compression"]
    able = ["margin_compression"]           # liquidity was never computable
    recall, _ = signal_findings({"findings": [{"signal": "margin_compression"}]},
                                present, able)
    assert recall.total == 1 and recall.rate == 1.0


def test_prose_without_the_taxonomy_still_scores():
    """The prompt never names the signals, so scoring cannot require the keys."""
    present = ["revenue_up_cashflow_down"]
    able = [d.key for d in SIGNALS]
    prose = {"summary": ("Revenue grew 8% year on year while operating cash flow "
                         "declined by 12%, and that cash conversion gap is the first "
                         "thing to explain.")}
    recall, _ = signal_findings(prose, present, able)
    assert recall.rate == 1.0


def test_one_passing_keyword_is_not_enough_to_claim_a_finding():
    present = ["revenue_up_cashflow_down"]
    able = [d.key for d in SIGNALS]
    vague = {"summary": "The cash flow statement is included in Item 8."}
    recall, _ = signal_findings(vague, present, able)
    assert recall.rate == 0.0


def test_pooling_surfaces_quiet_case_behaviour_rather_than_hiding_it():
    able = [d.key for d in SIGNALS]
    recalls = [
        signal_findings({"summary": "nothing"}, [], able)[0],
        signal_findings({"findings": [{"signal": "operating_loss"}]},
                        ["operating_loss"], able)[0],
    ]
    pooled = aggregate(recalls)
    assert pooled.total == 1 and pooled.rate == 1.0
    assert pooled.detail["correctly_silent"] == 1


def test_score_case_wires_l3_through_the_harness(sample_case):
    metrics = score_case(
        sample_case,
        {"findings": [{"signal": "margin_compression"}]},
        level="l3",
        signals_present=["margin_compression"],
        signals_detectable=[d.key for d in SIGNALS],
    )
    assert set(metrics) == {"signal_recall", "signal_precision"}
    assert metrics["signal_recall"].rate == 1.0
    assert metrics["signal_precision"].rate == 1.0


def test_the_l3_prompt_does_not_hand_the_agent_the_taxonomy():
    """Naming the signals would turn analysis into filling in a form."""
    from tally.scenarios.dd_finance.spec import l3_objective

    prompt = l3_objective("AAPL", 2024, 2023)
    for key in SIGNALS_BY_KEY:
        assert key not in prompt
    assert "red_flag" in prompt   # the severity vocabulary is fine to state
