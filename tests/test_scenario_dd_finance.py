"""Scenario-level correctness: concept mapping, ground truth, and the eval contract."""

from __future__ import annotations

import datetime as dt

import pytest

from tally.execution.registry import ToolParam, ToolRegistry, ToolSpec
from tally.scenarios.dd_finance.concepts import (
    ConceptResolver,
    Fact,
    parse_company_facts,
    resolve_identity_components,
)
from tally.scenarios.dd_finance.fields import (
    L1_FIELDS,
    L2_BY_KEY,
    L2_RATIOS,
    Period,
)
from tally.scenarios.dd_finance.groundtruth import check_balance_sheet_identity
from tally.scenarios.dd_finance.sec_client import SecClient, fiscal_year_of
from tally.scenarios.dd_finance.spec import assert_no_truth_leak, build_skills

FY_END = dt.date(2024, 9, 28)


def facts_payload(entries: dict[str, list[dict]]) -> dict:
    return {"facts": {"us-gaap": {
        concept: {"units": {"USD": rows}} for concept, rows in entries.items()
    }}}


# --- the eval's integrity ------------------------------------------------
def test_a_tool_that_would_leak_xbrl_is_rejected():
    """XBRL is the answer key; reaching it from the agent invalidates everything."""
    registry = ToolRegistry().add(ToolSpec(
        module="sec", func="company_facts", summary="Get XBRL company facts",
        params=[ToolParam("cik", "str")], returns="dict", requires_network=True,
        handler=lambda cik: {},
    ))
    with pytest.raises(AssertionError, match="leak ground truth"):
        assert_no_truth_leak(registry)


def test_a_tool_merely_mentioning_xbrl_is_rejected():
    registry = ToolRegistry().add(ToolSpec(
        module="sec", func="lookup", summary="Read a value from companyfacts",
        params=[], returns="dict", local_source="return {}",
    ))
    with pytest.raises(AssertionError):
        assert_no_truth_leak(registry)


# --- field definitions ---------------------------------------------------
def test_the_locked_field_set_matches_the_spec():
    assert len(L1_FIELDS) == 12
    assert len(L2_RATIOS) == 8
    assert sum(1 for f in L1_FIELDS if f.period is Period.INSTANT) == 6


def test_every_ratio_reports_its_own_provenance():
    values = {f.key: 100.0 for f in L1_FIELDS}
    for ratio in L2_RATIOS:
        out = ratio.compute(values)
        assert out["numerator"] is not None
        assert out["denominator"] is not None
        assert out["numerator_keys"] and out["denominator_keys"]


def test_a_ratio_abstains_rather_than_treating_a_gap_as_zero():
    out = L2_BY_KEY["current_ratio"].compute({"current_assets": None,
                                              "current_liabilities": 100.0})
    assert out["value"] is None and out["reason"] == "missing_inputs"


def test_a_zero_denominator_abstains():
    out = L2_BY_KEY["roe"].compute({"net_income": 10.0, "total_equity": 0.0})
    assert out["value"] is None and out["reason"] == "zero_denominator"


# --- concept resolution --------------------------------------------------
def test_quarterly_duration_facts_are_rejected():
    """A Q4 figure has the same shape as an annual one; only the span differs."""
    payload = facts_payload({"Revenues": [
        {"start": "2024-06-30", "end": "2024-09-28", "val": 94_930_000_000,
         "form": "10-K", "filed": "2024-11-01", "fy": 2024, "fp": "Q4"},
        {"start": "2023-10-01", "end": "2024-09-28", "val": 391_035_000_000,
         "form": "10-K", "filed": "2024-11-01", "fy": 2024, "fp": "FY"},
    ]})
    resolver = ConceptResolver(parse_company_facts(payload))
    assert resolver.resolve_field("revenue", FY_END).value == 391_035_000_000


def test_a_fact_from_a_different_year_is_not_selected():
    """`fy` labels the filing, so comparatives carry the filing's year."""
    payload = facts_payload({"Revenues": [
        {"start": "2022-09-25", "end": "2023-09-30", "val": 383_285_000_000,
         "form": "10-K", "filed": "2024-11-01", "fy": 2024, "fp": "FY"},
        {"start": "2023-10-01", "end": "2024-09-28", "val": 391_035_000_000,
         "form": "10-K", "filed": "2024-11-01", "fy": 2024, "fp": "FY"},
    ]})
    resolver = ConceptResolver(parse_company_facts(payload))
    assert resolver.resolve_field("revenue", FY_END).value == 391_035_000_000


def test_the_latest_filing_wins_a_restatement_and_the_loser_is_kept():
    payload = facts_payload({"Revenues": [
        {"start": "2023-10-01", "end": "2024-09-28", "val": 390_000_000_000,
         "form": "10-K", "filed": "2024-11-01", "fy": 2024, "fp": "FY"},
        {"start": "2023-10-01", "end": "2024-09-28", "val": 391_035_000_000,
         "form": "10-K", "filed": "2025-11-01", "fy": 2025, "fp": "FY"},
    ]})
    resolution = ConceptResolver(parse_company_facts(payload)).resolve_field("revenue", FY_END)
    assert resolution.value == 391_035_000_000
    assert resolution.restated and resolution.superseded


def test_concept_preference_order_is_honoured():
    payload = facts_payload({
        "Revenues": [{"start": "2023-10-01", "end": "2024-09-28", "val": 1.0,
                      "form": "10-K", "filed": "2024-11-01"}],
        "RevenueFromContractWithCustomerExcludingAssessedTax": [
            {"start": "2023-10-01", "end": "2024-09-28", "val": 2.0,
             "form": "10-K", "filed": "2024-11-01"}],
    })
    resolution = ConceptResolver(parse_company_facts(payload)).resolve_field("revenue", FY_END)
    assert resolution.concept == "RevenueFromContractWithCustomerExcludingAssessedTax"


def test_an_undisclosed_field_resolves_to_an_explicit_abstention():
    resolver = ConceptResolver(parse_company_facts(facts_payload({})))
    resolution = resolver.resolve_field("current_assets", FY_END)
    assert resolution.value is None and resolution.reason == "not_disclosed"
    assert resolution.candidates_tried, "the attempted tags must be recorded"


def test_capex_is_normalised_positive():
    payload = facts_payload({"PaymentsToAcquirePropertyPlantAndEquipment": [
        {"start": "2023-10-01", "end": "2024-09-28", "val": -9_447_000_000,
         "form": "10-K", "filed": "2024-11-01"}]})
    resolver = ConceptResolver(parse_company_facts(payload))
    assert resolver.resolve_field("capex", FY_END).value == 9_447_000_000


# --- balance-sheet identity ----------------------------------------------
def test_noncontrolling_interests_explain_the_residual_rather_than_failing():
    l1 = {"total_assets": 61_077.4, "total_liabilities": 51_428.7, "total_equity": 3_382.2}
    components = {"minority_interest": 6_266.5, "liabilities_and_equity": 61_077.4}
    result = check_balance_sheet_identity(l1, components,
                                          equity_concept="StockholdersEquity")
    assert result["passes"], result


def test_nci_is_not_added_twice_when_equity_already_includes_it():
    """The bug: a negative residual exactly equal to the NCI amount."""
    l1 = {"total_assets": 100.0, "total_liabilities": 60.0, "total_equity": 40.0}
    components = {"minority_interest": 5.0, "liabilities_and_equity": 100.0}
    result = check_balance_sheet_identity(
        l1, components,
        equity_concept="StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    )
    assert result["equity_includes_nci"] is True
    assert result["passes"]


def test_the_filers_own_total_is_the_strongest_cross_check():
    l1 = {"total_assets": 100.0, "total_liabilities": 60.0, "total_equity": 40.0}
    result = check_balance_sheet_identity(l1, {"liabilities_and_equity": 999.0},
                                          equity_concept="StockholdersEquity")
    assert result["direct_check"]["passes"] is False


# --- fiscal calendar -----------------------------------------------------
@pytest.mark.parametrize(
    ("end", "expected"),
    [
        (dt.date(2024, 12, 31), 2024),
        (dt.date(2024, 9, 28), 2024),
        (dt.date(2025, 2, 1), 2025),
        (dt.date(2025, 1, 28), 2024),   # 52/53-week retailer closing FY2024
    ],
)
def test_fiscal_year_label(end: dt.date, expected: int):
    assert fiscal_year_of(end) == expected


def test_cik_padding():
    assert SecClient.pad_cik("320193") == "0000320193"
    assert SecClient.pad_cik("CIK0000320193") == "0000320193"


# --- skills --------------------------------------------------------------
def test_scenario_skills_are_selected_for_the_steps_they_belong_to():
    library = build_skills()
    picked = {s.name for s in library.select(
        "read the balance sheet section of the annual report filing", k=2)}
    assert picked & {"read-a-10k", "abstain-on-absence", "persist-then-summarise"}
    assert library.select("hello there", k=2) == []
