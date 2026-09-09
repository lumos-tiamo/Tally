"""The `workflow-c` arm: a fixed task graph scored by the same harness.

Two claims are under test. First, that the deterministic arm is genuinely
interchangeable with the agent arm from the harness's point of view — same
runner interface, same CaseRunResult, same metrics. Second, that the validity
guard tells the difference between an arm that needed no model and a run where
no model was reachable, because those produce identical-looking tables of zeros
and mean opposite things.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tally.evaluation.harness import CaseOutcome, EvalRun
from tally.evaluation.metrics import RunMetrics
from tally.evaluation.runner import ARMS, ARMS_BY_NAME, RunnerContext, make_case_runner
from tally.evaluation.workflow_arm import (
    CAPTIONS,
    build_graph,
    caption_is_plausible,
    statement_caption_sample,
)
from tally.models.providers.fake import FakeProvider
from tally.models.router import Registry, Router
from tally.observability.trace import Tracer, new_run_id
from tally.orchestration.graph import TaskGraph
from tally.scenarios.dd_finance.fields import L1_KEYS


# --- registration and dispatch -------------------------------------------
def test_the_arm_is_registered_and_dispatches_to_the_graph_runner(tmp_path: Path):
    from tally.scenarios.dd_finance.sec_client import SecClient

    assert "workflow-c" in ARMS_BY_NAME
    assert ARMS_BY_NAME["workflow-c"].deterministic_workflow

    graph_runner = make_case_runner(RunnerContext(
        registry=Registry(), client=SecClient(), runs_root=tmp_path,
        workspaces_root=tmp_path, arm=ARMS_BY_NAME["workflow-c"],
    ))
    agent_runner = make_case_runner(RunnerContext(
        registry=Registry(), client=SecClient(), runs_root=tmp_path,
        workspaces_root=tmp_path, arm=ARMS_BY_NAME["full"],
    ))
    assert "make_workflow_runner" in graph_runner.__qualname__
    assert "make_workflow_runner" not in agent_runner.__qualname__


def test_exactly_one_arm_is_a_deterministic_workflow():
    """The comparison is agent-vs-workflow; more than one workflow arm muddies it."""
    assert sum(1 for a in ARMS if a.deterministic_workflow) == 1


def test_the_arm_config_records_which_mode_produced_a_score():
    payload = ARMS_BY_NAME["workflow-c"].to_json()
    assert payload["deterministic_workflow"] is True
    assert ARMS_BY_NAME["full"].to_json()["deterministic_workflow"] is False


# --- graph shape ----------------------------------------------------------
def test_the_graph_calls_a_model_at_exactly_one_node(tmp_path: Path, sample_case):
    """The whole premise: code everywhere except where judgement is required."""
    from tally.execution.workspace import Workspace
    from tally.scenarios.dd_finance.sec_client import SecClient

    tracer = Tracer(new_run_id(), tmp_path / "runs")
    registry = Registry(providers={"fake": FakeProvider()})
    for purpose in list(registry.policy):
        registry.policy[purpose] = ("fake",)
    graph = build_graph(
        sample_case, router=Router(registry, tracer=tracer),
        workspace=Workspace.create(tmp_path / "ws", "g"),
        client=SecClient(), tracer=tracer, level="l1",
    )
    tracer.close()

    assert isinstance(graph, TaskGraph)
    order = graph.topological_order()
    assert order == ["download", "to_text", "deterministic_pass", "model_pass",
                     "assemble", "validate"]
    # The model node is optional: a case the caption table settles entirely must
    # still reach `assemble`.
    assert graph.nodes["model_pass"].optional
    assert graph.nodes["validate"].optional


def test_the_caption_table_covers_every_contract_field():
    assert set(CAPTIONS) == set(L1_KEYS)
    for field, captions in CAPTIONS.items():
        assert captions, f"{field} has no captions"


def test_a_failed_download_skips_the_rest_rather_than_running_on_nothing():
    graph = (TaskGraph()
             .add("download", lambda s: (_ for _ in ()).throw(LookupError("no filing")))
             .add("to_text", lambda s: {"path": "x"}, depends_on=["download"])
             .add("assemble", lambda s: {}, depends_on=["to_text"]))
    result = graph.run()
    assert not result.ok
    assert result.results["to_text"].skipped
    assert result.results["assemble"].skipped


# --- validity of a model-free arm ----------------------------------------
def _run_with(**kwargs) -> EvalRun:
    run = EvalRun(arm="workflow-c", level="l1")
    run.outcomes.append(CaseOutcome(case_id="AAPL-FY2024", level="l1", **kwargs))
    return run


def test_a_workflow_that_needed_no_model_is_a_deterministic_measurement():
    """Zero tokens because none were required is not the same as none available."""
    run = _run_with(finished=True, run=RunMetrics(steps=6))
    headline = run.headline()
    assert headline["measurement_kind"] == "deterministic"
    assert headline["valid_measurement"] is True
    assert headline["model_calls_made"] is False
    assert "DETERMINISTIC MEASUREMENT" in headline["validity_note"]


def test_a_run_where_the_model_was_unreachable_is_still_refused():
    run = _run_with(finished=False, run=RunMetrics(), error="NoProviderAvailable: none")
    headline = run.headline()
    assert headline["measurement_kind"] == "none"
    assert headline["valid_measurement"] is False
    assert "NOT A MEASUREMENT" in headline["validity_note"]


def test_a_deterministic_label_is_never_applied_when_a_case_errored():
    run = EvalRun(arm="workflow-c", level="l1")
    run.outcomes.append(CaseOutcome(case_id="A", level="l1", finished=True,
                                    run=RunMetrics(steps=6)))
    run.outcomes.append(CaseOutcome(case_id="B", level="l1", finished=False,
                                    run=RunMetrics(), error="LookupError: no filing"))
    assert run.measurement_kind == "none"
    assert not run.publishable


def test_a_workflow_arm_that_did_call_a_model_is_a_model_measurement():
    run = _run_with(finished=True, providers_used=("glm",),
                    run=RunMetrics(steps=6, tokens_in=400, tokens_out=60))
    assert run.headline()["measurement_kind"] == "model"
    assert run.headline()["validity_note"] == ""


@pytest.mark.parametrize("kind", ["model", "deterministic"])
def test_only_these_two_kinds_are_publishable(kind: str):
    run = (_run_with(finished=True, providers_used=("glm",),
                     run=RunMetrics(steps=6, tokens_in=1, tokens_out=1))
           if kind == "model" else _run_with(finished=True, run=RunMetrics(steps=6)))
    assert run.publishable


# --- the caption gate ----------------------------------------------------
BANK_VOCAB = [
    "Net interest income", "Noninterest income", "Total revenue, net of interest expense",
    "Provision for credit losses", "Noninterest expense", "Total assets",
    "Accounts receivable, net", "Total current assets", "Cost of sales",
]


def test_a_caption_the_filing_does_not_contain_is_rejected():
    """The prompt asks for verbatim captions; the code enforces it."""
    allowed, why = caption_is_plausible("Invented Line Nobody Wrote", "revenue", BANK_VOCAB)
    assert not allowed
    assert "not present in the filing" in why


def test_a_caption_about_a_different_line_item_is_rejected():
    allowed, why = caption_is_plausible("Total assets", "accounts_receivable", BANK_VOCAB)
    assert not allowed


def test_the_gate_is_discriminative_not_merely_non_zero():
    """The failure two weaker rules let through.

    'Total revenue, net of interest expense' shares the word 'revenue' with
    `cost_of_revenue`, so a non-zero-overlap test accepts it and the arm reports
    a revenue figure as cost of revenue. Counting shared words *ties* the two —
    the caption contains nothing that distinguishes them — and an accepted tie is
    a field chosen arbitrarily. Jaccard breaks it correctly by also penalising
    the words the caption is missing.
    """
    ok_for_revenue, _ = caption_is_plausible(
        "Total revenue, net of interest expense", "revenue", BANK_VOCAB)
    ok_for_cost, why = caption_is_plausible(
        "Total revenue, net of interest expense", "cost_of_revenue", BANK_VOCAB)
    assert ok_for_revenue
    assert not ok_for_cost
    assert "does not distinguish" in why


def test_a_tie_between_two_fields_is_rejected_not_resolved_arbitrarily():
    """An accepted tie means the field was picked by dictionary order."""
    ok, why = caption_is_plausible("Total assets", "current_assets", BANK_VOCAB)
    assert not ok
    assert "distinguish" in why or "ambiguous" in why


def test_the_stopword_list_keeps_the_words_that_actually_discriminate():
    """`net` and `current` look like noise and are the only discriminators.

    A generic financial stopword list drops both, which collapses
    net_income/operating_income and current_assets/total_assets into ties the
    gate then has to reject — rejecting captions that are exactly right.
    """
    vocab = ["Net income", "Operating income", "Total assets", "Total current assets"]
    assert caption_is_plausible("Net income", "net_income", vocab)[0]
    assert not caption_is_plausible("Net income", "operating_income", vocab)[0]
    assert caption_is_plausible("Total current assets", "current_assets", vocab)[0]
    assert not caption_is_plausible("Total current assets", "total_assets", vocab)[0]


def test_the_matching_caption_is_allowed_for_its_own_field():
    assert caption_is_plausible("Cost of sales", "cost_of_revenue", BANK_VOCAB)[0]
    assert caption_is_plausible("Total current assets", "current_assets", BANK_VOCAB)[0]
    assert caption_is_plausible("Accounts receivable, net", "accounts_receivable",
                                BANK_VOCAB)[0]


def test_a_stub_model_answering_every_field_the_same_way_resolves_nothing(
    tmp_path: Path, sample_case
):
    """The defect that motivated the gate, as a regression test.

    Before it existed, a model that replied 'Total revenue' for every unresolved
    field produced that one figure for cost_of_revenue, operating_income,
    current_assets and four others — each confidently wrong and attributed to the
    wrong field.
    """
    from tally.execution.workspace import Workspace
    from tally.scenarios.dd_finance.sec_client import SecClient

    statements = "\n".join([
        "Item 8. Financial Statements",
        "CONSOLIDATED STATEMENTS OF OPERATIONS (In millions)",
        "Total net sales",
        "391,035 383,285 394,328",
        "Total cost of sales",
        "210,352 214,137 223,546",
        "Total assets",
        "364,980 352,583",
    ])
    filing = tmp_path / "f.txt"
    filing.write_text(statements, encoding="utf-8")

    tracer = Tracer(new_run_id(), tmp_path / "runs")
    fake = FakeProvider(name="fake",
                        default='{"captions": ["Total net sales"], "reason": "same"}')
    registry = Registry(providers={"fake": fake})
    for purpose in list(registry.policy):
        registry.policy[purpose] = ("fake",)

    graph = build_graph(
        sample_case, router=Router(registry, tracer=tracer),
        workspace=Workspace.create(tmp_path / "ws", "g"),
        client=SecClient(), tracer=tracer, level="l1",
    )
    state = {"to_text": {"path": str(filing), "document_scale_hint": "millions"}}
    state["deterministic_pass"] = graph.nodes["deterministic_pass"].fn(state)
    decided = graph.nodes["model_pass"].fn(state)
    tracer.close()

    wrongly_accepted = {
        field: entry["value"] for field, entry in decided.items()
        if entry.get("value") is not None and field != "revenue"
    }
    assert not wrongly_accepted, f"one caption resolved several fields: {wrongly_accepted}"


def test_no_vocabulary_means_no_model_call_at_all(tmp_path: Path, sample_case):
    """Asking a model to choose from an empty list invites invention."""
    from tally.execution.workspace import Workspace
    from tally.scenarios.dd_finance.sec_client import SecClient

    empty = tmp_path / "empty.txt"
    empty.write_text("Item 8. Financial Statements\nNothing tabular here.\n", encoding="utf-8")
    tracer = Tracer(new_run_id(), tmp_path / "runs")
    fake = FakeProvider(name="fake", default='{"captions": ["Anything"]}')
    registry = Registry(providers={"fake": fake})
    for purpose in list(registry.policy):
        registry.policy[purpose] = ("fake",)

    graph = build_graph(
        sample_case, router=Router(registry, tracer=tracer),
        workspace=Workspace.create(tmp_path / "ws", "g2"),
        client=SecClient(), tracer=tracer, level="l1",
    )
    state = {"to_text": {"path": str(empty), "document_scale_hint": None}}
    state["deterministic_pass"] = graph.nodes["deterministic_pass"].fn(state)
    decided = graph.nodes["model_pass"].fn(state)
    tracer.close()

    assert fake.calls == [], "no captions to choose from should mean no call"
    assert all(entry["value"] is None for entry in decided.values())


def test_the_sampler_finds_statement_rows_not_narrative_tables(tmp_path: Path):
    """Captions sit on their own line; the figures follow on the next ones."""
    import sys

    sandbox = (Path(__file__).resolve().parents[1] / "src" / "tally" /
               "scenarios" / "dd_finance" / "sandbox_tools")
    sys.path.insert(0, str(sandbox))
    import doc  # noqa: PLC0415

    filing = tmp_path / "f.txt"
    filing.write_text("\n".join([
        "Item 2. Properties",
        "Financial centers",
        "3,800",
        "Item 8. Financial Statements",
        "Total net sales",
        "391,035 383,285 394,328",
        "Total cost of sales",
        "210,352 214,137",
        "Narrative sentence about the year that mentions 1,234 once.",
    ]), encoding="utf-8")

    sample = statement_caption_sample(doc, str(filing))
    assert "Total net sales" in sample
    assert "Total cost of sales" in sample
    # Item 2's one-figure table must not become financial vocabulary.
    assert "Financial centers" not in sample
