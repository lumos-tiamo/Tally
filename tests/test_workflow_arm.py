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

from teamclaw.evaluation.harness import CaseOutcome, EvalRun
from teamclaw.evaluation.metrics import RunMetrics
from teamclaw.evaluation.runner import ARMS, ARMS_BY_NAME, RunnerContext, make_case_runner
from teamclaw.evaluation.workflow_arm import CAPTIONS, build_graph
from teamclaw.models.providers.fake import FakeProvider
from teamclaw.models.router import Registry, Router
from teamclaw.observability.trace import Tracer, new_run_id
from teamclaw.orchestration.graph import TaskGraph
from teamclaw.scenarios.dd_finance.fields import L1_KEYS


# --- registration and dispatch -------------------------------------------
def test_the_arm_is_registered_and_dispatches_to_the_graph_runner(tmp_path: Path):
    from teamclaw.scenarios.dd_finance.sec_client import SecClient

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
    from teamclaw.execution.workspace import Workspace
    from teamclaw.scenarios.dd_finance.sec_client import SecClient

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
