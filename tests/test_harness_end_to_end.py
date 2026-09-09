"""The harness driven all the way through, offline.

This exercises the path `teamclaw eval` takes — load a case, run an agent, parse
the deliverable, score it, pool the results — using a scripted provider so it
needs no credentials and no network. Its purpose is to prove the plumbing works
*and* that the harness refuses to call the result a measurement.

That second half is the point. The first real invocation of `teamclaw eval` on a
machine with no credentials produced a full report of zeros marked
`valid_measurement: True`, which is indistinguishable from a genuine score of
zero. Both ways of not being a measurement — a test double doing the work, and no
model doing any work — are asserted here.
"""

from __future__ import annotations

import json
from pathlib import Path

from teamclaw.evaluation.harness import CaseRunResult, Harness
from teamclaw.evaluation.metrics import RunMetrics
from teamclaw.execution.workspace import Workspace
from teamclaw.models.providers.fake import ScriptedProvider
from teamclaw.models.router import Registry, Router
from teamclaw.observability.accounting import Accountant
from teamclaw.observability.trace import Tracer, new_run_id
from teamclaw.orchestration.agent import Agent
from teamclaw.scenarios.dd_finance.fields import L1_KEYS
from teamclaw.scenarios.dd_finance.spec import build_spec, l1_objective, parse_output


def perfect_l1_script(truth: dict[str, float | None]) -> list[str]:
    """A trajectory that writes exactly the right answer, with citations."""
    payload = {
        key: (
            {"value": value, "unit": "USD", "fiscal_year": 2024, "page": 31,
             "quote": f"Total {key.replace('_', ' ')} {value:,.0f} in the statement"}
            if value is not None else {"value": None, "reason": "not_disclosed"}
        )
        for key, value in truth.items()
    }
    literal = json.dumps(payload)
    return [
        "THOUGHT: write the extraction\n```python\n"
        f"import json\nresult = json.loads({literal!r})\n"
        "json.dump(result, open('l1.json','w'))\n"
        "print('fields', len(result))\n```",
        "DONE\nExtracted all twelve fields.",
    ]


def run_harness(tmp_path: Path, case, script: list[str]):  # noqa: ANN001
    tracer = Tracer(new_run_id(), tmp_path / "runs")
    provider = ScriptedProvider(script, name="scripted", loop_last=True)
    registry = Registry(providers={"scripted": provider})
    for purpose in list(registry.policy):
        registry.policy[purpose] = ("scripted",)

    def runner(target, level):  # noqa: ANN001
        workspace = Workspace.create(tmp_path / "ws", f"{target.case_id}_{level}")
        accountant = Accountant()
        router = Router(registry, accountant=accountant, tracer=tracer)
        # No SEC tools: the scripted trajectory writes the deliverable directly,
        # because what is under test is the harness, not the extraction stack.
        from teamclaw.execution.registry import ToolRegistry

        spec = build_spec(tools=ToolRegistry(), max_steps=4, prefer_docker=False)
        agent = Agent(spec, router=router, workspace=workspace, tracer=tracer,
                      run_dir=tmp_path / f"run_{target.case_id}")
        result = agent.run(l1_objective(target.ticker, target.fiscal_year))
        totals = accountant.total()
        return CaseRunResult(
            predicted=parse_output(workspace, level),
            run=RunMetrics(steps=result.step_count, finished=result.finished,
                           tokens_in=totals.tokens_in, tokens_out=totals.tokens_out),
            finished=result.finished,
            providers_used=tuple(sorted({e.provider for e in accountant.entries})),
            source_text="",
        )

    harness = Harness([case], arm="full", level="l1", conditions={"offline": True})
    out = harness.run(runner)
    tracer.close()
    return out


def test_a_perfect_answer_scores_perfectly_and_is_still_not_a_measurement(
    tmp_path: Path, sample_case
):
    result = run_harness(tmp_path, sample_case, perfect_l1_script(sample_case.l1))
    headline = result.headline()

    # The plumbing works: the deliverable was parsed and scored.
    assert headline["cases"] == 1
    assert headline["completed"] == 1
    assert headline["metrics"]["numeric_accuracy@strict"] == 1.0

    # And it is refused as a result.
    assert headline["valid_measurement"] is False
    assert "fake/scripted provider" in headline["validity_note"]


def test_a_wrong_answer_is_scored_wrong(tmp_path: Path, sample_case):
    wrong = {k: (v * 2 if v is not None else None) for k, v in sample_case.l1.items()}
    result = run_harness(tmp_path, sample_case, perfect_l1_script(wrong))
    assert result.headline()["metrics"]["numeric_accuracy@strict"] == 0.0


def test_a_missing_deliverable_scores_zero_rather_than_crashing(
    tmp_path: Path, sample_case
):
    """Numbers reported in prose but never written to the file are not an answer."""
    result = run_harness(tmp_path, sample_case, [
        "DONE\nRevenue was 391,035 million and net income 93,736 million."
    ])
    headline = result.headline()
    assert headline["completed"] == 1
    assert headline["metrics"]["numeric_accuracy@strict"] == 0.0


def test_only_the_contract_keys_survive_parsing(tmp_path: Path, sample_case):
    payload = {**{k: {"value": 1.0} for k in L1_KEYS}, "made_up_field": {"value": 2.0}}
    literal = json.dumps(payload)
    result = run_harness(tmp_path, sample_case, [
        "```python\n"
        f"import json\njson.dump(json.loads({literal!r}), open('l1.json','w'))\n"
        "print('written')\n```",
        "DONE\ndone",
    ])
    predicted = result.outcomes[0].predicted
    assert "made_up_field" not in predicted
    assert set(predicted) == set(L1_KEYS)


def test_a_run_where_no_model_was_reachable_is_not_a_measurement():
    """The failure that read as a genuine score of zero."""
    from teamclaw.evaluation.harness import CaseOutcome, EvalRun

    run = EvalRun(arm="full", level="l1")
    run.outcomes.append(CaseOutcome(
        case_id="AAPL-FY2024", level="l1", finished=False, run=RunMetrics(),
        error="NoProviderAvailable: no provider available for purpose=code",
    ))
    assert run.publishable is False
    note = run.headline()["validity_note"]
    assert "no model tokens were consumed" in note
    assert "NoProviderAvailable" in note


def test_the_report_records_the_conditions_it_was_produced_under(
    tmp_path: Path, sample_case
):
    """A score without its conditions cannot be compared to another score."""
    result = run_harness(tmp_path, sample_case, perfect_l1_script(sample_case.l1))
    payload = result.to_json()
    assert payload["conditions"]["offline"] is True
    assert payload["dataset_digest"]
    assert payload["arm"] == "full" and payload["level"] == "l1"
