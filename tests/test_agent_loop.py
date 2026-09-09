"""End-to-end agent loop behaviour, driven by scripted model replies."""

from __future__ import annotations

from pathlib import Path

from teamclaw.context.slots import SlotName
from teamclaw.orchestration.actions import ActionKind, parse_action
from teamclaw.orchestration.agent import Agent
from teamclaw.orchestration.checkpoint import CheckpointStore, check_drift
from teamclaw.orchestration.hitl import (
    AutoDeny,
    Decision,
    InterruptResponse,
    ScriptedResolver,
)


def build_agent(spec, router, workspace, tracer, tmp_path: Path, **kwargs):  # noqa: ANN001
    return Agent(spec, router=router, workspace=workspace, tracer=tracer,
                 run_dir=tmp_path / "run", **kwargs)


def test_a_run_completes_and_persists_its_deliverable(
    agent_spec, scripted, workspace, tracer, tmp_path
):
    router, _ = scripted([
        "THOUGHT: fetch\n```python\nimport json\nfrom tools import sec\n"
        "d = sec.fetch(cik='0000320193')\njson.dump(d, open('filing.json','w'))\nprint(d)\n```",
        "THOUGHT: compute\n```python\nimport json\nfrom tools import fin\n"
        "d = json.load(open('filing.json'))\n"
        "r = fin.gross_margin(revenue=float(d['revenue']),"
        " cost_of_revenue=float(d['cost_of_revenue']))\n"
        "json.dump(r, open('margin.json','w'))\nprint(r)\n```",
        "DONE\nGross margin 46.21% (page 31).",
    ], tracer=tracer)
    agent = build_agent(agent_spec, router, workspace, tracer, tmp_path)
    run = agent.run("Compute Apple FY2024 gross margin.")

    assert run.finished and run.stop_reason == "done"
    assert run.step_count == 3
    assert workspace.exists("filing.json") and workspace.exists("margin.json")
    assert "46.21" in run.final_answer


def test_a_failed_step_is_retried_with_structured_feedback(
    agent_spec, scripted, workspace, tracer, tmp_path
):
    router, provider = scripted([
        "```python\nfrom tools import sec\nprint(sec.nonexistent())\n```",   # wrong name
        "```python\nfrom tools import sec\nprint(sec.fetch(cik='1'))\n```",  # corrected
        "DONE\nok",
    ], tracer=tracer)
    agent = build_agent(agent_spec, router, workspace, tracer, tmp_path)
    run = agent.run("Fetch a filing.")

    assert run.finished
    assert run.steps[0].attempts == 2
    assert run.recovered_steps == 1
    # The retry prompt must have carried the real tool surface back to the model.
    retry_prompt = provider.calls[1][-1].content
    assert "fetch" in retry_prompt and "AttributeError" in retry_prompt


def test_only_printed_output_returns_to_the_model(
    agent_spec, scripted, workspace, tracer, tmp_path
):
    """The property the whole context strategy rests on."""
    router, provider = scripted([
        "```python\nbig = 'X' * 200000\nopen('big.txt','w').write(big)\n"
        "print('wrote', len(big), 'chars')\n```",
        "DONE\ndone",
    ], tracer=tracer)
    agent = build_agent(agent_spec, router, workspace, tracer, tmp_path)
    agent.run("Write a large artefact.")

    second_prompt = "\n".join(m.content for m in provider.calls[1])
    assert "wrote 200000 chars" in second_prompt
    assert "X" * 5000 not in second_prompt
    assert workspace.exists("big.txt")


def test_a_malformed_reply_is_corrected_rather_than_crashing(
    agent_spec, scripted, workspace, tracer, tmp_path
):
    router, provider = scripted([
        "I would call sec.fetch and then compute the margin.",   # prose, no fence
        "```python\nprint('ok')\n```",
        "DONE\nfine",
    ], tracer=tracer)
    agent = build_agent(agent_spec, router, workspace, tracer, tmp_path)
    run = agent.run("Do the thing.")

    assert run.finished
    assert run.steps[0].action.kind is ActionKind.MALFORMED
    assert "fenced Python block" in provider.calls[1][-1].content


def test_repeated_failure_reaches_a_human_and_an_unattended_run_stops(
    agent_spec, scripted, workspace, tracer, tmp_path
):
    router, _ = scripted(["```python\nfrom tools import sec\nsec.wrong()\n```"],
                         tracer=tracer, loop_last=True)
    resolver = AutoDeny()
    agent = build_agent(agent_spec, router, workspace, tracer, tmp_path, resolver=resolver)
    run = agent.run("Fetch a filing.")

    assert not run.finished
    assert run.stop_reason.startswith("hitl_")
    assert resolver.log, "the interrupt must be recorded, not silently swallowed"


def test_a_human_can_unblock_a_stuck_run(
    agent_spec, scripted, workspace, tracer, tmp_path
):
    router, _ = scripted([
        "```python\nfrom tools import sec\nsec.wrong()\n```",
        "```python\nfrom tools import sec\nsec.wrong()\n```",
        "```python\nfrom tools import sec\nsec.wrong()\n```",
        "```python\nprint('took the hint')\n```",
        "DONE\nrecovered",
    ], tracer=tracer)
    resolver = ScriptedResolver([
        InterruptResponse(Decision.GUIDANCE, "Call help(sec) before guessing.")
    ])
    agent = build_agent(agent_spec, router, workspace, tracer, tmp_path, resolver=resolver)
    run = agent.run("Fetch a filing.")

    assert resolver.log
    assert run.finished


def test_persona_and_objective_are_written_to_memory(
    agent_spec, scripted, workspace, tracer, tmp_path
):
    router, _ = scripted(["DONE\nnothing to do"], tracer=tracer)
    agent = build_agent(agent_spec, router, workspace, tracer, tmp_path)
    agent.run("Analyse FY2024 margins.")

    kinds = agent.memory.stats()["by_kind"]
    assert kinds.get("persona") == 1
    assert agent.memory.focus() == "Analyse FY2024 margins."


def test_every_step_writes_a_checkpoint_that_can_be_resumed(
    agent_spec, scripted, workspace, tracer, tmp_path
):
    router, _ = scripted([
        "```python\nopen('a.txt','w').write('1')\nprint('step one')\n```",
        "```python\nopen('b.txt','w').write('2')\nprint('step two')\n```",
        "DONE\nfinished",
    ], tracer=tracer)
    agent = build_agent(agent_spec, router, workspace, tracer, tmp_path)
    run = agent.run("Two steps then stop.")

    store = CheckpointStore(tmp_path / "run")
    assert store.steps(), "checkpoints must exist"
    latest = store.load_latest()
    assert latest is not None and latest.finished
    assert latest.final_answer == run.final_answer
    # The manifest must reflect what is actually on disk.
    assert check_drift(latest, workspace).clean


def test_resume_detects_a_workspace_edited_behind_the_agents_back(
    agent_spec, scripted, workspace, tracer, tmp_path
):
    router, _ = scripted([
        "```python\nopen('a.txt','w').write('1')\nprint('one')\n```",
        "DONE\ndone",
    ], tracer=tracer)
    agent = build_agent(agent_spec, router, workspace, tracer, tmp_path)
    agent.run("One step.")

    store = CheckpointStore(tmp_path / "run")
    checkpoint = store.load_latest()
    assert checkpoint is not None
    workspace.write_text("a.txt", "tampered")
    drift = check_drift(checkpoint, workspace)
    assert not drift.clean
    assert "a.txt" in drift.modified


def test_context_is_rebuilt_each_step_not_appended(
    agent_spec, scripted, workspace, tracer, tmp_path
):
    """Prompts must be re-decided per step, not grown monotonically."""
    router, _ = scripted([
        "```python\nprint('a')\n```",
        "```python\nprint('b')\n```",
        "DONE\nz",
    ], tracer=tracer)
    agent = build_agent(agent_spec, router, workspace, tracer, tmp_path)
    run = agent.run("Two steps.")
    for step in run.steps:
        record = step.ledger
        assert record.used <= record.budget * 1.05, "a step must stay inside its budget"
        assert record.slot(SlotName.SYSTEM) is not None, "instructions are never dropped"
    # Later steps carry more history, but the ledger keeps the total bounded.
    assert all(s.ledger.utilisation <= 1.05 for s in run.steps)


def test_parse_action_handles_small_model_quirks():
    assert parse_action("```py\nprint(1)\n```").kind is ActionKind.RUN
    assert parse_action("done:\nthe answer").kind is ActionKind.DONE
    assert parse_action("```python\nprint(1)").code == "print(1)"   # unterminated
    assert parse_action("just talking").kind is ActionKind.MALFORMED
