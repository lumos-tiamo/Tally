"""Task graph, sub-agent isolation, compaction gating, and harness validity."""

from __future__ import annotations

import pytest

from teamclaw.context.compactor import Compactor
from teamclaw.context.slots import Item
from teamclaw.context.tokenizer import count_tokens
from teamclaw.evaluation.harness import CaseOutcome, EvalRun
from teamclaw.orchestration.graph import CyclicGraph, TaskGraph, UnknownDependency
from teamclaw.orchestration.subagent import DelegationDepthExceeded, SubAgentFactory


# --- task graph -----------------------------------------------------------
def test_a_failed_node_short_circuits_its_descendants():
    graph = (TaskGraph()
             .add("fetch", lambda s: {"ok": True})
             .add("extract", lambda s: {"revenue": 1}, depends_on=["fetch"])
             .add("verify", lambda s: (_ for _ in ()).throw(ValueError("mismatch")),
                  depends_on=["extract"])
             .add("report", lambda s: "md", depends_on=["verify"]))
    result = graph.run()
    assert not result.ok
    assert result.results["report"].skipped
    assert "verify" in result.results["report"].skip_reason


def test_an_optional_node_does_not_block_descendants():
    graph = (TaskGraph()
             .add("a", lambda s: 1)
             .add("flaky", lambda s: (_ for _ in ()).throw(RuntimeError("x")),
                  depends_on=["a"], optional=True)
             .add("b", lambda s: 2, depends_on=["flaky"]))
    result = graph.run()
    assert result.results["b"].ok


def test_ordering_is_deterministic():
    """An arm that reorders between runs is not reproducible."""
    def build() -> TaskGraph:
        graph = TaskGraph()
        for name in ("d", "b", "c", "a"):
            graph.add(name, lambda s: name)
        graph.add("last", lambda s: 1, depends_on=["a", "b", "c", "d"])
        return graph

    assert build().topological_order() == build().topological_order()


def test_a_cycle_is_rejected():
    graph = TaskGraph().add("a", lambda s: 1, depends_on=["b"]).add(
        "b", lambda s: 2, depends_on=["a"])
    with pytest.raises(CyclicGraph):
        graph.topological_order()


def test_an_unknown_dependency_is_rejected():
    graph = TaskGraph().add("a", lambda s: 1, depends_on=["nope"])
    with pytest.raises(UnknownDependency):
        graph.topological_order()


# --- sub-agents -----------------------------------------------------------
def test_a_subagent_returns_a_digest_not_its_transcript(
    agent_spec, scripted, workspace, tracer, tmp_path
):
    """Delegation must reduce context pressure, not relocate it."""
    from teamclaw.orchestration.agent import Agent

    router, _ = scripted([
        "```python\nopen('extract.json','w').write('{\"revenue\": 1}')\nprint('wrote it')\n```",
        "DONE\nExtracted revenue to extract.json.",
    ], tracer=tracer, loop_last=True)
    parent = Agent(agent_spec, router=router, workspace=workspace, tracer=tracer,
                   run_dir=tmp_path / "run")
    factory = SubAgentFactory(parent=parent)
    result = factory.delegate(name="extractor", objective="Extract revenue.")

    observation = result.as_observation()
    assert "Sub-agent `extractor`" in observation
    assert "```python" not in observation, "the child's code must not reach the parent"
    assert any("extract.json" in path for path in result.artifacts)
    assert len(observation) < 1_600


def test_a_subagent_gets_a_separate_memory_by_default(
    agent_spec, scripted, workspace, tracer, tmp_path
):
    from teamclaw.orchestration.agent import Agent

    router, _ = scripted(["DONE\nnothing"], tracer=tracer, loop_last=True)
    parent = Agent(agent_spec, router=router, workspace=workspace, tracer=tracer,
                   run_dir=tmp_path / "run")
    parent.memory.write("a belief the parent holds", key="p")
    factory = SubAgentFactory(parent=parent)
    factory.delegate(name="child", objective="Do a thing.")
    assert parent.memory.stats()["active"] >= 1


def test_delegation_depth_is_capped(agent_spec, scripted, workspace, tracer, tmp_path):
    from teamclaw.orchestration.agent import Agent

    router, _ = scripted(["DONE\nx"], tracer=tracer, loop_last=True)
    parent = Agent(agent_spec, router=router, workspace=workspace, tracer=tracer,
                   run_dir=tmp_path / "run")
    factory = SubAgentFactory(parent=parent, depth=2, max_depth=2)
    with pytest.raises(DelegationDepthExceeded):
        factory.delegate(name="too-deep", objective="Recurse.")


def test_subagent_cost_is_attributed_separately(
    agent_spec, scripted, workspace, tracer, tmp_path
):
    from teamclaw.orchestration.agent import Agent

    router, _ = scripted(["DONE\nx"], tracer=tracer, loop_last=True)
    parent = Agent(agent_spec, router=router, workspace=workspace, tracer=tracer,
                   run_dir=tmp_path / "run")
    parent.run("Parent objective.")
    SubAgentFactory(parent=parent).delegate(name="kid", objective="Child objective.")
    agents = set(router.accountant.by("agent"))
    assert any("kid" in a for a in agents)


# --- compaction gating ----------------------------------------------------
def test_compaction_never_fires_mid_reasoning():
    compactor = Compactor()
    assert not compactor.should_compact(utilisation=0.99, at_step_boundary=False, turns=50)
    assert compactor.should_compact(utilisation=0.99, at_step_boundary=True, turns=50)


def test_compaction_below_the_threshold_is_skipped():
    compactor = Compactor()
    assert not compactor.should_compact(utilisation=0.5, at_step_boundary=True, turns=50)


def test_a_compaction_that_would_not_shrink_is_aborted():
    """Spending a call to make the window just as full is worse than nothing."""
    compactor = Compactor()
    items = [Item(text=f"line {i} with figure {i}000", tokens=8, kind="user",
                  label=f"t{i}") for i in range(15)]
    kept, result = compactor.compact(items, counter=count_tokens, task="t")
    assert result.replaced == 0
    assert len(kept) == len(items)
    assert any("aborted" in note for note in result.notes)


def test_a_redundant_middle_is_genuinely_compressed():
    compactor = Compactor()
    filler = "Understood, I will continue with the requested analysis now. " * 6
    items = [Item(text=filler, tokens=count_tokens(filler), kind="assistant",
                  label=f"c{i}") for i in range(15)]
    kept, result = compactor.compact(items, counter=count_tokens, task="t")
    assert result.replaced > 0
    assert result.compression_ratio < 0.5
    assert len(kept) < len(items)


# --- harness validity guard ----------------------------------------------
def test_a_run_served_by_a_fake_provider_is_not_a_measurement():
    """The guard that stops a plumbing check from being reported as a result."""
    run = EvalRun(arm="full", level="l1")
    run.outcomes.append(CaseOutcome(case_id="X-FY2024", level="l1", finished=True,
                                    providers_used=("fake",)))
    assert not run.publishable
    headline = run.headline()
    assert headline["valid_measurement"] is False
    assert "NOT A MEASUREMENT" in headline["validity_note"]


def test_a_run_served_by_real_providers_is_publishable():
    run = EvalRun(arm="full", level="l1")
    run.outcomes.append(CaseOutcome(case_id="X-FY2024", level="l1", finished=True,
                                    providers_used=("gemini",)))
    assert run.publishable
    assert run.headline()["validity_note"] == ""


def test_an_empty_run_is_not_publishable():
    assert not EvalRun(arm="full", level="l1").publishable
