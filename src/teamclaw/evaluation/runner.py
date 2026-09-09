"""Wiring a ground-truth case into an agent run and back into a score.

This is the seam between the platform and the eval, and it is where the ablation
arms differ. An arm is a :class:`ArmConfig`: which components are active, which
provider pool is allowed, how much window. Everything else — the tools, the
corpus, the metrics — is held constant, which is what makes two arms comparable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from teamclaw.context.ledger import Budget
from teamclaw.context.memory import MemoryStore
from teamclaw.context.slots import DEFAULT_POLICIES, SlotName, SlotPolicy
from teamclaw.evaluation.harness import CaseRunResult, run_metrics_from
from teamclaw.evaluation.metrics import RunMetrics
from teamclaw.execution.workspace import Workspace
from teamclaw.models.router import Registry, Router
from teamclaw.observability.accounting import Accountant
from teamclaw.observability.trace import Tracer, new_run_id
from teamclaw.orchestration.agent import Agent
from teamclaw.scenarios.dd_finance.groundtruth import GroundTruthCase
from teamclaw.scenarios.dd_finance.sec_client import SecClient
from teamclaw.scenarios.dd_finance.spec import (
    build_spec,
    build_tools,
    l1_objective,
    l2_objective,
    parse_output,
)


# Policy set with the ledger's discipline removed: no hard floors, no
# elasticity distinction, and history free to take everything. This is the
# `-Ledger` arm — naive tail truncation, which is what a system without a
# budget mechanism does by default.
NAIVE_POLICIES: tuple[SlotPolicy, ...] = tuple(
    SlotPolicy(p.name, hard_floor_tokens=0, elasticity=0.5,
               share=(0.60 if p.name is SlotName.HISTORY else 0.08))
    for p in DEFAULT_POLICIES
)


@dataclass
class ArmConfig:
    """One ablation arm. The name is what appears in the results table."""

    name: str
    description: str = ""
    ledger: bool = True            # False -> NAIVE_POLICIES
    memory: bool = True
    compaction: bool = True
    tool_retrieval: bool = True    # False -> every tool signature every step
    skills: bool = True
    window: int = 32_000
    max_steps: int = 14
    allow_paid: bool = False
    prefer_docker: bool = True

    def policies(self) -> tuple[SlotPolicy, ...]:
        return DEFAULT_POLICIES if self.ledger else NAIVE_POLICIES

    def to_json(self) -> dict[str, Any]:
        return {
            "arm": self.name,
            "description": self.description,
            "ledger": self.ledger,
            "memory": self.memory,
            "compaction": self.compaction,
            "tool_retrieval": self.tool_retrieval,
            "skills": self.skills,
            "window": self.window,
            "max_steps": self.max_steps,
            "allow_paid": self.allow_paid,
        }


ARMS: tuple[ArmConfig, ...] = (
    ArmConfig("full", "every component active — the baseline for comparison"),
    ArmConfig("minus-ledger", "naive tail truncation instead of slot bidding",
              ledger=False),
    ArmConfig("minus-tool-retrieval", "all tool signatures injected every step",
              tool_retrieval=False),
    ArmConfig("minus-memory", "no cross-session memory recall", memory=False),
    ArmConfig("minus-compaction", "history never compacted", compaction=False),
    ArmConfig("minus-skills", "no skill bodies, index only", skills=False),
    ArmConfig("strong-naked", "paid model, minimal scaffold — architecture vs model",
              ledger=False, memory=False, compaction=False, tool_retrieval=False,
              skills=False, allow_paid=True, max_steps=8),
)
ARMS_BY_NAME: dict[str, ArmConfig] = {a.name: a for a in ARMS}


@dataclass
class RunnerContext:
    """Shared, expensive handles reused across cases."""

    registry: Registry
    client: SecClient
    runs_root: Path
    workspaces_root: Path
    arm: ArmConfig
    echo: bool = False
    providers_seen: set[str] = field(default_factory=set)


def make_case_runner(ctx: RunnerContext) -> Callable[[GroundTruthCase, str], CaseRunResult]:
    """Build the callable the harness drives, closed over shared handles."""

    def run_case(case: GroundTruthCase, level: str) -> CaseRunResult:
        run_id = new_run_id()
        run_dir = ctx.runs_root / f"{ctx.arm.name}__{case.case_id}__{run_id}"
        tracer = Tracer(run_id, run_dir, echo=ctx.echo)
        workspace = Workspace.create(ctx.workspaces_root, f"{ctx.arm.name}__{case.case_id}")
        workspace.clear()

        accountant = Accountant(sink=run_dir / "usage.jsonl")
        router = Router(
            ctx.registry, accountant=accountant, tracer=tracer,
            agent="dd-analyst", scenario="dd_finance",
        )

        tools = build_tools(ctx.client, workspace)
        spec = build_spec(
            tools=tools, window=ctx.arm.window, max_steps=ctx.arm.max_steps,
            prefer_docker=ctx.arm.prefer_docker,
        )
        spec.policies = ctx.arm.policies()
        if not ctx.arm.tool_retrieval:
            # Inject everything: the arm's whole point is the cost of not choosing.
            spec.tool_k = len(tools)
        if not ctx.arm.skills:
            spec.skill_k = 0
        if not ctx.arm.compaction:
            spec.compaction_threshold = 10.0   # unreachable: never compact
        memory = MemoryStore(agent="dd-analyst") if ctx.arm.memory else MemoryStore(agent="none")

        objective = (
            l1_objective(case.ticker, case.fiscal_year) if level == "l1"
            else l2_objective(case.ticker, case.fiscal_year)
        )

        agent = Agent(
            spec, router=router, workspace=workspace, tracer=tracer,
            run_dir=run_dir, memory=memory,
        )
        if not ctx.arm.memory:
            # Disable recall without disabling the persona write, which is part of
            # the system slot rather than of recall.
            agent.spec.memory_k = 0

        error = ""
        try:
            run = agent.run(objective)
        except Exception as exc:  # noqa: BLE001 - recorded as a failed case
            tracer.close()
            return CaseRunResult(
                predicted={}, run=RunMetrics(), finished=False,
                providers_used=tuple(sorted(ctx.providers_seen)),
                error=f"{type(exc).__name__}: {exc}",
            )

        predicted = parse_output(workspace, level)
        metrics = run_metrics_from(run, tracer, accountant)

        providers = {e.provider for e in accountant.entries}
        ctx.providers_seen |= providers

        # The citation metric needs the plain text the agent actually read.
        source_text = ""
        for candidate in workspace.artifacts():
            if candidate.rel.endswith(".txt") and "filings/" in candidate.rel:
                source_text = workspace.read_text(candidate.rel)
                break

        report = agent.report()
        (run_dir / "agent_report.json").write_text(
            json.dumps({"run": run.to_json(), "report": report,
                        "arm": ctx.arm.to_json()}, indent=1, default=str),
            encoding="utf-8",
        )
        tracer.close()

        return CaseRunResult(
            predicted=predicted, run=metrics, finished=run.finished,
            providers_used=tuple(sorted(providers)), source_text=source_text, error=error,
        )

    return run_case
