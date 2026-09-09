"""Sub-agent delegation with context isolation.

The reason to spawn a sub-agent is *not* parallelism — it is that the parent
should not have to hold the child's working context. A parent that delegates
"extract the income statement from this 300-page filing" and receives back 40
steps of the child's reasoning has gained nothing: the tokens moved, they did not
disappear.

So the contract is deliberately narrow:

* the child gets a **fresh ledger and empty history**, its own workspace subtree,
  and only the objective text the parent wrote;
* the parent gets back a **digest and artefact paths**, never the child's
  transcript;
* file handoff happens through the filesystem, since the child's subtree lives
  inside the parent's workspace and is readable from it.

That is what makes delegation reduce context pressure rather than relocate it.

Cost is attributed to the child (``Router.child``), so the accounting can answer
"which sub-agent burned the free quota" — the question a flat token counter
cannot.

Delegation depth is capped. An agent that can spawn agents that can spawn agents
will, given a hard task, build a tree instead of solving it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from tally.context.memory import MemoryStore
from tally.execution.workspace import Workspace
from tally.observability.trace import SpanKind, Tracer
from tally.orchestration.agent import Agent, AgentSpec, RunResult

MAX_DEPTH = 2


class DelegationDepthExceeded(RuntimeError):
    pass


@dataclass
class DelegationResult:
    name: str
    objective: str
    finished: bool
    summary: str
    artifacts: list[str] = field(default_factory=list)
    steps: int = 0
    stop_reason: str = ""
    workspace_rel: str = ""

    def as_observation(self, *, max_chars: int = 1200) -> str:
        """What the parent actually sees. Bounded, and never the transcript."""
        head = f"Sub-agent `{self.name}` {'completed' if self.finished else 'did not complete'} " \
               f"in {self.steps} steps ({self.stop_reason})."
        files = (
            "Artefacts (readable from your workspace):\n"
            + "\n".join(f"  {p}" for p in self.artifacts[:20])
        ) if self.artifacts else "It produced no artefacts."
        body = self.summary.strip()[:max_chars]
        return f"{head}\n\n{files}\n\nIts report:\n{body}"

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "objective": self.objective[:300],
            "finished": self.finished,
            "steps": self.steps,
            "stop_reason": self.stop_reason,
            "artifacts": self.artifacts[:40],
            "workspace": self.workspace_rel,
            "summary_chars": len(self.summary),
        }


@dataclass
class SubAgentFactory:
    """Builds isolated children from a parent's runtime handles."""

    parent: Agent
    depth: int = 0
    max_depth: int = MAX_DEPTH
    results: list[DelegationResult] = field(default_factory=list)

    def delegate(
        self,
        *,
        name: str,
        objective: str,
        spec: AgentSpec | None = None,
        inherit_memory: bool = False,
        max_steps: int | None = None,
    ) -> DelegationResult:
        if self.depth >= self.max_depth:
            raise DelegationDepthExceeded(
                f"delegation depth {self.depth} would exceed max_depth={self.max_depth}"
            )

        child_spec = spec or self.parent.spec
        child_spec = AgentSpec(
            name=f"{child_spec.name}:{name}",
            scenario=child_spec.scenario,
            persona=child_spec.persona,
            tools=child_spec.tools,
            skills=child_spec.skills,
            budget=child_spec.budget,
            policies=child_spec.policies,
            max_steps=max_steps or min(child_spec.max_steps, 12),
            always_tools=child_spec.always_tools,
            tool_k=child_spec.tool_k,
            skill_k=child_spec.skill_k,
            memory_k=child_spec.memory_k,
            compaction_threshold=child_spec.compaction_threshold,
            sandbox_limits=child_spec.sandbox_limits,
            approval_required=child_spec.approval_required,
            prefer_docker=child_spec.prefer_docker,
        )

        child_ws: Workspace = self.parent.workspace.child(name)
        # A fresh store unless explicitly inherited: a child that recalls the
        # parent's whole memory is not isolated, and cross-contaminated memory is
        # how one sub-agent's wrong belief becomes every sibling's wrong belief.
        child_memory = (
            self.parent.memory if inherit_memory else MemoryStore(agent=child_spec.name)
        )

        with self.parent.tracer.span(SpanKind.SUBAGENT, name) as span:
            child = Agent(
                child_spec,
                router=self.parent.router.child(agent=child_spec.name),
                workspace=child_ws,
                tracer=self.parent.tracer,
                run_dir=self.parent.run_dir / "subagents" / name,
                memory=child_memory,
                resolver=self.parent.resolver,
                tools_dir=self.parent.tools_dir,
            )
            run: RunResult = child.run(objective)
            rel_root = child_ws.root.relative_to(self.parent.workspace.root)
            artifacts = [str(rel_root / a.rel) for a in child_ws.artifacts()]
            result = DelegationResult(
                name=name,
                objective=objective,
                finished=run.finished,
                summary=run.final_answer or "(no report produced)",
                artifacts=artifacts,
                steps=run.step_count,
                stop_reason=run.stop_reason,
                workspace_rel=str(rel_root),
            )
            span.set(**result.to_json())

        self.results.append(result)
        return result

    def fan_out(self, jobs: Sequence[tuple[str, str]], **kwargs: Any) -> list[DelegationResult]:
        """Run several children in sequence.

        Sequential on purpose: the binding constraint on a zero-budget run is
        provider requests-per-minute, so firing children concurrently converts
        available parallelism into 429s. The router's quota tracker would degrade
        them all onto the same fallback anyway.
        """
        return [self.delegate(name=name, objective=obj, **kwargs) for name, obj in jobs]

    def stats(self) -> dict[str, Any]:
        return {
            "delegations": len(self.results),
            "completed": sum(1 for r in self.results if r.finished),
            "total_child_steps": sum(r.steps for r in self.results),
            "artifacts": sum(len(r.artifacts) for r in self.results),
        }
