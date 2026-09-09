"""The agent loop.

One step is: build context under budget → ask for an action → execute it in the
sandbox → fold the result back in. Everything interesting is in the seams
between those four, and each seam is a decision the spec commits to:

**Context is rebuilt from scratch every step, never appended to.**
An append-only prompt grows monotonically and the only lever left is truncation.
Rebuilding means the ledger re-decides every step which tools, memories and
turns deserve space *for this step*, so a run's 40th prompt can be smaller than
its 4th. This is the whole reason the ledger exists.

**Only stdout returns to the model.** The sandbox writes artefacts; the agent
sees the digest it chose to print. Context growth is therefore bounded by what
the agent decided was worth seeing, not by the size of the data it touched.

**Compaction is checked at step boundaries, after the result is folded in.**
Never mid-step, never mid-reasoning.

**Escalation is bounded.** Repeated identical failures move to a stronger model
once, then to a human, then abort. A model that cannot do a step must not be
allowed to spend the day proving it.

**A checkpoint is written after every step.** Combined with the workspace being
the state, that makes resume a re-read rather than a replay.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from tally.context.compactor import Compactor
from tally.context.ledger import Budget, ContextLedger, LedgerRecord
from tally.context.memory import MemoryKind, MemoryStore
from tally.context.skills import SkillLibrary, default_library
from tally.context.slots import DEFAULT_POLICIES, Item, SlotBid, SlotName, SlotPolicy
from tally.execution.bridge import ToolBridge
from tally.execution.convergence import ConvergenceLoop, Escalation
from tally.execution.registry import ToolRegistry
from tally.execution.sandbox import ExecResult, Sandbox, SandboxLimits, build_sandbox
from tally.execution.stubgen import write_package
from tally.execution.workspace import Workspace
from tally.models.base import Message, Purpose
from tally.models.router import Router
from tally.observability.trace import SpanKind, Tracer
from tally.orchestration.actions import (
    Action,
    ActionKind,
    MALFORMED_FEEDBACK,
    PROTOCOL_PROMPT,
    parse_action,
)
from tally.orchestration.checkpoint import Checkpoint, CheckpointStore, check_drift
from tally.orchestration.hitl import (
    AutoDeny,
    Decision,
    InterruptLog,
    InterruptReason,
    InterruptRequest,
    Resolver,
)


@dataclass
class AgentSpec:
    """Everything a scenario declares. No runtime code belongs here.

    The platform's acceptance criterion is that adding a scenario means writing
    one of these plus its tool modules — if a new scenario needs a runtime
    change, the abstraction is wrong.
    """

    name: str
    scenario: str
    persona: str
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    skills: SkillLibrary = field(default_factory=default_library)
    budget: Budget = field(default_factory=lambda: Budget(window=32_000, max_output=2_048))
    policies: Sequence[SlotPolicy] = DEFAULT_POLICIES
    max_steps: int = 20
    always_tools: tuple[str, ...] = ()
    tool_k: int = 8
    skill_k: int = 2
    memory_k: int = 6
    compaction_threshold: float = 0.70
    sandbox_limits: SandboxLimits = field(default_factory=SandboxLimits)
    approval_required: frozenset[str] = frozenset()
    prefer_docker: bool = True

    def system_prompt(self) -> str:
        return f"{self.persona.strip()}\n\n{PROTOCOL_PROMPT}"


@dataclass
class StepRecord:
    step: int
    action: Action
    result: ExecResult | None
    ledger: LedgerRecord
    attempts: int = 1
    escalated: str = "none"
    observation: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action.to_json(),
            "exec": self.result.to_json() if self.result else None,
            "ledger": self.ledger.to_json(),
            "attempts": self.attempts,
            "escalated": self.escalated,
            "observation_chars": len(self.observation),
        }


@dataclass
class RunResult:
    run_id: str
    objective: str
    finished: bool
    final_answer: str
    steps: list[StepRecord] = field(default_factory=list)
    stop_reason: str = ""
    duration_s: float = 0.0

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def peak_utilisation(self) -> float:
        return max((s.ledger.utilisation for s in self.steps), default=0.0)

    @property
    def failed_steps(self) -> int:
        return sum(1 for s in self.steps if s.result is not None and not s.result.ok)

    @property
    def recovered_steps(self) -> int:
        """Steps that failed at least once and then succeeded."""
        return sum(
            1 for s in self.steps
            if s.attempts > 1 and s.result is not None and s.result.ok
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "objective": self.objective,
            "finished": self.finished,
            "stop_reason": self.stop_reason,
            "steps": self.step_count,
            "failed_steps": self.failed_steps,
            "recovered_steps": self.recovered_steps,
            "peak_utilisation": round(self.peak_utilisation, 4),
            "duration_s": round(self.duration_s, 2),
            "final_answer_chars": len(self.final_answer),
        }


class Agent:
    def __init__(
        self,
        spec: AgentSpec,
        *,
        router: Router,
        workspace: Workspace,
        tracer: Tracer,
        run_dir: Path,
        memory: MemoryStore | None = None,
        sandbox: Sandbox | None = None,
        resolver: Resolver | None = None,
        tools_dir: Path | None = None,
        datasets: Path | None = None,
    ) -> None:
        self.spec = spec
        self.router = router
        self.workspace = workspace
        self.tracer = tracer
        self.run_dir = Path(run_dir)
        self.memory = memory or MemoryStore(agent=spec.name)
        self.resolver = resolver or AutoDeny()
        self.interrupts = InterruptLog(self.run_dir / "interrupts.jsonl")
        self.checkpoints = CheckpointStore(self.run_dir)

        self.ledger = ContextLedger(budget=spec.budget, policies=spec.policies)
        self.compactor = Compactor(
            router=router,
            workspace_dir=workspace.root,
            tracer=tracer,
            max_summary_tokens=min(700, spec.budget.max_output),
        )

        # Materialise the tool package where the sandbox can import it.
        self.tools_dir = Path(tools_dir) if tools_dir else self.run_dir / "toolpkg"
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        write_package(self.tools_dir, spec.tools)

        self.bridge = ToolBridge(
            rpc_dir=workspace.root / ".rpc", tracer=tracer
        ).register_registry(spec.tools)

        self.sandbox = sandbox or build_sandbox(
            workspace=workspace.root,
            tools_dir=self.tools_dir,
            datasets=datasets,
            limits=spec.sandbox_limits,
            tracer=tracer,
            prefer_docker=spec.prefer_docker,
        )

        self.history: list[Item] = []
        self._persona_written = False

    # -- context assembly --------------------------------------------------
    def _bids(self, objective: str, step_hint: str) -> dict[SlotName, SlotBid]:
        led = self.ledger

        system = [led.item(self.spec.system_prompt(), pinned=True, label="system")]
        persona_memory = self.memory.persona()
        if persona_memory:
            system.append(led.item(persona_memory, pinned=True, label="persona-memory"))

        skills = self.spec.skills.select(step_hint, k=self.spec.skill_k)
        skill_items = [
            led.item(s.rendered(), score=1.0, label=f"skill:{s.name}") for s in skills
        ]
        index_text = self.spec.skills.index_text()

        def degrade_skills(evicted, remaining):  # noqa: ANN001
            # Dropped skill bodies collapse to the index: the agent still learns
            # the skill exists and can ask for it, at a fraction of the tokens.
            return Item(text=index_text, tokens=0, label="skill-index", pinned=False)

        selected_tools = self.spec.tools.select(
            step_hint, k=self.spec.tool_k, always=self.spec.always_tools
        )
        tool_items = [
            led.item(self.spec.tools.signature_block([t]), score=1.0, label=t.name)
            for t in selected_tools
        ]

        workspace_items = [led.item(self.workspace.digest(), score=1.0, label="workspace")]

        recalls = self.memory.recall(
            f"{objective}\n{step_hint}",
            k=self.spec.memory_k,
            kinds=(MemoryKind.FACT, MemoryKind.PROCEDURE, MemoryKind.EPISODE, MemoryKind.FOCUS),
        )
        memory_items = [
            led.item(r.record.text, score=r.score, label=f"mem:{r.record.record_id}")
            for r in recalls
        ]

        return {
            SlotName.SYSTEM: SlotBid(led.policy_for(SlotName.SYSTEM), system),
            SlotName.SKILL: SlotBid(
                led.policy_for(SlotName.SKILL), skill_items, degrade=degrade_skills
            ),
            SlotName.TOOLS: SlotBid(led.policy_for(SlotName.TOOLS), tool_items),
            SlotName.WORKSPACE: SlotBid(led.policy_for(SlotName.WORKSPACE), workspace_items),
            SlotName.MEMORY: SlotBid(led.policy_for(SlotName.MEMORY), memory_items),
            SlotName.HISTORY: SlotBid(led.policy_for(SlotName.HISTORY), list(self.history)),
        }

    def _turn(self, objective: str, step: int, feedback: str | None) -> str:
        if feedback:
            return feedback
        if step == 0:
            return f"Objective:\n{objective}\n\nTake the first step."
        return "Continue. Take the next step, or reply DONE if the objective is met."

    # -- the loop ----------------------------------------------------------
    def run(self, objective: str, *, resume: bool = False) -> RunResult:
        started = time.time()
        steps: list[StepRecord] = []
        stop_reason = "max_steps"
        finished = False
        final_answer = ""
        first_step = 0

        if resume and (cp := self.checkpoints.load_latest()) is not None:
            drift = check_drift(cp, self.workspace)
            self.history = [
                Item(text=h["text"], tokens=self.ledger.count(h["text"]),
                     score=float(i), kind=h.get("kind", "user"), label=h.get("label", ""))
                for i, h in enumerate(cp.history)
            ]
            first_step = cp.step
            objective = cp.objective or objective
            with self.tracer.span(SpanKind.STEP, "resume") as sp:
                sp.set(from_step=cp.step, drift=drift.to_json(), summary=drift.summary())
            if cp.finished:
                return RunResult(
                    run_id=self.tracer.run_id, objective=objective, finished=True,
                    final_answer=cp.final_answer, stop_reason="already_finished",
                    duration_s=0.0,
                )

        if not self._persona_written:
            self.memory.write(
                self.spec.persona.strip()[:2000],
                kind=MemoryKind.PERSONA,
                key=f"persona:{self.spec.name}",
                confidence=1.0,
                source="agent-spec",
            )
            self.memory.set_focus(objective, source="run-start")
            self._persona_written = True

        with self.bridge.serving():
            feedback: str | None = None
            step_hint = objective

            for step in range(first_step, self.spec.max_steps):
                convergence = ConvergenceLoop(registry=self.spec.tools)
                record = self._step(
                    objective=objective, step=step, step_hint=step_hint,
                    feedback=feedback, convergence=convergence,
                )
                steps.append(record)
                feedback = None

                if record.action.kind is ActionKind.MALFORMED:
                    feedback = MALFORMED_FEEDBACK
                    step_hint = objective
                elif record.action.is_terminal:
                    finished, final_answer, stop_reason = True, record.action.final, "done"
                    self._save_checkpoint(step + 1, objective, finished=True,
                                          final_answer=final_answer)
                    break
                elif record.escalated == Escalation.ABORT.value:
                    stop_reason = "convergence_abort"
                    self._save_checkpoint(step + 1, objective)
                    break
                elif record.escalated == Escalation.HITL.value:
                    response = self._interrupt(
                        InterruptReason.REPEATED_FAILURE, step,
                        "The agent has repeated the same failure and cannot proceed.",
                        record.observation,
                    )
                    if not response.continues:
                        stop_reason = f"hitl_{response.decision.value}"
                        self._save_checkpoint(step + 1, objective)
                        break
                    feedback = response.message or "A reviewer asked you to try a different approach."
                else:
                    step_hint = record.action.thought or objective

                self._fold_in(record)
                self._maybe_compact(objective, record)
                self._save_checkpoint(step + 1, objective)

        if not finished and stop_reason == "max_steps":
            response = self._interrupt(
                InterruptReason.BUDGET_EXHAUSTED, self.spec.max_steps,
                f"Step budget of {self.spec.max_steps} exhausted with no final answer.",
                steps[-1].observation if steps else "",
            )
            stop_reason = f"max_steps_{response.decision.value}"

        return RunResult(
            run_id=self.tracer.run_id, objective=objective, finished=finished,
            final_answer=final_answer, steps=steps, stop_reason=stop_reason,
            duration_s=time.time() - started,
        )

    # -- one step ----------------------------------------------------------
    def _step(
        self, *, objective: str, step: int, step_hint: str,
        feedback: str | None, convergence: ConvergenceLoop,
    ) -> StepRecord:
        with self.tracer.span(SpanKind.STEP, f"step-{step}") as span:
            bids = self._bids(objective, step_hint)
            with self.tracer.span(SpanKind.CONTEXT_BUILD, f"step-{step}") as cspan:
                built = self.ledger.build(bids, user_turn=self._turn(objective, step, feedback))
                cspan.set(**built.record.to_json())

            purpose = Purpose.CODE
            allow_paid = False
            action = self._ask(built.messages, purpose=purpose, task=objective,
                               allow_paid=allow_paid)

            if action.kind is not ActionKind.RUN:
                span.set(**action.to_json())
                return StepRecord(step=step, action=action, result=None,
                                  ledger=built.record, observation=action.final or action.raw[:500])

            # Attempt / feedback / escalate until the step converges or gives up.
            attempts = 0
            escalated = Escalation.NONE
            result: ExecResult | None = None
            messages = list(built.messages)

            while attempts < convergence.max_attempts:
                attempts += 1
                result = self.sandbox.run(action.code, limits=self.spec.sandbox_limits)
                fb = convergence.observe(result, code=action.code)
                if fb is None:
                    break
                escalated = fb.escalation
                if fb.escalation in {Escalation.HITL, Escalation.ABORT}:
                    break
                # Feed the structured failure back and let the model rewrite.
                messages = [*messages, Message.assistant(action.raw), Message.user(fb.text)]
                action = self._ask(
                    messages, purpose=purpose, task=objective,
                    allow_paid=fb.escalation is Escalation.MODEL,
                    escalate=fb.escalation is Escalation.MODEL,
                )
                if action.kind is not ActionKind.RUN:
                    break

            observation = self._observation(result, action)
            span.set(attempts=attempts, escalated=escalated.value, **action.to_json())
            return StepRecord(step=step, action=action, result=result, ledger=built.record,
                              attempts=attempts, escalated=escalated.value,
                              observation=observation)

    def _ask(
        self, messages: list[Message], *, purpose: Purpose, task: str,
        allow_paid: bool = False, escalate: bool = False,
    ) -> Action:
        # Escalation raises the purpose tier: PLAN routes to the strongest
        # configured free model, which is a different pool from CODE.
        effective = Purpose.PLAN if escalate else purpose
        completion = self.router.complete(
            effective, messages, max_tokens=self.spec.budget.max_output,
            task=task, allow_paid=allow_paid,
        )
        return parse_action(completion.text)

    # -- folding results back ----------------------------------------------
    @staticmethod
    def _observation(result: ExecResult | None, action: Action) -> str:
        if result is None:
            return ""
        if result.ok:
            body = result.stdout.strip() or "(no output)"
            return f"stdout:\n{body}"
        tail = (result.stderr or "").strip().splitlines()
        return "stderr (last lines):\n" + "\n".join(tail[-12:])

    def _fold_in(self, record: StepRecord) -> None:
        """Append the action and its observation to history as two turns.

        Scores are the step index, so the ``HISTORY`` slot's highest-score-first
        eviction naturally keeps the recent tail and drops the oldest turns.
        """
        if record.action.code:
            text = f"THOUGHT: {record.action.thought}\n```python\n{record.action.code}\n```"
            self.history.append(
                Item(text=text, tokens=self.ledger.count(text), score=float(record.step),
                     kind="assistant", label=f"action-{record.step}")
            )
        if record.observation:
            self.history.append(
                Item(text=record.observation, tokens=self.ledger.count(record.observation),
                     score=float(record.step), kind="user", label=f"obs-{record.step}")
            )

    def _maybe_compact(self, objective: str, record: StepRecord) -> None:
        if not self.compactor.should_compact(
            utilisation=record.ledger.utilisation,
            at_step_boundary=True,
            turns=len(self.history),
            threshold=self.spec.compaction_threshold,
        ):
            return
        self.history, _ = self.compactor.compact(
            self.history, counter=self.ledger.count, task=objective
        )

    # -- checkpoint / hitl -------------------------------------------------
    def _save_checkpoint(
        self, step: int, objective: str, *, finished: bool = False, final_answer: str = ""
    ) -> None:
        self.checkpoints.save(
            Checkpoint(
                run_id=self.tracer.run_id,
                step=step,
                objective=objective,
                history=[
                    {"text": i.text, "kind": i.kind, "label": i.label} for i in self.history
                ],
                manifest=self.workspace.snapshot_manifest(),
                finished=finished,
                final_answer=final_answer,
            )
        )

    def _interrupt(
        self, reason: InterruptReason, step: int, question: str, context: str
    ):
        request = InterruptRequest(reason=reason, step=step, question=question, context=context)
        with self.tracer.span(SpanKind.HITL, reason.value) as sp:
            response = self.resolver.resolve(request)
            sp.set(decision=response.decision.value)
        self.interrupts.record(request, response)
        return response

    # -- reporting ---------------------------------------------------------
    def report(self) -> dict[str, Any]:
        return {
            "sandbox": {"backend": self.sandbox.backend,
                        "isolation": self.sandbox.isolation_level()},
            "tools": {"count": len(self.spec.tools), "modules": self.spec.tools.modules()},
            "bridge": self.bridge.stats(),
            "compaction": self.compactor.stats(),
            "memory": self.memory.stats(),
            "interrupts": self.interrupts.stats(),
            "router": self.router.report(),
        }
