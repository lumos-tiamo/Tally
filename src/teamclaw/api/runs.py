"""Run orchestration for the API: threads, event fan-out, cancellation, HITL.

The impedance mismatch, and how it is handled
---------------------------------------------
The agent loop is synchronous and blocking by design — it spawns containers and
waits on them. FastAPI serves on an event loop. Running an agent on that loop
would stall every other request for the length of a step, so runs execute on a
bounded thread pool and communicate back through per-run queues.

Three consequences are handled explicitly rather than hoped about:

**Event delivery must never break a run.** The tracer's subscription is called
from the worker thread; it hands the payload to an ``asyncio.Queue`` via
``call_soon_threadsafe``. If no one is listening, or the loop is gone, the
enqueue is dropped. A browser closing a tab cannot fail an agent mid-step.

**Cancellation is cooperative, because it has to be.** A thread cannot be killed
safely while it holds a subprocess and a workspace. So cancel sets a flag the
loop checks between steps, and the caller is told the run will stop at the next
boundary rather than immediately. Claiming instant cancellation would be a lie
that leaves containers running.

**Replay matters more than live streaming.** A client that connects mid-run, or
reconnects, gets the spans already emitted before the live feed — read from the
run's trace file, which is the durable record. Without that, a refresh loses the
first half of a run.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from teamclaw.context.memory import MemoryStore
from teamclaw.evaluation.metrics import RunMetrics
from teamclaw.execution.workspace import Workspace
from teamclaw.observability.accounting import Accountant
from teamclaw.observability.trace import Tracer, new_run_id
from teamclaw.orchestration.agent import Agent, AgentSpec
from teamclaw.orchestration.hitl import (
    Decision,
    InterruptRequest,
    InterruptResponse,
)
from teamclaw.store import AgentDef, AgentStore, RunState

CANCEL_CHECK_NOTE = "cancellation takes effect at the next step boundary"


class RunCancelled(RuntimeError):
    pass


@dataclass
class SessionResolver:
    """A HITL resolver that asks through the session store and waits.

    The API cannot block a worker thread indefinitely waiting for a human, so the
    wait is bounded. On timeout it declines — the same default as an unattended
    run — because a run that silently waits forever holds a container and a
    thread slot for nothing.
    """

    session: Any
    run_id: str
    timeout_s: float = 300.0
    poll_s: float = 1.0
    log: list[InterruptRequest] = field(default_factory=list)

    def resolve(self, request: InterruptRequest) -> InterruptResponse:
        self.log.append(request)
        self.session.push_hitl(self.run_id, {
            "reason": request.reason.value,
            "step": request.step,
            "question": request.question,
            "context": request.context[:2000],
            "asked_at": time.time(),
        })
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            answer = self.session.hitl_answer(self.run_id)
            if answer:
                decision = str(answer.get("decision", "deny"))
                try:
                    parsed = Decision(decision)
                except ValueError:
                    parsed = Decision.DENY
                return InterruptResponse(parsed, str(answer.get("message", "")))
            time.sleep(self.poll_s)
        self.session.resolve_hitl(self.run_id, "deny", "no answer within the timeout")
        return InterruptResponse(
            Decision.DENY,
            f"No one answered within {self.timeout_s:.0f}s. Treat the step as "
            "failed and finish with a statement of what blocked you rather than "
            "guessing.",
        )


@dataclass
class RunHandle:
    run_id: str
    agent_id: str
    objective: str
    queues: list[asyncio.Queue] = field(default_factory=list)
    cancelled: threading.Event = field(default_factory=threading.Event)
    started_at: float = field(default_factory=time.time)
    state: RunState = RunState.QUEUED
    trace_path: Path | None = None

    def request_cancel(self) -> None:
        self.cancelled.set()


class RunManager:
    """Owns the worker pool, the live handles, and the run lifecycle."""

    def __init__(
        self,
        *,
        store: AgentStore,
        session: Any,
        registry_factory: Callable[[], Any],
        spec_factory: Callable[[AgentDef, Workspace], AgentSpec],
        runs_root: Path,
        workspaces_root: Path,
        max_workers: int = 2,
        hitl_timeout_s: float = 300.0,
    ) -> None:
        self.store = store
        self.session = session
        self.registry_factory = registry_factory
        self.spec_factory = spec_factory
        self.runs_root = Path(runs_root)
        self.workspaces_root = Path(workspaces_root)
        self.hitl_timeout_s = hitl_timeout_s
        # Bounded on purpose: each concurrent run may hold a container, and the
        # free-tier quota is the real ceiling anyway. Queued runs wait rather
        # than all contending for the same rate limit.
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="teamclaw-run")
        self._handles: dict[str, RunHandle] = {}
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Called from the app's lifespan so worker threads can reach the loop."""
        self._loop = loop

    def shutdown(self) -> None:
        for handle in list(self._handles.values()):
            handle.request_cancel()
        self._pool.shutdown(wait=False, cancel_futures=True)

    # -- lifecycle ---------------------------------------------------------
    def start(self, agent: AgentDef, objective: str,
              conversation_id: str | None = None) -> RunHandle:
        run_id = new_run_id()
        handle = RunHandle(run_id=run_id, agent_id=agent.id, objective=objective)
        with self._lock:
            self._handles[run_id] = handle
        self.store.create_run(run_id, agent.id, objective, conversation_id)
        if conversation_id:
            self.session.set_active_run(conversation_id, run_id)
        self._pool.submit(self._execute, handle, agent, conversation_id)
        return handle

    def handle(self, run_id: str) -> RunHandle | None:
        return self._handles.get(run_id)

    def cancel(self, run_id: str) -> dict[str, Any]:
        handle = self._handles.get(run_id)
        if handle is None:
            return {"cancelled": False, "reason": "no live run with that id"}
        handle.request_cancel()
        return {"cancelled": True, "note": CANCEL_CHECK_NOTE}

    def live(self) -> list[dict[str, Any]]:
        return [
            {"run_id": h.run_id, "agent_id": h.agent_id, "state": h.state.value,
             "objective": h.objective[:160], "listeners": len(h.queues),
             "elapsed_s": round(time.time() - h.started_at, 1),
             "cancelling": h.cancelled.is_set()}
            for h in self._handles.values()
        ]

    # -- event fan-out -----------------------------------------------------
    def attach(self, run_id: str) -> asyncio.Queue | None:
        handle = self._handles.get(run_id)
        if handle is None:
            return None
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        handle.queues.append(queue)
        return queue

    def detach(self, run_id: str, queue: asyncio.Queue) -> None:
        handle = self._handles.get(run_id)
        if handle and queue in handle.queues:
            handle.queues.remove(queue)

    def _emit(self, handle: RunHandle, event: dict[str, Any]) -> None:
        """Hand an event to every listener. Never raises into the run."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        for queue in list(handle.queues):
            def _put(q: asyncio.Queue = queue, payload: dict[str, Any] = event) -> None:
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    # A listener that cannot keep up loses events rather than
                    # applying backpressure to the agent.
                    pass

            try:
                loop.call_soon_threadsafe(_put)
            except RuntimeError:
                # Loop shut down mid-run; drop the event.
                return

    def replay(self, run_id: str) -> list[dict[str, Any]]:
        """Spans already emitted, from the durable trace, for a late joiner."""
        handle = self._handles.get(run_id)
        path = handle.trace_path if handle and handle.trace_path else None
        if path is None:
            record = self.store.get_run(run_id)
            path = Path(record.trace_path) if record and record.trace_path else None
        return Tracer.read(path) if path else []

    # -- the worker --------------------------------------------------------
    def _execute(self, handle: RunHandle, agent_def: AgentDef,
                 conversation_id: str | None) -> None:
        run_id = handle.run_id
        run_dir = self.runs_root / run_id
        workspace = Workspace.create(self.workspaces_root, run_id)
        tracer = Tracer(run_id, run_dir)
        handle.trace_path = tracer.path
        handle.state = RunState.RUNNING
        self.store.update_run(run_id, state=RunState.RUNNING,
                              workspace_path=str(workspace.root),
                              trace_path=str(tracer.path))
        self._emit(handle, {"type": "state", "state": "running", "run_id": run_id})

        unsubscribe = tracer.subscribe(
            lambda payload: self._emit(handle, {"type": "span", **payload})
        )

        accountant = Accountant(sink=run_dir / "usage.jsonl")
        from teamclaw.models.router import Router  # noqa: PLC0415 - import cycle

        router = Router(self.registry_factory(), accountant=accountant, tracer=tracer,
                        agent=agent_def.name, scenario=agent_def.scenario)
        spec = self.spec_factory(agent_def, workspace)

        try:
            runner = Agent(
                spec, router=router, workspace=workspace, tracer=tracer,
                run_dir=run_dir,
                memory=MemoryStore(path=run_dir.parent / "memory" / f"{agent_def.name}.jsonl",
                                   agent=agent_def.name),
                resolver=SessionResolver(self.session, run_id,
                                         timeout_s=self.hitl_timeout_s),
            )
            self._install_cancellation(runner, handle)
            result = runner.run(handle.objective)
            totals = accountant.total()
            report = runner.report()

            final_state = RunState.FINISHED if result.finished else (
                RunState.CANCELLED if handle.cancelled.is_set() else RunState.FAILED
            )
            if result.stop_reason.startswith("hitl_") and not result.finished:
                final_state = RunState.NEEDS_HUMAN

            handle.state = final_state
            self.store.update_run(
                run_id, state=final_state, stop_reason=result.stop_reason,
                final_answer=result.final_answer, steps=result.step_count,
                tokens_in=totals.tokens_in, tokens_out=totals.tokens_out,
                cost_usd=totals.cost_usd, peak_utilisation=result.peak_utilisation,
                duration_s=result.duration_s,
                sandbox_backend=runner.sandbox.backend,
                sandbox_isolation=runner.sandbox.isolation_level(),
                report=report,
            )
            if conversation_id and result.final_answer:
                self.store.add_message(conversation_id, "assistant",
                                       result.final_answer, run_id=run_id)
            self._emit(handle, {"type": "done", "run_id": run_id,
                                "state": final_state.value,
                                "result": result.to_json()})
        except Exception as exc:  # noqa: BLE001 - recorded, never re-raised into the pool
            handle.state = RunState.FAILED
            self.store.update_run(run_id, state=RunState.FAILED,
                                  error=f"{type(exc).__name__}: {exc}")
            self._emit(handle, {"type": "error", "run_id": run_id,
                                "error": f"{type(exc).__name__}: {exc}"})
        finally:
            unsubscribe()
            tracer.close()
            if conversation_id:
                self.session.clear_active_run(conversation_id)
            self.session.drop("hitl", run_id)
            # Keep the handle briefly so a client can fetch the terminal event,
            # then let it go; the durable record is in the database and on disk.
            threading.Timer(60.0, lambda: self._handles.pop(run_id, None)).start()

    @staticmethod
    def _install_cancellation(runner: Agent, handle: RunHandle) -> None:
        """Make the agent stop at its next step boundary when cancel is set.

        Wrapping the step function is the least invasive place to check: it runs
        once per step, outside the sandbox, so a cancelled run never leaves a
        container half-finished.
        """
        original_step = runner._step

        def guarded(*args: Any, **kwargs: Any) -> Any:
            if handle.cancelled.is_set():
                raise RunCancelled("cancelled by request")
            return original_step(*args, **kwargs)

        runner._step = guarded  # type: ignore[method-assign]
