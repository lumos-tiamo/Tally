"""Run control, the live WebSocket feed, and run introspection.

The introspection endpoints are the point of this module. A run's trace already
records every prompt's slot allocation and every sandbox exit code; these turn
that into something a browser can draw, because the context-engineering work is
the hardest part of the system to explain in words and the easiest to show.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    status,
)

from teamclaw.api.deps import AppState, app_state
from teamclaw.api.schemas import HitlAnswer, RunCreate
from teamclaw.observability.trace import Tracer

router = APIRouter(prefix="/api", tags=["runs"])


@router.get("/runs")
def list_runs(agent_id: str | None = None, limit: int = 50,
              state: AppState = Depends(app_state)) -> dict[str, Any]:
    return {
        "runs": [r.to_json() for r in state.store.list_runs(agent_id=agent_id,
                                                            limit=min(limit, 200))],
        "live": state.run_manager.live(),
    }


@router.post("/agents/{agent_id}/runs", status_code=status.HTTP_202_ACCEPTED)
def start_run(agent_id: str, payload: RunCreate,
              state: AppState = Depends(app_state)) -> dict[str, Any]:
    agent = state.store.get_agent(agent_id)
    if agent is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such agent")
    if not agent.enabled:
        raise HTTPException(status.HTTP_409_CONFLICT,
                            f"agent {agent.name!r} is disabled")
    handle = state.run_manager.start(agent, payload.objective, payload.conversation_id)
    return {"run_id": handle.run_id, "state": handle.state.value}


@router.get("/runs/{run_id}")
def get_run(run_id: str, state: AppState = Depends(app_state)) -> dict[str, Any]:
    record = state.store.get_run(run_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such run")
    handle = state.run_manager.handle(run_id)
    return {
        **record.to_json(),
        "live": handle is not None,
        "cancelling": handle.cancelled.is_set() if handle else False,
    }


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str, state: AppState = Depends(app_state)) -> dict[str, Any]:
    """Ask a run to stop. It stops at its next step boundary, not immediately.

    Killing the thread mid-step would leave a container running and a workspace
    half-written, so the response says when it will actually take effect rather
    than implying it already has.
    """
    return state.run_manager.cancel(run_id)


@router.get("/runs/{run_id}/trace")
def run_trace(run_id: str, state: AppState = Depends(app_state)) -> dict[str, Any]:
    """The whole span tree, from the durable trace file."""
    record = state.store.get_run(run_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such run")
    spans = Tracer.read(Path(record.trace_path)) if record.trace_path else []
    return {"run_id": run_id, "spans": spans, "count": len(spans)}


@router.get("/runs/{run_id}/ledger")
def run_ledger(run_id: str, state: AppState = Depends(app_state)) -> dict[str, Any]:
    """Per-step token allocation, shaped for a stacked bar chart.

    This is the endpoint the dashboard exists for: it turns "slot bidding with
    hard floors" from a sentence into a picture of where every token in every
    prompt went, and which slot lost what.
    """
    record = state.store.get_run(run_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such run")
    spans = Tracer.read(Path(record.trace_path)) if record.trace_path else []

    steps: list[dict[str, Any]] = []
    for span in spans:
        if span.get("kind") != "context_build":
            continue
        attrs = span.get("attrs") or {}
        steps.append({
            "name": span.get("name", ""),
            "budget": attrs.get("budget", 0),
            "used": attrs.get("used", 0),
            "utilisation": attrs.get("utilisation", 0.0),
            "evicted_tokens": attrs.get("evicted_tokens", 0),
            "slots": attrs.get("slots", []),
            "notes": attrs.get("notes", []),
        })

    compactions = [
        {"name": s.get("name"), **(s.get("attrs") or {})}
        for s in spans if s.get("kind") == "compaction"
    ]
    return {
        "run_id": run_id,
        "steps": steps,
        "compactions": compactions,
        "peak_utilisation": max((s["utilisation"] for s in steps), default=0.0),
        "total_evicted": sum(s["evicted_tokens"] for s in steps),
    }


@router.get("/runs/{run_id}/workspace")
def run_workspace(run_id: str, state: AppState = Depends(app_state)) -> dict[str, Any]:
    """The artefacts a run produced — the digest, never file bodies.

    Same discipline the agent itself is held to: a listing and a preview, because
    serving a 40MB parquet through the dashboard would be the browser making the
    mistake the platform exists to avoid.
    """
    record = state.store.get_run(run_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such run")
    if not record.workspace_path:
        return {"run_id": run_id, "artifacts": [], "digest": ""}

    from teamclaw.execution.workspace import Workspace

    root = Path(record.workspace_path)
    if not root.exists():
        return {"run_id": run_id, "artifacts": [],
                "digest": "(workspace no longer on disk)"}
    workspace = Workspace(root=root)
    return {
        "run_id": run_id,
        "artifacts": [a.to_json() for a in workspace.artifacts()],
        "digest": workspace.digest(),
    }


@router.get("/runs/{run_id}/artifact")
def run_artifact(run_id: str, path: str, limit: int = 20_000,
                 state: AppState = Depends(app_state)) -> dict[str, Any]:
    """One artefact's text, capped. Path traversal is refused by the workspace."""
    record = state.store.get_run(run_id)
    if record is None or not record.workspace_path:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such run")

    from teamclaw.execution.workspace import Workspace

    workspace = Workspace(root=Path(record.workspace_path))
    try:
        resolved = workspace.resolve(path)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such artefact")
    if resolved.stat().st_size > 5_000_000:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            "artefact is too large to preview; read it from disk")
    text = resolved.read_text(encoding="utf-8", errors="replace")
    return {"path": path, "bytes": resolved.stat().st_size,
            "truncated": len(text) > limit, "text": text[:limit]}


# --- human in the loop ----------------------------------------------------
@router.get("/hitl")
def pending_hitl(state: AppState = Depends(app_state)) -> dict[str, Any]:
    return {"pending": state.session.pending_hitl()}


@router.post("/hitl/{run_id}")
def answer_hitl(run_id: str, payload: HitlAnswer,
                state: AppState = Depends(app_state)) -> dict[str, Any]:
    state.session.resolve_hitl(run_id, payload.decision, payload.message)
    return {"run_id": run_id, "decision": payload.decision}


# --- live feed ------------------------------------------------------------
@router.websocket("/ws/runs/{run_id}")
async def run_events(websocket: WebSocket, run_id: str) -> None:
    """Stream a run's spans, replaying what already happened first.

    Replay matters: a client that connects mid-run or reloads the page would
    otherwise see only the tail. The durable trace is the replay source, so a
    reconnect reconstructs the whole run rather than resuming blind.
    """
    await websocket.accept()
    state: AppState = websocket.app.state.teamclaw

    record = state.store.get_run(run_id)
    if record is None:
        await websocket.send_json({"type": "error", "error": "no such run"})
        await websocket.close()
        return

    queue = state.run_manager.attach(run_id)
    try:
        for span in state.run_manager.replay(run_id):
            await websocket.send_json({"type": "span", "replay": True, **span})
        await websocket.send_json({
            "type": "state", "run_id": run_id, "replayed": True,
            "state": record.state.value if hasattr(record.state, "value")
            else str(record.state),
        })

        if queue is None:
            # Already finished: send the terminal record and close, rather than
            # holding a socket open on a run that will never emit again.
            await websocket.send_json({"type": "done", "run_id": run_id,
                                       "state": record.state.value,
                                       "result": record.to_json()})
            await websocket.close()
            return

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=25.0)
            except asyncio.TimeoutError:
                # A keepalive, because idle proxies close silent sockets and a
                # long sandbox step is silent by nature.
                await websocket.send_json({"type": "ping"})
                continue
            await websocket.send_json(event)
            if event.get("type") in {"done", "error"}:
                break
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001 - a broken socket must not surface as a 500
        pass
    finally:
        if queue is not None:
            state.run_manager.detach(run_id, queue)
        try:
            await websocket.close()
        except RuntimeError:
            pass
