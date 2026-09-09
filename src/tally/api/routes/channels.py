"""Inbound channel webhooks.

The route does the two things every channel needs identically, so no adapter can
forget them: verify the delivery, then deduplicate it. Both happen before an
agent is touched.

The reply is sent from a background task rather than by holding the webhook
open. Channels time out in seconds and an agent run takes minutes, so a
synchronous reply guarantees a timeout, a retry, and — without the dedupe — a
second run.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status

from tally.api.deps import AppState, app_state
from tally.channels import InboundMessage, build_channels

router = APIRouter(prefix="/api/channels", tags=["channels"])

# A tenant may only start so many runs per window. Each run can hold a container
# and burn free-tier quota, so an unbounded channel is a way to exhaust both.
RATE_LIMIT = 20
RATE_WINDOW_S = 60


@router.get("")
def list_channels(state: AppState = Depends(app_state)) -> dict[str, Any]:
    channels = build_channels(state.cfg)
    return {
        "configured": sorted(channels),
        "available": ["feishu"],
        "note": ("An unconfigured adapter is not registered at all, so its webhook "
                 "returns 404 rather than a confusing 401."),
    }


@router.post("/{channel_name}/webhook")
async def webhook(channel_name: str, request: Request,
                  background: BackgroundTasks,
                  state: AppState = Depends(app_state)) -> Any:
    channels = build_channels(state.cfg)
    channel = channels.get(channel_name)
    if channel is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"channel {channel_name!r} is not configured on this server",
        )

    body = await request.body()
    verdict = channel.verify(headers=dict(request.headers), body=body)
    if not verdict:
        # 401 rather than 400: the request was well-formed and unauthenticated.
        raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                            f"verification failed: {verdict.reason}")
    if verdict.challenge_response is not None:
        return verdict.challenge_response

    try:
        payload = json.loads(body.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "body is not JSON") from None

    message = channel.parse(payload)
    if message is None or not message.usable:
        # Acknowledged, not an error: channels send event types we do not act on,
        # and a non-2xx makes them retry something we will ignore again.
        return {"ok": True, "ignored": "no actionable message in this event"}

    if state.session.seen_before(channel_name, message.event_id):
        return {"ok": True, "deduplicated": message.event_id}

    allowed, count = state.session.rate_limit(
        f"channel:{channel_name}:{message.tenant}",
        limit=RATE_LIMIT, window_s=RATE_WINDOW_S,
    )
    if not allowed:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"tenant {message.tenant!r} is over {RATE_LIMIT} runs per "
            f"{RATE_WINDOW_S}s (seen {count})",
        )

    agent = _route_to_agent(state, message)
    if agent is None:
        return {"ok": True, "ignored": "no enabled agent is bound to this channel"}

    conversation = state.store.find_conversation(channel_name, message.thread_id)
    if conversation is None:
        conversation = state.store.create_conversation(
            agent_id=agent.id, tenant=message.tenant, channel=channel_name,
            external_id=message.thread_id, title=message.text[:120],
        )

    if (live := state.session.active_run(conversation.id)) is not None:
        # Tell the human rather than silently dropping their message.
        background.add_task(_safe_send, channel, message.thread_id,
                            f"I am still working on the previous request "
                            f"(run {live}). I will answer that one first.")
        return {"ok": True, "busy": live}

    state.store.add_message(conversation.id, "user", message.text)
    handle = state.run_manager.start(agent, message.text, conversation.id)
    background.add_task(_reply_when_done, state, channel, message.thread_id,
                        handle.run_id)
    return {"ok": True, "run_id": handle.run_id, "conversation_id": conversation.id}


def _route_to_agent(state: AppState, message: InboundMessage):  # noqa: ANN202
    """Pick the agent for an inbound message.

    Deliberately simple: an existing conversation keeps its agent, otherwise the
    first enabled agent wins. Routing by intent would be a model call before
    authorisation, which is the wrong order — bind agents to channels explicitly
    when that matters.
    """
    conversation = state.store.find_conversation(message.channel, message.thread_id)
    if conversation is not None:
        agent = state.store.get_agent(conversation.agent_id)
        if agent is not None and agent.enabled:
            return agent
    agents = state.store.list_agents(enabled_only=True)
    return agents[0] if agents else None


def _safe_send(channel, thread_id: str, text: str) -> None:  # noqa: ANN001
    """Outbound failures are logged into the run record, never raised.

    A background task that raises would be an unhandled exception in the event
    loop, and the platform would lose the reply *and* the error.
    """
    try:
        channel.send(thread_id, text)
    except Exception:  # noqa: BLE001
        pass


def _reply_when_done(state: AppState, channel, thread_id: str,  # noqa: ANN001
                     run_id: str) -> None:
    """Wait for the run, then send its answer back to the thread.

    Bounded: a run that outlives the wait leaves the thread without a reply
    rather than holding a task forever. The run itself is unaffected — its result
    is in the database either way, and the dashboard shows it.
    """
    import time

    deadline = time.monotonic() + 900.0
    while time.monotonic() < deadline:
        record = state.store.get_run(run_id)
        if record is None:
            return
        value = record.state.value if hasattr(record.state, "value") else str(record.state)
        if value in {"finished", "failed", "cancelled", "needs_human"}:
            if value == "finished" and record.final_answer:
                _safe_send(channel, thread_id, record.final_answer)
            elif value == "needs_human":
                _safe_send(channel, thread_id,
                           "I stopped and need a person to look at this. "
                           f"See run {run_id} in the console.")
            else:
                _safe_send(channel, thread_id,
                           f"I could not complete that ({record.stop_reason or value}). "
                           f"Run {run_id} has the detail.")
            return
        time.sleep(2.0)
