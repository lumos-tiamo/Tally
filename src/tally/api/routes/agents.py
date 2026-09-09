"""Agent declarations, A2A relations, and conversations."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from tally.api.deps import AppState, app_state
from tally.api.schemas import (
    AgentCreate,
    AgentUpdate,
    ChatMessage,
    DelegationCheck,
    RelationCreate,
    scenario_catalogue,
)

router = APIRouter(prefix="/api", tags=["agents"])


@router.get("/scenarios")
def list_scenarios() -> dict[str, Any]:
    """Scenario catalogue: personas, skills, and whether a scenario needs egress."""
    return {"scenarios": scenario_catalogue()}


@router.get("/scenarios/{name}/tools")
def scenario_tools(name: str, state: AppState = Depends(app_state)) -> dict[str, Any]:
    """The tool surface a scenario actually offers, with its token cost.

    Built by instantiating the scenario's registry, so what the editor offers is
    exactly what the sandbox can import — never a stored copy that has drifted.
    """
    from tally.context.tokenizer import count_tokens
    from tally.execution.workspace import Workspace
    from tally.scenarios import SCENARIOS

    if name not in SCENARIOS:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            f"unknown scenario {name!r}")
    module = SCENARIOS[name].spec
    workspace = Workspace.create(state.cfg.paths.workspaces, "_catalogue")
    if name == "dd_finance":
        from tally.scenarios.dd_finance.sec_client import SecClient

        registry = module.build_tools(SecClient(cfg=state.cfg), workspace)
    elif name == "bi_analyst":
        registry = module.build_tools(state.cfg.paths.data / "bi_demo.db")
    elif name == "deep_research":
        from tally.api.deps import _offline_web_handlers

        registry = module.build_tools(_offline_web_handlers(module))
    else:
        registry = module.build_tools()

    return {
        "scenario": name,
        "tools": [
            {"name": spec.name, "module": spec.module, "summary": spec.summary,
             "signature": spec.signature(), "bridged": spec.bridged,
             "requires_network": spec.requires_network, "tags": list(spec.tags)}
            for spec in sorted(registry.tools.values(), key=lambda s: s.name)
        ],
        "token_cost": registry.token_cost_comparison(
            count_tokens,
            step_description="extract the total revenue figure from the statement",
            k=8,
        ),
    }


@router.get("/agents")
def list_agents(state: AppState = Depends(app_state)) -> dict[str, Any]:
    return {"agents": [a.to_json() for a in state.store.list_agents()]}


@router.post("/agents", status_code=status.HTTP_201_CREATED)
def create_agent(payload: AgentCreate,
                 state: AppState = Depends(app_state)) -> dict[str, Any]:
    if state.store.get_agent_by_name(payload.name) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT,
                            f"an agent named {payload.name!r} already exists")
    agent = state.store.create_agent(**payload.model_dump())
    return agent.to_json()


@router.get("/agents/{agent_id}")
def get_agent(agent_id: str, state: AppState = Depends(app_state)) -> dict[str, Any]:
    agent = state.store.get_agent(agent_id)
    if agent is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such agent")
    return agent.to_json()


@router.patch("/agents/{agent_id}")
def update_agent(agent_id: str, payload: AgentUpdate,
                 state: AppState = Depends(app_state)) -> dict[str, Any]:
    fields = {k: v for k, v in payload.model_dump().items() if v is not None}
    agent = state.store.update_agent(agent_id, **fields)
    if agent is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such agent")
    return agent.to_json()


@router.delete("/agents/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_agent(agent_id: str, state: AppState = Depends(app_state)) -> None:
    if not state.store.delete_agent(agent_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such agent")


# --- A2A ------------------------------------------------------------------
@router.get("/agents/{agent_id}/relations")
def list_relations(agent_id: str,
                   state: AppState = Depends(app_state)) -> dict[str, Any]:
    return {"relations": [r.to_json() for r in state.store.relations(agent_id)]}


@router.post("/agents/{agent_id}/relations", status_code=status.HTTP_201_CREATED)
def create_relation(agent_id: str, payload: RelationCreate,
                    state: AppState = Depends(app_state)) -> dict[str, Any]:
    relation = state.store.relate(
        agent_id, payload.target_id, tenant=payload.tenant,
        may_delegate=payload.may_delegate, may_share_files=payload.may_share_files,
    )
    if relation is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "source or target agent does not exist")
    return relation.to_json()


@router.post("/a2a/check")
def check_delegation(payload: DelegationCheck,
                     state: AppState = Depends(app_state)) -> dict[str, Any]:
    """Ask whether one agent may delegate to another, and why not if not.

    Exposed deliberately. The authorisation rule is the part of A2A worth
    inspecting, and an endpoint that explains a refusal is more useful than one
    that only ever answers during a run.
    """
    verdict = state.store.may_delegate(
        payload.source_id, payload.target_id, need_files=payload.need_files
    )
    return {"allowed": verdict.allowed, "reason": verdict.reason}


# --- conversations --------------------------------------------------------
@router.get("/conversations")
def list_conversations(agent_id: str | None = None,
                       state: AppState = Depends(app_state)) -> dict[str, Any]:
    conversations = state.store.list_conversations(agent_id)
    return {
        "conversations": [
            {**c.to_json(),
             "active_run": state.session.active_run(c.id),
             "message_count": len(state.store.messages(c.id, limit=1000))}
            for c in conversations
        ]
    }


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str,
                     state: AppState = Depends(app_state)) -> dict[str, Any]:
    messages = state.store.messages(conversation_id)
    return {
        "conversation_id": conversation_id,
        "active_run": state.session.active_run(conversation_id),
        "messages": [m.to_json() for m in messages],
    }


@router.post("/agents/{agent_id}/chat", status_code=status.HTTP_202_ACCEPTED)
def chat(agent_id: str, payload: ChatMessage, conversation_id: str | None = None,
         state: AppState = Depends(app_state)) -> dict[str, Any]:
    """Send a message to an agent, starting a run for it.

    One run per conversation at a time. A second message while a run is live is
    rejected rather than queued: two agents writing the same workspace would
    corrupt each other's artefacts, and the workspace is the run's state.
    """
    agent = state.store.get_agent(agent_id)
    if agent is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such agent")
    if not agent.enabled:
        raise HTTPException(status.HTTP_409_CONFLICT,
                            f"agent {agent.name!r} is disabled")

    if conversation_id is None:
        conversation = state.store.create_conversation(
            agent_id=agent_id, channel="web",
            title=payload.content[:120],
        )
        conversation_id = conversation.id
    elif (live := state.session.active_run(conversation_id)) is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"run {live} is still active on this conversation; one run at a time "
            "per conversation, because they would share a workspace",
        )

    state.store.add_message(conversation_id, "user", payload.content)
    handle = state.run_manager.start(agent, payload.content, conversation_id)
    return {"run_id": handle.run_id, "conversation_id": conversation_id}
