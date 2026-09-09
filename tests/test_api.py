"""The serving layer, driven through real HTTP and a real WebSocket.

Two things get most of the attention. Validation, because an agent's window and
step ceiling are resource limits rather than preferences — a bad value there is a
way to exhaust a quota, so it must be refused at the edge with a message that
says why. And the auth boundary, because this server starts containers and runs
generated code.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from teamclaw.api import create_app
from teamclaw.config import Paths, Settings
from teamclaw.models.providers.fake import ScriptedProvider
from teamclaw.models.router import Registry

SCRIPT = [
    "THOUGHT: write the answer\n```python\n"
    "open('answer.txt','w').write('46.21%')\nprint('wrote answer.txt')\n```",
    "DONE\nGross margin 46.21%.",
]


def scripted_registry() -> Registry:
    provider = ScriptedProvider(SCRIPT, name="scripted", loop_last=True)
    registry = Registry(providers={"scripted": provider})
    for purpose in list(registry.policy):
        registry.policy[purpose] = ("scripted",)
    return registry


def settings_for(tmp_path: Path, **overrides) -> Settings:
    paths = Paths(root=tmp_path, data=tmp_path / "data", cache=tmp_path / "cache",
                  filings=tmp_path / "f", datasets=tmp_path / "ds",
                  runs=tmp_path / "runs", workspaces=tmp_path / "ws").ensure()
    return Settings(paths=paths, **overrides)


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(cfg=settings_for(tmp_path), db_path=tmp_path / "t.db",
                     registry_factory=scripted_registry, max_workers=1, seed=True)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def agent_id(client: TestClient) -> str:
    agents = client.get("/api/agents").json()["agents"]
    return next(a["id"] for a in agents if a["name"] == "code_engineer")


# --- shape ----------------------------------------------------------------
def test_seeding_happens_inside_the_lifespan(client: TestClient):
    """@app.on_event does not fire when a lifespan is set — this is that bug's test."""
    agents = client.get("/api/agents").json()["agents"]
    assert {a["name"] for a in agents} == {
        "dd_finance", "bi_analyst", "deep_research", "code_engineer"
    }


def test_api_responses_are_never_cached(client: TestClient):
    """A cached /api/agents shows agents that no longer exist. Presents as a broken UI."""
    response = client.get("/api/agents")
    assert "no-store" in response.headers.get("cache-control", "")


def test_the_console_is_served(client: TestClient):
    assert client.get("/").status_code == 200
    assert "TeamClaw" in client.get("/").text


def test_the_scenario_catalogue_comes_from_the_code(client: TestClient):
    """What the editor offers must be what the sandbox can import."""
    scenarios = client.get("/api/scenarios").json()["scenarios"]
    assert {s["name"] for s in scenarios} == {
        "dd_finance", "bi_analyst", "deep_research", "code_engineer"
    }
    tools = client.get("/api/scenarios/code_engineer/tools").json()
    assert tools["tools"], "the scenario should publish a tool surface"
    assert tools["token_cost"]["retrieved_signatures"] < tools["token_cost"]["full_json_schemas"]


def test_an_unknown_scenario_is_a_404(client: TestClient):
    assert client.get("/api/scenarios/nope/tools").status_code == 404


# --- validation -----------------------------------------------------------
@pytest.mark.parametrize(
    ("payload", "needle"),
    [
        ({"name": "Bad Name", "scenario": "dd_finance", "persona": "x" * 20}, "pattern"),
        ({"name": "tiny", "scenario": "dd_finance", "persona": "x" * 20, "window": 500},
         "greater than or equal"),
        ({"name": "hog", "scenario": "dd_finance", "persona": "x" * 20,
          "window": 8000, "max_output": 6000}, "half the window"),
        ({"name": "nope", "scenario": "not_a_scenario", "persona": "x" * 20},
         "unknown scenario"),
        ({"name": "marathon", "scenario": "dd_finance", "persona": "x" * 20,
          "max_steps": 500}, "less than or equal"),
    ],
)
def test_resource_limits_are_refused_at_the_edge(client: TestClient, payload, needle):
    response = client.post("/api/agents", json=payload)
    assert response.status_code == 422
    assert needle in str(response.json()["detail"])


def test_a_duplicate_name_is_a_conflict(client: TestClient):
    body = {"name": "dd_finance", "scenario": "dd_finance", "persona": "x" * 20}
    assert client.post("/api/agents", json=body).status_code == 409


def test_a_valid_agent_round_trips(client: TestClient):
    created = client.post("/api/agents", json={
        "name": "analyst-2", "scenario": "dd_finance",
        "persona": "You are careful and you cite everything.",
        "window": 16_000, "max_output": 1_200, "max_steps": 6,
    }).json()
    fetched = client.get(f"/api/agents/{created['id']}").json()
    assert fetched["window"] == 16_000 and fetched["max_steps"] == 6

    patched = client.patch(f"/api/agents/{created['id']}",
                           json={"max_steps": 9}).json()
    assert patched["max_steps"] == 9
    assert client.delete(f"/api/agents/{created['id']}").status_code == 204
    assert client.get(f"/api/agents/{created['id']}").status_code == 404


# --- A2A over HTTP --------------------------------------------------------
def test_delegation_is_denied_until_granted(client: TestClient):
    agents = client.get("/api/agents").json()["agents"]
    a, b = agents[0]["id"], agents[1]["id"]
    before = client.post("/api/a2a/check", json={"source_id": a, "target_id": b}).json()
    assert before["allowed"] is False and "explicitly" in before["reason"]

    client.post(f"/api/agents/{a}/relations", json={"target_id": b, "tenant": "acme"})
    after = client.post("/api/a2a/check", json={"source_id": a, "target_id": b}).json()
    assert after["allowed"] is True

    files = client.post("/api/a2a/check",
                        json={"source_id": a, "target_id": b, "need_files": True}).json()
    assert files["allowed"] is False


def test_relating_an_unknown_agent_is_a_400(client: TestClient, agent_id: str):
    response = client.post(f"/api/agents/{agent_id}/relations",
                           json={"target_id": "agent_nope"})
    assert response.status_code == 400


# --- runs -----------------------------------------------------------------
def wait_for(client: TestClient, run_id: str, timeout_s: float = 25.0) -> dict:
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        record = client.get(f"/api/runs/{run_id}").json()
        if record["state"] in {"finished", "failed", "cancelled", "needs_human"}:
            return record
        time.sleep(0.2)
    raise AssertionError(f"run {run_id} did not settle within {timeout_s}s")


def test_a_run_completes_and_records_where_its_artefacts_live(client: TestClient, agent_id: str):
    started = client.post(f"/api/agents/{agent_id}/runs",
                          json={"objective": "Write the answer to a file."})
    assert started.status_code == 202
    record = wait_for(client, started.json()["run_id"])

    assert record["state"] == "finished"
    assert record["steps"] >= 1
    assert record["workspace_path"] and record["trace_path"]
    assert record["sandbox"]["isolation"], "a result must carry its isolation level"

    workspace = client.get(f"/api/runs/{record['id']}/workspace").json()
    assert "answer.txt" in [a["path"] for a in workspace["artifacts"]]

    artifact = client.get(f"/api/runs/{record['id']}/artifact",
                          params={"path": "answer.txt"}).json()
    assert artifact["text"] == "46.21%"


def test_the_ledger_endpoint_exposes_per_step_slot_allocation(client: TestClient, agent_id: str):
    """The endpoint the console exists for."""
    started = client.post(f"/api/agents/{agent_id}/runs",
                          json={"objective": "Write the answer."})
    run_id = started.json()["run_id"]
    wait_for(client, run_id)

    ledger = client.get(f"/api/runs/{run_id}/ledger").json()
    assert ledger["steps"], "every step should have a recorded allocation"
    first = ledger["steps"][0]
    assert first["budget"] > 0 and first["used"] > 0
    slots = {s["slot"] for s in first["slots"]}
    assert "system" in slots, "the operating contract is always present"
    for slot in first["slots"]:
        assert slot["used"] <= slot["allowance"] or slot["overflowed"]


def test_artifact_paths_cannot_escape_the_workspace(client: TestClient, agent_id: str):
    started = client.post(f"/api/agents/{agent_id}/runs", json={"objective": "Write it."})
    run_id = started.json()["run_id"]
    wait_for(client, run_id)
    response = client.get(f"/api/runs/{run_id}/artifact",
                          params={"path": "../../../etc/passwd"})
    assert response.status_code == 400


def test_cancelling_says_when_it_takes_effect(client: TestClient, agent_id: str):
    """Killing a thread mid-step would leave a container running."""
    started = client.post(f"/api/agents/{agent_id}/runs", json={"objective": "Write it."})
    run_id = started.json()["run_id"]
    out = client.post(f"/api/runs/{run_id}/cancel").json()
    assert "step boundary" in out.get("note", "") or out["cancelled"] is False
    wait_for(client, run_id)


def test_a_run_on_an_unknown_agent_is_a_404(client: TestClient):
    assert client.post("/api/agents/agent_nope/runs",
                       json={"objective": "hello"}).status_code == 404


def test_a_disabled_agent_refuses_work(client: TestClient, agent_id: str):
    client.patch(f"/api/agents/{agent_id}", json={"enabled": False})
    response = client.post(f"/api/agents/{agent_id}/runs", json={"objective": "hello"})
    assert response.status_code == 409


# --- chat and conversations ----------------------------------------------
def test_one_run_at_a_time_per_conversation(client: TestClient, agent_id: str):
    """Two runs would share a workspace, and the workspace is the run's state."""
    first = client.post(f"/api/agents/{agent_id}/chat",
                        json={"content": "Do the first thing."})
    conversation_id = first.json()["conversation_id"]
    second = client.post(f"/api/agents/{agent_id}/chat",
                         params={"conversation_id": conversation_id},
                         json={"content": "And another."})
    assert second.status_code == 409
    assert "one run at a time" in second.json()["detail"]
    wait_for(client, first.json()["run_id"])


def test_a_finished_run_writes_its_answer_into_the_conversation(client: TestClient, agent_id: str):
    started = client.post(f"/api/agents/{agent_id}/chat",
                          json={"content": "Compute the margin."})
    conversation_id = started.json()["conversation_id"]
    wait_for(client, started.json()["run_id"])
    messages = client.get(f"/api/conversations/{conversation_id}").json()["messages"]
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant"]
    assert "46.21" in messages[1]["content"]


# --- websocket ------------------------------------------------------------
def test_the_websocket_replays_then_streams_to_a_terminal_event(client: TestClient, agent_id: str):
    """A client that reloads mid-run must not lose the first half."""
    started = client.post(f"/api/agents/{agent_id}/runs", json={"objective": "Write it."})
    run_id = started.json()["run_id"]

    kinds: list[str] = []
    with client.websocket_connect(f"/api/ws/runs/{run_id}") as socket:
        for _ in range(80):
            event = socket.receive_json()
            if event["type"] == "ping":
                continue
            kinds.append(event.get("kind") or event["type"])
            if event["type"] in {"done", "error"}:
                assert event.get("state") in {"finished", "failed", "cancelled",
                                              "needs_human"}
                break
    assert "context_build" in kinds
    assert kinds[-1] in {"done", "error"}


def test_connecting_to_an_unknown_run_is_reported_not_hung(client: TestClient):
    with client.websocket_connect("/api/ws/runs/run_nope") as socket:
        assert socket.receive_json()["type"] == "error"


# --- auth -----------------------------------------------------------------
def test_reads_are_open_and_mutations_need_the_token(tmp_path: Path):
    """This server can execute code, so the verbs that start work are gated."""
    cfg = settings_for(tmp_path, api_token="s3cret")
    app = create_app(cfg=cfg, db_path=tmp_path / "t.db",
                     registry_factory=scripted_registry, max_workers=1, seed=True)
    with TestClient(app) as client:
        assert client.get("/api/agents").status_code == 200

        blocked = client.post("/api/agents", json={
            "name": "sneaky", "scenario": "dd_finance", "persona": "x" * 20})
        assert blocked.status_code == 401
        assert "token" in blocked.json()["detail"].lower()

        allowed = client.post("/api/agents", headers={"X-Teamclaw-Token": "s3cret"},
                              json={"name": "sneaky", "scenario": "dd_finance",
                                    "persona": "x" * 20})
        assert allowed.status_code == 201


def test_a_bearer_token_is_also_accepted(tmp_path: Path):
    cfg = settings_for(tmp_path, api_token="s3cret")
    app = create_app(cfg=cfg, db_path=tmp_path / "t.db",
                     registry_factory=scripted_registry, max_workers=1)
    with TestClient(app) as client:
        response = client.post("/api/agents",
                               headers={"Authorization": "Bearer s3cret"},
                               json={"name": "ok", "scenario": "dd_finance",
                                     "persona": "x" * 20})
        assert response.status_code == 201


def test_health_surfaces_the_configuration_it_would_be_dangerous_to_hide(tmp_path: Path):
    cfg = settings_for(tmp_path, api_host="0.0.0.0", api_workers=4)
    app = create_app(cfg=cfg, db_path=tmp_path / "t.db",
                     registry_factory=scripted_registry, max_workers=1)
    with TestClient(app) as client:
        health = client.get("/api/insight/health").json()
        joined = " ".join(health["config_warnings"])
        assert "remote code execution" in joined
        assert "TEAMCLAW_REDIS_URL" in joined


def test_insight_reports_no_usable_provider_when_only_fakes_are_present(client: TestClient):
    """The console must say so rather than letting runs fail unexplained."""
    health = client.get("/api/insight/health").json()
    assert health["usable_providers"] == []
    assert any("NoProviderAvailable" in w for w in health["warnings"])
