"""Persistence and the A2A authorisation rule.

Most of this file is about :meth:`AgentStore.may_delegate`, because it is the
entire A2A permission model and the failure modes are the interesting part:
default-allow, tenant taken from the request, and direction ignored are the three
ways this gets built wrong.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from teamclaw.store import (
    AgentStore,
    RunState,
    SchemaTooNew,
    build_engine,
    init_schema,
    seed_default_agents,
)
from teamclaw.store.schema import SCHEMA_VERSION, SchemaMeta
from sqlalchemy.orm import Session


@pytest.fixture
def store(tmp_path: Path) -> AgentStore:
    engine = build_engine(db_path=tmp_path / "t.db")
    init_schema(engine)
    return AgentStore(engine)


@pytest.fixture
def pair(store: AgentStore):
    a = store.create_agent(name="alpha", scenario="dd_finance", persona="Analyst A.")
    b = store.create_agent(name="beta", scenario="bi_analyst", persona="Analyst B.")
    return a, b


# --- schema ---------------------------------------------------------------
def test_init_is_idempotent(tmp_path: Path):
    engine = build_engine(db_path=tmp_path / "t.db")
    assert init_schema(engine) == SCHEMA_VERSION
    assert init_schema(engine) == SCHEMA_VERSION


def test_a_database_from_the_future_is_refused(tmp_path: Path):
    """Better to stop than to write rows a newer schema will misread."""
    engine = build_engine(db_path=tmp_path / "t.db")
    init_schema(engine)
    with Session(engine) as session:
        session.get(SchemaMeta, 1).version = SCHEMA_VERSION + 5
        session.commit()
    with pytest.raises(SchemaTooNew):
        init_schema(engine)


def test_seeding_is_idempotent(store: AgentStore):
    first = seed_default_agents(store)
    second = seed_default_agents(store)
    assert len(first) == 4
    assert second == []


# --- A2A ------------------------------------------------------------------
def test_an_unlisted_pair_is_denied(store: AgentStore, pair):
    """Absence is denial. There is no default-allow path.

    This is the one that matters: a permission model whose default is allow
    grants every delegation nobody thought to forbid.
    """
    a, b = pair
    verdict = store.may_delegate(a.id, b.id)
    assert not verdict
    assert "must be granted explicitly" in verdict.reason


def test_delegation_is_directional(store: AgentStore, pair):
    a, b = pair
    store.relate(a.id, b.id)
    assert store.may_delegate(a.id, b.id)
    assert not store.may_delegate(b.id, a.id), "a grant must not be symmetric"


def test_file_transfer_is_a_separate_permission(store: AgentStore, pair):
    a, b = pair
    store.relate(a.id, b.id, may_delegate=True, may_share_files=False)
    assert store.may_delegate(a.id, b.id)
    denied = store.may_delegate(a.id, b.id, need_files=True)
    assert not denied and "not file transfer" in denied.reason


def test_self_delegation_is_refused(store: AgentStore, pair):
    """An agent that can delegate to itself recurses without bound."""
    a, _ = pair
    verdict = store.may_delegate(a.id, a.id)
    assert not verdict and "itself" in verdict.reason


def test_a_disabled_target_cannot_be_delegated_to(store: AgentStore, pair):
    a, b = pair
    store.relate(a.id, b.id)
    store.update_agent(b.id, enabled=False)
    verdict = store.may_delegate(a.id, b.id)
    assert not verdict and "disabled" in verdict.reason


def test_the_tenant_comes_from_the_stored_relation(store: AgentStore, pair):
    """A caller cannot authorise a cross-tenant delegation by asserting a tenant.

    ``may_delegate`` takes no tenant argument at all — the only tenant in play is
    the one on a row that must already exist.
    """
    a, b = pair
    relation = store.relate(a.id, b.id, tenant="acme")
    assert relation.tenant == "acme"
    import inspect

    signature = inspect.signature(store.may_delegate)
    assert "tenant" not in signature.parameters


def test_withholding_delegation_on_an_existing_relation_is_honoured(store: AgentStore, pair):
    a, b = pair
    store.relate(a.id, b.id, may_delegate=False)
    verdict = store.may_delegate(a.id, b.id)
    assert not verdict and "withheld" in verdict.reason


def test_relating_twice_updates_rather_than_duplicates(store: AgentStore, pair):
    a, b = pair
    store.relate(a.id, b.id, tenant="one")
    store.relate(a.id, b.id, tenant="two", may_share_files=True)
    relations = store.relations(a.id)
    assert len(relations) == 1
    assert relations[0].tenant == "two" and relations[0].may_share_files


def test_relating_an_unknown_agent_fails_closed(store: AgentStore, pair):
    a, _ = pair
    assert store.relate(a.id, "agent_does_not_exist") is None


# --- runs and conversations ----------------------------------------------
def test_a_run_row_points_at_its_artefacts_rather_than_copying_them(store: AgentStore, pair):
    a, _ = pair
    store.create_run("run_1", a.id, "do the thing")
    store.update_run("run_1", state=RunState.FINISHED, workspace_path="/w/run_1",
                     trace_path="/w/run_1/trace.jsonl", steps=4,
                     tokens_in=900, tokens_out=120)
    run = store.get_run("run_1")
    assert run.workspace_path == "/w/run_1"
    assert run.tokens_total == 1020
    assert run.finished_at is not None, "a terminal state must stamp finished_at"


def test_conversations_are_found_by_channel_thread(store: AgentStore, pair):
    a, _ = pair
    store.create_conversation(agent_id=a.id, channel="feishu", external_id="chat_9")
    assert store.find_conversation("feishu", "chat_9") is not None
    assert store.find_conversation("feishu", "chat_other") is None
    assert store.find_conversation("slack", "chat_9") is None


def test_deleting_an_agent_takes_its_runs(store: AgentStore, pair):
    a, _ = pair
    store.create_run("run_x", a.id, "objective")
    assert store.delete_agent(a.id)
    assert store.get_run("run_x") is None


def test_totals_are_computed_not_guessed(store: AgentStore, pair):
    a, _ = pair
    store.create_run("r1", a.id, "one")
    store.update_run("r1", state=RunState.FINISHED, steps=3, tokens_in=100, tokens_out=20)
    store.create_run("r2", a.id, "two")
    store.update_run("r2", state=RunState.FAILED)
    totals = store.totals()
    assert totals["agents"] == 2 and totals["runs"] == 2
    assert totals["finished"] == 1 and totals["failed"] == 1
    assert totals["tokens_total"] == 120
