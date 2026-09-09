"""Data access, plus the one place A2A authorisation is decided.

:meth:`AgentStore.may_delegate` is the whole of the A2A permission model, and it
is deliberately a database lookup with no arguments taken from the message. Two
properties follow from that:

* **Tenant comes from the stored relation, not the request.** A caller cannot
  authorise a cross-tenant delegation by asserting a tenant, because the tenant
  is a column on the row that must already exist.
* **Absence is denial.** There is no default-allow path and no wildcard. An
  unlisted pair is refused, and the refusal names which condition failed so the
  caller is not left guessing.

Everything else here is ordinary CRUD.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from sqlalchemy import create_engine, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from tally.store.schema import (
    SCHEMA_VERSION,
    AgentDef,
    AgentRelation,
    Base,
    Conversation,
    Message,
    Run,
    RunState,
    SchemaMeta,
)


class SchemaTooNew(RuntimeError):
    """The database was written by a newer version than this code understands."""


def build_engine(url: str | None = None, *, db_path: Path | None = None) -> Engine:
    if url:
        return create_engine(url, future=True)
    path = db_path or Path("data/tally.db")
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: runs execute on a worker thread while the API
    # serves on the event loop, and each gets its own Session from the factory.
    return create_engine(
        f"sqlite+pysqlite:///{path}", future=True,
        connect_args={"check_same_thread": False},
    )


def init_schema(engine: Engine) -> int:
    """Create tables if absent and refuse a database from the future."""
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        meta = session.get(SchemaMeta, 1)
        if meta is None:
            session.add(SchemaMeta(id=1, version=SCHEMA_VERSION))
            session.commit()
            return SCHEMA_VERSION
        if meta.version > SCHEMA_VERSION:
            raise SchemaTooNew(
                f"database schema is v{meta.version} but this build understands "
                f"v{SCHEMA_VERSION}; upgrade the code rather than migrating down"
            )
        return meta.version


@dataclass
class Denial:
    allowed: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.allowed


class AgentStore:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._session = sessionmaker(engine, expire_on_commit=False, future=True)

    def session(self) -> Session:
        return self._session()

    # -- agents ------------------------------------------------------------
    def create_agent(self, **fields: Any) -> AgentDef:
        with self.session() as s:
            agent = AgentDef(**fields)
            s.add(agent)
            s.commit()
            s.refresh(agent)
            return agent

    def update_agent(self, agent_id: str, **fields: Any) -> AgentDef | None:
        with self.session() as s:
            agent = s.get(AgentDef, agent_id)
            if agent is None:
                return None
            for key, value in fields.items():
                if value is not None and hasattr(agent, key):
                    setattr(agent, key, value)
            s.commit()
            s.refresh(agent)
            return agent

    def get_agent(self, agent_id: str) -> AgentDef | None:
        with self.session() as s:
            return s.get(AgentDef, agent_id)

    def get_agent_by_name(self, name: str) -> AgentDef | None:
        with self.session() as s:
            return s.scalar(select(AgentDef).where(AgentDef.name == name))

    def list_agents(self, *, enabled_only: bool = False) -> list[AgentDef]:
        with self.session() as s:
            stmt = select(AgentDef).order_by(AgentDef.created_at.desc())
            if enabled_only:
                stmt = stmt.where(AgentDef.enabled.is_(True))
            return list(s.scalars(stmt))

    def delete_agent(self, agent_id: str) -> bool:
        with self.session() as s:
            agent = s.get(AgentDef, agent_id)
            if agent is None:
                return False
            s.delete(agent)
            s.commit()
            return True

    # -- A2A authorisation -------------------------------------------------
    def relate(
        self, source_id: str, target_id: str, *, tenant: str = "default",
        may_delegate: bool = True, may_share_files: bool = False,
    ) -> AgentRelation | None:
        with self.session() as s:
            if s.get(AgentDef, source_id) is None or s.get(AgentDef, target_id) is None:
                return None
            existing = s.scalar(
                select(AgentRelation).where(
                    AgentRelation.source_id == source_id,
                    AgentRelation.target_id == target_id,
                )
            )
            if existing is not None:
                existing.tenant = tenant
                existing.may_delegate = may_delegate
                existing.may_share_files = may_share_files
                s.commit()
                s.refresh(existing)
                return existing
            relation = AgentRelation(
                source_id=source_id, target_id=target_id, tenant=tenant,
                may_delegate=may_delegate, may_share_files=may_share_files,
            )
            s.add(relation)
            s.commit()
            s.refresh(relation)
            return relation

    def relations(self, source_id: str | None = None) -> list[AgentRelation]:
        with self.session() as s:
            stmt = select(AgentRelation)
            if source_id:
                stmt = stmt.where(AgentRelation.source_id == source_id)
            return list(s.scalars(stmt))

    def may_delegate(
        self, source_id: str, target_id: str, *, need_files: bool = False
    ) -> Denial:
        """Decide one delegation. Absence is denial; the tenant comes from the row.

        Self-delegation is refused before anything else: an agent that can
        delegate to itself can recurse without bound, and the depth cap in the
        sub-agent factory is a second line of defence rather than the first.
        """
        if source_id == target_id:
            return Denial(False, "an agent may not delegate to itself")
        with self.session() as s:
            source = s.get(AgentDef, source_id)
            target = s.get(AgentDef, target_id)
            if source is None:
                return Denial(False, f"unknown source agent {source_id!r}")
            if target is None:
                return Denial(False, f"unknown target agent {target_id!r}")
            if not target.enabled:
                return Denial(False, f"target agent {target.name!r} is disabled")
            relation = s.scalar(
                select(AgentRelation).where(
                    AgentRelation.source_id == source_id,
                    AgentRelation.target_id == target_id,
                )
            )
            if relation is None:
                return Denial(
                    False,
                    f"no relation from {source.name!r} to {target.name!r}; "
                    "delegation must be granted explicitly",
                )
            if not relation.may_delegate:
                return Denial(False, "the relation exists but delegation is withheld")
            if need_files and not relation.may_share_files:
                return Denial(
                    False, "the relation permits delegation but not file transfer"
                )
            return Denial(True)

    # -- conversations -----------------------------------------------------
    def create_conversation(self, **fields: Any) -> Conversation:
        with self.session() as s:
            conversation = Conversation(**fields)
            s.add(conversation)
            s.commit()
            s.refresh(conversation)
            return conversation

    def find_conversation(self, channel: str, external_id: str) -> Conversation | None:
        with self.session() as s:
            return s.scalar(
                select(Conversation).where(
                    Conversation.channel == channel,
                    Conversation.external_id == external_id,
                )
            )

    def list_conversations(self, agent_id: str | None = None) -> list[Conversation]:
        with self.session() as s:
            stmt = select(Conversation).order_by(Conversation.created_at.desc())
            if agent_id:
                stmt = stmt.where(Conversation.agent_id == agent_id)
            return list(s.scalars(stmt))

    def add_message(self, conversation_id: str, role: str, content: str,
                    run_id: str | None = None) -> Message:
        with self.session() as s:
            message = Message(conversation_id=conversation_id, role=role,
                              content=content, run_id=run_id)
            s.add(message)
            s.commit()
            s.refresh(message)
            return message

    def messages(self, conversation_id: str, limit: int = 200) -> list[Message]:
        with self.session() as s:
            stmt = (select(Message)
                    .where(Message.conversation_id == conversation_id)
                    .order_by(Message.created_at)
                    .limit(limit))
            return list(s.scalars(stmt))

    # -- runs --------------------------------------------------------------
    def create_run(self, run_id: str, agent_id: str, objective: str,
                   conversation_id: str | None = None) -> Run:
        with self.session() as s:
            run = Run(id=run_id, agent_id=agent_id, objective=objective,
                      conversation_id=conversation_id, state=RunState.QUEUED)
            s.add(run)
            s.commit()
            s.refresh(run)
            return run

    def update_run(self, run_id: str, **fields: Any) -> Run | None:
        with self.session() as s:
            run = s.get(Run, run_id)
            if run is None:
                return None
            for key, value in fields.items():
                if hasattr(run, key):
                    setattr(run, key, value)
            if fields.get("state") in {RunState.FINISHED, RunState.FAILED,
                                       RunState.CANCELLED}:
                run.finished_at = datetime.now(timezone.utc)
            s.commit()
            s.refresh(run)
            return run

    def get_run(self, run_id: str) -> Run | None:
        with self.session() as s:
            return s.get(Run, run_id)

    def list_runs(self, *, agent_id: str | None = None, limit: int = 50) -> list[Run]:
        with self.session() as s:
            stmt = select(Run).order_by(Run.created_at.desc()).limit(limit)
            if agent_id:
                stmt = stmt.where(Run.agent_id == agent_id)
            return list(s.scalars(stmt))

    # -- aggregates for the dashboard --------------------------------------
    def totals(self) -> dict[str, Any]:
        with self.session() as s:
            runs = list(s.scalars(select(Run)))
            finished = [r for r in runs if r.state is RunState.FINISHED]
            return {
                "agents": s.scalar(select(func.count()).select_from(AgentDef)) or 0,
                "runs": len(runs),
                "finished": len(finished),
                "failed": sum(1 for r in runs if r.state is RunState.FAILED),
                "needs_human": sum(1 for r in runs if r.state is RunState.NEEDS_HUMAN),
                "tokens_total": sum(r.tokens_total for r in runs),
                "cost_usd": round(sum(r.cost_usd or 0.0 for r in runs), 6),
                "mean_steps": round(sum(r.steps for r in finished) / len(finished), 2)
                if finished else 0.0,
                "conversations": s.scalar(
                    select(func.count()).select_from(Conversation)) or 0,
            }


def seed_default_agents(store: AgentStore) -> list[AgentDef]:
    """One agent per scenario, so a fresh install has something to run.

    Personas and tool lists come from the scenario modules rather than being
    retyped here — the scenario is the source of truth for what its agent is,
    and a copy in the seed would drift from it.
    """
    from tally.scenarios import SCENARIOS

    created: list[AgentDef] = []
    for scenario_name, module in SCENARIOS.items():
        if store.get_agent_by_name(scenario_name) is not None:
            continue
        persona = getattr(module, "PERSONA", "").strip()
        spec_module = module.spec
        created.append(store.create_agent(
            name=scenario_name,
            display_name=scenario_name.replace("_", " ").title(),
            scenario=scenario_name,
            persona=persona,
            tools=[],
            skills=[s.name for s in getattr(spec_module, "build_skills")().skills.values()]
            if hasattr(spec_module, "build_skills") else [],
            always_tools=list(getattr(spec_module, "ALWAYS_TOOLS", ())),
        ))
    return created
