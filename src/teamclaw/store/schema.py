"""Persistence schema for agent definitions, runs and conversations.

Scope of what is stored, and what deliberately is not
-----------------------------------------------------
Stored: the *declarations* — an agent's persona, which tools and skills it may
use, its budget — plus a record of every run and the conversation it belonged to.

Not stored: an agent's working state. That lives in its workspace on disk, which
is the platform's central bet (see :mod:`teamclaw.execution.workspace`). Putting
run state in the database would create two sources of truth for the same thing
and make resume a merge rather than a re-read.

So the database answers "which agents exist and what happened", and the
filesystem answers "what does this run know". A run row points at its workspace
and its trace; it does not duplicate them.

SQLite by default, Postgres by URL. Nothing here uses SQLite-specific SQL, and
JSON columns are portable across both.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

SCHEMA_VERSION = 1


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Base(DeclarativeBase):
    pass


class RunState(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"
    NEEDS_HUMAN = "needs_human"


class SchemaMeta(Base):
    """One row. Lets startup refuse a database written by a newer version."""

    __tablename__ = "schema_meta"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AgentDef(Base):
    """A declarative agent: the same thing an ``AgentSpec`` carries, persisted.

    ``tools`` and ``skills`` hold *names*, not definitions. A tool's behaviour
    belongs to its scenario module and is versioned with the code; storing a copy
    here would let the database drift from what the sandbox can actually import.
    """

    __tablename__ = "agent_defs"
    __table_args__ = (UniqueConstraint("name", name="uq_agent_name"),)

    id: Mapped[str] = mapped_column(String(40), primary_key=True,
                                    default=lambda: new_id("agent"))
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    display_name: Mapped[str] = mapped_column(String(200), default="")
    scenario: Mapped[str] = mapped_column(String(60), nullable=False)
    persona: Mapped[str] = mapped_column(Text, nullable=False)

    tools: Mapped[list] = mapped_column(JSON, default=list)
    skills: Mapped[list] = mapped_column(JSON, default=list)
    always_tools: Mapped[list] = mapped_column(JSON, default=list)

    window: Mapped[int] = mapped_column(Integer, default=32_000)
    max_output: Mapped[int] = mapped_column(Integer, default=2_048)
    max_steps: Mapped[int] = mapped_column(Integer, default=14)
    tool_k: Mapped[int] = mapped_column(Integer, default=8)
    skill_k: Mapped[int] = mapped_column(Integer, default=2)
    memory_k: Mapped[int] = mapped_column(Integer, default=5)
    compaction_threshold: Mapped[float] = mapped_column(Float, default=0.70)

    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow,
                                                 onupdate=utcnow)

    runs: Mapped[list["Run"]] = relationship(back_populates="agent",
                                             cascade="all, delete-orphan")
    # Relationships an agent is permitted to initiate. A2A authorisation is a
    # lookup here, never an inference from the message.
    peers_out: Mapped[list["AgentRelation"]] = relationship(
        back_populates="source", foreign_keys="AgentRelation.source_id",
        cascade="all, delete-orphan",
    )

    def to_json(self) -> dict:
        return {
            "id": self.id, "name": self.name,
            "display_name": self.display_name or self.name,
            "scenario": self.scenario, "persona": self.persona,
            "tools": list(self.tools or []), "skills": list(self.skills or []),
            "always_tools": list(self.always_tools or []),
            "window": self.window, "max_output": self.max_output,
            "max_steps": self.max_steps, "tool_k": self.tool_k,
            "skill_k": self.skill_k, "memory_k": self.memory_k,
            "compaction_threshold": self.compaction_threshold,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class AgentRelation(Base):
    """Who may delegate to whom, and what they may pass.

    A2A authorisation is a row here rather than a judgement made from the
    message, and the tenant is on the row rather than on the request — so a
    cross-tenant delegation cannot be authorised by anything the caller says.
    """

    __tablename__ = "agent_relations"
    __table_args__ = (
        UniqueConstraint("source_id", "target_id", name="uq_relation_pair"),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True,
                                    default=lambda: new_id("rel"))
    source_id: Mapped[str] = mapped_column(ForeignKey("agent_defs.id"), nullable=False)
    target_id: Mapped[str] = mapped_column(ForeignKey("agent_defs.id"), nullable=False)
    tenant: Mapped[str] = mapped_column(String(80), default="default")
    may_delegate: Mapped[bool] = mapped_column(Boolean, default=True)
    may_share_files: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    source: Mapped["AgentDef"] = relationship(back_populates="peers_out",
                                              foreign_keys=[source_id])

    def to_json(self) -> dict:
        return {
            "id": self.id, "source_id": self.source_id, "target_id": self.target_id,
            "tenant": self.tenant, "may_delegate": self.may_delegate,
            "may_share_files": self.may_share_files,
        }


class Conversation(Base):
    """A thread of interaction, possibly arriving from a channel."""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(40), primary_key=True,
                                    default=lambda: new_id("conv"))
    agent_id: Mapped[str] = mapped_column(ForeignKey("agent_defs.id"), nullable=False)
    tenant: Mapped[str] = mapped_column(String(80), default="default")
    channel: Mapped[str] = mapped_column(String(40), default="web")
    external_id: Mapped[str] = mapped_column(String(200), default="")
    title: Mapped[str] = mapped_column(String(300), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan",
        order_by="Message.created_at",
    )

    def to_json(self) -> dict:
        return {
            "id": self.id, "agent_id": self.agent_id, "tenant": self.tenant,
            "channel": self.channel, "external_id": self.external_id,
            "title": self.title,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(40), primary_key=True,
                                    default=lambda: new_id("msg"))
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"),
                                                 nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, default="")
    run_id: Mapped[str | None] = mapped_column(String(60), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")

    def to_json(self) -> dict:
        return {
            "id": self.id, "conversation_id": self.conversation_id, "role": self.role,
            "content": self.content, "run_id": self.run_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Run(Base):
    """A record that a run happened, and where its real artefacts live.

    ``workspace_path`` and ``trace_path`` are pointers, not copies. The run's
    state is on disk; this row is the index into it.
    """

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(60), primary_key=True)
    agent_id: Mapped[str] = mapped_column(ForeignKey("agent_defs.id"), nullable=False)
    conversation_id: Mapped[str | None] = mapped_column(
        ForeignKey("conversations.id"), nullable=True
    )
    objective: Mapped[str] = mapped_column(Text, default="")
    state: Mapped[RunState] = mapped_column(Enum(RunState), default=RunState.QUEUED)
    stop_reason: Mapped[str] = mapped_column(String(80), default="")
    final_answer: Mapped[str] = mapped_column(Text, default="")

    steps: Mapped[int] = mapped_column(Integer, default=0)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    peak_utilisation: Mapped[float] = mapped_column(Float, default=0.0)
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)

    sandbox_backend: Mapped[str] = mapped_column(String(40), default="")
    sandbox_isolation: Mapped[str] = mapped_column(String(60), default="")
    workspace_path: Mapped[str] = mapped_column(String(500), default="")
    trace_path: Mapped[str] = mapped_column(String(500), default="")
    report: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str] = mapped_column(Text, default="")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True),
                                                         nullable=True)

    agent: Mapped["AgentDef"] = relationship(back_populates="runs")

    @property
    def tokens_total(self) -> int:
        return (self.tokens_in or 0) + (self.tokens_out or 0)

    def to_json(self) -> dict:
        return {
            "id": self.id, "agent_id": self.agent_id,
            "conversation_id": self.conversation_id,
            "objective": self.objective,
            "state": self.state.value if isinstance(self.state, RunState) else self.state,
            "stop_reason": self.stop_reason, "final_answer": self.final_answer,
            "steps": self.steps, "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out, "tokens_total": self.tokens_total,
            "cost_usd": round(self.cost_usd or 0.0, 6),
            "peak_utilisation": round(self.peak_utilisation or 0.0, 4),
            "duration_s": round(self.duration_s or 0.0, 2),
            "sandbox": {"backend": self.sandbox_backend,
                        "isolation": self.sandbox_isolation},
            "workspace_path": self.workspace_path, "trace_path": self.trace_path,
            "report": self.report or {}, "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }
