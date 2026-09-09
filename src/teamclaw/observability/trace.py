"""Append-only span tracing.

Design notes
------------
* JSONL, one line per span *close*, so a crashed run still leaves a readable
  trace up to the crash point. Checkpoint/resume relies on this.
* Spans nest via ``parent_id`` and a contextvar stack, so instrumentation does
  not have to thread a span object through every call site.
* ``attrs`` is free-form but a handful of keys are conventional and consumed by
  the reporting layer: ``model``, ``provider``, ``tokens_in``, ``tokens_out``,
  ``cost_usd``, ``slots`` (context ledger breakdown), ``exit_code``.
"""

from __future__ import annotations

import contextvars
import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator


class SpanKind(str, Enum):
    RUN = "run"
    STEP = "step"
    CONTEXT_BUILD = "context_build"
    LLM_CALL = "llm_call"
    SANDBOX_EXEC = "sandbox_exec"
    TOOL_CALL = "tool_call"
    COMPACTION = "compaction"
    SUBAGENT = "subagent"
    EVAL_CASE = "eval_case"
    HITL = "hitl"


def new_run_id() -> str:
    return f"run_{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"


@dataclass
class Span:
    span_id: str
    kind: SpanKind
    name: str
    parent_id: str | None = None
    start: float = field(default_factory=time.time)
    end: float | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    @property
    def duration_s(self) -> float:
        return (self.end or time.time()) - self.start

    def set(self, **attrs: Any) -> "Span":
        self.attrs.update(attrs)
        return self

    def event(self, name: str, **data: Any) -> "Span":
        self.events.append({"t": time.time(), "name": name, **data})
        return self

    def to_json(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "kind": self.kind.value,
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "duration_s": round(self.duration_s, 4),
            "attrs": self.attrs,
            "events": self.events,
            "error": self.error,
        }


_CURRENT: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "teamclaw_span", default=None
)


class Tracer:
    """Writes spans to ``<runs>/<run_id>/trace.jsonl`` and keeps them in memory."""

    def __init__(self, run_id: str, run_dir: Path, *, echo: bool = False) -> None:
        self.run_id = run_id
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "trace.jsonl"
        self.spans: list[Span] = []
        self._echo = echo
        self._fh = self.path.open("a", encoding="utf-8")

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- span creation -----------------------------------------------------
    @contextmanager
    def span(self, kind: SpanKind, name: str, **attrs: Any) -> Iterator[Span]:
        sp = Span(
            span_id=uuid.uuid4().hex[:12],
            kind=kind,
            name=name,
            parent_id=_CURRENT.get(),
            attrs=dict(attrs),
        )
        token = _CURRENT.set(sp.span_id)
        try:
            yield sp
        except BaseException as exc:  # noqa: BLE001 - trace then re-raise
            sp.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _CURRENT.reset(token)
            sp.end = time.time()
            self._write(sp)

    def _write(self, sp: Span) -> None:
        self.spans.append(sp)
        if not self._fh.closed:
            self._fh.write(json.dumps(sp.to_json(), ensure_ascii=False) + "\n")
            self._fh.flush()
        if self._echo:
            mark = "!" if sp.error else " "
            print(f"[trace]{mark}{sp.kind.value:>14} {sp.name} ({sp.duration_s:.2f}s)")

    # -- queries -----------------------------------------------------------
    def of_kind(self, kind: SpanKind) -> list[Span]:
        return [s for s in self.spans if s.kind is kind]

    def sum_attr(self, key: str, kind: SpanKind | None = None) -> float:
        pool = self.spans if kind is None else self.of_kind(kind)
        return float(sum(s.attrs.get(key, 0) or 0 for s in pool))

    def summary(self) -> dict[str, Any]:
        errors = [s for s in self.spans if s.error]
        return {
            "run_id": self.run_id,
            "spans": len(self.spans),
            "steps": len(self.of_kind(SpanKind.STEP)),
            "llm_calls": len(self.of_kind(SpanKind.LLM_CALL)),
            "sandbox_execs": len(self.of_kind(SpanKind.SANDBOX_EXEC)),
            "compactions": len(self.of_kind(SpanKind.COMPACTION)),
            "tokens_in": int(self.sum_attr("tokens_in")),
            "tokens_out": int(self.sum_attr("tokens_out")),
            "cost_usd": round(self.sum_attr("cost_usd"), 6),
            "errors": len(errors),
            "error_samples": [s.error for s in errors[:5]],
        }

    @staticmethod
    def read(path: Path) -> list[dict[str, Any]]:
        """Read a trace back, tolerating a truncated final line from a crash."""
        out: list[dict[str, Any]] = []
        if not path.exists():
            return out
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                break  # truncated tail: everything after is unusable
        return out
