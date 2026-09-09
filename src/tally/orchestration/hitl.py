"""Human-in-the-loop interruption points.

An autonomous agent needs a defined way to stop and ask, or it will either give
up silently or loop until the budget is gone. Three triggers, all reached from
the agent loop:

* the convergence loop escalated to :class:`~tally.execution.convergence.Escalation.HITL`
* a policy gate matched (a tool the spec marks as requiring approval)
* the step budget ran out with no final answer

The default resolver is :class:`AutoDeny`, which *records the request and
declines*. That is the correct default for an eval harness: a run that would
have needed a human is a run that did not succeed autonomously, and letting it
silently proceed would inflate the score. Interactive use supplies
:class:`ConsoleResolver`; scripted tests supply :class:`ScriptedResolver`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol, Sequence


class InterruptReason(str, Enum):
    REPEATED_FAILURE = "repeated_failure"
    POLICY_GATE = "policy_gate"
    BUDGET_EXHAUSTED = "budget_exhausted"
    AGENT_REQUEST = "agent_request"


class Decision(str, Enum):
    APPROVE = "approve"
    DENY = "deny"
    GUIDANCE = "guidance"   # not a yes/no: a human supplied a hint, keep going
    ABORT = "abort"


@dataclass
class InterruptRequest:
    reason: InterruptReason
    step: int
    question: str
    context: str = ""
    at: float = field(default_factory=time.time)

    def to_json(self) -> dict[str, object]:
        return {
            "reason": self.reason.value,
            "step": self.step,
            "question": self.question,
            "context": self.context[:1000],
            "at": self.at,
        }


@dataclass
class InterruptResponse:
    decision: Decision
    message: str = ""

    @property
    def continues(self) -> bool:
        return self.decision in {Decision.APPROVE, Decision.GUIDANCE}

    def to_json(self) -> dict[str, object]:
        return {"decision": self.decision.value, "message": self.message[:500]}


class Resolver(Protocol):
    def resolve(self, request: InterruptRequest) -> InterruptResponse: ...


@dataclass
class AutoDeny:
    """Record and decline. The honest default for unattended runs."""

    log: list[InterruptRequest] = field(default_factory=list)

    def resolve(self, request: InterruptRequest) -> InterruptResponse:
        self.log.append(request)
        return InterruptResponse(
            Decision.DENY,
            "No human is attached to this run. Treat the step as failed and, if you "
            "cannot proceed another way, finish with an explicit statement of what "
            "blocked you rather than guessing.",
        )


@dataclass
class ScriptedResolver:
    """Replays a fixed list of responses. For tests."""

    responses: Sequence[InterruptResponse]
    log: list[InterruptRequest] = field(default_factory=list)
    index: int = 0

    def resolve(self, request: InterruptRequest) -> InterruptResponse:
        self.log.append(request)
        if self.index < len(self.responses):
            out = self.responses[self.index]
            self.index += 1
            return out
        return InterruptResponse(Decision.ABORT, "scripted resolver exhausted")


@dataclass
class ConsoleResolver:
    """Prompts on stdin. Interactive use only."""

    log: list[InterruptRequest] = field(default_factory=list)

    def resolve(self, request: InterruptRequest) -> InterruptResponse:
        self.log.append(request)
        print(f"\n[HITL] step {request.step} — {request.reason.value}")
        print(request.question)
        if request.context:
            print(f"--- context ---\n{request.context[:2000]}\n---------------")
        try:
            raw = input("[a]pprove / [d]eny / [g]uidance / [x]abort > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return InterruptResponse(Decision.ABORT, "no console input available")
        if raw.startswith("a"):
            return InterruptResponse(Decision.APPROVE)
        if raw.startswith("g"):
            return InterruptResponse(Decision.GUIDANCE, input("guidance > ").strip())
        if raw.startswith("x"):
            return InterruptResponse(Decision.ABORT, "aborted by operator")
        return InterruptResponse(Decision.DENY, input("reason (optional) > ").strip())


class InterruptLog:
    """Persists every request/response pair for the run report."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.entries: list[dict[str, object]] = []

    def record(self, request: InterruptRequest, response: InterruptResponse) -> None:
        entry = {"request": request.to_json(), "response": response.to_json()}
        self.entries.append(entry)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def stats(self) -> dict[str, object]:
        by_reason: dict[str, int] = {}
        for e in self.entries:
            reason = str(e["request"]["reason"])  # type: ignore[index]
            by_reason[reason] = by_reason.get(reason, 0) + 1
        return {"interrupts": len(self.entries), "by_reason": by_reason}
