"""Error-feedback convergence loop.

A small model writing sandbox code fails constantly, so the loop that turns a
traceback back into a corrected attempt is not a detail — it is most of whether
the agent works at all. Three design choices carry that weight:

**Feedback is structured, not a raw traceback.** The model gets the exception
type, the offending line quoted from its own code, and — critically — the
docstring of the tool it misused, re-injected. "AttributeError on line 7" is not
actionable; "line 7 called ``sec.fetch()``, which does not exist; ``sec`` exports
fetch_filing(cik, form)" is.

**Repetition is detected by *signature*, not by count.** Two different errors in
a row is progress; the same error twice is a loop. The signature deliberately
excludes the message text — ``NameError: name 'df' is not defined`` and
``NameError: name 'tbl' is not defined`` are the same mistake being repeated.

**Escalation is bounded and ordered.** Same signature twice → retry on a
stronger model (:class:`Escalation.MODEL`). Three times → stop and ask a human
(:class:`Escalation.HITL`). Without a ceiling, a model that cannot solve a step
burns the entire free-tier quota rediscovering that fact.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from teamclaw.execution.registry import ToolRegistry
from teamclaw.execution.sandbox import ExecResult

TRACE_LINE = re.compile(r'File "[^"]*step\.py", line (\d+)')
EXC_LINE = re.compile(r"^(\w+(?:Error|Exception|Exit|Interrupt|Warning))\b:?\s*(.*)$", re.MULTILINE)
ATTR_MISS = re.compile(r"module '([\w.]+)' has no attribute '(\w+)'")
# A wrong keyword or arity on a tool call. This is the single most common way a
# model misuses a tool — it invents a parameter — and the first real model run
# produced four of them. Without attributing it to a module the tool surface was
# never re-injected, so the correction the model most needed was the one it did
# not get.
BAD_KWARG = re.compile(r"(\w+)\(\) got an unexpected keyword argument '(\w+)'")
BAD_ARITY = re.compile(
    r"(\w+)\(\) (?:takes|missing) .*?(?:argument|positional)", re.DOTALL
)
NAME_MISS = re.compile(r"name '(\w+)' is not defined")
IMPORT_MISS = re.compile(r"No module named '([\w.]+)'")
TOOL_ERR = re.compile(r"ToolError: ([\w.]+): (.*)")


class Escalation(str, Enum):
    NONE = "none"        # retry on the same model with feedback
    MODEL = "model"      # retry on a stronger model
    HITL = "hitl"        # stop; a human must look
    ABORT = "abort"      # unrecoverable (e.g. budget exhausted)


@dataclass(frozen=True)
class Failure:
    kind: str
    message: str
    line: int | None = None
    symbol: str = ""       # the missing name / attribute / module, when known
    module: str = ""       # the tool module involved, when known
    timed_out: bool = False

    @property
    def signature(self) -> str:
        """Identity of the *mistake class*, not of the specific symbol.

        Three things are deliberately excluded:

        *The message text.* ``name 'df' is not defined`` and ``name 'tbl' is not
        defined`` are one misconception — that state survives between steps —
        being repeated. Treating them as distinct lets a model cycle through
        variable names forever without ever tripping the escalation.

        *The missing symbol.* Likewise ``sec.fetch()`` then ``sec.download()`` is
        one misconception: guessing tool names instead of reading ``help()``.

        *The line number.* A model that rewrites its code hits the same
        conceptual error on a different line, and counting that as progress is
        how loops go undetected.

        The *module* is kept, because it identifies which tool surface was
        misunderstood — misusing ``sec`` and then misusing ``doc`` really are two
        different problems, and each deserves its own feedback before escalating.
        """
        return f"{self.kind}:{self.module}" if self.module else self.kind

    def to_json(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "message": self.message[:300],
            "line": self.line,
            "symbol": self.symbol,
            "module": self.module,
            "timed_out": self.timed_out,
            "signature": self.signature,
        }


def registry_module_for(func: str, registry: ToolRegistry | None) -> str:
    """Which tool module exports ``func``, if any.

    A TypeError names the function and not its module, and two modules may both
    export a ``save_json``. An ambiguous name resolves to nothing rather than to
    a guess: re-injecting the wrong module's signatures would be worse than
    re-injecting none.
    """
    if registry is None:
        return ""
    owners = {spec.module for spec in registry.tools.values() if spec.func == func}
    return owners.pop() if len(owners) == 1 else ""


def parse_failure(result: ExecResult, registry: ToolRegistry | None = None) -> Failure | None:
    """Extract a structured failure from a sandbox result, or None on success."""
    if result.ok:
        return None
    if result.timed_out:
        return Failure(kind="Timeout", message=result.stderr.strip() or "wall clock exceeded",
                       timed_out=True)

    stderr = result.stderr or ""

    # Static rejection from the local sandbox's pre-check.
    if result.exit_code == 126:
        symbol = ""
        if (m := re.search(r"(?:import|call) in local sandbox: ([\w.()]+)", stderr)):
            symbol = m.group(1).rstrip("()")
        return Failure(kind="SandboxPolicy", message=stderr.strip(), symbol=symbol)

    line = None
    matches = TRACE_LINE.findall(stderr)
    if matches:
        line = int(matches[-1])  # innermost frame in the agent's own file

    if (m := TOOL_ERR.search(stderr)):
        tool, msg = m.group(1), m.group(2)
        module = tool.split(".")[0]
        return Failure(kind="ToolError", message=msg.strip(), line=line,
                       symbol=tool.split(".")[-1], module=module)

    kind, message = "UnknownError", stderr.strip()[-400:]
    if (m := EXC_LINE.findall(stderr)):
        kind, message = m[-1][0], m[-1][1].strip()

    symbol, module = "", ""
    if (m := ATTR_MISS.search(stderr)):
        module, symbol = m.group(1).replace("tools.", ""), m.group(2)
    elif (m := NAME_MISS.search(stderr)):
        symbol = m.group(1)
    elif (m := IMPORT_MISS.search(stderr)):
        symbol, module = m.group(1), m.group(1).split(".")[0]
    elif (m := BAD_KWARG.search(stderr)) or (m := BAD_ARITY.search(stderr)):
        # The traceback names the function but not its module, so it is looked
        # up in the registry — which is what lets the feedback show the real
        # signature instead of only naming the bad keyword.
        symbol = m.group(1)
        module = registry_module_for(symbol, registry)

    return Failure(kind=kind, message=message, line=line, symbol=symbol, module=module)


@dataclass
class Feedback:
    text: str
    escalation: Escalation
    failure: Failure
    repeat_count: int

    def to_json(self) -> dict[str, object]:
        return {
            "escalation": self.escalation.value,
            "repeat_count": self.repeat_count,
            "failure": self.failure.to_json(),
            "feedback_chars": len(self.text),
        }


@dataclass
class ConvergenceLoop:
    registry: ToolRegistry | None = None
    escalate_after: int = 2      # same signature this many times -> stronger model
    hitl_after: int = 3          # ... this many times -> ask a human
    max_attempts: int = 6
    seen: Counter[str] = field(default_factory=Counter)
    failures: list[Failure] = field(default_factory=list)

    # -- state -------------------------------------------------------------
    def reset(self) -> None:
        self.seen.clear()
        self.failures.clear()

    @property
    def attempts(self) -> int:
        return len(self.failures)

    # -- main entry point --------------------------------------------------
    def observe(self, result: ExecResult, *, code: str) -> Feedback | None:
        """Record a failure and produce the feedback to send back, or None if OK."""
        failure = parse_failure(result, self.registry)
        if failure is None:
            return None

        self.failures.append(failure)
        self.seen[failure.signature] += 1
        repeats = self.seen[failure.signature]

        if self.attempts >= self.max_attempts:
            escalation = Escalation.ABORT
        elif repeats >= self.hitl_after:
            escalation = Escalation.HITL
        elif repeats >= self.escalate_after:
            escalation = Escalation.MODEL
        else:
            escalation = Escalation.NONE

        return Feedback(
            text=self.render(failure, code=code, repeats=repeats, escalation=escalation),
            escalation=escalation,
            failure=failure,
            repeat_count=repeats,
        )

    # -- feedback rendering ------------------------------------------------
    def render(
        self, failure: Failure, *, code: str, repeats: int, escalation: Escalation
    ) -> str:
        parts: list[str] = [f"Your code failed: **{failure.kind}**"]
        if failure.message:
            parts.append(f"Message: {failure.message[:400]}")

        quoted = self._quote(code, failure.line)
        if quoted:
            parts.append(f"The failing line ({failure.line}) in your code was:\n{quoted}")

        if (hint := self._hint(failure)):
            parts.append(hint)

        if (surface := self._tool_surface(failure)):
            parts.append(surface)

        if repeats >= self.escalate_after:
            parts.append(
                f"You have now hit this same error {repeats} times. Do not retry the "
                "same approach: change strategy, or print intermediate values to find "
                "out what the data actually looks like before acting on it."
            )
        if escalation is Escalation.HITL:
            parts.append("This step is being escalated to a human reviewer.")

        return "\n\n".join(parts)

    @staticmethod
    def _quote(code: str, line: int | None, context: int = 1) -> str:
        if not line:
            return ""
        lines = code.splitlines()
        if not 1 <= line <= len(lines):
            return ""
        lo, hi = max(1, line - context), min(len(lines), line + context)
        out = []
        for n in range(lo, hi + 1):
            mark = ">>" if n == line else "  "
            out.append(f"{mark} {n:>3} | {lines[n - 1]}")
        return "\n".join(out)

    def _hint(self, failure: Failure) -> str:
        """Actionable instruction per failure class."""
        match failure.kind:
            case "Timeout":
                return (
                    "The step exceeded its wall clock. Split the work: write partial "
                    "results to the workspace, print a digest, and continue in the next "
                    "step. Do not restart the whole computation."
                )
            case "SandboxPolicy":
                return (
                    f"`{failure.symbol}` is not permitted in the sandbox. Network access "
                    "and process control are unavailable by design — use the bridged "
                    "tool modules for anything that needs the outside world."
                )
            case "ModuleNotFoundError" | "ImportError":
                return (
                    f"`{failure.symbol}` is not installed and you cannot install it. "
                    "The available set is fixed: pandas, numpy, pyarrow, lxml, "
                    "beautifulsoup4, plus the generated `tools` package."
                )
            case "NameError":
                return (
                    f"`{failure.symbol}` was never assigned in this step. Each step runs "
                    "in a fresh interpreter — nothing carries over in memory. Re-read what "
                    "you need from the workspace at the start of every step."
                )
            case "AttributeError":
                return (
                    f"`{failure.module or 'that object'}` has no `{failure.symbol}`. "
                    "Call `dir()` on it, or `help()` on the module, before guessing a name."
                )
            case "KeyError":
                return (
                    f"The key {failure.message.strip() or 'you used'} is not in that "
                    "object. Print its keys before subscripting — tool results are "
                    "documented in the signature but their exact shape is worth "
                    "checking once, and filings differ in structure between companies "
                    "and years so a key that worked for one may be absent in another."
                )
            case "ToolError":
                return (
                    f"The tool `{failure.module}.{failure.symbol}` rejected the call. "
                    "Check its signature and argument types with `help()` before retrying."
                )
            case "TypeError":
                return (
                    f"`{failure.symbol}` was called with arguments it does not accept. "
                    "Use only the parameters in the signature below — do not invent "
                    "keywords such as `year` or `item` to narrow a search. Filter the "
                    "result yourself after the call."
                )
            case "IndexError":
                return (
                    "You indexed past the end of a list. Tool results carry a count; "
                    "check it before subscripting, because a lookup that found "
                    "nothing returns an empty list rather than raising."
                )
            case "ZeroDivisionError":
                return (
                    "A denominator was zero — most often a field that was absent and "
                    "defaulted to 0. Verify the inputs were actually extracted before "
                    "computing a ratio, and abstain rather than reporting a fabricated one."
                )
            case _:
                return ""

    def _tool_surface(self, failure: Failure) -> str:
        """Re-inject the docstring of the module the agent got wrong.

        This is the cheapest correction available: the model does not need to be
        smarter, it needs the actual signature in front of it.
        """
        if self.registry is None or not failure.module:
            return ""
        specs = self.registry.by_module(failure.module)
        if not specs:
            return ""
        lines = "\n".join(s.signature_line() for s in specs)
        return f"`{failure.module}` actually exports:\n{lines}"

    # -- reporting ---------------------------------------------------------
    def stats(self) -> dict[str, object]:
        return {
            "attempts": self.attempts,
            "distinct_signatures": len(self.seen),
            "repeats": {k: v for k, v in self.seen.items() if v > 1},
            "by_kind": dict(Counter(f.kind for f in self.failures)),
        }


def recovery_rate(loops: Sequence[ConvergenceLoop], successes: Sequence[bool]) -> float:
    """Share of steps that failed at least once but eventually succeeded.

    Reported by the eval harness as "sandbox exception recovery rate" — the
    number that says whether the feedback loop actually converges.
    """
    eligible = [(l, ok) for l, ok in zip(loops, successes, strict=False) if l.attempts]
    if not eligible:
        return 0.0
    return round(sum(1 for _, ok in eligible if ok) / len(eligible), 4)
