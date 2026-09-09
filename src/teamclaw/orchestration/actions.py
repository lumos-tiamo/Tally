"""The agent's action protocol and its parser.

The agent has exactly two moves: run Python, or finish. Keeping the surface that
small is what makes the loop debuggable — there is no tool-selection step to go
wrong, no argument schema to violate. Choosing a tool becomes an import, and
calling it becomes a function call, both checked by the interpreter rather than
by us.

Parsing is written to be *forgiving of small models*, because that is the model
class this platform targets. A 4B–8B model reliably produces a fenced code
block and unreliably produces anything else, so:

* fences may be ```python, ```py or bare ```
* ``DONE`` may appear with or without a colon, in any case, on its own line
* prose outside the fence is treated as the thought, not as an error
* a response with neither a fence nor DONE is a :class:`Malformed` action that
  the loop feeds back as a format correction rather than crashing on

The one thing we refuse to guess at is code hidden in prose without a fence:
executing text a model did not mark as code is how you run its commentary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

FENCE = re.compile(
    r"```(?:python|py|Python|PY)?[ \t]*\r?\n(.*?)(?:```|\Z)", re.DOTALL
)
DONE = re.compile(r"^\s*(?:DONE|FINISHED|COMPLETE)\s*:?\s*$", re.IGNORECASE | re.MULTILINE)
THOUGHT = re.compile(r"^\s*(?:THOUGHT|PLAN|REASONING)\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)


class ActionKind(str, Enum):
    RUN = "run"
    DONE = "done"
    MALFORMED = "malformed"


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    code: str = ""
    thought: str = ""
    final: str = ""
    raw: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.kind is ActionKind.DONE

    def to_json(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "thought": self.thought[:200],
            "code_lines": len(self.code.splitlines()) if self.code else 0,
            "final_chars": len(self.final),
        }


PROTOCOL_PROMPT = """\
You work by writing Python that runs in a sandbox. Each step, reply in exactly \
one of these two forms and nothing else.

To run code:

THOUGHT: one sentence on what this step establishes
```python
# your code
print(...)   # only what you print comes back to you
```

To finish:

DONE
<your final answer>

Rules that follow from how the sandbox works:

* Each step is a **fresh interpreter**. Nothing stays in memory between steps.
  Persist to the workspace and re-read what you need.
* Only what you `print` returns to you. Everything else stays on disk. Print
  digests — shapes, column names, a few rows — never whole documents.
* Import tools with `from tools import <module>`. Use `help(<module>)` when you
  are unsure of a signature; guessing costs a whole step.
* Write large intermediates to `workspace/` as parquet/json/markdown.
* Never state a computed number you did not actually compute in code.
"""


def parse_action(text: str) -> Action:
    """Turn a model response into an :class:`Action`."""
    raw = text or ""
    thought_match = THOUGHT.search(raw)
    thought = thought_match.group(1).strip() if thought_match else ""

    fences = FENCE.findall(raw)
    code = "\n\n".join(block.strip() for block in fences if block.strip())

    done_match = DONE.search(raw)

    # A response containing both is a model that solved the step and announced
    # completion in one breath. Run the code first — discarding it would lose
    # the work — and let the next step see the result and re-declare DONE.
    if code:
        if not thought:
            lead = FENCE.split(raw)[0].strip() if FENCE.search(raw) else ""
            thought = " ".join(lead.split())[:300]
        return Action(kind=ActionKind.RUN, code=code, thought=thought, raw=raw)

    if done_match:
        final = raw[done_match.end():].strip()
        if not final:
            # DONE with nothing after it: take whatever preceded the marker.
            final = raw[: done_match.start()].strip()
        return Action(kind=ActionKind.DONE, final=final, thought=thought, raw=raw)

    return Action(kind=ActionKind.MALFORMED, thought=thought, raw=raw)


MALFORMED_FEEDBACK = """\
Your reply matched neither required form, so nothing could be executed.

Reply with either a fenced Python block:

THOUGHT: what this step establishes
```python
print("...")
```

or the single word DONE on its own line followed by your final answer.

Do not describe code in prose — code outside a fenced block is never executed.\
"""
