"""Deterministic providers for tests, offline development and record/replay.

``FakeProvider`` answers with a rule table so the whole platform — router,
ledger, sandbox loop, eval harness — can be exercised with no network and no
credentials. ``ScriptedProvider`` replays a fixed list of responses, which is
how the agent-loop tests assert on multi-step behaviour.

These are test doubles, not a simulation of model quality. Any number produced
with them is a plumbing check, never an eval result; the eval harness refuses to
publish scores from a fake provider (see evaluation/harness.py).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from teamclaw.models.base import Completion, Message, ProviderUnavailable

Rule = tuple[str, str | Callable[[list[Message]], str]]


def estimate_tokens(text: str) -> int:
    """Rough token count used only for fake usage figures."""
    return max(1, len(text) // 4)


class FakeProvider:
    is_paid = False

    def __init__(
        self,
        *,
        name: str = "fake",
        model: str = "fake-model",
        rules: Sequence[Rule] | None = None,
        default: str = "OK",
        fail_times: int = 0,
        failure: Exception | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.rules = list(rules or [])
        self.default = default
        self.calls: list[list[Message]] = []
        self._fail_times = fail_times
        self._failure = failure or ProviderUnavailable(f"{name}: injected failure")

    def available(self) -> bool:
        return True

    def generate(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> Completion:
        if self._fail_times > 0:
            self._fail_times -= 1
            raise self._failure

        self.calls.append(list(messages))
        blob = "\n".join(m.content for m in messages)
        text = self.default
        for pattern, response in self.rules:
            if re.search(pattern, blob, re.IGNORECASE | re.DOTALL):
                text = response(messages) if callable(response) else response
                break
        return Completion(
            text=text,
            model=self.model,
            provider=self.name,
            tokens_in=estimate_tokens(blob),
            tokens_out=estimate_tokens(text),
        )


class ScriptedProvider:
    """Returns responses in order; raises once the script is exhausted."""

    is_paid = False

    def __init__(
        self,
        responses: Sequence[str],
        *,
        name: str = "scripted",
        model: str = "fake-model",
        loop_last: bool = False,
    ) -> None:
        self.responses = list(responses)
        self.name = name
        self.model = model
        self.loop_last = loop_last
        self.index = 0
        self.calls: list[list[Message]] = []

    def available(self) -> bool:
        return True

    def generate(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> Completion:
        self.calls.append(list(messages))
        if self.index < len(self.responses):
            text = self.responses[self.index]
            self.index += 1
        elif self.loop_last and self.responses:
            text = self.responses[-1]
        else:
            raise ProviderUnavailable(
                f"{self.name}: script exhausted after {len(self.responses)} responses"
            )
        blob = "\n".join(m.content for m in messages)
        return Completion(
            text=text,
            model=self.model,
            provider=self.name,
            tokens_in=estimate_tokens(blob),
            tokens_out=estimate_tokens(text),
        )
