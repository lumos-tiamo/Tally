"""Core model-layer types shared by every provider.

``Purpose`` is the routing key. The spec's cost strategy is capability-tiered
routing: a cheap local model handles extraction, a free cloud model handles
planning, and the paid model is reachable only from the ablation harness. Making
purpose an explicit argument (rather than letting call sites pick a model name)
is what makes that policy enforceable in one place.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class Purpose(str, Enum):
    """What the call is *for*, not which model should serve it."""

    PLAN = "plan"          # decide the next step; needs the strongest free model
    DECIDE = "decide"      # pick among options mid-task
    REFLECT = "reflect"    # critique own output
    CODE = "code"          # write sandbox python
    EXTRACT = "extract"    # pull structured fields out of text; local 8B is fine
    CLASSIFY = "classify"  # short label output; local 8B
    REWRITE = "rewrite"    # query rewriting / normalisation; local 8B
    JUDGE = "judge"        # LLM-as-judge for eval; must be independent of the actor
    SUMMARIZE = "summarize"  # compaction


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class Message:
    role: Role
    content: str

    def to_json(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}

    @staticmethod
    def system(text: str) -> "Message":
        return Message(Role.SYSTEM, text)

    @staticmethod
    def user(text: str) -> "Message":
        return Message(Role.USER, text)

    @staticmethod
    def assistant(text: str) -> "Message":
        return Message(Role.ASSISTANT, text)


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    provider: str
    tokens_in: int = 0
    tokens_out: int = 0
    cached: bool = False
    finish_reason: str = "stop"
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def with_cached(self, cached: bool) -> "Completion":
        return Completion(
            text=self.text,
            model=self.model,
            provider=self.provider,
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            cached=cached,
            finish_reason=self.finish_reason,
            raw=self.raw,
        )


class ProviderError(RuntimeError):
    """Base class for provider failures."""


class RateLimited(ProviderError):
    """429 or an explicit quota rejection. Router should降级, not retry forever."""

    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class ProviderUnavailable(ProviderError):
    """Timeout, connection error, 5xx. Retryable then degradable."""


class NotConfigured(ProviderError):
    """Missing credentials. Never retried; the router skips the provider."""


class PaidCallBlocked(ProviderError):
    """A paid model was requested without TALLY_ALLOW_PAID=1."""


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    model: str
    is_paid: bool

    def available(self) -> bool: ...

    def generate(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> Completion: ...


def request_fingerprint(
    model: str,
    messages: list[Message],
    *,
    max_tokens: int,
    temperature: float,
    stop: list[str] | None,
) -> str:
    """Stable hash of everything that can change the answer.

    This is the key that makes 7 ablation arms affordable on a free tier: the
    arms share most prompts, so the second arm onward mostly reads from disk.
    Temperature is part of the key, so a sampled run never masquerades as a
    deterministic one.
    """
    payload = json.dumps(
        {
            "model": model,
            "messages": [m.to_json() for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop or [],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
