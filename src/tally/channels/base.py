"""What a channel adapter has to provide, and the two things it must get right.

An inbound webhook is untrusted input arriving over the public internet, and the
two failures that actually happen in production are both structural rather than
clever:

**Unverified delivery.** Anyone who learns the URL can make an agent run. Every
adapter must implement :meth:`Channel.verify`, and the router refuses an adapter
that cannot verify — an unverified webhook on a server that executes code is the
same exposure as an open API with no token.

**Retried delivery.** Feishu, Slack and every other platform redeliver when a
200 is slow, and an agent run is slow by nature. Without idempotency the same
message starts two runs on the same workspace. The adapter supplies a stable
``event_id`` and the router does the deduplication once, in the session store,
so it works across workers.

Outbound is deliberately narrow: send a message back to a thread. Anything
richer belongs in the adapter, not in the platform's idea of a channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class InboundMessage:
    """A normalised inbound event.

    ``event_id`` is what makes retries safe, and ``thread_id`` is what maps a
    channel conversation onto a stored one. An adapter that cannot supply a
    stable event id should say so rather than inventing one per delivery.
    """

    channel: str
    event_id: str
    thread_id: str
    sender_id: str
    text: str
    tenant: str = "default"
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def usable(self) -> bool:
        return bool(self.text.strip()) and bool(self.thread_id)


@dataclass
class VerificationResult:
    ok: bool
    reason: str = ""
    # Some platforms send a one-off handshake that must be echoed rather than
    # treated as a message. When present the router returns it and stops.
    challenge_response: dict[str, Any] | None = None

    def __bool__(self) -> bool:
        return self.ok


class Channel(Protocol):
    name: str

    def configured(self) -> bool: ...

    def verify(self, *, headers: dict[str, str], body: bytes) -> VerificationResult: ...

    def parse(self, payload: dict[str, Any]) -> InboundMessage | None: ...

    def send(self, thread_id: str, text: str) -> dict[str, Any]: ...


class NotConfigured(RuntimeError):
    pass
