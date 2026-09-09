"""Session state, in Redis when it is there and in memory when it is not.

What belongs here and what does not
-----------------------------------
Session state is the small, hot, *shared* things: which run a channel thread is
currently waiting on, a per-tenant rate-limit counter, a pending HITL question, a
short-lived idempotency key for an inbound webhook. All of it is fine to lose.

An agent's working state is not here — that is the workspace on disk. Putting it
in Redis would give the same fact two homes and make resume a merge.

Why the fallback exists
-----------------------
The rest of this project degrades honestly rather than failing when an optional
dependency is absent: the sandbox falls back from a container to a subprocess,
retrieval falls back from dense to lexical, and both report which one they used.
Redis gets the same treatment, and :attr:`SessionStore.backend` says which is
live so a run's conditions record it.

The in-memory backend is single-process and therefore *wrong* for a deployment
with more than one worker — it will silently not share state. It says so, and
:meth:`SessionStore.warnings` surfaces it for the dashboard rather than leaving
it to be discovered under load.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol


class Backend(Protocol):
    name: str

    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str, ttl_s: int | None = None) -> None: ...
    def delete(self, key: str) -> None: ...
    def incr(self, key: str, ttl_s: int | None = None) -> int: ...
    def keys(self, prefix: str) -> list[str]: ...
    def ping(self) -> bool: ...


@dataclass
class _Entry:
    value: str
    expires_at: float | None = None

    def alive(self, now: float) -> bool:
        return self.expires_at is None or self.expires_at > now


class MemoryBackend:
    """Process-local dict with TTLs. Correct for one worker, not for several."""

    name = "memory"

    def __init__(self) -> None:
        self._data: dict[str, _Entry] = {}
        self._lock = threading.Lock()

    def _sweep(self, now: float) -> None:
        dead = [k for k, e in self._data.items() if not e.alive(now)]
        for key in dead:
            self._data.pop(key, None)

    def get(self, key: str) -> str | None:
        now = time.time()
        with self._lock:
            self._sweep(now)
            entry = self._data.get(key)
            return entry.value if entry and entry.alive(now) else None

    def set(self, key: str, value: str, ttl_s: int | None = None) -> None:
        with self._lock:
            self._data[key] = _Entry(value, time.time() + ttl_s if ttl_s else None)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def incr(self, key: str, ttl_s: int | None = None) -> int:
        with self._lock:
            now = time.time()
            entry = self._data.get(key)
            current = int(entry.value) if entry and entry.alive(now) else 0
            current += 1
            expires = entry.expires_at if entry and entry.alive(now) else (
                now + ttl_s if ttl_s else None
            )
            self._data[key] = _Entry(str(current), expires)
            return current

    def keys(self, prefix: str) -> list[str]:
        now = time.time()
        with self._lock:
            self._sweep(now)
            return sorted(k for k in self._data if k.startswith(prefix))

    def ping(self) -> bool:
        return True


class RedisBackend:
    name = "redis"

    def __init__(self, url: str) -> None:
        import redis  # noqa: PLC0415 - optional at import time

        self._client = redis.Redis.from_url(url, decode_responses=True,
                                            socket_connect_timeout=2.0,
                                            socket_timeout=2.0)

    def get(self, key: str) -> str | None:
        return self._client.get(key)

    def set(self, key: str, value: str, ttl_s: int | None = None) -> None:
        if ttl_s:
            self._client.setex(key, ttl_s, value)
        else:
            self._client.set(key, value)

    def delete(self, key: str) -> None:
        self._client.delete(key)

    def incr(self, key: str, ttl_s: int | None = None) -> int:
        pipe = self._client.pipeline()
        pipe.incr(key)
        if ttl_s:
            # NX so a refill does not extend a window that is already counting —
            # otherwise a steady stream of requests keeps resetting the TTL and
            # the rate limit never resets.
            pipe.expire(key, ttl_s, nx=True)
        result = pipe.execute()
        return int(result[0])

    def keys(self, prefix: str) -> list[str]:
        return sorted(self._client.scan_iter(match=f"{prefix}*", count=200))

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:  # noqa: BLE001
            return False


PREFIX = "teamclaw"


@dataclass
class SessionStore:
    backend: Backend
    degraded_reason: str = ""
    namespace: str = PREFIX

    # -- construction ------------------------------------------------------
    @classmethod
    def connect(cls, url: str | None) -> "SessionStore":
        """Redis if reachable, memory otherwise, and it says which and why."""
        if not url:
            return cls(MemoryBackend(),
                       degraded_reason="no TEAMCLAW_REDIS_URL configured")
        try:
            backend = RedisBackend(url)
            if backend.ping():
                return cls(backend)
            return cls(MemoryBackend(),
                       degraded_reason=f"redis at {url} did not answer PING")
        except Exception as exc:  # noqa: BLE001
            return cls(MemoryBackend(),
                       degraded_reason=f"redis unavailable ({type(exc).__name__})")

    def _key(self, *parts: str) -> str:
        return ":".join((self.namespace, *parts))

    # -- typed helpers -----------------------------------------------------
    def set_json(self, *parts: str, value: Any, ttl_s: int | None = None) -> None:
        self.backend.set(self._key(*parts), json.dumps(value, default=str), ttl_s)

    def get_json(self, *parts: str) -> Any:
        raw = self.backend.get(self._key(*parts))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def drop(self, *parts: str) -> None:
        self.backend.delete(self._key(*parts))

    # -- channel thread -> active run --------------------------------------
    ACTIVE_RUN_TTL_S = 3_600

    def set_active_run(self, conversation_id: str, run_id: str) -> None:
        self.set_json("active_run", conversation_id, value=run_id,
                      ttl_s=self.ACTIVE_RUN_TTL_S)

    def active_run(self, conversation_id: str) -> str | None:
        value = self.get_json("active_run", conversation_id)
        return str(value) if value else None

    def clear_active_run(self, conversation_id: str) -> None:
        self.drop("active_run", conversation_id)

    # -- inbound idempotency ------------------------------------------------
    DEDUPE_TTL_S = 600

    def seen_before(self, channel: str, event_id: str) -> bool:
        """True if this inbound event was already handled.

        Channels retry aggressively — Feishu and Slack both redeliver on a slow
        200 — and without this a retried message starts a second agent run on
        the same request. First caller wins by atomic increment, so two workers
        racing the same delivery cannot both proceed.
        """
        if not event_id:
            return False
        count = self.backend.incr(self._key("seen", channel, event_id),
                                  ttl_s=self.DEDUPE_TTL_S)
        return count > 1

    # -- rate limiting ------------------------------------------------------
    def rate_limit(self, scope: str, *, limit: int, window_s: int) -> tuple[bool, int]:
        """Fixed-window counter. Returns (allowed, current count)."""
        window = int(time.time() // window_s)
        count = self.backend.incr(self._key("rate", scope, str(window)),
                                  ttl_s=window_s + 5)
        return count <= limit, count

    # -- pending human questions -------------------------------------------
    HITL_TTL_S = 86_400

    def push_hitl(self, run_id: str, payload: dict[str, Any]) -> None:
        self.set_json("hitl", run_id, value=payload, ttl_s=self.HITL_TTL_S)

    def pending_hitl(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for key in self.backend.keys(self._key("hitl", "")):
            raw = self.backend.get(key)
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            payload["run_id"] = key.rsplit(":", 1)[-1]
            out.append(payload)
        return out

    def resolve_hitl(self, run_id: str, decision: str, message: str = "") -> None:
        self.set_json("hitl_answer", run_id,
                      value={"decision": decision, "message": message},
                      ttl_s=self.HITL_TTL_S)
        self.drop("hitl", run_id)

    def hitl_answer(self, run_id: str) -> dict[str, Any] | None:
        return self.get_json("hitl_answer", run_id)

    # -- reporting ---------------------------------------------------------
    def warnings(self) -> list[str]:
        if self.backend.name != "memory":
            return []
        return [
            "Session state is in process memory"
            + (f" ({self.degraded_reason})" if self.degraded_reason else "")
            + ". This is correct for a single worker and silently wrong for more "
              "than one: state will not be shared. Set TEAMCLAW_REDIS_URL before "
              "running multiple workers."
        ]

    def status(self) -> dict[str, Any]:
        return {
            "backend": self.backend.name,
            "healthy": self.backend.ping(),
            "degraded_reason": self.degraded_reason,
            "warnings": self.warnings(),
        }
