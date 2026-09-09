"""Host-side half of the tool bridge.

The sandbox runs with no network. Tools that genuinely need egress (SEC, web
search) are executed *here*, on the host, and the sandbox reaches them by
writing a request file into the one directory both sides share. That makes the
host broker a single chokepoint where every outbound call can be authorised,
rate-limited, traced and, if need be, refused.

Concurrency model
-----------------
The broker runs on a daemon thread for exactly as long as one sandbox execution
lasts (:meth:`ToolBridge.serving`). It polls rather than uses inotify/FSEvents
because the mount crosses a VM boundary under Docker Desktop, where filesystem
events are unreliable; a 20ms poll is cheap and portable.

Requests are written by the sandbox as ``<id>.tmp`` then renamed to
``<id>.req.json``, so the broker can never read a half-written request. The
broker replies the same way.

Safety properties
-----------------
* Only registered tool names are dispatchable. An unknown name is an error, not
  a passthrough.
* Handler exceptions are converted to an error payload; the sandbox raises
  ``ToolError`` and the convergence loop gets a real message to react to. A
  broker crash would hang the sandbox until its RPC timeout, so nothing here is
  allowed to propagate.
* Every call is appended to ``.rpc/audit.jsonl`` with arguments, duration and
  outcome — the audit trail for anything the run touched outside the container.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from teamclaw.observability.trace import SpanKind, Tracer

Handler = Callable[..., object]


@dataclass
class BridgeCall:
    call_id: str
    tool: str
    kwargs: dict[str, object]
    ok: bool
    duration_s: float
    error: str = ""

    def to_json(self) -> dict[str, object]:
        return {
            "id": self.call_id,
            "tool": self.tool,
            "kwargs": self.kwargs,
            "ok": self.ok,
            "duration_s": round(self.duration_s, 4),
            "error": self.error,
            "at": time.time(),
        }


@dataclass
class ToolBridge:
    rpc_dir: Path
    handlers: dict[str, Handler] = field(default_factory=dict)
    tracer: Tracer | None = None
    poll_s: float = 0.02
    max_calls: int = 500

    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    calls: list[BridgeCall] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.rpc_dir = Path(self.rpc_dir)
        self.rpc_dir.mkdir(parents=True, exist_ok=True)

    # -- registration ------------------------------------------------------
    def register(self, name: str, handler: Handler) -> "ToolBridge":
        self.handlers[name] = handler
        return self

    def register_registry(self, registry) -> "ToolBridge":
        """Register every bridged tool in a ToolRegistry that has a handler."""
        for spec in registry.tools.values():
            if spec.bridged and spec.handler is not None:
                self.register(spec.name, spec.handler)
        return self

    # -- lifecycle ---------------------------------------------------------
    @contextmanager
    def serving(self) -> Iterator["ToolBridge"]:
        self.start()
        try:
            yield self
        finally:
            self.stop()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="teamclaw-bridge", daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    # -- the loop ----------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._drain()
            except Exception:  # noqa: BLE001
                # The broker must never die: a dead broker hangs the sandbox
                # until its RPC timeout, turning a tool bug into a lost run.
                pass
            self._stop.wait(self.poll_s)
        # One final pass so a request written just before the sandbox exited
        # still gets answered.
        try:
            self._drain()
        except Exception:  # noqa: BLE001
            pass

    def _drain(self) -> None:
        for req in sorted(self.rpc_dir.glob("*.req.json")):
            if len(self.calls) >= self.max_calls:
                self._respond(req, error=f"bridge call cap ({self.max_calls}) exceeded")
                continue
            self._service(req)

    def _service(self, req: Path) -> None:
        try:
            payload = json.loads(req.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            self._respond(req, error=f"unreadable request: {exc}")
            return

        call_id = str(payload.get("id") or req.stem.split(".")[0])
        tool = str(payload.get("tool") or "")
        kwargs = payload.get("kwargs") or {}
        if not isinstance(kwargs, dict):
            self._respond(req, error="kwargs must be an object")
            return
        # Drop keys the caller left as None so handler defaults apply.
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        handler = self.handlers.get(tool)
        if handler is None:
            self._finish(req, call_id, tool, kwargs, None,
                         f"unknown tool {tool!r}; registered: {sorted(self.handlers)}", 0.0)
            return

        started = time.time()
        try:
            result = handler(**kwargs)
            error = ""
        except TypeError as exc:
            result, error = None, f"bad arguments: {exc}"
        except Exception as exc:  # noqa: BLE001 - surfaced to the agent verbatim
            result, error = None, f"{type(exc).__name__}: {exc}"
        self._finish(req, call_id, tool, kwargs, result, error, time.time() - started)

    def _finish(
        self, req: Path, call_id: str, tool: str, kwargs: dict[str, object],
        result: object, error: str, duration: float,
    ) -> None:
        record = BridgeCall(call_id, tool, kwargs, ok=not error, duration_s=duration, error=error)
        self.calls.append(record)
        self._audit(record)
        if self.tracer is not None:
            with self.tracer.span(SpanKind.TOOL_CALL, tool) as sp:
                sp.set(bridged=True, ok=record.ok, duration_s=round(duration, 4))
                if error:
                    sp.set(error_message=error)
        self._respond(req, result=result, error=error)

    def _respond(self, req: Path, *, result: object = None, error: str = "") -> None:
        base = req.name[: -len(".req.json")]
        res = self.rpc_dir / f"{base}.res.json"
        payload: dict[str, object] = {"error": error} if error else {"result": self._jsonable(result)}
        tmp = res.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
            os.replace(tmp, res)
            req.unlink(missing_ok=True)
        except OSError:
            tmp.unlink(missing_ok=True)

    @staticmethod
    def _jsonable(value: object) -> object:
        """Coerce handler output into something JSON can carry.

        Handlers legitimately return DataFrames and Paths; forcing every handler
        to serialise by hand would be noise, and letting json.dumps fail would
        surface as an opaque bridge error.
        """
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            to_dict = getattr(value, "to_dict", None)
            if callable(to_dict):
                try:
                    return to_dict(orient="records")  # pandas DataFrame
                except TypeError:
                    return to_dict()
            return str(value)

    # -- reporting ---------------------------------------------------------
    def _audit(self, record: BridgeCall) -> None:
        path = self.rpc_dir / "audit.jsonl"
        try:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.to_json(), ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass

    def stats(self) -> dict[str, object]:
        failures = [c for c in self.calls if not c.ok]
        by_tool: dict[str, int] = {}
        for c in self.calls:
            by_tool[c.tool] = by_tool.get(c.tool, 0) + 1
        return {
            "calls": len(self.calls),
            "failures": len(failures),
            "failure_rate": round(len(failures) / len(self.calls), 4) if self.calls else 0.0,
            "by_tool": by_tool,
            "total_duration_s": round(sum(c.duration_s for c in self.calls), 3),
        }
