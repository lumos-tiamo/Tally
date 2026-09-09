"""Local inference via Ollama.

On a 16GB M1 Pro the practical ceiling is an 8B model at 4-bit (~5GB resident).
That is enough for extraction / classification / rewriting, and deliberately not
enough for planning — the router reflects that split rather than pretending a
local 8B can drive the agent loop.
"""

from __future__ import annotations

from typing import Any

import httpx

from teamclaw.models.base import (
    Completion,
    Message,
    NotConfigured,
    ProviderUnavailable,
    Role,
)


class OllamaProvider:
    is_paid = False

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:8b",
        name: str = "ollama",
        timeout_s: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = name
        self.timeout_s = timeout_s
        self._available: bool | None = None

    def available(self) -> bool:
        """Probe once and memoise; a dead daemon should not cost a probe per call."""
        if self._available is not None:
            return self._available
        try:
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(f"{self.base_url}/api/tags")
            self._available = resp.status_code == 200
        except (httpx.TimeoutException, httpx.TransportError):
            self._available = False
        return self._available

    def generate(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> Completion:
        if not self.available():
            raise NotConfigured(f"ollama: daemon not reachable at {self.base_url}")

        options: dict[str, Any] = {"temperature": temperature, "num_predict": max_tokens}
        if stop:
            options["stop"] = stop
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "assistant" if m.role is Role.ASSISTANT else m.role.value,
                    "content": m.content,
                }
                for m in messages
            ],
            "stream": False,
            "options": options,
        }
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                resp = client.post(f"{self.base_url}/api/chat", json=payload)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ProviderUnavailable(f"ollama: {type(exc).__name__}: {exc}") from exc

        if resp.status_code == 404:
            raise NotConfigured(
                f"ollama: model {self.model!r} not pulled; run `ollama pull {self.model}`"
            )
        if resp.status_code >= 400:
            raise ProviderUnavailable(f"ollama: {resp.status_code} {resp.text[:300]}")

        data = resp.json()
        return Completion(
            text=(data.get("message") or {}).get("content", ""),
            model=data.get("model", self.model),
            provider=self.name,
            tokens_in=int(data.get("prompt_eval_count", 0) or 0),
            tokens_out=int(data.get("eval_count", 0) or 0),
            finish_reason=data.get("done_reason", "stop"),
            raw=data,
        )
