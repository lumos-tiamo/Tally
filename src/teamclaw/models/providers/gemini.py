"""Google AI Studio (Gemini) provider.

Gemini's generateContent API differs from OpenAI's in three ways that matter
here: the system prompt is a separate ``system_instruction`` field, roles are
``user``/``model``, and quota rejections arrive as 429 with a RESOURCE_EXHAUSTED
status body. All three are handled below.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from teamclaw.models.base import (
    Completion,
    Message,
    NotConfigured,
    ProviderUnavailable,
    RateLimited,
    Role,
)

DEFAULT_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider:
    is_paid = False

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-2.5-flash",
        name: str = "gemini",
        base_url: str = DEFAULT_BASE,
        timeout_s: float = 90.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def available(self) -> bool:
        return bool(self.api_key)

    def generate(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> Completion:
        if not self.available():
            raise NotConfigured("gemini: missing TEAMCLAW_GEMINI_API_KEY")

        system_parts = [m.content for m in messages if m.role is Role.SYSTEM]
        contents = [
            {
                "role": "model" if m.role is Role.ASSISTANT else "user",
                "parts": [{"text": m.content}],
            }
            for m in messages
            if m.role is not Role.SYSTEM
        ]
        payload: dict[str, Any] = {
            "contents": contents or [{"role": "user", "parts": [{"text": ""}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_parts:
            payload["system_instruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        if stop:
            payload["generationConfig"]["stopSequences"] = stop

        url = f"{self.base_url}/models/{self.model}:generateContent"
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    resp = client.post(
                        url, json=payload, headers={"x-goog-api-key": self.api_key}
                    )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = ProviderUnavailable(f"gemini: {type(exc).__name__}: {exc}")
                if attempt < self.max_retries:
                    time.sleep(min(2.0**attempt, 8.0))
                continue

            if resp.status_code == 429:
                raise RateLimited("gemini: 429 RESOURCE_EXHAUSTED", self._retry_after(resp))
            if resp.status_code in {401, 403}:
                raise NotConfigured(f"gemini: auth rejected ({resp.status_code})")
            if resp.status_code >= 500:
                last_error = ProviderUnavailable(f"gemini: {resp.status_code}")
                if attempt < self.max_retries:
                    time.sleep(min(2.0**attempt, 8.0))
                continue
            if resp.status_code >= 400:
                raise ProviderUnavailable(f"gemini: {resp.status_code} {resp.text[:300]}")

            return self._parse(resp.json())

        raise last_error or ProviderUnavailable("gemini: exhausted retries")

    def _parse(self, data: dict[str, Any]) -> Completion:
        candidates = data.get("candidates") or []
        if not candidates:
            # A safety block returns no candidates; surface it rather than返回空串.
            feedback = data.get("promptFeedback", {})
            raise ProviderUnavailable(f"gemini: no candidates, feedback={feedback}")
        parts = candidates[0].get("content", {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)
        usage = data.get("usageMetadata") or {}
        return Completion(
            text=text,
            model=data.get("modelVersion", self.model),
            provider=self.name,
            tokens_in=int(usage.get("promptTokenCount", 0) or 0),
            tokens_out=int(usage.get("candidatesTokenCount", 0) or 0),
            finish_reason=candidates[0].get("finishReason", "STOP"),
            raw=data,
        )

    @staticmethod
    def _retry_after(resp: httpx.Response) -> float | None:
        raw = resp.headers.get("retry-after")
        try:
            return float(raw) if raw else None
        except ValueError:
            return None
