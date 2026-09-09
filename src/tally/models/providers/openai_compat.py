"""OpenAI-compatible chat-completions provider.

One class covers Zhipu GLM, Groq, Cerebras, SiliconFlow, DeepSeek and any
self-hosted vLLM endpoint, because they all speak /chat/completions. Keeping
them in one implementation means quota handling and error mapping are written
once rather than five times.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from tally.models.base import (
    Completion,
    Message,
    NotConfigured,
    PaidCallBlocked,
    ProviderUnavailable,
    RateLimited,
)


class OpenAICompatProvider:
    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 90.0,
        is_paid: bool = False,
        allow_paid: bool = False,
        max_retries: int = 2,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.is_paid = is_paid
        self.allow_paid = allow_paid
        self.max_retries = max_retries

    def available(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def generate(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> Completion:
        if not self.available():
            raise NotConfigured(f"{self.name}: missing api key / base url / model")
        if self.is_paid and not self.allow_paid:
            raise PaidCallBlocked(
                f"{self.name} is a paid provider; set TALLY_ALLOW_PAID=1 to enable it"
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_json() for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    resp = client.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                    )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = ProviderUnavailable(f"{self.name}: {type(exc).__name__}: {exc}")
                self._backoff(attempt)
                continue

            if resp.status_code == 429:
                raise RateLimited(
                    f"{self.name}: 429 rate limited", self._retry_after(resp)
                )
            if resp.status_code in {401, 403}:
                # Credentials are wrong; retrying cannot help and would burn quota.
                raise NotConfigured(f"{self.name}: auth rejected ({resp.status_code})")
            if resp.status_code >= 500:
                last_error = ProviderUnavailable(f"{self.name}: {resp.status_code}")
                self._backoff(attempt)
                continue
            if resp.status_code >= 400:
                raise ProviderUnavailable(
                    f"{self.name}: {resp.status_code} {resp.text[:300]}"
                )

            return self._parse(resp.json())

        raise last_error or ProviderUnavailable(f"{self.name}: exhausted retries")

    def _parse(self, data: dict[str, Any]) -> Completion:
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
            finish = choice.get("finish_reason") or "stop"
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderUnavailable(
                f"{self.name}: unexpected response shape ({exc})"
            ) from exc
        usage = data.get("usage") or {}
        return Completion(
            text=text,
            model=data.get("model", self.model),
            provider=self.name,
            tokens_in=int(usage.get("prompt_tokens", 0) or 0),
            tokens_out=int(usage.get("completion_tokens", 0) or 0),
            finish_reason=finish,
            raw=data,
        )

    @staticmethod
    def _retry_after(resp: httpx.Response) -> float | None:
        raw = resp.headers.get("retry-after")
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    def _backoff(self, attempt: int) -> None:
        if attempt < self.max_retries:
            time.sleep(min(2.0**attempt, 8.0))
