"""Feishu / Lark adapter.

Three details in Feishu's protocol are where an implementation goes wrong, and
each is handled explicitly below.

**The URL-verification handshake arrives on the same endpoint as messages.**
A first-time subscription posts ``{"type": "url_verification", "challenge": ...}``
and expects the challenge echoed. Treating it as a message means the
subscription never activates; treating messages as handshakes means nothing
works. It is detected before verification, because the handshake carries the
verification token rather than a signature.

**Signature verification covers the timestamp and the nonce, not just the body.**
``sha256(timestamp + nonce + encrypt_key + body)``. Omitting the timestamp makes
a captured request replayable forever, so the timestamp is also checked against a
skew window — a signature that is valid but three days old is a replay.

**Message content is JSON inside a JSON string.** ``event.message.content`` is a
serialised object, not text, so a naive read yields ``'{"text":"hi"}'`` as the
user's message.

Comparisons use :func:`hmac.compare_digest` throughout: a signature check that
short-circuits on the first wrong byte leaks its own answer through timing.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any

import httpx

from tally.channels.base import InboundMessage, NotConfigured, VerificationResult

TOKEN_URL = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
REPLY_URL = "https://open.feishu.cn/open-apis/im/v1/messages"
MAX_SKEW_S = 300


@dataclass
class FeishuChannel:
    name: str = "feishu"
    app_id: str = ""
    app_secret: str = ""
    verification_token: str = ""
    encrypt_key: str = ""
    timeout_s: float = 15.0
    _token: str = ""
    _token_expires_at: float = 0.0

    def configured(self) -> bool:
        # A verification token alone is enough to *receive* safely; sending needs
        # the app credentials. Receiving without being able to reply is a valid
        # configuration for a read-only integration, so it is not refused here.
        return bool(self.verification_token or self.encrypt_key)

    # -- inbound -----------------------------------------------------------
    def verify(self, *, headers: dict[str, str], body: bytes) -> VerificationResult:
        lower = {k.lower(): v for k, v in headers.items()}

        try:
            payload = json.loads(body.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            return VerificationResult(False, "body is not JSON")

        # The subscription handshake, which must be answered before any message
        # ever arrives. It authenticates with the verification token, not a
        # signature, so it is handled first.
        if payload.get("type") == "url_verification":
            token = str(payload.get("token", ""))
            if not self.verification_token:
                return VerificationResult(
                    False, "url_verification received but no verification token is set"
                )
            if not hmac.compare_digest(token, self.verification_token):
                return VerificationResult(False, "handshake token mismatch")
            return VerificationResult(
                True, "url_verification",
                challenge_response={"challenge": payload.get("challenge", "")},
            )

        if self.encrypt_key:
            signature = lower.get("x-lark-signature", "")
            timestamp = lower.get("x-lark-request-timestamp", "")
            nonce = lower.get("x-lark-request-nonce", "")
            if not signature or not timestamp:
                return VerificationResult(False, "missing signature headers")

            try:
                age = abs(time.time() - float(timestamp))
            except ValueError:
                return VerificationResult(False, "unparseable request timestamp")
            if age > MAX_SKEW_S:
                # A valid signature on an old request is a replay, not a message.
                return VerificationResult(
                    False, f"request timestamp is {age:.0f}s old (max {MAX_SKEW_S}s)"
                )

            digest = hashlib.sha256(
                (timestamp + nonce + self.encrypt_key).encode("utf-8") + body
            ).hexdigest()
            if not hmac.compare_digest(digest, signature):
                return VerificationResult(False, "signature mismatch")
            return VerificationResult(True)

        token = str((payload.get("header") or {}).get("token")
                    or payload.get("token") or "")
        if not self.verification_token:
            return VerificationResult(
                False,
                "no encrypt key and no verification token configured; refusing an "
                "unverifiable webhook on a server that executes code",
            )
        if not hmac.compare_digest(token, self.verification_token):
            return VerificationResult(False, "verification token mismatch")
        return VerificationResult(True)

    def parse(self, payload: dict[str, Any]) -> InboundMessage | None:
        header = payload.get("header") or {}
        event = payload.get("event") or {}
        message = event.get("message") or {}
        sender = (event.get("sender") or {}).get("sender_id") or {}

        if header.get("event_type") not in {"im.message.receive_v1", None}:
            return None

        # content is a JSON string, not text. Reading it raw hands the agent
        # '{"text":"hello"}' as the user's message.
        text = ""
        raw_content = message.get("content")
        if isinstance(raw_content, str):
            try:
                text = str((json.loads(raw_content) or {}).get("text", ""))
            except json.JSONDecodeError:
                text = raw_content
        elif isinstance(raw_content, dict):
            text = str(raw_content.get("text", ""))

        # Mentions arrive inline as @_user_1 placeholders; strip them so the
        # objective does not begin with a token the agent cannot interpret.
        for mention in message.get("mentions") or []:
            key = str(mention.get("key", ""))
            if key:
                text = text.replace(key, "")

        thread_id = str(message.get("chat_id") or message.get("root_id") or "")
        event_id = str(header.get("event_id") or message.get("message_id") or "")
        return InboundMessage(
            channel=self.name, event_id=event_id, thread_id=thread_id,
            sender_id=str(sender.get("open_id") or sender.get("user_id") or ""),
            text=text.strip(),
            tenant=str(header.get("tenant_key") or "default"),
            raw=payload,
        )

    # -- outbound ----------------------------------------------------------
    def _access_token(self) -> str:
        if not (self.app_id and self.app_secret):
            raise NotConfigured("feishu app_id/app_secret are required to send")
        if self._token and time.time() < self._token_expires_at - 60:
            return self._token
        with httpx.Client(timeout=self.timeout_s) as client:
            response = client.post(TOKEN_URL, json={"app_id": self.app_id,
                                                    "app_secret": self.app_secret})
        response.raise_for_status()
        data = response.json()
        if data.get("code") != 0:
            raise NotConfigured(f"feishu token request failed: {data.get('msg')}")
        self._token = str(data.get("tenant_access_token", ""))
        self._token_expires_at = time.time() + float(data.get("expire", 1800))
        return self._token

    def send(self, thread_id: str, text: str) -> dict[str, Any]:
        token = self._access_token()
        with httpx.Client(timeout=self.timeout_s) as client:
            response = client.post(
                REPLY_URL,
                params={"receive_id_type": "chat_id"},
                headers={"Authorization": f"Bearer {token}"},
                json={"receive_id": thread_id, "msg_type": "text",
                      "content": json.dumps({"text": text})},
            )
        return {"status": response.status_code, "body": response.json()
                if response.headers.get("content-type", "").startswith("application/json")
                else response.text[:400]}
