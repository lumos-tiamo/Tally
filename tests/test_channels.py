"""Channel adapters: verification, the handshake, and replay.

An inbound webhook is untrusted input on a server that executes code, so these
tests are mostly about what must be *refused*. Feishu's protocol has three
details that are easy to get wrong and each has a test named after it.
"""

from __future__ import annotations

import hashlib
import json
import time

from teamclaw.channels import FeishuChannel, build_channels
from teamclaw.config import Settings

ENCRYPT_KEY = "test-encrypt-key"
VERIFICATION_TOKEN = "test-verification-token"


def signed(body: dict, *, key: str = ENCRYPT_KEY, timestamp: str | None = None,
           nonce: str = "n1") -> tuple[dict[str, str], bytes]:
    raw = json.dumps(body).encode("utf-8")
    stamp = timestamp or str(int(time.time()))
    digest = hashlib.sha256((stamp + nonce + key).encode("utf-8") + raw).hexdigest()
    return {
        "X-Lark-Signature": digest,
        "X-Lark-Request-Timestamp": stamp,
        "X-Lark-Request-Nonce": nonce,
    }, raw


def message_event(text: str = "hello", *, event_id: str = "evt_1") -> dict:
    return {
        "header": {"event_id": event_id, "event_type": "im.message.receive_v1",
                   "token": VERIFICATION_TOKEN, "tenant_key": "acme"},
        "event": {
            "sender": {"sender_id": {"open_id": "ou_1"}},
            "message": {"chat_id": "chat_1", "message_id": "om_1",
                        "content": json.dumps({"text": text})},
        },
    }


# --- registration ---------------------------------------------------------
def test_an_unconfigured_channel_is_absent_rather_than_broken():
    """A webhook to a channel nobody set up should 404, not a confusing 401."""
    assert build_channels(Settings()) == {}


def test_a_channel_with_only_a_verification_token_is_registered():
    """Receiving without being able to reply is a valid read-only integration."""
    cfg = Settings(feishu_verification_token=VERIFICATION_TOKEN)
    assert "feishu" in build_channels(cfg)


# --- verification ---------------------------------------------------------
def test_a_valid_signature_is_accepted():
    channel = FeishuChannel(encrypt_key=ENCRYPT_KEY)
    headers, body = signed(message_event())
    assert channel.verify(headers=headers, body=body)


def test_a_tampered_body_is_rejected():
    channel = FeishuChannel(encrypt_key=ENCRYPT_KEY)
    headers, _ = signed(message_event())
    assert not channel.verify(headers=headers, body=b'{"event":"tampered"}')


def test_an_old_but_correctly_signed_request_is_a_replay():
    """The signature covers the timestamp; without a skew check it is valid forever."""
    channel = FeishuChannel(encrypt_key=ENCRYPT_KEY)
    stale = str(int(time.time()) - 4000)
    headers, body = signed(message_event(), timestamp=stale)
    verdict = channel.verify(headers=headers, body=body)
    assert not verdict and "old" in verdict.reason


def test_missing_signature_headers_are_rejected():
    channel = FeishuChannel(encrypt_key=ENCRYPT_KEY)
    _, body = signed(message_event())
    assert not channel.verify(headers={}, body=body)


def test_an_unparseable_timestamp_is_rejected():
    channel = FeishuChannel(encrypt_key=ENCRYPT_KEY)
    _, body = signed(message_event())
    verdict = channel.verify(
        headers={"X-Lark-Signature": "x", "X-Lark-Request-Timestamp": "not-a-number"},
        body=body,
    )
    assert not verdict and "timestamp" in verdict.reason


def test_a_channel_with_no_secret_at_all_refuses_everything():
    """An unverifiable webhook on a code-executing server is not acceptable."""
    channel = FeishuChannel()
    verdict = channel.verify(headers={}, body=json.dumps(message_event()).encode())
    assert not verdict
    assert "unverifiable" in verdict.reason


def test_token_mode_checks_the_token():
    channel = FeishuChannel(verification_token=VERIFICATION_TOKEN)
    body = json.dumps(message_event()).encode()
    assert channel.verify(headers={}, body=body)

    wrong = message_event()
    wrong["header"]["token"] = "not-the-token"
    assert not channel.verify(headers={}, body=json.dumps(wrong).encode())


def test_a_non_json_body_is_rejected():
    channel = FeishuChannel(encrypt_key=ENCRYPT_KEY)
    verdict = channel.verify(headers={}, body=b"\xff\xfe not json")
    assert not verdict


# --- the handshake --------------------------------------------------------
def test_the_url_verification_handshake_is_answered_not_treated_as_a_message():
    """Same endpoint as messages. Mishandling it means the subscription never activates."""
    channel = FeishuChannel(verification_token=VERIFICATION_TOKEN)
    body = json.dumps({"type": "url_verification", "token": VERIFICATION_TOKEN,
                       "challenge": "abc123"}).encode()
    verdict = channel.verify(headers={}, body=body)
    assert verdict
    assert verdict.challenge_response == {"challenge": "abc123"}


def test_a_handshake_with_the_wrong_token_is_refused():
    channel = FeishuChannel(verification_token=VERIFICATION_TOKEN)
    body = json.dumps({"type": "url_verification", "token": "wrong",
                       "challenge": "abc"}).encode()
    assert not channel.verify(headers={}, body=body)


# --- parsing --------------------------------------------------------------
def test_message_content_is_json_inside_a_json_string():
    """Reading it raw hands the agent '{"text":"hello"}' as the user's message."""
    channel = FeishuChannel(encrypt_key=ENCRYPT_KEY)
    parsed = channel.parse(message_event("compute the gross margin"))
    assert parsed is not None
    assert parsed.text == "compute the gross margin"
    assert parsed.thread_id == "chat_1"
    assert parsed.tenant == "acme"
    assert parsed.usable


def test_inline_mention_placeholders_are_stripped():
    """Otherwise the objective starts with a token the agent cannot interpret."""
    channel = FeishuChannel()
    payload = message_event()
    payload["event"]["message"]["content"] = json.dumps({"text": "@_user_1 do the thing"})
    payload["event"]["message"]["mentions"] = [{"key": "@_user_1"}]
    parsed = channel.parse(payload)
    assert parsed.text == "do the thing"


def test_an_event_type_we_do_not_act_on_is_ignored():
    channel = FeishuChannel()
    payload = message_event()
    payload["header"]["event_type"] = "im.chat.member.user.added_v1"
    assert channel.parse(payload) is None


def test_an_empty_message_is_parsed_but_not_usable():
    channel = FeishuChannel()
    payload = message_event("   ")
    parsed = channel.parse(payload)
    assert parsed is not None and not parsed.usable


def test_malformed_content_falls_back_to_the_raw_string():
    channel = FeishuChannel()
    payload = message_event()
    payload["event"]["message"]["content"] = "not json at all"
    assert channel.parse(payload).text == "not json at all"


def test_sending_without_credentials_raises_rather_than_failing_silently():
    from teamclaw.channels import NotConfigured
    import pytest

    channel = FeishuChannel(verification_token=VERIFICATION_TOKEN)
    with pytest.raises(NotConfigured):
        channel.send("chat_1", "hello")
