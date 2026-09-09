"""Inbound channel adapters. Verification and idempotency are not optional."""

from tally.channels.base import (
    Channel,
    InboundMessage,
    NotConfigured,
    VerificationResult,
)
from tally.channels.feishu import FeishuChannel


def build_channels(cfg) -> dict[str, Channel]:  # noqa: ANN001
    """Only configured adapters are registered.

    An unconfigured adapter is absent rather than present-and-broken, so a
    webhook to a channel nobody set up gets a 404 instead of a confusing 401.
    """
    channels: dict[str, Channel] = {}
    feishu = FeishuChannel(
        app_id=cfg.feishu_app_id, app_secret=cfg.feishu_app_secret,
        verification_token=cfg.feishu_verification_token,
        encrypt_key=cfg.feishu_encrypt_key,
    )
    if feishu.configured():
        channels[feishu.name] = feishu
    return channels


__all__ = ["Channel", "InboundMessage", "NotConfigured", "VerificationResult",
           "FeishuChannel", "build_channels"]
