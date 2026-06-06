"""Configuration for the OpsRAG Slack chatbot.

Wired into the top-level OpsRAGConfig as `slack_bot: SlackBotConfig`. The
bot only starts when `enabled=True` AND the process role is `slack-bot`
(see SESSION-SLACK-BOT-PLAN.md section 2.1 for the lifespan wiring).
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

# Slack channel IDs always start with C (public) / G (private group) /
# D (DM). We restrict the allowlist to channel IDs (C...) only -- DMs do
# not need an allowlist, and private groups would also start with G but
# the v1 design is "public channels we explicitly invite the bot to".
_CHANNEL_ID_RE = re.compile(r"^C[A-Z0-9]{6,}$")


class SlackBotConfig(BaseModel):
    """Pydantic v2 config for the Slack chatbot subsystem."""

    enabled: bool = False
    bot_token_env: str = "OPSRAG_SLACK_BOT_TOKEN"
    app_token_env: str = "OPSRAG_SLACK_APP_TOKEN"
    channels_allowlist: list[str] = Field(default_factory=list)
    thread_context_message_cap: int = 20
    streaming_enabled: bool = True
    streaming_min_update_interval_s: float = 1.5
    per_user_daily_quota: int = 200
    # Per Constitution Principle VI, no example default URL; operators
    # set this to their workspace URL or leave it None.
    workspace_url: str | None = None
    web_ui_base_url: str = ""

    @field_validator("channels_allowlist")
    @classmethod
    def _validate_channel_ids(cls, v: list[str]) -> list[str]:
        for ch in v:
            if not isinstance(ch, str) or not _CHANNEL_ID_RE.match(ch):
                raise ValueError(
                    f"channels_allowlist entry {ch!r} is not a valid Slack channel ID "
                    "(expected 'C' followed by 6+ uppercase alphanumerics, e.g. 'C0ABCDEFG')."
                )
        return v

    @field_validator("streaming_min_update_interval_s")
    @classmethod
    def _validate_min_interval(cls, v: float) -> float:
        if v < 0:
            raise ValueError(
                "streaming_min_update_interval_s must be non-negative "
                f"(got {v}); Slack rate-limit is per minute, not per call."
            )
        return v

    @field_validator("thread_context_message_cap")
    @classmethod
    def _validate_thread_cap(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"thread_context_message_cap must be >= 0 (got {v}).")
        return v

    @field_validator("per_user_daily_quota")
    @classmethod
    def _validate_quota(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"per_user_daily_quota must be >= 0 (got {v}).")
        return v
