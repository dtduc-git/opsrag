"""Unified per-channel config (pydantic v2).

The ``channels:`` block on ``OpsRAGConfig`` (wired in P1, not here). Each
sub-config:
  * carries ``enabled`` + ``allowlist`` + ``per_user_daily_quota`` +
    ``web_ui_base_url``.
  * references secrets ONLY via ``*_env`` field names (Constitution
    Principle VI -- never an inline token).
  * validates its allowlist loosely (channel-id strings); Slack keeps the
    strict ``C...`` validator it already had.

See design doc ``specs/002-channel-bots/design.md`` section 5.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

# Slack channel IDs start with C (public) / G (private group) / D (DM). We
# restrict the Slack allowlist to public channel IDs (C...) -- DMs bypass
# the allowlist and the v1 design is "public channels we explicitly invite
# the bot to".
_SLACK_CHANNEL_ID_RE = re.compile(r"^C[A-Z0-9]{6,}$")

# Slack subteam (user-group) id: "S" + uppercase alphanumerics. Config also
# accepts the fully-formed "<!subteam^S...>" mention token.
_SLACK_SUBTEAM_ID_RE = re.compile(r"^S[A-Z0-9]{6,}$")
_SLACK_SUBTEAM_MENTION_RE = re.compile(r"^<!subteam\^S[A-Z0-9]{6,}>$")
# Slack app id: "A" + uppercase alphanumerics. bot ids ("B...") are also
# permitted in request_app_allowlist since classify() matches either.
_SLACK_APP_OR_BOT_ID_RE = re.compile(r"^[AB][A-Z0-9]{6,}$")


def _validate_quota(v: int) -> int:
    if v < 0:
        raise ValueError(f"per_user_daily_quota must be >= 0 (got {v}).")
    return v


def _validate_loose_allowlist(v: list[str]) -> list[str]:
    """Loose allowlist validation -- non-empty channel-id strings.

    Telegram chat ids may be negative integers (groups), Discord/Teams
    are opaque snowflakes/conversation ids -- so we only require that each
    entry is a non-empty string.
    """
    for ch in v:
        if not isinstance(ch, str) or not ch.strip():
            raise ValueError(f"allowlist entry {ch!r} must be a non-empty string.")
    return v


class _BaseChannelConfig(BaseModel):
    """Common fields every channel shares."""

    enabled: bool = False
    allowlist: list[str] = Field(default_factory=list)
    # Per-user DM allowlist (platform user ids). DENY-BY-DEFAULT: empty => no
    # one may DM the bot (DMs aren't covered by ``allowlist``). List ids to
    # allow them, or ["*"] to allow anyone. Unauthorized DMs are denied silently.
    dm_allowlist: list[str] = Field(default_factory=list)
    per_user_daily_quota: int = 200
    web_ui_base_url: str = ""
    # How many prior thread messages to pull for context (mentions in a
    # thread only). Matches the legacy SlackBotConfig default.
    thread_context_message_cap: int = 20

    @field_validator("per_user_daily_quota")
    @classmethod
    def _check_quota(cls, v: int) -> int:
        return _validate_quota(v)

    @field_validator("thread_context_message_cap")
    @classmethod
    def _check_thread_cap(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"thread_context_message_cap must be >= 0 (got {v}).")
        return v


class FirstResponderChannelConfig(BaseModel):
    """First-responder rules for ONE channel.

    ``request_app_allowlist`` lists the app_id (A...) or bot_id (B...) of the
    request apps whose posts we auto-answer (e.g. the SRE-Support Slack
    Workflow). ``oncall_handle`` is the subteam always cc'd on replies.
    ``include_direct`` also answers plain human posts in the channel.
    ``daily_quota`` bounds each principal (the workflow source, or a human)
    against the shared channel quota bucket.
    """

    request_app_allowlist: list[str] = Field(default_factory=list)
    oncall_handle: str = ""
    include_direct: bool = True
    reply_in_thread: bool = True
    daily_quota: int = 500
    # Display name the ack greeting introduces ("I am *OpsRAG*"). Empty -> default.
    agent_name: str = "OpsRAG"
    # Readable label for the on-call subteam used ONLY to clean tokens out of
    # the ingested query text (never pings). Distinct from ``oncall_handle``,
    # which is the S…/<!subteam^S…> PING that drives the cc block. Empty ->
    # the normalizer falls back to "on-call".
    oncall_display: str = ""

    @field_validator("request_app_allowlist")
    @classmethod
    def _check_app_ids(cls, v: list[str]) -> list[str]:
        for a in v:
            if not isinstance(a, str) or not _SLACK_APP_OR_BOT_ID_RE.match(a):
                raise ValueError(
                    f"request_app_allowlist entry {a!r} is not a valid Slack "
                    "app/bot id (expected 'A' or 'B' + 6+ uppercase alphanumerics)."
                )
        return v

    @field_validator("oncall_handle")
    @classmethod
    def _check_oncall(cls, v: str) -> str:
        if v and not (
            _SLACK_SUBTEAM_ID_RE.match(v) or _SLACK_SUBTEAM_MENTION_RE.match(v)
        ):
            raise ValueError(
                f"oncall_handle {v!r} must be a subteam id ('S...') or a "
                "'<!subteam^S...>' mention token."
            )
        return v

    @field_validator("daily_quota")
    @classmethod
    def _check_quota(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"daily_quota must be >= 0 (got {v}).")
        return v

    @field_validator("agent_name")
    @classmethod
    def _check_agent_name(cls, v: str) -> str:
        return (v or "").strip() or "OpsRAG"

    @field_validator("oncall_display")
    @classmethod
    def _check_oncall_display(cls, v: str) -> str:
        return (v or "").strip()  # empty allowed; use-site falls back


class FirstResponderConfig(BaseModel):
    """Feature block: enabled flag + per-channel mappings."""

    enabled: bool = False
    channels: dict[str, FirstResponderChannelConfig] = Field(default_factory=dict)

    @field_validator("channels")
    @classmethod
    def _check_channel_keys(
        cls, v: dict[str, FirstResponderChannelConfig]
    ) -> dict[str, FirstResponderChannelConfig]:
        for ch in v:
            if not _SLACK_CHANNEL_ID_RE.match(ch):
                raise ValueError(
                    f"first_responder channel key {ch!r} is not a valid Slack "
                    "channel id (expected 'C' + 6+ uppercase alphanumerics)."
                )
        return v


class SlackChannelConfig(_BaseChannelConfig):
    """Slack adapter config (Socket Mode)."""

    bot_token_env: str = "OPSRAG_SLACK_BOT_TOKEN"
    app_token_env: str = "OPSRAG_SLACK_APP_TOKEN"
    streaming_enabled: bool = True
    workspace_url: str | None = None
    first_responder: FirstResponderConfig = Field(default_factory=FirstResponderConfig)

    @field_validator("allowlist")
    @classmethod
    def _check_slack_ids(cls, v: list[str]) -> list[str]:
        for ch in v:
            if not isinstance(ch, str) or not _SLACK_CHANNEL_ID_RE.match(ch):
                raise ValueError(
                    f"allowlist entry {ch!r} is not a valid Slack channel ID "
                    "(expected 'C' + 6+ uppercase alphanumerics, e.g. 'C0ABCDEFG')."
                )
        return v


class TelegramChannelConfig(_BaseChannelConfig):
    """Telegram adapter config (Bot API long-poll over httpx)."""

    bot_token_env: str = "OPSRAG_TELEGRAM_BOT_TOKEN"

    @field_validator("allowlist")
    @classmethod
    def _check_ids(cls, v: list[str]) -> list[str]:
        return _validate_loose_allowlist(v)


class DiscordChannelConfig(_BaseChannelConfig):
    """Discord adapter config (gateway via discord.py)."""

    bot_token_env: str = "OPSRAG_DISCORD_BOT_TOKEN"

    @field_validator("allowlist")
    @classmethod
    def _check_ids(cls, v: list[str]) -> list[str]:
        return _validate_loose_allowlist(v)


class TeamsChannelConfig(_BaseChannelConfig):
    """Teams adapter config (Bot Framework webhook on the API role)."""

    app_id_env: str = "OPSRAG_TEAMS_APP_ID"
    app_password_env: str = "OPSRAG_TEAMS_APP_PASSWORD"
    # Bot identity model. Microsoft deprecated *multi-tenant* bot creation
    # (2025), so new Azure Bots are SingleTenant (or UserAssignedMSI). For a
    # SingleTenant bot the SDK needs MicrosoftAppType=SingleTenant + the home
    # tenant id, else outbound token acquisition + inbound validation fail.
    # Defaults keep the legacy MultiTenant behaviour for existing deployments.
    app_type_env: str = "OPSRAG_TEAMS_APP_TYPE"          # MultiTenant | SingleTenant | UserAssignedMSI
    app_tenant_id_env: str = "OPSRAG_TEAMS_APP_TENANT_ID"  # required when app_type=SingleTenant

    @field_validator("allowlist")
    @classmethod
    def _check_ids(cls, v: list[str]) -> list[str]:
        return _validate_loose_allowlist(v)


class ChannelsConfig(BaseModel):
    """The unified ``channels:`` block.

    Not wired into ``OpsRAGConfig`` yet -- that is P1.
    """

    slack: SlackChannelConfig = Field(default_factory=SlackChannelConfig)
    telegram: TelegramChannelConfig = Field(default_factory=TelegramChannelConfig)
    discord: DiscordChannelConfig = Field(default_factory=DiscordChannelConfig)
    teams: TeamsChannelConfig = Field(default_factory=TeamsChannelConfig)
