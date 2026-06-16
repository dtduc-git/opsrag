"""``SlackAdapter`` -- the Slack implementation of the ``ChannelAdapter`` port.

This adapter is a *thin wrapper* over the EXISTING Slack leaf modules; those
modules are kept byte-for-byte:

  * ``opsrag.slack_bot.client.SlackBotClient`` -- Socket Mode transport
    (``post_message`` / ``update_message`` / ``add_reaction`` /
    ``fetch_thread_replies`` / ``get_user_info``).
  * ``opsrag.slack_bot.render.format_answer_as_slack_blocks`` -- Block Kit
    rendering (answer + sources + 👍/👎 + footer).
  * ``opsrag.slack_bot.thread_context`` -- the self-filter + display-name
    helpers reused when mapping raw thread replies to ``ThreadMessage``.
  * ``opsrag.slack_bot.identity.slack_user_to_current_user`` -- the synthetic
    ``slack-bot:<workspace>:<user>`` identity.

Port mapping (design §4.1):

  ============================  ===================================================
  port method                   Slack leaf call
  ============================  ===================================================
  ``post_placeholder``          ``client.post_message(channel, text, thread_ts)``
  ``edit``                      ``client.update_message(channel, ts, text)``
  ``finalize``                  ``format_answer_as_slack_blocks`` -> ``update_message(blocks=...)``
  ``react`` ACK/DONE/ERROR      ``client.add_reaction`` eyes / white_check_mark / x
  ``fetch_thread``              ``client.fetch_thread_replies`` -> ``[ThreadMessage]``
  ``resolve_identity``          ``slack_user_to_current_user``
  ``send_denial``               ``client.post_message(channel=user_id, ...)`` (DM)
  ``confirm_feedback``          ``httpx`` POST the ``response_url`` (ephemeral)
  ============================  ===================================================

Inbound normalization (the adapter owns it; the dispatcher owns the flow):
``connect(sink)`` wires ``SlackBotClient.start`` to an internal shim object
whose ``on_app_mention`` / ``on_message_im`` build an ``InboundMessage`` (with
``<@MENTION>`` stripped + ``is_dm`` set) and push it to ``sink.on_message``,
and whose ``on_block_action`` builds a ``FeedbackEvent`` and pushes it to
``sink.on_feedback``. ``SlackBotClient.start``'s ack-everything + bot-loop
filter stay exactly as they were.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from opsrag.auth.pomerium import CurrentUser
from opsrag.channels.base import ChannelAdapter, CoreSink
from opsrag.channels.types import (
    AgentResult,
    FeedbackEvent,
    ImageRef,
    InboundMessage,
    MessageHandle,
    ReactionKind,
    ThreadMessage,
)

# WRAP, don't rewrite: the leaf modules are imported and used unchanged.
from opsrag.slack_bot.client import SlackBotClient
from opsrag.slack_bot.identity import slack_user_to_current_user
from opsrag.slack_bot.render import format_answer_as_slack_blocks
from opsrag.slack_bot.thread_context import (
    _display_name_from_info,
    _is_our_own_message,
)

_log = logging.getLogger("opsrag.channels.adapters.slack")

# Slack puts the bot mention as ``<@U0BOTID>`` at the start of an
# ``app_mention`` payload's text. Strip it so the agent doesn't see it.
_MENTION_RE = re.compile(r"<@[A-Z0-9]+>")

# ReactionKind -> Slack emoji name (same glyphs the old handler used).
_REACTION_EMOJI: dict[ReactionKind, str] = {
    ReactionKind.ACK: "eyes",
    ReactionKind.DONE: "white_check_mark",
    ReactionKind.ERROR: "x",
}


@dataclass(frozen=True)
class _SlackHandle:
    """Opaque per-message handle: a ``(channel, ts)`` pair.

    The core treats this as a token it hands back to ``edit`` / ``finalize``.
    """

    channel: str
    ts: str


class SlackAdapter(ChannelAdapter):
    """Slack adapter over the unchanged ``slack_bot`` leaf modules."""

    name = "slack"

    def __init__(self, config: Any) -> None:
        """Build the adapter from a Slack channel sub-config.

        ``config`` is a ``SlackChannelConfig`` (the unified ``channels.slack``
        block). Tokens are read from the env vars it names (Principle VI --
        never inline). The ``SlackBotClient`` itself is constructed lazily in
        ``connect`` so importing this module never touches ``slack_sdk``.
        """
        self._config = config
        self._web_ui_base_url = (getattr(config, "web_ui_base_url", "") or "").rstrip("/")
        self._client: SlackBotClient | None = None
        self._sink: CoreSink | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def connect(self, sink: CoreSink) -> None:
        """Build the Socket Mode client and start it wired to ``sink``.

        Reads the bot/app tokens from the env vars named by the config and
        registers an internal shim (see :class:`_SlackEventShim`) that turns
        Slack events into neutral inbound types and pushes them into ``sink``.
        """
        self._sink = sink
        bot_token = os.environ.get(getattr(self._config, "bot_token_env", ""), "").strip()
        app_token = os.environ.get(getattr(self._config, "app_token_env", ""), "").strip()
        if not bot_token or not app_token:
            raise RuntimeError(
                "Slack adapter: bot/app token env unset "
                f"({getattr(self._config, 'bot_token_env', '?')} / "
                f"{getattr(self._config, 'app_token_env', '?')})"
            )
        self._client = SlackBotClient(bot_token=bot_token, app_token=app_token)
        shim = _SlackEventShim(self, sink)
        # SlackBotClient.start keeps its ack-everything + bot-loop filter; the
        # shim just receives the already-filtered app_mention / message.im /
        # block_actions callbacks.
        await self._client.start(shim)
        _log.info("slack adapter connected via Socket Mode")

    async def close(self) -> None:
        if self._client is not None:
            await self._client.stop()
            self._client = None

    # ------------------------------------------------------------------
    # Outbound primitives
    # ------------------------------------------------------------------
    async def post_placeholder(
        self, channel_id: str, thread_id: str | None, text: str,
    ) -> MessageHandle:
        client = self._require_client()
        ts = await client.post_message(
            channel=channel_id, text=text, thread_ts=thread_id,
        )
        return _SlackHandle(channel=channel_id, ts=ts)

    async def edit(self, handle: MessageHandle, text: str) -> None:
        client = self._require_client()
        h = _as_handle(handle)
        await client.update_message(channel=h.channel, ts=h.ts, text=text)

    async def finalize(self, handle: MessageHandle, result: AgentResult) -> None:
        client = self._require_client()
        h = _as_handle(handle)
        fallback_text, blocks = format_answer_as_slack_blocks(
            result.answer,
            result.sources,
            diagram_present=result.diagram_present,
            web_ui_base_url=self._web_ui_base_url,
            session_id=result.session_id,
            investigation_id=result.investigation_id,
        )
        await client.update_message(
            channel=h.channel, ts=h.ts, text=fallback_text, blocks=blocks,
        )

    async def react(
        self, channel_id: str, message_id: str, kind: ReactionKind,
    ) -> None:
        client = self._client
        if client is None:
            return
        emoji = _REACTION_EMOJI.get(kind)
        if not emoji:
            return
        # add_reaction is already best-effort (swallows its own errors).
        await client.add_reaction(channel_id, message_id, emoji)

    async def fetch_thread(
        self, channel_id: str, thread_id: str, *, cap: int,
    ) -> list[ThreadMessage]:
        """Fetch the thread's replies and map them to neutral ``ThreadMessage``.

        Reuses the EXISTING ``thread_context`` self-filter
        (``_is_our_own_message``) + display-name helper
        (``_display_name_from_info``) so the "drop our own past replies, keep
        other bots' alerts" rule is identical to the old Slack path. The core
        does the serialization + truncation (``_serialize_thread_context``);
        this method only fetches + maps.
        """
        client = self._require_client()
        raw_messages = await client.fetch_thread_replies(
            channel=channel_id, thread_ts=thread_id, limit=cap,
        )
        self_user_id = getattr(client, "self_user_id", None)
        self_bot_id = getattr(client, "self_bot_id", None)

        out: list[ThreadMessage] = []
        for msg in raw_messages:
            is_self = _is_our_own_message(msg, self_user_id, self_bot_id)
            text = (msg.get("text") or "").strip()
            if not text:
                continue
            user_id = msg.get("user") or ""
            if user_id:
                try:
                    info = await client.get_user_info(user_id)
                    author = _display_name_from_info(info, user_id)
                except Exception:  # noqa: BLE001
                    author = user_id
            else:
                # Bot messages (Rootly, Datadog) carry a ``username`` field.
                author = msg.get("username") or "bot"
            # source_id = the Slack ts so the core can drop the triggering
            # message (conversations.replies returns the whole thread, the
            # current message included).
            out.append(
                ThreadMessage(
                    author=author,
                    text=text,
                    is_self=is_self,
                    source_id=msg.get("ts") or None,
                )
            )
        return out

    async def fetch_image(self, ref: ImageRef) -> bytes | None:
        """Download a Slack ``url_private`` -- requires the bot token as bearer.

        Slack file URLs are NOT public: the request must carry
        ``Authorization: Bearer <bot token>`` (the same token the
        ``SlackBotClient`` uses for Web API calls), else Slack returns an HTML
        sign-in page instead of the bytes. Returns ``None`` when the client or
        url is missing.
        """
        if not ref.url:
            return None
        client = self._client
        bot_token = getattr(client, "_bot_token", None) if client is not None else None
        if not bot_token:
            return None
        # Routed through the shared hardened helper (FIX 3/4): https-only + SSRF
        # IP block + size ceiling, and the bearer token in the header is never
        # echoed into any error/log line.
        from opsrag.channels.image_fetch import fetch_image_bytes
        return await fetch_image_bytes(
            ref.url, headers={"Authorization": f"Bearer {bot_token}"},
        )

    async def resolve_identity(self, msg: InboundMessage) -> CurrentUser:
        # Rebuild the minimal Slack event shape the identity helper expects.
        event = {"user": msg.user_id, "team": msg.workspace}
        return await slack_user_to_current_user(event, client=self._client)

    async def send_denial(self, msg: InboundMessage, reason: str) -> None:
        client = self._client
        if client is None or not msg.user_id:
            return
        # DM the user privately (their user_id is a valid channel target) so we
        # don't pollute the public channel with denial noise.
        await client.post_message(channel=msg.user_id, text=reason)

    async def confirm_feedback(self, fb: FeedbackEvent, *, accepted: bool) -> None:
        """Ephemeral confirm via the Slack ``response_url`` (best-effort).

        The shim stashes the click's ``response_url`` in ``fb.raw`` so this
        method can POST the same ephemeral note the old handler did.
        """
        response_url = (fb.raw or {}).get("response_url") or ""
        if not response_url:
            return
        confirm_text = (
            "👍 Thanks -- recorded as helpful."
            if fb.thumbs == "up"
            else "👎 Thanks -- recorded as wrong. We'll learn from this."
        )
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as cx:
                await cx.post(
                    response_url,
                    json={
                        "text": confirm_text,
                        "response_type": "ephemeral",
                        "replace_original": False,
                    },
                )
        except Exception as exc:  # noqa: BLE001
            _log.warning("slack feedback confirm POST failed: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _require_client(self) -> SlackBotClient:
        if self._client is None:
            raise RuntimeError("SlackAdapter.connect() must run before this call")
        return self._client


def _as_handle(handle: MessageHandle) -> _SlackHandle:
    if not isinstance(handle, _SlackHandle):
        raise TypeError(f"expected _SlackHandle, got {type(handle)!r}")
    return handle


class _SlackEventShim:
    """Bridges ``SlackBotClient``'s callback shape to the neutral ``CoreSink``.

    ``SlackBotClient.start`` calls ``on_app_mention`` / ``on_message_im`` /
    ``on_block_action`` (its existing dispatcher protocol). This shim builds
    neutral ``InboundMessage`` / ``FeedbackEvent`` from those Slack payloads
    and forwards them to the dispatcher. The adapter owns all inbound
    normalization; the dispatcher owns the flow.
    """

    def __init__(self, adapter: SlackAdapter, sink: CoreSink) -> None:
        self._adapter = adapter
        self._sink = sink

    async def on_app_mention(self, event: dict[str, Any]) -> None:
        await self._sink.on_message(_event_to_inbound(event, is_dm=False))

    async def on_message_im(self, event: dict[str, Any]) -> None:
        await self._sink.on_message(_event_to_inbound(event, is_dm=True))

    async def on_block_action(self, payload: dict[str, Any]) -> None:
        fb = _payload_to_feedback(payload)
        if fb is None:
            return
        await self._sink.on_feedback(fb)


def _event_to_inbound(event: dict[str, Any], *, is_dm: bool) -> InboundMessage:
    """Map a Slack ``app_mention`` / ``message.im`` event to ``InboundMessage``.

    Strips the ``<@MENTION>`` token (the adapter owns mention-stripping per the
    port contract). ``thread_id`` is the Slack ``thread_ts`` (``None`` when the
    message is not in a thread); ``message_id`` is the message ``ts``.
    """
    raw_text = (event or {}).get("text", "") or ""
    text = _MENTION_RE.sub("", raw_text).strip()
    image_refs = tuple(
        ImageRef(
            url=f.get("url_private"),
            mime_type=f.get("mimetype", "image/png"),
            size=f.get("size"),
        )
        for f in ((event or {}).get("files") or [])
        if str(f.get("mimetype", "") or "").startswith("image/") and f.get("url_private")
    )
    return InboundMessage(
        channel_id=(event or {}).get("channel", "") or "",
        user_id=(event or {}).get("user", "") or "",
        text=text,
        message_id=(event or {}).get("ts", "") or "",
        thread_id=(event or {}).get("thread_ts") or None,
        is_dm=is_dm,
        workspace=(event or {}).get("team") or None,
        images=image_refs,
        raw=dict(event or {}),
    )


def _payload_to_feedback(payload: dict[str, Any]) -> FeedbackEvent | None:
    """Parse a Slack ``block_actions`` payload into a ``FeedbackEvent``.

    Returns ``None`` for actions that are not our feedback buttons or that
    carry a malformed ``value`` (the core also rejects malformed events, but
    bailing here avoids a needless sink call).
    """
    if not isinstance(payload, dict):
        return None
    actions = payload.get("actions") or []
    if not actions:
        return None
    action = actions[0] or {}
    action_id = action.get("action_id") or ""
    if not action_id.startswith("opsrag_feedback_"):
        return None
    value = (action.get("value") or "").strip()
    if ":" not in value:
        return None
    thumbs, investigation_id = value.split(":", 1)
    if thumbs not in ("up", "down") or not investigation_id:
        return None

    user_block = payload.get("user") or {}
    user_id = user_block.get("id") or "slack-unknown"
    container = payload.get("container") or {}
    thread_id = container.get("thread_ts") or container.get("message_ts")
    return FeedbackEvent(
        thumbs=thumbs,
        investigation_id=investigation_id,
        user_id=user_id,
        thread_id=thread_id,
        raw={"response_url": payload.get("response_url") or ""},
    )
