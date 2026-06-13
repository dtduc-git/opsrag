"""``TeamsAdapter`` -- the Microsoft Teams implementation of ``ChannelAdapter``.

Teams is fundamentally an **inbound-via-webhook** channel: the Microsoft Bot
Framework PUSHES ``Activity`` payloads to a public HTTPS endpoint
(``POST /api/channels/teams/messages``) rather than the adapter polling a
gateway. So unlike the Slack/Telegram/Discord workers, ``connect(sink)`` does
NOT spin up a transport loop -- it just stores the sink. Inbound Activities are
parsed by the FastAPI router (:mod:`opsrag.channels.adapters.teams.router`),
which calls ``dispatcher.on_message`` / ``dispatcher.on_feedback`` directly.

Outbound primitives go back out over the Bot Framework connector: the adapter
keeps a ``ConversationReference`` (+ the sent activity id) per posted message as
its :class:`MessageHandle`, replays into the same conversation with
``continue_conversation``, and edits in place with ``update_activity``.

Port mapping (design 4.4):

  ===========================  ====================================================
  port method                  Teams / Bot Framework call
  ===========================  ====================================================
  ``connect``                  store the sink (no transport loop)
  ``post_placeholder``         send an ``Activity`` -> handle = (conv ref, activity id)
  ``edit``                     ``update_activity`` (heartbeat tick)
  ``finalize``                 Adaptive Card (answer + sources + 👍/👎) via ``update_activity``
  ``react``                    no-op (Teams has no message-reaction affordance)
  ``fetch_thread``             ``[]`` (no simple thread-replies fetch; see below)
  ``resolve_identity``         synthetic ``teams-bot:<tenant>:<user>`` (anonymous)
  ``send_denial``              private ``Activity`` back into the conversation
  ``confirm_feedback``         small confirm ``Activity``
  ===========================  ====================================================

The ``botbuilder`` SDK (``teams`` extra) is imported **lazily** inside
``connect`` / the outbound methods so importing this module on the ``api`` role
(which mounts the webhook router) never requires the extra.

``fetch_thread`` returns ``[]``: the Bot Framework exposes no simple
"fetch all replies for this conversation/thread" REST call the way Slack's
``conversations.replies`` does. Teams channel-message replies would require
Microsoft Graph + delegated permissions (out of scope for the synthetic-bot v1).
The core treats an empty thread context as "no prior messages", so this is a
clean degradation rather than an error.

See design doc ``specs/002-channel-bots/design.md`` section 4.4.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from typing import Any

from opsrag.auth.pomerium import CurrentUser
from opsrag.channels.base import ChannelAdapter, CoreSink
from opsrag.channels.types import (
    AgentResult,
    FeedbackEvent,
    InboundMessage,
    MessageHandle,
    ReactionKind,
    ThreadMessage,
)

_log = logging.getLogger("opsrag.channels.adapters.teams")

# Cap on the answer body we put in the Adaptive Card TextBlock. Teams cards
# have generous limits but very long blocks render poorly on mobile, so we
# truncate with a deep link the same way Slack does.
_ANSWER_CHARS_CAP = 3500
# Cap on source rows we surface in the card (matches the Slack 10 cap).
_MAX_SOURCES = 10


@dataclass(frozen=True)
class _TeamsHandle:
    """Opaque per-message handle for a sent Teams Activity.

    Carries the serialized ``ConversationReference`` (so ``edit`` /
    ``finalize`` can ``continue_conversation`` back into the same chat) plus
    the activity id of the message we sent (the ``update_activity`` target).
    The core treats this purely as a token it hands back to ``edit`` /
    ``finalize``.
    """

    conversation_reference: Any
    activity_id: str | None


class TeamsAdapter(ChannelAdapter):
    """Teams adapter over the Bot Framework ``CloudAdapter`` / connector."""

    name = "teams"

    def __init__(self, config: Any) -> None:
        """Build the adapter from a Teams channel sub-config.

        ``config`` is a ``TeamsChannelConfig`` (the unified ``channels.teams``
        block). The Microsoft app id + password are read from the env vars it
        names (Principle VI -- never inline). The ``CloudAdapter`` itself is
        built lazily in :meth:`connect` so importing this module never touches
        ``botbuilder``.
        """
        self._config = config
        self._web_ui_base_url = (
            getattr(config, "web_ui_base_url", "") or ""
        ).rstrip("/")
        self._app_id: str = ""
        self._app_password: str = ""
        self._app_type: str = "MultiTenant"
        self._app_tenant_id: str = ""
        self._cloud_adapter: Any = None
        self._sink: CoreSink | None = None
        # Per-conversation ConversationReference cache, keyed by the platform
        # conversation id (== InboundMessage.channel_id). The router registers
        # the reference on every inbound activity so ``post_placeholder`` -- which
        # the dispatcher drives with the channel_id ALONE -- can resolve the
        # proactive reply target.
        self._refs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def connect(self, sink: CoreSink) -> None:
        """Store the sink and build the Bot Framework ``CloudAdapter``.

        There is NO transport loop here (Teams is inbound-via-webhook -- the
        router drives ``sink.on_message`` / ``on_feedback``). We only build the
        outbound connector so the bot can reply / edit. ``botbuilder`` is
        imported lazily so the ``api`` role imports this module without the
        ``teams`` extra; if the extra is missing the adapter still *connects*
        (so the router can normalize inbound activities) but outbound calls
        no-op with a warning.
        """
        self._sink = sink
        self._app_id = os.environ.get(
            getattr(self._config, "app_id_env", ""), "",
        ).strip()
        self._app_password = os.environ.get(
            getattr(self._config, "app_password_env", ""), "",
        ).strip()
        self._app_type = (
            os.environ.get(getattr(self._config, "app_type_env", ""), "").strip()
            or "MultiTenant"
        )
        self._app_tenant_id = os.environ.get(
            getattr(self._config, "app_tenant_id_env", ""), "",
        ).strip()
        self._cloud_adapter = self._build_cloud_adapter()
        _log.info(
            "teams adapter connected (webhook-driven; app_type=%s; outbound=%s)",
            self._app_type,
            "ready" if self._cloud_adapter is not None else "disabled",
        )

    async def close(self) -> None:
        self._cloud_adapter = None
        self._sink = None

    # ------------------------------------------------------------------
    # Outbound primitives
    # ------------------------------------------------------------------
    async def post_placeholder(
        self, channel_id: str, thread_id: str | None, text: str,
    ) -> MessageHandle:
        """Send the placeholder Activity and return a handle for later edits.

        The conversation reference comes off ``raw["conversation_reference"]``
        (the router stashes a serialized ``ConversationReference`` there when it
        normalizes the inbound Activity). We send the text into that
        conversation and capture the resulting activity id so ``edit`` /
        ``finalize`` can target it with ``update_activity``.
        """
        conv_ref = self._conversation_reference_for(channel_id)
        if self._cloud_adapter is None:
            # Outbound disabled (no ``teams`` extra) -- return a handle with no
            # activity id so edit/finalize degrade to clean no-ops. Never touch
            # the SDK-backed Activity builders in this state.
            return _TeamsHandle(conversation_reference=conv_ref, activity_id=None)
        activity_id = await self._send_activity(conv_ref, self._text_activity(text))
        return _TeamsHandle(conversation_reference=conv_ref, activity_id=activity_id)

    async def edit(self, handle: MessageHandle, text: str) -> None:
        """Heartbeat tick -- replace the placeholder text via ``update_activity``."""
        h = _as_handle(handle)
        if h.activity_id is None or self._cloud_adapter is None:
            return
        activity = self._text_activity(text)
        activity.id = h.activity_id
        await self._update_activity(h.conversation_reference, activity)

    async def finalize(self, handle: MessageHandle, result: AgentResult) -> None:
        """Render the answer as an Adaptive Card and update in place.

        The card body is: an answer ``TextBlock`` (markdown), an optional
        diagram callout, a sources ``FactSet`` (+ link buttons), a footer
        ``TextBlock``, and -- when we have an ``investigation_id`` -- a pair of
        ``Action.Submit`` buttons carrying ``{"feedback":"up"|"down",
        "id":"<investigation_id>"}`` (Teams delivers a card submit as a message
        activity with that ``.value``, which the router turns into a
        ``FeedbackEvent``).
        """
        h = _as_handle(handle)
        if self._cloud_adapter is None:
            # Outbound disabled -- nothing to send. The card is still rendered
            # by ``render_adaptive_card`` (pure) in tests; here we no-op.
            return
        card = render_adaptive_card(
            result, web_ui_base_url=self._web_ui_base_url,
        )
        activity = self._card_activity(card, fallback_text=result.answer)
        if h.activity_id is not None:
            activity.id = h.activity_id
            await self._update_activity(h.conversation_reference, activity)
        else:
            await self._send_activity(h.conversation_reference, activity)

    async def react(
        self, channel_id: str, message_id: str, kind: ReactionKind,
    ) -> None:
        """No-op: Teams exposes no message-reaction affordance to bots.

        Reaction status is conveyed instead through the heartbeat text + the
        final Adaptive Card. Implemented as a clean no-op per the port contract
        (``react`` is best-effort).
        """
        return None

    async def fetch_thread(
        self, channel_id: str, thread_id: str, *, cap: int,
    ) -> list[ThreadMessage]:
        """Return ``[]`` -- Teams has no simple thread-replies fetch.

        The Bot Framework connector exposes no equivalent of Slack's
        ``conversations.replies``. Pulling a Teams channel thread's history
        requires Microsoft Graph + delegated permissions, which is out of scope
        for the synthetic-bot v1. The core treats ``[]`` as "no prior thread
        context", so this degrades cleanly.
        """
        return []

    async def resolve_identity(self, msg: InboundMessage) -> CurrentUser:
        """Synthetic anonymous identity ``teams-bot:<tenant>:<user>``.

        ``workspace`` carries the AAD tenant id (the router fills it from
        ``channelData.tenant.id``). Real AAD-backed identity is a future
        enhancement (design D4) -- for now the oid is traceable but anonymous,
        so admin-gated actions stay fail-closed.
        """
        tenant = msg.workspace or "unknown"
        user = msg.user_id or "unknown-user"
        base = CurrentUser.anonymous()
        return replace(base, oid=f"{self.name}-bot:{tenant}:{user}")

    async def send_denial(self, msg: InboundMessage, reason: str) -> None:
        """Privately tell the user why we won't answer (a plain Activity)."""
        conv_ref = (msg.raw or {}).get("conversation_reference")
        if conv_ref is None or self._cloud_adapter is None:
            return
        await self._send_activity(conv_ref, self._text_activity(reason))

    async def confirm_feedback(self, fb: FeedbackEvent, *, accepted: bool) -> None:
        """Send a small confirm Activity acknowledging the 👍/👎 vote."""
        conv_ref = (fb.raw or {}).get("conversation_reference")
        if conv_ref is None or self._cloud_adapter is None:
            return
        confirm_text = (
            "👍 Thanks -- recorded as helpful."
            if fb.thumbs == "up"
            else "👎 Thanks -- recorded as wrong. We'll learn from this."
        )
        await self._send_activity(conv_ref, self._text_activity(confirm_text))

    # ------------------------------------------------------------------
    # botbuilder plumbing (lazy import)
    # ------------------------------------------------------------------
    def _build_cloud_adapter(self) -> Any:
        """Lazily build a ``CloudAdapter`` from the app id/password.

        Returns ``None`` (outbound disabled) if the ``teams`` extra is not
        installed -- importing must never hard-fail on the ``api`` role.
        """
        try:
            from botbuilder.core import CloudAdapter, ConfigurationBotFrameworkAuthentication
            from botbuilder.core.integration import ConfigurationServiceClientCredentialFactory
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "teams: botbuilder not installed (%s) -- outbound disabled, "
                "inbound normalization still works", exc,
            )
            return None
        try:
            settings = _BotSettings(
                self._app_id, self._app_password,
                app_type=self._app_type, tenant_id=self._app_tenant_id,
            )
            credentials = ConfigurationServiceClientCredentialFactory(settings)
            auth = ConfigurationBotFrameworkAuthentication(
                settings, credentials_factory=credentials,
            )
            return CloudAdapter(auth)
        except Exception as exc:  # noqa: BLE001
            _log.warning("teams: CloudAdapter build failed (%s)", exc)
            return None

    def register_conversation(self, channel_id: str, conversation_reference: Any) -> None:
        """Cache the ``ConversationReference`` for a conversation.

        Called by the router on every inbound activity so the adapter can later
        reply proactively: the dispatcher drives ``post_placeholder`` with the
        ``channel_id`` only, so the adapter needs this side-channel to recover
        the reply target.
        """
        if channel_id:
            self._refs[channel_id] = conversation_reference

    def _conversation_reference_for(self, channel_id: str) -> Any:
        """Resolve the conversation reference for ``channel_id``.

        Returns the reference the router registered for this conversation, or
        the raw ``channel_id`` token as a last-resort fallback (outbound is then
        a best-effort no-op if the SDK can't resume from it).
        """
        return self._refs.get(channel_id, channel_id)

    def _text_activity(self, text: str) -> Any:
        """Build a plain-text ``Activity`` (lazy import)."""
        from botbuilder.schema import Activity, ActivityTypes
        return Activity(type=ActivityTypes.message, text=text)

    def _card_activity(self, card: dict, *, fallback_text: str) -> Any:
        """Build a ``message`` Activity carrying an Adaptive Card attachment."""
        from botbuilder.schema import (
            Activity,
            ActivityTypes,
            Attachment,
        )
        attachment = Attachment(
            content_type="application/vnd.microsoft.card.adaptive",
            content=card,
        )
        return Activity(
            type=ActivityTypes.message,
            text=_plain_summary(fallback_text),
            attachments=[attachment],
        )

    async def _send_activity(self, conv_ref: Any, activity: Any) -> str | None:
        """Send ``activity`` into the conversation ``conv_ref`` references.

        Returns the resulting activity id (for later ``update_activity``), or
        ``None`` when outbound is disabled (no ``teams`` extra) so the rest of
        the flow degrades cleanly.
        """
        if self._cloud_adapter is None:
            _log.debug("teams: outbound disabled -- dropping activity")
            return None
        captured: dict[str, str | None] = {"id": None}

        async def _logic(turn_context: Any) -> None:
            resource = await turn_context.send_activity(activity)
            captured["id"] = getattr(resource, "id", None)

        await self._continue_conversation(conv_ref, _logic)
        return captured["id"]

    async def _update_activity(self, conv_ref: Any, activity: Any) -> None:
        """Edit a previously-sent activity in place via ``update_activity``."""
        if self._cloud_adapter is None:
            return

        async def _logic(turn_context: Any) -> None:
            await turn_context.update_activity(activity)

        await self._continue_conversation(conv_ref, _logic)

    async def _continue_conversation(self, conv_ref: Any, logic: Any) -> None:
        """Resume the proactive conversation and run ``logic`` in its turn."""
        try:
            await self._cloud_adapter.continue_conversation(
                conv_ref, logic, self._app_id,
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("teams: continue_conversation failed: %s", exc)


def _as_handle(handle: MessageHandle) -> _TeamsHandle:
    if not isinstance(handle, _TeamsHandle):
        raise TypeError(f"expected _TeamsHandle, got {type(handle)!r}")
    return handle


class _BotSettings:
    """Minimal settings object exposing the keys botbuilder's factories read.

    ``ConfigurationServiceClientCredentialFactory`` /
    ``ConfigurationBotFrameworkAuthentication`` read ``MicrosoftAppId`` /
    ``MicrosoftAppPassword`` via a ``.get(key)`` accessor on the settings
    object. We satisfy that shape without pulling in a config framework.
    """

    def __init__(
        self,
        app_id: str,
        app_password: str,
        app_type: str = "MultiTenant",
        tenant_id: str = "",
    ) -> None:
        self._values = {
            "MicrosoftAppId": app_id,
            "MicrosoftAppPassword": app_password,
            "MicrosoftAppType": app_type or "MultiTenant",
            "MicrosoftAppTenantId": tenant_id or "",
        }

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)


# =====================================================================
# Adaptive Card rendering (pure -- no SDK, testable without botbuilder)
# =====================================================================


def _plain_summary(answer: str, *, cap: int = 200) -> str:
    """Best-effort plain-text summary used as the Activity's ``text`` fallback."""
    import re
    if not answer:
        return "OpsRAG answer"
    work = re.sub(r"```.*?```", "", answer, flags=re.DOTALL)
    work = re.sub(r"[#*_`>]", "", work)
    work = re.sub(r"\s+", " ", work).strip()
    if len(work) > cap:
        work = work[: cap - 3] + "..."
    return work or "OpsRAG answer"


def _deep_link(web_ui_base_url: str, session_id: str | None) -> str | None:
    base = (web_ui_base_url or "").rstrip("/")
    if not base:
        return None
    if session_id:
        return f"{base}/#chat/{session_id}"
    return base


def _truncate_answer(text: str, cap: int, deep_link: str | None) -> str:
    """Clip the answer to ``cap`` chars, appending a truncation marker."""
    if len(text) <= cap:
        return text
    suffix = (
        f"\n\n_...answer truncated -- [view full in OpsRAG UI]({deep_link})_"
        if deep_link
        else "\n\n_...answer truncated_"
    )
    head = text[: max(0, cap - len(suffix))]
    if head.count("```") % 2 == 1:
        last_fence = head.rfind("```")
        if last_fence > 0:
            head = head[:last_fence].rstrip()
    return head.rstrip() + suffix


def render_adaptive_card(
    result: AgentResult,
    *,
    web_ui_base_url: str = "",
    answer_chars_cap: int = _ANSWER_CHARS_CAP,
) -> dict[str, Any]:
    """Render an ``AgentResult`` as an Adaptive Card payload (a plain dict).

    Layout (top -> bottom):
      1. Answer ``TextBlock`` (markdown, truncated with a deep link).
      2. (optional) diagram callout ``TextBlock`` + "View in OpsRAG UI" action.
      3. Sources ``FactSet`` (titles) + link ``Action.OpenUrl`` buttons.
      4. Footer ``TextBlock`` (attribution + timestamp).
      5. (when ``investigation_id``) 👍/👎 ``Action.Submit`` pair carrying
         ``{"feedback":"up"|"down","id":"<investigation_id>"}``.

    Returns the Adaptive Card JSON (a dict) -- pure, no ``botbuilder`` needed,
    so tests can assert the shape without the SDK installed. The adapter wraps
    it in an ``Attachment`` at send time.
    """
    from datetime import UTC, datetime

    deep_link = _deep_link(web_ui_base_url, result.session_id)
    body: list[dict[str, Any]] = []

    # 1. Answer body.
    answer_text = _truncate_answer(result.answer or "", answer_chars_cap, deep_link)
    if answer_text:
        body.append({
            "type": "TextBlock",
            "text": answer_text,
            "wrap": True,
        })

    # 2. Diagram callout.
    if result.diagram_present:
        callout = "Diagram available -- open in OpsRAG UI for the full visual."
        body.append({"type": "TextBlock", "text": callout, "wrap": True, "isSubtle": True})

    # 3. Sources -- a FactSet of titles + link buttons.
    actions: list[dict[str, Any]] = []
    facts: list[dict[str, str]] = []
    sources = result.sources or []
    for idx, src in enumerate(sources[:_MAX_SOURCES], start=1):
        if not isinstance(src, dict):
            continue
        title = (src.get("title") or src.get("source") or "source").strip()
        url = (src.get("url") or "").strip()
        path = (src.get("source") or "").strip()
        facts.append({"title": f"{idx}.", "value": title or path or "source"})
        if url:
            actions.append({"type": "Action.OpenUrl", "title": title[:60] or "source", "url": url})
    if facts:
        body.append({"type": "TextBlock", "text": f"**Sources** ({len(sources)})", "wrap": True})
        body.append({"type": "FactSet", "facts": facts})
        extra = len(sources) - _MAX_SOURCES
        if extra > 0:
            body.append({"type": "TextBlock", "text": f"_+{extra} more_", "wrap": True, "isSubtle": True})

    # 4. Footer.
    footer = "OpsRAG · " + datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    body.append({
        "type": "TextBlock",
        "text": footer,
        "wrap": True,
        "isSubtle": True,
        "spacing": "Medium",
        "separator": True,
    })

    # 5. Feedback Action.Submit pair (only with an anchor id).
    if result.investigation_id:
        actions.append({
            "type": "Action.Submit",
            "title": "👍 Helpful",
            "data": {"feedback": "up", "id": result.investigation_id},
        })
        actions.append({
            "type": "Action.Submit",
            "title": "👎 Wrong",
            "data": {"feedback": "down", "id": result.investigation_id},
        })

    card: dict[str, Any] = {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.4",
        "body": body,
    }
    if actions:
        card["actions"] = actions
    return card
