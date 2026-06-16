"""Teams inbound webhook router -- the real Bot Framework endpoint.

The Bot Framework delivers Teams activities by POSTing them to a public HTTPS
endpoint. This router is mounted on the ``api`` role (see
``opsrag.api.server``) only when ``cfg.channels.teams.enabled``. It:

  * validates the inbound Bot Connector JWT (rejects missing/invalid with 401);
  * parses the ``Activity`` and normalizes it to ``InboundMessage`` (a
    ``type=='message'`` activity with no card-action ``value``) or a
    ``FeedbackEvent`` (an ``Action.Submit`` activity carrying ``.value``);
  * drives the shared :class:`~opsrag.channels.dispatcher.ChannelDispatcher`
    over a single :class:`~opsrag.channels.adapters.teams.adapter.TeamsAdapter`.

:func:`build_teams_router` is the factory the API lifespan calls; it constructs
the adapter + dispatcher (wired to the agent graph / providers / caches /
permission from ``teams_cfg``) and returns an ``APIRouter``.

The ``botbuilder`` SDK (``teams`` extra) is NOT imported at module top: JWT
validation imports it **lazily** inside :func:`validate_bot_jwt`, so the ``api``
role -- which mounts this router -- imports cleanly without the extra (and unit
tests monkeypatch the validator). Activity parsing here works on the raw JSON
dict, so it needs no SDK either.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse, Response

from opsrag.channels.adapters.teams.adapter import TeamsAdapter
from opsrag.channels.dispatcher import ChannelDispatcher
from opsrag.channels.permission import ChannelPermission
from opsrag.channels.types import FeedbackEvent, ImageRef, InboundMessage

_log = logging.getLogger("opsrag.channels.adapters.teams.router")


class TeamsAuthError(Exception):
    """Raised when the inbound Bot Connector JWT is missing or invalid."""


async def validate_bot_jwt(
    auth_header: str | None,
    activity: dict[str, Any],
    *,
    app_id: str,
) -> None:
    """Validate the Bot Connector JWT on an inbound activity.

    Rejects a missing/empty ``Authorization`` header immediately. When the
    ``botbuilder`` SDK is available it delegates to
    ``JwtTokenValidation.authenticate_request`` (the canonical Bot Framework
    check); the SDK import is lazy so this module imports without the ``teams``
    extra. Raises :class:`TeamsAuthError` on any failure -- the handler maps
    that to HTTP 401.

    Kept as a small standalone function on purpose: unit tests monkeypatch
    ``opsrag.channels.adapters.teams.router.validate_bot_jwt`` to assert the
    401 path without a real signed token.
    """
    if not auth_header or not auth_header.strip():
        raise TeamsAuthError("missing Authorization header")
    try:
        from botbuilder.connector.auth import (
            AuthenticationConfiguration,
            JwtTokenValidation,
            SimpleCredentialProvider,
        )
        from botbuilder.schema import Activity
    except Exception as exc:  # noqa: BLE001
        # Without the SDK we cannot cryptographically verify the token. Fail
        # CLOSED -- never accept an unverifiable token in a deployment that
        # somehow lacks the extra.
        raise TeamsAuthError(f"botbuilder unavailable for JWT validation: {exc}") from exc
    try:
        credentials = SimpleCredentialProvider(app_id, "")
        await JwtTokenValidation.authenticate_request(
            Activity().deserialize(activity),
            auth_header,
            credentials,
            channel_service_or_provider="",
            auth_configuration=AuthenticationConfiguration(),
        )
    except Exception as exc:  # noqa: BLE001
        raise TeamsAuthError(f"JWT validation failed: {exc}") from exc


# =====================================================================
# Activity -> neutral type normalization (pure dict work -- no SDK)
# =====================================================================

# A Teams bot mention arrives as ``<at>Bot Name</at>`` in the text.
_AT_MENTION_RE = re.compile(r"<at\b[^>]*>.*?</at>", re.DOTALL | re.IGNORECASE)


def _conversation_reference(activity: dict[str, Any]) -> dict[str, Any]:
    """Build a (serialized) ConversationReference dict from an activity.

    Mirrors ``TurnContext.get_conversation_reference``: enough of the activity
    to resume a proactive conversation later (the adapter passes this back into
    ``continue_conversation``). We keep it as a plain dict so the router needs
    no SDK; the adapter's ``CloudAdapter`` accepts the deserialized form.
    """
    return {
        "activity_id": activity.get("id"),
        "user": activity.get("from") or {},
        "bot": activity.get("recipient") or {},
        "conversation": activity.get("conversation") or {},
        "channel_id": activity.get("channelId"),
        "locale": activity.get("locale"),
        "service_url": activity.get("serviceUrl"),
    }


def activity_to_inbound(activity: dict[str, Any]) -> InboundMessage:
    """Normalize a Teams message ``Activity`` to an ``InboundMessage``.

    * ``conversation.id`` -> ``channel_id``;
    * ``from.id`` -> ``user_id``;
    * ``conversation.conversationType == 'personal'`` (or ``isGroup`` false in a
      1:1) -> ``is_dm`` -- group/channel chats are not DMs;
    * ``<at>...</at>`` bot-mention spans stripped from the text;
    * ``channelData.tenant.id`` -> ``workspace`` (namespaces the synthetic oid);
    * the conversation reference is stashed in ``raw`` so the adapter can reply
      / DM a denial back into the same conversation.
    """
    conversation = activity.get("conversation") or {}
    from_block = activity.get("from") or {}
    channel_data = activity.get("channelData") or {}
    tenant = (channel_data.get("tenant") or {}).get("id")

    conv_type = (conversation.get("conversationType") or "").strip().lower()
    is_group_flag = conversation.get("isGroup")
    is_dm = conv_type == "personal" or (conv_type == "" and is_group_flag is False)

    raw_text = activity.get("text") or ""
    text = _AT_MENTION_RE.sub("", raw_text).strip()

    image_refs = tuple(
        ImageRef(url=att.get("contentUrl"), mime_type=att.get("contentType", "image/png"))
        for att in (activity.get("attachments") or [])
        if str(att.get("contentType", "") or "").startswith("image/") and att.get("contentUrl")
    )

    return InboundMessage(
        channel_id=conversation.get("id") or "",
        user_id=from_block.get("id") or "",
        text=text,
        message_id=activity.get("id") or "",
        thread_id=None,  # Teams has no Slack-style thread_ts in the v1 model
        is_dm=is_dm,
        workspace=tenant or None,
        images=image_refs,
        raw={
            "activity": activity,
            "conversation_reference": _conversation_reference(activity),
        },
    )


def activity_to_feedback(activity: dict[str, Any]) -> FeedbackEvent | None:
    """Parse an ``Action.Submit`` activity into a ``FeedbackEvent``.

    Teams delivers a card submit as a ``message`` activity whose ``.value`` is
    the action's ``data`` dict -- ``{"feedback":"up"|"down","id":"<id>"}``.
    Returns ``None`` for anything that is not a well-formed feedback submit (the
    core also rejects malformed events, but bailing here avoids a needless sink
    call and lets the handler treat it as a no-op 200).
    """
    value = activity.get("value")
    if not isinstance(value, dict):
        return None
    thumbs = (value.get("feedback") or "").strip().lower()
    investigation_id = (value.get("id") or "").strip()
    if thumbs not in ("up", "down") or not investigation_id:
        return None

    from_block = activity.get("from") or {}
    return FeedbackEvent(
        thumbs=thumbs,
        investigation_id=investigation_id,
        user_id=from_block.get("id") or "teams-unknown",
        thread_id=None,
        raw={
            "activity": activity,
            "conversation_reference": _conversation_reference(activity),
        },
    )


# =====================================================================
# Router factory
# =====================================================================


def build_teams_router(
    agent_graph: Any,
    providers: Any,
    caches: Any,
    teams_cfg: Any,
    vision: Any = None,
) -> APIRouter:
    """Construct the Teams webhook router + its adapter/dispatcher.

    Builds ONE :class:`TeamsAdapter` and a :class:`ChannelDispatcher` (wired to
    the agent graph / providers / caches / permission derived from
    ``teams_cfg``), then returns an ``APIRouter`` exposing
    ``POST /api/channels/teams/messages``. The dispatcher is the
    :class:`~opsrag.channels.base.CoreSink` the handler pushes normalized
    events into.

    ``caches`` is a namespace exposing ``qa_cache`` / ``investigation_cache`` /
    ``semantic_router`` / ``feedback_store`` (any may be ``None``).
    """
    router = APIRouter(prefix="/api/channels/teams", tags=["channels", "teams"])

    adapter = TeamsAdapter(teams_cfg)
    permission = ChannelPermission(
        allowed_channels=set(getattr(teams_cfg, "allowlist", []) or []),
        per_user_daily_quota=int(getattr(teams_cfg, "per_user_daily_quota", 200)),
    )
    dispatcher = ChannelDispatcher(
        adapter=adapter,
        agent_graph=agent_graph,
        providers=providers,
        permission=permission,
        web_ui_base_url=getattr(teams_cfg, "web_ui_base_url", "") or "",
        thread_context_message_cap=int(
            getattr(teams_cfg, "thread_context_message_cap", 20),
        ),
        qa_cache=getattr(caches, "qa_cache", None),
        investigation_cache=getattr(caches, "investigation_cache", None),
        semantic_router=getattr(caches, "semantic_router", None),
        feedback_store=getattr(caches, "feedback_store", None),
        vision=vision,
    )
    app_id = os.environ.get(getattr(teams_cfg, "app_id_env", ""), "").strip()

    # Expose the built objects so the lifespan (or a test) can introspect /
    # connect them. ``connect`` for Teams has no transport loop -- it just stores
    # the sink + builds the outbound CloudAdapter -- so we run it lazily on the
    # first inbound activity (idempotent) rather than via a fragile router-level
    # startup hook.
    router.adapter = adapter  # type: ignore[attr-defined]
    router.dispatcher = dispatcher  # type: ignore[attr-defined]
    _connected = {"done": False}
    # Double-checked locking: concurrent first activities must not both call
    # adapter.connect(). The fast path skips the lock once connected.
    _connect_lock = asyncio.Lock()

    async def _ensure_connected() -> None:
        if _connected["done"]:
            return
        async with _connect_lock:
            if not _connected["done"]:
                await adapter.connect(dispatcher)
                _connected["done"] = True

    @router.post("/messages")
    async def teams_messages(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> Response:
        """Inbound Teams Activity endpoint.

        Validates the Bot Connector JWT (401 on missing/invalid), then routes a
        message activity to ``dispatcher.on_message`` and a card-submit activity
        to ``dispatcher.on_feedback``. Returns 200 (empty body) on success --
        the Bot Framework expects a fast 2xx; the reply is sent proactively via
        the connector, not in this response body.
        """
        try:
            activity = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse(
                status_code=400,
                content={"error": {"code": "BAD_REQUEST", "message": "invalid JSON"}},
            )
        if not isinstance(activity, dict):
            return JSONResponse(
                status_code=400,
                content={"error": {"code": "BAD_REQUEST", "message": "expected an Activity object"}},
            )

        # -- 1. Auth: reject missing/invalid Bot Connector JWT.
        try:
            await validate_bot_jwt(authorization, activity, app_id=app_id)
        except TeamsAuthError as exc:
            _log.info("teams: rejected inbound activity (auth): %s", exc)
            return JSONResponse(
                status_code=401,
                content={"error": {"code": "UNAUTHORIZED", "message": "invalid bot token"}},
            )

        # -- 2. Only message activities carry user input / card submits.
        if (activity.get("type") or "").strip().lower() != "message":
            return Response(status_code=200)

        # Lazily wire the adapter's outbound connector on the first activity.
        await _ensure_connected()

        # Register the conversation so the adapter can reply proactively.
        try:
            adapter.register_conversation(
                (activity.get("conversation") or {}).get("id") or "",
                _conversation_reference(activity),
            )
        except Exception:  # noqa: BLE001
            pass

        # -- 3a. Card submit (Action.Submit) -> feedback.
        fb = activity_to_feedback(activity)
        if fb is not None:
            await dispatcher.on_feedback(fb)
            return Response(status_code=200)

        # -- 3b. Plain message -> inbound query.
        inbound = activity_to_inbound(activity)
        await dispatcher.on_message(inbound)
        return Response(status_code=200)

    return router


# Back-compat: ``opsrag.api.server`` historically imported a bare
# ``teams_router``. It now builds the real router via ``build_teams_router``;
# this module-level name is intentionally NOT defined so a stale import fails
# loudly rather than silently mounting a stub.
