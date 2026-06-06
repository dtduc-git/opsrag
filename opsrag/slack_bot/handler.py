"""Slack event -> OpsRAG agent dispatcher.

The :class:`SlackEventDispatcher` is the central bridge between
Slack's Socket Mode event stream and the OpsRAG agent:

* event arrives via ``client.start(dispatcher)`` ->
  :meth:`on_app_mention` / :meth:`on_message_im`
* permission check (channel allowlist + per-user quota)
* placeholder message + 👀 reaction
* thread-context assembly (mentions only -- DMs are flat)
* :func:`opsrag.agent.graph.query_with_session_events` is invoked
  *directly* (not via HTTP) so we don't pay extra latency or
  double-auth
* progress events are streamed back to Slack via
  :class:`opsrag.slack_bot.streaming.SlackProgressStreamer`
* final answer rendered through
  :func:`opsrag.slack_bot.render.format_answer_as_slack_blocks`
"""
from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING, Any

from opsrag.agent.graph import query_with_session_events
from opsrag.slack_bot.identity import slack_user_to_current_user
from opsrag.slack_bot.render import format_answer_as_slack_blocks

if TYPE_CHECKING:  # pragma: no cover
    from opsrag.slack_bot.client import SlackBotClient
    from opsrag.slack_bot.config import SlackBotConfig
    from opsrag.slack_bot.permission import SlackBotPermission

_log = logging.getLogger("opsrag.slack_bot.handler")

# Slack puts the bot mention as ``<@U0BOTID>`` at the start of an
# ``app_mention`` payload's ``text``. Strip it so the agent doesn't see
# it as part of the question.
_MENTION_RE = re.compile(r"<@[A-Z0-9]+>")

# Placeholder + error copy live in streaming.py so the heartbeat
# ladder and the initial post agree on phrasing.
from opsrag.slack_bot.streaming import (  # noqa: E402
    ERROR_TEXT as _ERROR_TEXT,
)
from opsrag.slack_bot.streaming import (
    PLACEHOLDER_TEXT as _PLACEHOLDER_TEXT,
)


class SlackEventDispatcher:
    """Routes Slack events to the OpsRAG agent and posts answers back.

    Instantiated once at lifespan startup and registered with
    :meth:`SlackBotClient.start`. The client invokes
    :meth:`on_app_mention` for ``app_mention`` events and
    :meth:`on_message_im` for ``message.im`` DM events.

    Thread-safety
    -------------
    A single dispatcher instance may be invoked concurrently by the
    Slack client when bursts of events arrive. All state held by the
    dispatcher (config refs, providers, caches) is either immutable
    or has its own internal locking; the dispatcher itself adds no
    coordination, by design.
    """

    def __init__(
        self,
        *,
        client: SlackBotClient,
        agent_graph: Any,
        providers: Any,
        config: SlackBotConfig,
        permission: SlackBotPermission,
        web_ui_base_url: str,
        qa_cache: Any = None,
        investigation_cache: Any = None,
        semantic_router: Any = None,
        feedback_store: Any = None,
    ) -> None:
        self._client = client
        self._graph = agent_graph
        self._providers = providers
        self._config = config
        self._permission = permission
        self._web_ui_base_url = (web_ui_base_url or "").rstrip("/")
        self._qa_cache = qa_cache
        self._investigation_cache = investigation_cache
        self._semantic_router = semantic_router
        self._feedback_store = feedback_store

    # ------------------------------------------------------------------
    # Public entrypoints
    # ------------------------------------------------------------------
    async def on_app_mention(self, event: dict[str, Any]) -> None:
        """Handle ``app_mention`` -- bot @-mentioned in a channel."""
        await self._dispatch(event, is_dm=False)

    async def on_message_im(self, event: dict[str, Any]) -> None:
        """Handle ``message.im`` -- direct message to the bot."""
        await self._dispatch(event, is_dm=True)

    async def on_block_action(self, payload: dict[str, Any]) -> None:
        """Handle ``block_actions`` -- user clicked 👍 Helpful / 👎 Wrong
        button on a bot answer.

        Slack delivers these over the same Socket Mode websocket as
        regular events. The action payload's `value` field encodes
        ``up:<investigation_id>`` or ``down:<investigation_id>`` (see
        :func:`opsrag.slack_bot.render._build_feedback_actions_block`).

        Side effects (best-effort, none block the others):
          - `investigation_cache.record_feedback` -- increments the
            up/down counter on the cached investigation, so
            high-feedback investigations rank higher in past-similar
            retrieval (see `investigation_cache.py`).
          - `feedback_store.record` -- append-only audit log row in
            Postgres `opsrag_feedback`, used by SRE eval dashboards.
          - POST to the payload's `response_url` -- ephemeral confirm
            visible only to the clicker so they get UI feedback that
            the click landed.
        """
        if not isinstance(payload, dict):
            return
        actions = payload.get("actions") or []
        if not actions:
            return
        action = actions[0]
        action_id = (action or {}).get("action_id") or ""
        if not action_id.startswith("opsrag_feedback_"):
            return
        value = ((action or {}).get("value") or "").strip()
        if ":" not in value:
            _log.warning(
                "slack block_action: malformed value=%r (expected up:<id> / down:<id>)",
                value,
            )
            return
        thumbs, investigation_id = value.split(":", 1)
        if thumbs not in ("up", "down") or not investigation_id:
            _log.warning(
                "slack block_action: bad thumbs/id thumbs=%r id=%r",
                thumbs, investigation_id,
            )
            return

        slack_user_block = payload.get("user") or {}
        slack_user = slack_user_block.get("id") or "slack-unknown"
        container = payload.get("container") or {}
        thread_ts = container.get("thread_ts") or container.get("message_ts")
        direction = 1 if thumbs == "up" else -1

        # -- 1. investigation_cache (best-effort)
        if self._investigation_cache is not None:
            try:
                await self._investigation_cache.record_feedback(
                    investigation_id,
                    thumbs=thumbs,
                    correction=None,
                )
            except Exception as exc:
                _log.warning(
                    "slack block_action: investigation_cache write failed "
                    "id=%s err=%s", investigation_id, exc,
                )

        # -- 2. feedback_store (best-effort)
        if self._feedback_store is not None:
            try:
                await self._feedback_store.record(
                    investigation_id=investigation_id,
                    direction=direction,
                    thread_id=thread_ts,
                    user_id=f"slack:{slack_user}",
                    note=None,
                    query_snippet=None,
                    answer_snippet=None,
                )
            except Exception as exc:
                _log.warning(
                    "slack block_action: feedback_store write failed "
                    "id=%s err=%s", investigation_id, exc,
                )

        # -- 3. Ephemeral confirm via response_url (best-effort).
        # Slack guarantees response_url stays valid for ~30 minutes after
        # the click. POSTing `{"text": ..., "response_type": "ephemeral",
        # "replace_original": false}` shows a private note to the
        # clicker. Failure is non-fatal -- the feedback is already
        # persisted by steps 1+2.
        response_url = payload.get("response_url") or ""
        if response_url:
            confirm_text = (
                "👍 Thanks -- recorded as helpful."
                if thumbs == "up" else
                "👎 Thanks -- recorded as wrong. We'll learn from this."
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
            except Exception as exc:
                _log.warning(
                    "slack block_action: response_url POST failed "
                    "id=%s err=%s", investigation_id, exc,
                )

        _log.info(
            "slack block_action ok: thumbs=%s investigation=%s slack_user=%s "
            "thread=%s",
            thumbs, investigation_id, slack_user, thread_ts,
        )

    # ------------------------------------------------------------------
    # Core flow
    # ------------------------------------------------------------------
    async def _dispatch(self, event: dict[str, Any], *, is_dm: bool) -> None:
        started = time.monotonic()
        channel = (event or {}).get("channel", "") or ""
        user_id = (event or {}).get("user", "") or ""
        thread_ts = (event or {}).get("thread_ts") or None
        msg_ts = (event or {}).get("ts") or None
        log_extra = {
            "channel": channel,
            "user_id": user_id,
            "thread_ts": thread_ts,
            "is_dm": is_dm,
        }

        # ---- 1. Permission check ------------------------------------------------
        ok, deny_reason = await self._permission.allow(event)
        if not ok:
            if deny_reason and user_id:
                # DM the user privately so we don't pollute the
                # channel with denial noise.
                try:
                    await self._client.post_message(
                        channel=user_id,  # DMing a user uses their user_id as the channel
                        text=deny_reason,
                    )
                except Exception as exc:
                    _log.warning(
                        "slack: denial DM failed user=%s err=%s",
                        user_id, exc,
                    )
            _log.info(
                "slack: deny channel=%s user=%s reason=%s",
                channel, user_id, deny_reason or "silent",
            )
            return

        # ---- 2. Extract the user query -----------------------------------------
        raw_text = (event or {}).get("text", "") or ""
        user_query = _MENTION_RE.sub("", raw_text).strip()
        if not user_query:
            # Nothing to answer -- politely no-op rather than calling
            # the agent with an empty string.
            _log.info("slack: empty query (post-mention-strip) channel=%s", channel)
            return

        # ---- 3. Post placeholder + reaction ------------------------------------
        # Replies in channels post into the thread (creating one off the
        # @mention if needed). DMs are flat -- no thread_ts.
        reply_thread_ts = None if is_dm else (thread_ts or msg_ts)
        try:
            placeholder_ts = await self._client.post_message(
                channel=channel,
                text=_PLACEHOLDER_TEXT,
                thread_ts=reply_thread_ts,
            )
        except Exception as exc:
            _log.warning(
                "slack: placeholder post failed channel=%s err=%s",
                channel, exc,
            )
            return

        # Best-effort: 👀 on the original message so the user sees we
        # picked it up.
        if msg_ts and not is_dm:
            try:
                await self._client.add_reaction(channel, msg_ts, "eyes")
            except Exception:
                pass  # add_reaction already swallows, but belt+braces

        # Streamer wraps the placeholder for in-place updates.
        # Imported lazily to keep the sister module out of import-time
        # cycles.
        from opsrag.slack_bot.streaming import SlackProgressStreamer

        streamer = SlackProgressStreamer(
            self._client,
            channel,
            placeholder_ts,
            min_update_interval_s=self._config.streaming_min_update_interval_s,
        )
        # Kick off the reassurance heartbeat so the placeholder doesn't
        # sit static if the agent takes longer than ~30s.
        streamer.start_heartbeat()

        # ---- 4. Thread context (mentions only) ---------------------------------
        thread_context = ""
        if not is_dm and thread_ts and thread_ts != msg_ts:
            try:
                from opsrag.slack_bot.thread_context import assemble_thread_context
                thread_context = await assemble_thread_context(
                    event,
                    self._client,
                    max_messages=self._config.thread_context_message_cap,
                    max_chars=2000,
                )
            except Exception as exc:
                _log.warning(
                    "slack: thread-context assembly failed channel=%s err=%s",
                    channel, exc,
                )
                thread_context = ""

        if thread_context:
            combined_query = f"{thread_context}\n\n{user_query}"
        else:
            combined_query = user_query

        # ---- 5. Resolve identity -----------------------------------------------
        try:
            current_user = await slack_user_to_current_user(
                event, client=self._client,
            )
        except Exception as exc:
            _log.warning("slack: identity resolution failed: %s", exc)
            current_user = None

        # ---- 6. Run the agent (streaming events) -------------------------------
        thread_id_for_session = self._session_thread_id(channel, msg_ts, thread_ts, is_dm)
        user_oid = getattr(current_user, "oid", None) or "slack-bot-anon"

        final: dict[str, Any] | None = None
        agent_error: Exception | None = None
        try:
            async for ev in query_with_session_events(
                self._graph,
                query=combined_query,
                user_id=user_oid,
                thread_id=thread_id_for_session,
                embedder=getattr(self._providers, "embedder", None),
                qa_cache=self._qa_cache,
                llm=getattr(self._providers, "llm", None),
                session_store=getattr(self._providers, "session_store", None),
                investigation_cache=self._investigation_cache,
                semantic_router=self._semantic_router,
                user_email=getattr(current_user, "email", None),
                user_name=getattr(current_user, "name", None),
            ):
                if ev.get("type") == "final":
                    final = ev
                elif ev.get("type") == "error":
                    agent_error = RuntimeError(
                        ev.get("detail") or "agent error"
                    )
        except Exception as exc:
            agent_error = exc

        # ---- 7. Finalize -------------------------------------------------------
        if agent_error is not None or final is None:
            err_kind = (
                type(agent_error).__name__ if agent_error else "EmptyResult"
            )
            _log.warning(
                "slack: agent failed channel=%s user=%s err=%s duration_ms=%d",
                channel, user_id, err_kind,
                int((time.monotonic() - started) * 1000),
            )
            try:
                await streamer.finalize(
                    f"{_ERROR_TEXT} ({err_kind})",
                    blocks=None,
                )
            except Exception as exc:
                _log.warning("slack: error-finalize failed: %s", exc)
            # Best-effort transition reaction to a "failed" indicator.
            if msg_ts and not is_dm:
                try:
                    await self._client.add_reaction(channel, msg_ts, "x")
                except Exception:
                    pass
            return

        answer = final.get("answer") or ""
        sources = final.get("sources") or []
        diagram_present = bool(
            final.get("diagram")
            or final.get("diagram_present")
            or final.get("has_diagram")
        )
        session_id = (
            final.get("thread_id")
            or final.get("session_id")
            or thread_id_for_session
        )

        fallback_text, blocks = format_answer_as_slack_blocks(
            answer,
            sources,
            diagram_present=diagram_present,
            web_ui_base_url=self._web_ui_base_url,
            session_id=session_id,
            investigation_id=final.get("investigation_id"),
        )

        try:
            await streamer.finalize(fallback_text, blocks=blocks)
        except Exception as exc:
            _log.warning("slack: finalize failed channel=%s err=%s", channel, exc)

        # Successful run -- count it for quota purposes.
        if user_id:
            self._permission.record_usage(user_id)

        # Swap 👀 -> ✅ on the original message (best-effort).
        if msg_ts and not is_dm:
            try:
                await self._client.add_reaction(channel, msg_ts, "white_check_mark")
            except Exception:
                pass

        duration_ms = int((time.monotonic() - started) * 1000)
        _log.info(
            "slack: ok channel=%s user=%s thread_ts=%s duration_ms=%d sources=%d",
            channel, user_id, thread_ts, duration_ms, len(sources),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _session_thread_id(
        self,
        channel: str,
        msg_ts: str | None,
        thread_ts: str | None,
        is_dm: bool,
    ) -> str:
        """Compose a stable session/thread id for the agent.

        Rules:
          - mention reply in an *existing* thread -> reuse the thread_ts
            so multi-turn questions in the same Slack thread share an
            OpsRAG session
          - mention reply that *creates* a new thread -> use the source
            message ts
          - DM -> flat conversation per user, scoped by the IM channel
        """
        if is_dm:
            return f"slack-dm:{channel}"
        anchor = thread_ts or msg_ts or "no-ts"
        return f"slack-thread:{channel}:{anchor}"
