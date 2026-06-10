"""The shared, channel-neutral flow -- ``ChannelDispatcher``.

This is ``slack_bot/handler.py::_dispatch`` (the seven-stage flow) and
``on_block_action`` (feedback) with every Slack call replaced by a
``ChannelAdapter`` port call. It IS the :class:`CoreSink` -- the adapter
pushes normalised inbound events into ``on_message`` / ``on_feedback``.

A fix to thread-context, quota, streaming or feedback lands here once and
every channel inherits it.

``query_with_session_events`` is imported at module top **on purpose** so
tests can monkeypatch ``opsrag.channels.dispatcher.query_with_session_events``
with an async-generator stub (no real agent run in unit tests).

See design doc ``specs/002-channel-bots/design.md`` section 3.4.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from opsrag.agent.graph import query_with_session_events
from opsrag.channels.base import ChannelAdapter
from opsrag.channels.feedback import record_feedback
from opsrag.channels.permission import ChannelPermission
from opsrag.channels.streaming import (
    ERROR_TEXT,
    PLACEHOLDER_TEXT,
    ProgressStreamer,
)
from opsrag.channels.types import (
    AgentResult,
    FeedbackEvent,
    InboundMessage,
    ThreadMessage,
)

_log = logging.getLogger("opsrag.channels.dispatcher")

# Header for the serialized prior-thread block prepended to the query.
_THREAD_HEADER = "PRIOR THREAD MESSAGES:"
# Char budget for the assembled thread context (matches Slack's 2000).
_THREAD_MAX_CHARS = 2000


def _serialize_thread_context(
    messages: list[ThreadMessage],
    *,
    max_chars: int = _THREAD_MAX_CHARS,
    exclude_id: str | None = None,
) -> str:
    """Serialize prior (non-self) thread messages into the context block.

    Lifted verbatim (in spirit) from ``slack_bot/thread_context.py``:
      * drop our OWN past replies (``msg.is_self``) -- feedback-loop
        avoidance; OTHER bots' messages stay (they ARE the context).
      * drop the *triggering* message (``msg.source_id == exclude_id``) --
        the thread fetch returns the whole thread including the message the
        user just sent, whose text is already the primary query; without
        this it would appear twice (the old ``assemble_thread_context``
        skipped ``ts == current_ts`` for exactly this reason).
      * skip empty messages.
      * greedy-from-newest truncation: if the assembled block exceeds
        ``max_chars``, drop OLDEST messages first (recency wins). If the
        single newest line alone overflows, truncate it but keep something.

    Returns ``""`` when there is no usable context.
    """
    lines = [
        f"{m.author}: {m.text.strip()}"
        for m in messages
        if not m.is_self
        and (m.text or "").strip()
        and not (exclude_id is not None and m.source_id == exclude_id)
    ]
    if not lines:
        return ""

    selected_reverse: list[str] = []
    running = len(_THREAD_HEADER) + 1  # +1 for newline after header
    for line in reversed(lines):
        cost = len(line) + 1  # +1 for trailing newline
        if running + cost > max_chars and selected_reverse:
            break
        if running + cost > max_chars and not selected_reverse:
            # Newest line alone overflows -- truncate it but keep something
            # rather than returning nothing.
            keep = max(0, max_chars - running - 1)
            if keep <= 0:
                break
            selected_reverse.append(line[:keep] + "...")
            running += keep + 2
            break
        selected_reverse.append(line)
        running += cost

    if not selected_reverse:
        return ""

    selected = list(reversed(selected_reverse))
    return "\n".join([_THREAD_HEADER, *selected])


class ChannelDispatcher:
    """Routes neutral inbound events to the OpsRAG agent over a port.

    Holds the adapter + agent graph + providers + caches and implements
    the :class:`CoreSink`. One instance per channel worker; may be invoked
    concurrently by the adapter when bursts of events arrive -- all held
    state is immutable or has its own locking (``ChannelPermission``).
    """

    def __init__(
        self,
        *,
        adapter: ChannelAdapter,
        agent_graph: Any,
        providers: Any,
        permission: ChannelPermission,
        web_ui_base_url: str = "",
        thread_context_message_cap: int = 20,
        heartbeat_interval_s: float = 30.0,
        qa_cache: Any = None,
        investigation_cache: Any = None,
        semantic_router: Any = None,
        feedback_store: Any = None,
    ) -> None:
        self._adapter = adapter
        self._graph = agent_graph
        self._providers = providers
        self._permission = permission
        self._web_ui_base_url = (web_ui_base_url or "").rstrip("/")
        self._thread_cap = int(thread_context_message_cap)
        self._heartbeat_interval_s = float(heartbeat_interval_s)
        self._qa_cache = qa_cache
        self._investigation_cache = investigation_cache
        self._semantic_router = semantic_router
        self._feedback_store = feedback_store

    @property
    def channel_name(self) -> str:
        return getattr(self._adapter, "name", "channel")

    # ------------------------------------------------------------------
    # CoreSink: inbound message
    # ------------------------------------------------------------------
    async def on_message(self, msg: InboundMessage) -> None:
        """Reproduce ``handler.py::_dispatch`` over the port (7 stages)."""
        started = time.monotonic()
        channel = msg.channel_id
        user_id = msg.user_id
        is_dm = msg.is_dm

        # ---- 1. Permission check ----------------------------------------
        ok, deny_reason = await self._permission.allow(msg)
        if not ok:
            if deny_reason and user_id:
                # Privately tell the user why, so we don't pollute the
                # channel with denial noise.
                try:
                    await self._adapter.send_denial(msg, deny_reason)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("denial send failed user=%s err=%s", user_id, exc)
            _log.info(
                "deny channel=%s user=%s reason=%s",
                channel, user_id, deny_reason or "silent",
            )
            return

        # ---- 2. Extract the user query (already mention-stripped) --------
        user_query = (msg.text or "").strip()
        if not user_query:
            # Nothing to answer -- politely no-op rather than calling the
            # agent with an empty string.
            _log.info("empty query channel=%s", channel)
            return

        # ---- 3. Post placeholder + ACK reaction + heartbeat -------------
        reply_thread_id = None if is_dm else (msg.thread_id or msg.message_id)
        try:
            handle = await self._adapter.post_placeholder(
                channel, reply_thread_id, PLACEHOLDER_TEXT,
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("placeholder post failed channel=%s err=%s", channel, exc)
            return

        # Best-effort ACK on the original message.
        if msg.message_id and not is_dm:
            try:
                from opsrag.channels.types import ReactionKind
                await self._adapter.react(channel, msg.message_id, ReactionKind.ACK)
            except Exception:  # noqa: BLE001
                pass

        streamer = ProgressStreamer(
            self._adapter, handle,
            heartbeat_interval_s=self._heartbeat_interval_s,
        )
        streamer.start_heartbeat()

        # ---- 4. Thread context (mentions in a thread only) --------------
        thread_context = ""
        if msg.thread_id and not is_dm and msg.thread_id != msg.message_id:
            try:
                prior = await self._adapter.fetch_thread(
                    channel, msg.thread_id, cap=self._thread_cap,
                )
                # Drop the triggering message (its text is already the
                # primary query) so the question isn't duplicated.
                thread_context = _serialize_thread_context(
                    prior, exclude_id=msg.message_id,
                )
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "thread-context assembly failed channel=%s err=%s",
                    channel, exc,
                )
                thread_context = ""

        combined_query = (
            f"{thread_context}\n\n{user_query}" if thread_context else user_query
        )

        # ---- 5. Resolve identity ----------------------------------------
        try:
            current_user = await self._adapter.resolve_identity(msg)
        except Exception as exc:  # noqa: BLE001
            _log.warning("identity resolution failed: %s", exc)
            current_user = None

        # ---- 6. Run the agent (streaming events) ------------------------
        thread_id_for_session = self._session_thread_id(msg)
        user_oid = (
            getattr(current_user, "oid", None)
            or f"{self.channel_name}-bot-anon"
        )

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
                    agent_error = RuntimeError(ev.get("detail") or "agent error")
        except Exception as exc:  # noqa: BLE001
            agent_error = exc

        # ---- 7. Finalize ------------------------------------------------
        if agent_error is not None or final is None:
            err_kind = type(agent_error).__name__ if agent_error else "EmptyResult"
            _log.warning(
                "agent failed channel=%s user=%s err=%s duration_ms=%d",
                channel, user_id, err_kind,
                int((time.monotonic() - started) * 1000),
            )
            try:
                await streamer.finalize_text(f"{ERROR_TEXT} ({err_kind})")
            except Exception as exc:  # noqa: BLE001
                _log.warning("error-finalize failed: %s", exc)
            # Best-effort transition reaction to a "failed" indicator.
            if msg.message_id and not is_dm:
                try:
                    from opsrag.channels.types import ReactionKind
                    await self._adapter.react(
                        channel, msg.message_id, ReactionKind.ERROR,
                    )
                except Exception:  # noqa: BLE001
                    pass
            # NOTE: quota is NOT burned on error -- record_usage stays uncalled.
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

        result = AgentResult(
            answer=answer,
            sources=sources,
            diagram_present=diagram_present,
            session_id=session_id,
            investigation_id=final.get("investigation_id"),
        )

        try:
            await streamer.finalize_result(result)
        except Exception as exc:  # noqa: BLE001
            _log.warning("finalize failed channel=%s err=%s", channel, exc)

        # Successful run -- count it for quota purposes.
        if user_id:
            self._permission.record_usage(user_id)

        # Swap ACK -> DONE on the original message (best-effort).
        if msg.message_id and not is_dm:
            try:
                from opsrag.channels.types import ReactionKind
                await self._adapter.react(channel, msg.message_id, ReactionKind.DONE)
            except Exception:  # noqa: BLE001
                pass

        duration_ms = int((time.monotonic() - started) * 1000)
        _log.info(
            "ok channel=%s user=%s thread=%s duration_ms=%d sources=%d",
            channel, user_id, msg.thread_id, duration_ms, len(sources),
        )

    # ------------------------------------------------------------------
    # CoreSink: feedback
    # ------------------------------------------------------------------
    async def on_feedback(self, fb: FeedbackEvent) -> None:
        """Persist feedback, then ephemerally confirm to the clicker."""
        accepted = await record_feedback(
            fb,
            investigation_cache=self._investigation_cache,
            feedback_store=self._feedback_store,
            channel_name=self.channel_name,
        )
        if not accepted:
            return
        try:
            await self._adapter.confirm_feedback(fb, accepted=True)
        except Exception as exc:  # noqa: BLE001
            _log.warning("feedback confirm failed: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _session_thread_id(self, msg: InboundMessage) -> str:
        """Compose a stable session/thread id for the agent.

        Generalizes the Slack rule with a ``<channel>`` prefix so sessions
        stay disjoint across platforms:
          * DM -> ``"<ch>-dm:<channel>"`` (flat per-channel conversation)
          * threaded reply in an existing thread -> reuse the thread id
            so multi-turn questions in the same thread share a session
          * threaded reply that creates a new thread -> use the source
            message id
        """
        ch = self.channel_name
        if msg.is_dm:
            return f"{ch}-dm:{msg.channel_id}"
        anchor = msg.thread_id or msg.message_id or "no-id"
        return f"{ch}-thread:{msg.channel_id}:{anchor}"
