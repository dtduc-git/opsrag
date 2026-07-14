"""Self-contained Slack first-responder for auto-answering #ops requests.

This pipeline is DELIBERATELY independent of the channel-neutral
``ChannelDispatcher.on_message`` so the shared @-mention / DM path stays
behaviorally identical: the first-responder owns its own gate, quota (with a
synthetic principal for the userless workflow bot), confidence label, and
always-on on-call cc -- none of which the shared render carries.

Order matters. Cheap pure predicates (feature-gate -> subtype -> self-guard
-> classify -> dedup -> quota) run BEFORE any agent/LLM call so the ~90% of
#ops bot noise is rejected without cost.

Posting contract (Phase-1 polish): TWO independent in-thread posts -- an
ack/greeting first, then a SEPARATE answer post. The answer is NOT an edit of
the ack. Loop-safety holds because both self-posts are dropped by _is_self AND
by classify's allowlist backstop (our own app_id/bot_id are never allowlisted).
"""
from __future__ import annotations

import logging
import re
from collections import deque
from typing import Any

from opsrag.agent.graph import query_with_session_events
from opsrag.channels.config import FirstResponderConfig
from opsrag.channels.permission import ChannelPermission
from opsrag.slack_bot.reply_format import build_reply, derive_confidence
from opsrag.slack_bot.request_extract import (
    RequestKind,
    classify,
    extract_query,
    extract_requester,
    is_ignorable_subtype,
)
from opsrag.slack_bot.slack_text import normalize_slack_text, slack_ids

_log = logging.getLogger("opsrag.slack_bot.first_responder")

# Cap distinct users.info lookups per message (a DIRECT human post can carry
# many <@U..> mentions -> serial, first-sight, uncached calls). Unresolved
# ids degrade to bare-id text; bounding protects the Slack rate budget.
_MAX_MENTION_LOOKUPS = 15

_SUBTEAM_ID_RE = re.compile(r"^<!subteam\^(S[A-Z0-9]+)>$")
_BARE_SUBTEAM_ID_RE = re.compile(r"^(S[A-Z0-9]+)$")


class FirstResponder:
    """Auto-answers mapped channels' workflow + direct requests, self-contained."""

    def __init__(
        self,
        *,
        graph: Any,
        providers: Any,
        permission: ChannelPermission,
        config: FirstResponderConfig,
        qa_cache: Any = None,
        investigation_cache: Any = None,
        semantic_router: Any = None,
        web_ui_base_url: str = "",
        dedup_cap: int = 512,
    ) -> None:
        self._graph = graph
        self._providers = providers
        self._permission = permission
        self._config = config
        self._qa_cache = qa_cache
        self._investigation_cache = investigation_cache
        self._semantic_router = semantic_router
        self._web_ui_base_url = (web_ui_base_url or "").rstrip("/")
        self._client: Any = None
        # Bounded dedup of "<channel>:<ts>" already handled.
        self._seen_keys: set[str] = set()
        self._seen_order: deque[str] = deque()
        self._dedup_cap = int(dedup_cap)

    def bind_client(self, client: Any) -> None:
        """Attach the live SlackBotClient (built in adapter.connect). Identity
        (self_bot_id/self_user_id) is read LIVE in _is_self, because the client
        only learns its own ids inside start(), which runs AFTER connect binds."""
        self._client = client

    async def on_channel_message(self, event: dict[str, Any]) -> None:
        """Full first-responder pipeline for one top-level channel message."""
        if self._client is None or not self._config.enabled:
            return
        event = event or {}
        channel = event.get("channel") or ""
        chan_cfg = self._config.channels.get(channel)
        if chan_cfg is None:
            return  # channel not mapped -- the channel gate

        # 1. Subtype filter FIRST (drops OpsRAG's own message_changed edits).
        if is_ignorable_subtype(event):
            return

        # 2. Only top-level messages (a thread reply is not a new request).
        ts = event.get("ts") or ""
        thread_ts = event.get("thread_ts")
        if not ts or (thread_ts and thread_ts != ts):
            return

        # 3. Best-effort self-guard (top-level AND nested under event.message).
        if self._is_self(event):
            return

        # 4. Classify (durable self-loop backstop: our app_id is not allowlisted).
        kind = classify(event, chan_cfg)
        if kind is RequestKind.IGNORE:
            return

        # 5. Dedup by channel:ts.
        key = f"{channel}:{ts}"
        if key in self._seen_keys:
            return
        self._remember(key)

        # 6. Quota against a stable principal (workflow bot carries empty user).
        if kind is RequestKind.WORKFLOW:
            principal = f"slack-wf:{event.get('app_id') or event.get('bot_id') or 'unknown'}"
        else:
            principal = event.get("user") or "slack-fr-anon"
        if self._permission.usage_count(principal) >= int(chan_cfg.daily_quota):
            _log.info("first-responder quota hit principal=%s channel=%s", principal, channel)
            return

        # 7. Extract the prompt.
        query = extract_query(event)
        if not query:
            return

        # 8. ACK first -- its own in-thread post. Fast: uses only the raw mention
        #    (no users.info round-trip); Slack resolves the name at render time.
        thread_ts_reply = ts if chan_cfg.reply_in_thread else None
        requester_id = extract_requester(event)
        ack = self._ack_text(requester_id, chan_cfg.agent_name)
        try:
            await self._client.post_message(
                channel=channel, text=ack, thread_ts=thread_ts_reply,
            )
        except Exception as exc:  # noqa: BLE001 -- hard-abort, mirrors old placeholder
            _log.warning("first-responder ack failed channel=%s err=%s", channel, exc)
            return  # no agent run, no quota

        # 9. Resolve sender identity for the web UI (fixes "You"), and build the
        #    id->name maps for query normalization. All get_user_info calls share
        #    the client's FIFO cache; failures degrade gracefully.
        user_name, user_email = await self._resolve_sender(event, requester_id)
        uids, _sids, _cids = slack_ids(query)
        user_names = await self._resolve_user_names(uids)
        subteam_names = self._subteam_map(chan_cfg)
        # channel_names={} in Phase 1: pipe-labelled <#C..|name> already renders
        # the name; a bare <#C..> degrades to "#C..".
        clean_query = normalize_slack_text(
            query,
            user_names=user_names,
            subteam_names=subteam_names,
            channel_names={},
        )

        # 10. Run the agent on the CLEAN query (agent + session store + web-UI
        #     transcript all see readable text). Identity/quota unchanged.
        final = await self._run_agent(
            clean_query, channel, ts, principal,
            user_name=user_name, user_email=user_email,
        )

        # 11. Answer as a SEPARATE second post (NOT an edit of the ack).
        if final is None:
            await self._safe_post(
                channel,
                text=self._error_text(chan_cfg.oncall_handle),
                blocks=None,
                thread_ts=thread_ts_reply,
            )
            return  # no quota on error

        confidence = derive_confidence(final)
        text, blocks = build_reply(
            answer=final.get("answer") or "",
            sources=final.get("sources") or [],
            confidence=confidence,
            oncall_handle=chan_cfg.oncall_handle,
            diagram_present=bool(
                final.get("diagram") or final.get("diagram_present") or final.get("has_diagram")
            ),
            web_ui_base_url=self._web_ui_base_url,
            session_id=final.get("thread_id") or final.get("session_id"),
            investigation_id=final.get("investigation_id"),
        )
        await self._safe_post(channel, text=text, blocks=blocks, thread_ts=thread_ts_reply)

        # 12. Record quota only after a successful answer.
        self._permission.record_usage(principal)
        _log.info(
            "first-responder answered channel=%s ts=%s kind=%s confidence=%s",
            channel, ts, kind.value, confidence.label,
        )

    # ------------------------------------------------------------------
    async def _run_agent(
        self, query: str, channel: str, ts: str, principal: str,
        *, user_name: str | None = None, user_email: str | None = None,
    ) -> dict[str, Any] | None:
        thread_id = f"slack-thread:{channel}:{ts}"  # session key; distinct from dedup key & post ts
        final: dict[str, Any] | None = None
        try:
            async for ev in query_with_session_events(
                self._graph,
                query=query,
                user_id=principal,
                thread_id=thread_id,
                embedder=getattr(self._providers, "embedder", None),
                qa_cache=self._qa_cache,
                llm=getattr(self._providers, "llm", None),
                session_store=getattr(self._providers, "session_store", None),
                investigation_cache=self._investigation_cache,
                semantic_router=self._semantic_router,
                user_email=user_email,
                user_name=user_name,
            ):
                if ev.get("type") == "final":
                    final = ev
        except Exception as exc:  # noqa: BLE001
            _log.warning("first-responder agent failed channel=%s err=%s", channel, exc)
            return None
        return final

    def _is_self(self, event: dict[str, Any]) -> bool:
        self_bot_id = getattr(self._client, "self_bot_id", None)
        self_user_id = getattr(self._client, "self_user_id", None)
        nested = event.get("message") or {}
        for src in (event, nested):
            if self_bot_id and src.get("bot_id") == self_bot_id:
                return True
            if self_user_id and src.get("user") == self_user_id:
                return True
        return False

    def _remember(self, key: str) -> None:
        self._seen_keys.add(key)
        self._seen_order.append(key)
        while len(self._seen_order) > self._dedup_cap:
            old = self._seen_order.popleft()
            self._seen_keys.discard(old)

    @staticmethod
    def _ack_text(requester_id: str | None, agent_name: str) -> str:
        greeting = f"Hi <@{requester_id}>," if requester_id else "Hi,"
        return (
            f"👋 {greeting} I am *{agent_name}* — starting the "
            "investigation, this may take ~2-3 min."
        )

    async def _resolve_sender(
        self, event: dict[str, Any], requester_id: str | None,
    ) -> tuple[str | None, str | None]:
        """(user_name, user_email) for the web-UI sender. Resolved real name,
        else the workflow's friendly 'username', else None. Never leaks a bare
        U… id as a name."""
        name: str | None = None
        email: str | None = None
        if requester_id:
            info = await self._safe_user_info(requester_id)
            name = self._name_from_info(info)
            profile = info.get("profile") or {}
            email = (profile.get("email") or "").strip() or None  # scope-gated; may be absent
        if not name:
            name = (event.get("username") or "").strip() or None
        return name, email

    async def _resolve_user_names(self, ids: set[str]) -> dict[str, str]:
        """Resolve up to _MAX_MENTION_LOOKUPS ids -> display name. Sorted for
        determinism; unresolved ids are simply omitted (normalizer keeps them
        as bare '@<id>')."""
        out: dict[str, str] = {}
        for uid in sorted(ids)[:_MAX_MENTION_LOOKUPS]:
            info = await self._safe_user_info(uid)
            nm = self._name_from_info(info)
            if nm:
                out[uid] = nm
        return out

    async def _safe_user_info(self, uid: str) -> dict:
        try:
            return await self._client.get_user_info(uid) or {}
        except Exception as exc:  # noqa: BLE001
            _log.debug("get_user_info(%s) failed: %s", uid, exc)
            return {}

    @staticmethod
    def _name_from_info(info: dict) -> str | None:
        profile = info.get("profile") or {}
        nm = (
            profile.get("display_name")
            or profile.get("real_name")
            or info.get("real_name")
            or info.get("name")
            or ""
        ).strip()
        return nm or None

    @staticmethod
    def _subteam_map(chan_cfg: Any) -> dict[str, str]:
        """{ oncall S-id : oncall_display } (falls back to 'on-call')."""
        h = (chan_cfg.oncall_handle or "").strip()
        m = _SUBTEAM_ID_RE.match(h) or _BARE_SUBTEAM_ID_RE.match(h)
        if not m:
            return {}
        disp = (getattr(chan_cfg, "oncall_display", "") or "").strip() or "on-call"
        return {m.group(1): disp}

    @staticmethod
    def _error_text(oncall_handle: str) -> str:
        from opsrag.slack_bot.reply_format import normalize_oncall_handle

        cc = normalize_oncall_handle(oncall_handle)
        base = "😔 I hit an error putting this together."
        return f"{base} cc {cc}" if cc else base

    async def _safe_post(
        self, channel: str, *, text: str, blocks: list | None, thread_ts: str | None,
    ) -> None:
        """Best-effort new post (replaces _safe_update). Swallows exceptions."""
        try:
            await self._client.post_message(
                channel=channel, text=text, thread_ts=thread_ts, blocks=blocks,
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("first-responder answer post failed channel=%s err=%s", channel, exc)
