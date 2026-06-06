"""SlackSource -- bridges `SlackClient` to `SourceProtocol`.

Document model: one Slack THREAD = one Markdown document. Threads with
fewer than `min_replies_per_thread` replies are skipped to drop the
high-volume single-message noise (alert pings, "ok thanks", etc).

Backfill window: `backfill_days` controls how far back to walk on the
first run. Daily delta runs use `updated_since`; nothing before that
timestamp is fetched.

Privacy gate (defense in depth):
  1. Channel allowlist enforced here AND at config level.
  2. Bot messages can be skipped via `skip_bot_messages`.
  3. Token-shaped strings are redacted in `formatter.render_thread`.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

from opsrag.interfaces.source import DocRef, SourceDocument
from opsrag.sources.slack.client import (
    ChannelInfo,
    SlackClient,
    slack_ts_to_datetime,
)
from opsrag.sources.slack.formatter import render_thread

_log = logging.getLogger("opsrag.slack.source")


class SlackSource:
    """SourceProtocol implementation for Slack channel archives."""

    source_type = "slack"

    def __init__(
        self,
        client: SlackClient,
        *,
        channels_allowlist: list[str],
        backfill_days: int = 30,
        min_replies_per_thread: int = 1,
        skip_bot_messages: bool = True,
    ):
        self._client = client
        # Allowlist is stored as a frozen set so membership lookup is O(1)
        # and callers can't accidentally mutate it.
        self._allowlist = frozenset(channels_allowlist or [])
        self._backfill_days = max(1, backfill_days)
        self._min_replies = max(0, min_replies_per_thread)
        self._skip_bots = skip_bot_messages
        # channel_id -> ChannelInfo cache so list_documents and
        # fetch_document don't both re-query conversations.info.
        self._channel_cache: dict[str, ChannelInfo] = {}

    def _is_channel_allowed(self, channel_id: str) -> bool:
        if not self._allowlist:
            return False  # never index without an explicit allowlist
        return channel_id in self._allowlist

    async def _get_channel(self, channel_id: str) -> ChannelInfo | None:
        if channel_id in self._channel_cache:
            return self._channel_cache[channel_id]
        try:
            info = await self._client.get_channel(channel_id)
        except Exception as exc:
            _log.warning("slack: conversations.info(%s) failed: %s", channel_id, exc)
            return None
        self._channel_cache[channel_id] = info
        return info

    async def list_documents(
        self,
        scope: str,
        *,
        updated_since: datetime | None = None,
    ) -> AsyncIterator[DocRef]:
        """Yield a `DocRef` per qualifying thread in the channel `scope`.

        `scope` is a Slack channel ID (e.g. `CC448TKTQ`). On the first
        run `updated_since` is None and we walk back `backfill_days`.
        Daily delta runs pass the previous run's high-water timestamp.
        """
        if not self._is_channel_allowed(scope):
            _log.warning("slack: channel=%s not in allowlist -- skipping", scope)
            return

        info = await self._get_channel(scope)
        if info is None:
            return
        if not info.is_member:
            _log.warning(
                "slack: bot is not a member of #%s (id=%s) -- invite it first",
                info.name, scope,
            )
            return

        oldest = updated_since or (
            datetime.now(tz=UTC) - timedelta(days=self._backfill_days)
        )

        async for root in self._client.list_thread_roots(scope, oldest=oldest):
            if root.subtype in ("channel_join", "channel_leave", "channel_topic", "channel_purpose"):
                # Membership / topic events have no informational value.
                continue
            # Bot-rooted threads: skip ONLY if there are no human
            # replies. An alert that triggered a debugging thread is
            # high-value content -- the replies usually contain the
            # resolution. fetch_document filters the bot root itself
            # out of the rendered Markdown when skip_bots is on.
            is_bot_root = bool(root.bot_id) or root.subtype == "bot_message"
            if self._skip_bots and is_bot_root and root.reply_count == 0:
                continue
            # `reply_count` on the root reflects the whole thread. Use
            # this to apply the min-replies filter without paying for
            # `conversations.replies` on every single-message post.
            if root.reply_count < self._min_replies:
                continue
            thread_ts = root.thread_ts or root.ts
            yield DocRef(
                source_type=self.source_type,
                scope=scope,
                doc_id=f"{scope}:{thread_ts}",
            )

    async def fetch_document(self, ref: DocRef) -> SourceDocument:
        """Fetch the full thread (root + replies) and render to Markdown."""
        # doc_id = "<channel_id>:<thread_ts>"
        channel_id, thread_ts = ref.doc_id.split(":", 1)
        info = await self._get_channel(channel_id)
        channel_name = info.name if info else channel_id

        thread = await self._client.get_thread(channel_id, thread_ts)
        # Re-apply the bot filter at thread level -- a thread root could
        # be a human message but every reply is a bot. Compute effective
        # message count after bot filtering and skip if it's all noise.
        effective_msgs = [
            m for m in (thread.root, *thread.replies)
            if not (self._skip_bots and (m.bot_id or m.subtype == "bot_message"))
        ]
        if not effective_msgs or all(not (m.text or "").strip() for m in effective_msgs):
            content = ""  # empty content -> parser pipeline no-ops
        else:
            content = await render_thread(
                thread,
                channel_name=channel_name,
                resolve_user=self._client.resolve_user,
            )

        last_modified = slack_ts_to_datetime(thread.latest_ts)
        # `path` is what shows up in indexed_files dedup AND in the
        # vector store payload (`source_path`). Keep it human-readable.
        # `.md` suffix so the markdown parser's extension check matches.
        path = f"{channel_name}/thread-{thread_ts}.md"

        # Use channel_id (not name) in repo so it stays stable across
        # Slack channel renames AND matches the tracker key registered
        # by the route handler (which only knows the ID at request
        # time). Human-readable channel name is in `path` and metadata
        # for citation display.
        return SourceDocument(
            path=path,
            content=content,
            sha=thread.latest_ts,  # changes whenever a new reply lands
            last_modified=last_modified,
            repo=f"{self.source_type}:{channel_id}",
            branch=self.source_type,
            metadata={
                "source_type": self.source_type,
                "channel_id": channel_id,
                "channel_name": channel_name,
                "thread_ts": thread_ts,
                "message_count": len(thread.replies) + 1,
                "reply_count": len(thread.replies),
                "private": info.is_private if info else False,
            },
        )
