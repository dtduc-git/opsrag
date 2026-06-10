"""Placeholder reveal + reassurance heartbeat, channel-neutral.

This is ``SlackProgressStreamer`` with ``client.update_message`` replaced
by ``adapter.edit`` and a new ``finalize_result`` that hands the neutral
:class:`~opsrag.channels.types.AgentResult` to ``adapter.finalize`` (the
adapter renders it). The error path uses ``finalize_text`` which goes
through ``adapter.edit`` (plain text).

Heartbeat ladder (same phrases / cadence as Slack)::

    t=0   🤔 Thinking...                                  (initial post)
    t=30  😅 I'm still fetching the data...
    t=60  🙏 Sorry for the wait -- give me a bit more time...
    t=90  🐢 Still on it -- this one's a bit tricky...     (final phrase, holds)

The heartbeat **stops advancing** after the last phrase (re-saying
"thinking..." forever looks stuck). ``finalize_*`` is idempotent and
cancels the heartbeat task.

See design doc ``specs/002-channel-bots/design.md`` section 3.6.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-only typing
    from opsrag.channels.base import ChannelAdapter
    from opsrag.channels.types import AgentResult, MessageHandle

_log = logging.getLogger("opsrag.channels.streaming")

# The placeholder ladder. Index 0 is the initial post (the dispatcher
# writes it directly via ``post_placeholder``). The heartbeat rewrites to
# indices 1, 2, 3 at 30s, 60s, 90s. After index 3 we hold.
_HEARTBEAT_PHRASES: tuple[str, ...] = (
    "🤔 Thinking...",
    "😅 I'm still fetching the data...",
    "🙏 Sorry for the wait -- give me a bit more time...",
    "🐢 Still on it -- this one's a bit tricky...",
)

# The first phrase is what the dispatcher posts as the placeholder so the
# user sees something instantly.
PLACEHOLDER_TEXT: str = _HEARTBEAT_PHRASES[0]

# Warm, non-grovelly error message when the agent fails after the bot
# acknowledged the question. The dispatcher appends an ``(ExceptionName)``
# suffix for debuggability.
ERROR_TEXT: str = (
    "😔 Sorry, I couldn't put together an answer this time. "
    "Please try again, or rephrase if it keeps happening."
)


class ProgressStreamer:
    """Wraps a posted placeholder handle for heartbeat + finalize.

    Per-query lifecycle:
      * constructed by the dispatcher once the placeholder is posted
      * ``start_heartbeat()`` kicks off the background reassurance loop
      * ``finalize_result(result)`` / ``finalize_text(text)`` cancel the
        heartbeat and write the final answer (or apology) in one update
    """

    def __init__(
        self,
        adapter: ChannelAdapter,
        handle: MessageHandle,
        *,
        heartbeat_interval_s: float = 30.0,
    ) -> None:
        self._adapter = adapter
        self._handle = handle
        self._heartbeat_interval = float(heartbeat_interval_s)
        self._finalized = False
        self._heartbeat_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------
    def start_heartbeat(self) -> None:
        """Spawn the background reassurance task.

        Idempotent -- calling twice is a no-op. Safe to call after
        finalize (silently ignored).
        """
        if self._finalized or self._heartbeat_task is not None:
            return
        self._heartbeat_task = asyncio.create_task(
            self._run_heartbeat(),
            name=f"channel-heartbeat:{id(self._handle)}",
        )

    async def _run_heartbeat(self) -> None:
        """Walk the placeholder ladder one rung at a time.

        Stops walking after the last phrase. Cancellation by
        :meth:`stop_heartbeat` is the normal exit path; we swallow
        ``CancelledError`` so the caller's flow isn't interrupted.
        """
        try:
            for phrase in _HEARTBEAT_PHRASES[1:]:
                await asyncio.sleep(self._heartbeat_interval)
                if self._finalized:
                    return
                try:
                    await self._adapter.edit(self._handle, phrase)
                except Exception as exc:  # noqa: BLE001
                    # Don't let a single failed edit crash the heartbeat;
                    # the next tick or finalize() will likely succeed.
                    _log.debug("heartbeat edit failed err=%s", exc)
        except asyncio.CancelledError:
            pass

    async def stop_heartbeat(self) -> None:
        """Cancel the background heartbeat task and await its exit."""
        task = self._heartbeat_task
        self._heartbeat_task = None
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------
    async def finalize_result(self, result: AgentResult) -> None:
        """Stop the heartbeat and hand the rendered result to the adapter.

        Idempotent -- a duplicate finalize is ignored.
        """
        if self._finalized:
            _log.debug("ignoring duplicate finalize_result")
            return
        self._finalized = True
        await self.stop_heartbeat()
        await self._adapter.finalize(self._handle, result)

    async def finalize_text(self, text: str) -> None:
        """Stop the heartbeat and replace the placeholder with plain ``text``.

        Used for the error path (no ``AgentResult`` to render). Idempotent.
        """
        if self._finalized:
            _log.debug("ignoring duplicate finalize_text")
            return
        self._finalized = True
        await self.stop_heartbeat()
        await self._adapter.edit(self._handle, text)
