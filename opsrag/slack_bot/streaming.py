"""Placeholder reveal + reassurance heartbeat for the OpsRAG Slack chatbot.

Flow
----
1. Handler posts a Slack placeholder message ("🤔 Thinking...") and creates
   a :class:`SlackProgressStreamer` around it.
2. Handler calls :meth:`start_heartbeat` -- a background task wakes every
   30s and rewrites the placeholder to the next "buying time" phrase:

      t=0   🤔 Thinking...                                  (initial post)
      t=30  😅 I'm still fetching the data...
      t=60  🙏 Sorry for the wait -- give me a bit more time...
      t=90  🐢 Still on it -- this one's a bit tricky...     (final phrase, holds)

3. When the agent finishes (or errors), :meth:`finalize` cancels the
   heartbeat and rewrites the placeholder one last time with the actual
   answer (or apology).

Design choices
--------------
* The agent's internal stages (triage / tool_calling / reasoning /
  writing) are deliberately NOT surfaced to end users -- keeps the bot
  publishable outside the organization and avoids leaking detail.
* The heartbeat **stops advancing** after the last phrase; we don't
  loop, because re-saying "thinking..." forever looks like the bot is
  stuck even when it's making progress.
* Slack `chat.update` rate-limit is 50/min; a 30s heartbeat is well
  under budget even with many concurrent queries in one channel.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opsrag.slack_bot.client import SlackBotClient

_log = logging.getLogger("opsrag.slack_bot.streaming")

# The placeholder ladder. Index 0 is the initial post (the handler
# writes it directly when posting the placeholder). The heartbeat
# rewrites to indices 1, 2, 3 at 30s, 60s, 90s. After index 3 we hold.
_HEARTBEAT_PHRASES: tuple[str, ...] = (
    "🤔 Thinking...",
    "😅 I'm still fetching the data...",
    "🙏 Sorry for the wait -- give me a bit more time...",
    "🐢 Still on it -- this one's a bit tricky...",
)

# The first phrase is what the handler posts as the placeholder so the
# user sees something instantly.
PLACEHOLDER_TEXT: str = _HEARTBEAT_PHRASES[0]

# Warm, non-grovelly error message when the agent fails after the bot
# acknowledged the question. The handler appends an `(ExceptionName)`
# suffix for debuggability.
ERROR_TEXT: str = (
    "😔 Sorry, I couldn't put together an answer this time. "
    "Please try again, or rephrase if it keeps happening."
)


class SlackProgressStreamer:
    """Wraps the Slack placeholder for heartbeat + finalize.

    Per-query lifecycle:
      * constructed by the handler when the placeholder is posted
      * ``start_heartbeat()`` kicks off the background reassurance loop
      * ``finalize(text, blocks)`` cancels the heartbeat and writes the
        final answer (or apology) in one update
    """

    def __init__(
        self,
        client: SlackBotClient,
        channel: str,
        ts: str,
        *,
        min_update_interval_s: float = 1.5,
        heartbeat_interval_s: float = 30.0,
    ) -> None:
        # ``min_update_interval_s`` accepted for backward-compat with
        # the config wiring; not used here (no per-call rate limit
        # because the heartbeat already self-paces at 30s).
        del min_update_interval_s
        self._client = client
        self._channel = channel
        self._ts = ts
        self._heartbeat_interval = float(heartbeat_interval_s)
        self._finalized = False
        self._heartbeat_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------
    def start_heartbeat(self) -> None:
        """Spawn the background reassurance task.

        Idempotent -- calling twice is a no-op. Safe to call after
        finalize() (silently ignored).
        """
        if self._finalized or self._heartbeat_task is not None:
            return
        self._heartbeat_task = asyncio.create_task(
            self._run_heartbeat(),
            name=f"slack-heartbeat:{self._channel}:{self._ts}",
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
                    await self._client.update_message(
                        channel=self._channel, ts=self._ts, text=phrase,
                    )
                except Exception as exc:  # noqa: BLE001
                    # Don't let a single failed chat.update crash the
                    # heartbeat; the next tick or finalize() will
                    # likely succeed.
                    _log.debug(
                        "heartbeat update failed channel=%s ts=%s err=%s",
                        self._channel, self._ts, exc,
                    )
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
    async def finalize(self, text: str, blocks: list | None = None) -> None:
        """Stop the heartbeat and replace the placeholder with ``text``."""
        if self._finalized:
            _log.debug(
                "ignoring duplicate finalize channel=%s ts=%s",
                self._channel, self._ts,
            )
            return
        self._finalized = True
        await self.stop_heartbeat()
        await self._client.update_message(
            channel=self._channel, ts=self._ts, text=text, blocks=blocks,
        )
