"""Memory saver node -- persists interaction patterns after generation.

The actual writes run FIRE-AND-FORGET (off the response critical path): with
the Mem0 backend each write is an LLM `infer` fact-extraction call (~1-2s), and
the answer is already produced by the time this node runs, so blocking on it
just adds latency the user feels for no benefit. We capture the needed values
synchronously, schedule the writes as a background task, and return
immediately. Best-effort throughout -- a write failure never affects the turn.

Trade-off: a follow-up sent within ~1-2s of the previous answer might not yet
see the just-stored fact. In practice the user spends longer than that reading
the answer before typing again, so the window is negligible.
"""
from __future__ import annotations

import asyncio
import logging

from opsrag.interfaces.memory import MemoryStore
from opsrag.interfaces.observability import ObservabilityProvider

_log = logging.getLogger("opsrag.agent.memory_saver")

# Hold references to in-flight background writes so they aren't garbage-
# collected mid-flight; drop each on completion. Module-level + shared across
# requests (single event loop -> set ops are safe).
_bg_tasks: set = set()


def _fire_and_forget(coro) -> None:
    try:
        task = asyncio.create_task(coro)
    except RuntimeError:
        # No running loop (shouldn't happen inside the async graph) -- run
        # inline as a fallback so the write isn't silently dropped.
        return
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


def save_memory_node(memory_store: MemoryStore, observability: ObservabilityProvider):
    async def _persist(
        user_id: str,
        query: str,
        generation: str,
        graded: list,
        query_type,
        thread_id,
    ) -> None:
        # Per-user conversational memory FIRST -- this is the recall-critical
        # write (the durable fact a follow-up turn looks up). On Mem0 each write
        # is a ~1-2s `infer` call, so doing this before the topic counters means
        # the fact is queryable quickly instead of queued behind them.
        # Store ONLY the user's own message. Mem0ServiceMemory.put flattens the
        # value into a single user-role message, so including the assistant's
        # answer made mem0 extract facts FROM THE ANSWER as if the user stated
        # them (e.g. a Slack-command tip stored as a "fact about the user").
        # Durable user facts come from what the USER says; a question turn
        # ("which service do I own?") simply yields no new fact.
        try:
            if query and user_id and user_id != "anonymous":
                await memory_store.put(
                    ("user", user_id),
                    thread_id or "conversation",
                    {"user_message": query},
                )
        except Exception as exc:  # noqa: BLE001
            _log.debug("conversational memory write failed: %s", exc)

        # Frequent-topic + query-type counters (per-user, legacy/best-effort).
        # On the Mem0 backend these also route through fact-extraction -- low
        # value there, so they run last and never delay the recall-critical fact.
        try:
            for chunk in graded:
                svc = chunk.metadata.get("service") or chunk.metadata.get("title", "")
                if not svc:
                    continue
                existing = await memory_store.get(("user", user_id, "topics"), svc)
                count = (existing.value.get("count", 0) if existing else 0) + 1
                await memory_store.put(
                    ("user", user_id, "topics"),
                    svc,
                    {"count": count, "last_query": query},
                )
            if query_type:
                existing_qt = await memory_store.get(
                    ("user", user_id, "query_types"), query_type
                )
                qt_count = (existing_qt.value.get("count", 0) if existing_qt else 0) + 1
                await memory_store.put(
                    ("user", user_id, "query_types"),
                    query_type,
                    {"count": qt_count},
                )
        except Exception as exc:  # noqa: BLE001
            _log.debug("topic/query-type memory write failed: %s", exc)

    async def _save(state: dict) -> dict:
        # Capture values now (state may be reused after the node returns), then
        # schedule the writes in the background so the response isn't blocked.
        _fire_and_forget(
            _persist(
                state.get("user_id", "anonymous"),
                state.get("query", ""),
                state.get("generation", ""),
                list(state.get("graded_chunks") or []),
                state.get("query_type"),
                state.get("thread_id"),
            )
        )
        return {"current_step": "memory_saved"}

    return _save
