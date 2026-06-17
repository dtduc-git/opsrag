"""Memory loader node -- loads user preferences and frequent topics."""
from __future__ import annotations

import asyncio

from opsrag.interfaces.memory import MemoryStore
from opsrag.interfaces.observability import ObservabilityProvider


def load_memory_node(memory_store: MemoryStore, observability: ObservabilityProvider):
    async def _load(state: dict) -> dict:
        user_id = state.get("user_id", "anonymous")
        prefs: dict = {}
        context_parts: list[str] = []
        user_memories: list = []

        # These three reads are independent and previously ran serially, adding
        # up to three sequential round-trips (one of which embeds the query) to
        # the memory backend on the critical path of every turn. They are
        # concurrency-safe READ ops (get_all / search) against the same store,
        # so fan them out with asyncio.gather. The underlying MemoryStore
        # methods already swallow their own exceptions and return {} / [] on
        # failure; return_exceptions=True is a belt-and-braces guard so one
        # read raising can never sink the others. Each result is isinstance-
        # checked and falls back to the exact same empty default as the old
        # serial path, so the loaded state is byte-for-byte identical.
        q = state.get("query") or ""
        # Per-user conversational memory (Mem0): semantically recall durable
        # facts about this user relevant to the CURRENT query, so generation
        # feels like a continuous, personalized chat. Best-effort: a no-op for
        # non-semantic stores / empty memory, never blocks the turn. Namespace
        # ("user", user_id) is distinct from the ("user", user_id, "topics")
        # counters above.
        pref_res, topics_res, mems_res = await asyncio.gather(
            memory_store.get(("user", user_id, "preferences"), "default"),
            memory_store.search(("user", user_id, "topics"), limit=5),
            memory_store.search(("user", user_id), query=q, limit=6),
            return_exceptions=True,
        )

        # (1) preferences -- mirror old behaviour: only adopt .value when a
        # truthy Memory came back; any exception leaves prefs as {}.
        if pref_res and not isinstance(pref_res, BaseException):
            prefs = pref_res.value

        # (2) frequent topics -- only build the context line on a truthy,
        # non-exception list; otherwise no line is appended (same as before).
        if topics_res and not isinstance(topics_res, BaseException):
            topic_names = [t.key for t in topics_res]
            context_parts.append(f"Frequent topics: {', '.join(topic_names)}")

        # (3) query-relevant memories -- `mems or []` previously; preserve the
        # empty-list fallback on falsy result or exception.
        if mems_res and not isinstance(mems_res, BaseException):
            user_memories = mems_res
        else:
            user_memories = []

        return {
            "user_preferences": prefs,
            "session_context": " | ".join(context_parts) if context_parts else "",
            "user_memories": user_memories,
            "current_step": "memory_loaded",
        }

    return _load
