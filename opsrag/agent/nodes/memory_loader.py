"""Memory loader node -- loads user preferences and frequent topics."""
from __future__ import annotations

from opsrag.interfaces.memory import MemoryStore
from opsrag.interfaces.observability import ObservabilityProvider


def load_memory_node(memory_store: MemoryStore, observability: ObservabilityProvider):
    async def _load(state: dict) -> dict:
        user_id = state.get("user_id", "anonymous")
        prefs: dict = {}
        context_parts: list[str] = []
        user_memories: list = []

        try:
            pref = await memory_store.get(("user", user_id, "preferences"), "default")
            if pref:
                prefs = pref.value

            topics = await memory_store.search(("user", user_id, "topics"), limit=5)
            if topics:
                topic_names = [t.key for t in topics]
                context_parts.append(f"Frequent topics: {', '.join(topic_names)}")
        except Exception:
            pass

        # Per-user conversational memory (Mem0): semantically recall durable
        # facts about this user relevant to the CURRENT query, so generation
        # feels like a continuous, personalized chat. Best-effort: a no-op for
        # non-semantic stores / empty memory, never blocks the turn. Namespace
        # ("user", user_id) is distinct from the ("user", user_id, "topics")
        # counters above.
        try:
            q = state.get("query") or ""
            mems = await memory_store.search(("user", user_id), query=q, limit=6)
            user_memories = mems or []
        except Exception:
            user_memories = []

        return {
            "user_preferences": prefs,
            "session_context": " | ".join(context_parts) if context_parts else "",
            "user_memories": user_memories,
            "current_step": "memory_loaded",
        }

    return _load
