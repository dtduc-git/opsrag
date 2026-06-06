"""In-memory session store -- dev and unit-test use only.

Uses LangGraph's InMemorySaver; sessions evaporate on process restart.
"""
from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver


class InMemorySessionStore:
    def __init__(self) -> None:
        self._saver = InMemorySaver()

    def get_checkpointer(self) -> Any:
        return self._saver

    async def list_sessions(self, user_id: str) -> list[dict]:
        # Mirror the postgres store: derive title/preview/timestamps/turns in
        # the same checkpoint walk (newest-first). See postgres.py for the
        # field-by-field rationale.
        sessions: dict[str, dict] = {}
        queries: dict[str, set[str]] = {}
        for cp_tuple in self._saver.list(None):
            cfg = cp_tuple.config.get("configurable", {})
            thread_id = cfg.get("thread_id")
            if not thread_id:
                continue
            if cfg.get("user_id") and cfg["user_id"] != user_id:
                continue
            entry = sessions.setdefault(
                thread_id,
                {
                    "thread_id": thread_id,
                    "user_id": cfg.get("user_id", user_id),
                    "checkpoint_count": 0,
                    "title": None,
                    "preview": None,
                    "updated_at": None,
                    "created_at": None,
                    "turn_count": 0,
                },
            )
            entry["checkpoint_count"] += 1
            values = cp_tuple.checkpoint.get("channel_values") or {}
            q = (values.get("query") or "").strip()
            gen = (values.get("generation") or "").strip()
            ts = cp_tuple.checkpoint.get("ts")
            if ts:
                if entry["updated_at"] is None:
                    entry["updated_at"] = ts
                entry["created_at"] = ts
            if q:
                entry["title"] = q
                queries.setdefault(thread_id, set()).add(q)
            if gen and entry["preview"] is None:
                entry["preview"] = gen[:160]
        for tid, entry in sessions.items():
            entry["turn_count"] = len(queries.get(tid, ()))
            if entry["preview"] is None and entry["title"]:
                entry["preview"] = entry["title"][:160]
        return list(sessions.values())

    async def delete_session(self, thread_id: str) -> bool:
        storage = getattr(self._saver, "storage", None)
        if storage is None:
            return False
        if thread_id in storage:
            del storage[thread_id]
            return True
        return False

    async def get_session_metadata(self, thread_id: str) -> dict | None:
        cp = self._saver.get({"configurable": {"thread_id": thread_id}})
        if cp is None:
            return None
        return {"thread_id": thread_id, "has_checkpoint": True}

    async def get_messages(self, thread_id: str) -> list[dict]:
        """Same dedupe-by-(query,generation) replay as the postgres impl."""
        seen: dict[tuple[str, str], dict] = {}
        order: list[tuple[str, str]] = []
        config = {"configurable": {"thread_id": thread_id}}
        for cp_tuple in self._saver.list(config):
            values = cp_tuple.checkpoint.get("channel_values") or {}
            query = (values.get("query") or "").strip()
            generation = (values.get("generation") or "").strip()
            if not query or not generation:
                continue
            key = (query, generation)
            if key in seen:
                continue
            sources = []
            for chunk in values.get("final_chunks") or values.get("graded_chunks") or []:
                src = chunk.get("source_path") if isinstance(chunk, dict) else getattr(chunk, "source_path", None)
                repo = chunk.get("repo") if isinstance(chunk, dict) else getattr(chunk, "repo", None)
                if src:
                    label = f"{repo}:{src}" if repo else src
                    if label not in sources:
                        sources.append(label)
            seen[key] = {
                "query": query,
                "generation": generation,
                "sources": sources[:8],
                "grounded": bool(values.get("generation_grounded")),
                "query_type": values.get("query_type"),
            }
            order.append(key)
        messages: list[dict] = []
        for key in reversed(order):
            entry = seen[key]
            messages.append({"role": "user", "content": entry["query"]})
            messages.append({
                "role": "assistant",
                "content": entry["generation"],
                "sources": entry["sources"],
                "grounded": entry["grounded"],
                "query_type": entry["query_type"],
            })
        return messages
