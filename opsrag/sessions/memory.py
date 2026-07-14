"""In-memory session store -- dev and unit-test use only.

Uses LangGraph's InMemorySaver; sessions evaporate on process restart.
"""
from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from opsrag.sessions.replay_components import rebuild_rich_components


class InMemorySessionStore:
    def __init__(self) -> None:
        self._saver = InMemorySaver()

    def get_checkpointer(self) -> Any:
        return self._saver

    def _collect_sessions(self, keep) -> list[dict]:
        # Mirror the postgres store: derive title/preview/timestamps/turns in
        # the same checkpoint walk (newest-first). ``keep(thread_id, owner)``
        # decides inclusion. See postgres.py for the field-by-field rationale.
        sessions: dict[str, dict] = {}
        queries: dict[str, set[str]] = {}
        for cp_tuple in self._saver.list(None):
            cfg = cp_tuple.config.get("configurable", {})
            thread_id = cfg.get("thread_id")
            if not thread_id:
                continue
            owner = cfg.get("user_id")
            if owner is None:
                owner = (cp_tuple.metadata or {}).get("user_id")
            if not keep(thread_id, owner):
                continue
            entry = sessions.setdefault(
                thread_id,
                {
                    "thread_id": thread_id,
                    "user_id": owner if owner is not None else "anonymous",
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

    async def list_sessions(
        self, user_id: str, *, include_all: bool = False
    ) -> list[dict]:
        return self._collect_sessions(
            lambda _tid, owner: include_all or owner == user_id
        )

    async def list_sessions_by_prefixes(
        self, prefixes: tuple[str, ...]
    ) -> list[dict]:
        """List sessions whose ``thread_id`` starts with any of ``prefixes``
        (owner-agnostic). See PostgresSessionStore for the rationale."""
        pfx = tuple(prefixes)
        if not pfx:
            return []
        return self._collect_sessions(lambda tid, _owner: tid.startswith(pfx))

    async def delete_session(self, thread_id: str) -> bool:
        storage = getattr(self._saver, "storage", None)
        if storage is None:
            return False
        if thread_id in storage:
            del storage[thread_id]
            return True
        return False

    async def get_session_owner(self, thread_id: str) -> str | None:
        """Return the recorded owner (checkpoint-metadata ``user_id``) for a
        thread, or None if the thread has no checkpoints. Mirrors how
        ``list_sessions`` reads ``cfg.get("user_id")``."""
        config = {"configurable": {"thread_id": thread_id}}
        for cp_tuple in self._saver.list(config):
            cfg = cp_tuple.config.get("configurable", {}) or {}
            owner = cfg.get("user_id")
            if owner is None:
                owner = (cp_tuple.metadata or {}).get("user_id")
            return owner
        return None

    async def get_session_metadata(self, thread_id: str) -> dict | None:
        cp = self._saver.get({"configurable": {"thread_id": thread_id}})
        if cp is None:
            return None
        return {"thread_id": thread_id, "has_checkpoint": True}

    async def get_messages(self, thread_id: str) -> list[dict]:
        """Collapse each contiguous same-query run to one turn (see the postgres
        impl for the rationale -- a turn writes several ``generation`` values, so
        a (query, generation) dedup duplicated messages)."""
        turns: list[dict] = []
        prev_query: str | None = None
        config = {"configurable": {"thread_id": thread_id}}
        for cp_tuple in self._saver.list(config):
            values = cp_tuple.checkpoint.get("channel_values") or {}
            query = (values.get("query") or "").strip()
            generation = (values.get("generation") or "").strip()
            if not query or not generation:
                continue
            # New turn iff the query changed from the previous qualifying
            # checkpoint (same query = same contiguous turn, already captured).
            if query == prev_query:
                continue
            prev_query = query
            sources = []
            for chunk in values.get("final_chunks") or values.get("graded_chunks") or []:
                src = chunk.get("source_path") if isinstance(chunk, dict) else getattr(chunk, "source_path", None)
                repo = chunk.get("repo") if isinstance(chunk, dict) else getattr(chunk, "repo", None)
                if src:
                    label = f"{repo}:{src}" if repo else src
                    if label not in sources:
                        sources.append(label)
            turns.append({
                "query": query,
                "generation": generation,
                "sources": sources[:8],
                "grounded": bool(values.get("generation_grounded")),
                "query_type": values.get("query_type"),
                # Source data for charts/plan -- rebuilt below so they survive
                # replay instead of vanishing after the live stream (mirrors the
                # postgres store).
                "tool_message_history": values.get("tool_message_history") or [],
                "plan": values.get("plan") or [],
            })
        messages: list[dict] = []
        for entry in reversed(turns):
            messages.append({"role": "user", "content": entry["query"]})
            assistant_msg = {
                "role": "assistant",
                "content": entry["generation"],
                "sources": entry["sources"],
                "grounded": entry["grounded"],
                "query_type": entry["query_type"],
            }
            rich = rebuild_rich_components(
                entry.get("tool_message_history"), entry.get("plan")
            )
            if rich:
                assistant_msg["rich_components"] = rich
            messages.append(assistant_msg)
        return messages
