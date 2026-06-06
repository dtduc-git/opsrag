"""PostgreSQL-backed session store via langgraph-checkpoint-postgres.

The underlying checkpointer requires an explicit ``setup()`` call the first
time it runs against a fresh database so it can create its tables.
"""
from __future__ import annotations

from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool


class PostgresSessionStore:
    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 10):
        self._dsn = dsn
        self._pool = AsyncConnectionPool(
            conninfo=dsn,
            min_size=min_size,
            max_size=max_size,
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        self._saver: AsyncPostgresSaver | None = None
        self._setup_done = False

    async def open(self) -> None:
        await self._pool.open()
        self._saver = AsyncPostgresSaver(self._pool)
        if not self._setup_done:
            await self._saver.setup()
            self._setup_done = True

    async def close(self) -> None:
        await self._pool.close()

    def get_checkpointer(self) -> Any:
        if self._saver is None:
            raise RuntimeError("PostgresSessionStore.open() must be awaited first")
        return self._saver

    async def list_sessions(self, user_id: str) -> list[dict]:
        if self._saver is None:
            return []
        # ``alist(None)`` yields every checkpoint across all threads,
        # NEWEST-FIRST. In this single pass we also derive, per thread:
        #   - title:      the FIRST user question (oldest query -> the last
        #                 one we overwrite, since we iterate newest-first).
        #   - preview:    the MOST RECENT answer (first generation we see).
        #   - updated_at: newest checkpoint ts (first ts seen).
        #   - created_at: oldest checkpoint ts (last ts overwritten).
        #   - turn_count: number of distinct user questions.
        # No extra DB round-trips -- it's the same walk that already counted
        # checkpoints, so the list endpoint stays a single query.
        seen: dict[str, dict] = {}
        queries: dict[str, set[str]] = {}
        async for cp_tuple in self._saver.alist(None):
            cfg = cp_tuple.config.get("configurable", {})
            thread_id = cfg.get("thread_id")
            if not thread_id:
                continue
            if cfg.get("user_id") and cfg["user_id"] != user_id:
                continue
            entry = seen.setdefault(
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
                    entry["updated_at"] = ts  # first seen = newest
                entry["created_at"] = ts       # keep overwriting -> oldest
            if q:
                entry["title"] = q             # keep overwriting -> first message
                queries.setdefault(thread_id, set()).add(q)
            if gen and entry["preview"] is None:
                entry["preview"] = gen[:160]   # first seen = most recent answer
        for tid, entry in seen.items():
            entry["turn_count"] = len(queries.get(tid, ()))
            # Fall back to the (oldest) question for the preview if no answer
            # was captured yet (e.g. a turn still streaming).
            if entry["preview"] is None and entry["title"]:
                entry["preview"] = entry["title"][:160]
        return list(seen.values())

    async def delete_session(self, thread_id: str) -> bool:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM checkpoints WHERE thread_id = %s",
                    (thread_id,),
                )
                await cur.execute(
                    "DELETE FROM checkpoint_blobs WHERE thread_id = %s",
                    (thread_id,),
                )
                await cur.execute(
                    "DELETE FROM checkpoint_writes WHERE thread_id = %s",
                    (thread_id,),
                )
        return True

    async def get_session_metadata(self, thread_id: str) -> dict | None:
        if self._saver is None:
            return None
        cp = await self._saver.aget({"configurable": {"thread_id": thread_id}})
        if cp is None:
            return None
        return {"thread_id": thread_id, "has_checkpoint": True}

    async def get_messages(self, thread_id: str) -> list[dict]:
        """Replay a session's chat history from the LangGraph checkpoints.

        Each ``ainvoke`` for a thread produces many intermediate checkpoints
        (one per node). We dedupe on (query, generation) so we end up with
        one (user, assistant) pair per turn, ordered oldest-first.

        Also returns the original author's email/name (from the checkpoint
        config) attached to every user-role message, so the UI can show the
        teammate's name instead of a generic "You" when a session is
        replayed by a different viewer.
        """
        if self._saver is None:
            return []

        # Walk every checkpoint for this thread. ``alist`` yields newest-first;
        # we collect into ``seen`` to keep the LAST occurrence of each pair
        # (which carries the final ``sources``/``grounded`` flags) and then
        # reverse for chronological display.
        config = {"configurable": {"thread_id": thread_id}}
        seen: dict[tuple[str, str], dict] = {}
        order: list[tuple[str, str]] = []
        # Author attribution: LangGraph's saver stores user-defined
        # configurable keys in ``cp_tuple.metadata`` -- NOT in
        # ``cp_tuple.config.configurable`` (which only carries canonical
        # keys: thread_id, checkpoint_id, checkpoint_ns). Read both as
        # belt-and-suspenders so future LangGraph versions that *do*
        # rehydrate custom keys won't surprise us.
        author_email: str | None = None
        author_name: str | None = None

        async for cp_tuple in self._saver.alist(config):
            cfg = cp_tuple.config.get("configurable", {}) or {}
            md = cp_tuple.metadata or {}
            if not author_email:
                author_email = md.get("user_email") or cfg.get("user_email")
            if not author_name:
                author_name = md.get("user_name") or cfg.get("user_name")
            values = cp_tuple.checkpoint.get("channel_values") or {}
            query = (values.get("query") or "").strip()
            generation = (values.get("generation") or "").strip()
            if not query or not generation:
                continue
            key = (query, generation)
            if key in seen:
                continue
            # Build a UI-shaped message pair. Sources/grounded come from
            # whatever final-stage state was captured in this checkpoint.
            sources = []
            for chunk in values.get("final_chunks") or values.get("graded_chunks") or []:
                # Chunks may be dicts (after JSON deser) or Chunk objects.
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
                "ts": cp_tuple.checkpoint.get("ts"),
            }
            order.append(key)

        # ``alist`` is newest-first -> reverse for chronological replay.
        messages: list[dict] = []
        for key in reversed(order):
            entry = seen[key]
            messages.append({
                "role": "user",
                "content": entry["query"],
                # Same author for every user turn in a thread (current
                # architecture -- one thread = one owner). UI compares to
                # `me.email` to decide "You" vs the actual author name.
                "author_email": author_email,
                "author_name": author_name,
                "ts": entry["ts"],
            })
            messages.append({
                "role": "assistant",
                "content": entry["generation"],
                "sources": entry["sources"],
                "grounded": entry["grounded"],
                "query_type": entry["query_type"],
                "ts": entry["ts"],
            })
        return messages
