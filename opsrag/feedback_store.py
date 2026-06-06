"""Postgres-backed persistence for thumbs-up/down user feedback.

The thumbs-up/down buttons in the chat UI hit
`POST /investigation/{id}/feedback`; that handler still updates the
Qdrant-side investigation cache so cache audit / quality
filters keep working. In ADDITION we now write a normalized row into
Postgres `opsrag_feedback` so SREs can later query

    SELECT created_at, query_snippet, answer_snippet, note
    FROM opsrag_feedback
    WHERE direction = -1
    ORDER BY created_at DESC LIMIT 50;

to find low-scored answers and author corrections into the SRE-KB.

Schema is created idempotently on startup. The partial index on
`direction = -1` gives SRE a cheap "what went wrong this week" view
without scanning all-time positive feedback.

Endpoint failures are GRACEFUL: if the pool is broken we log a warning
and return without raising -- feedback collection must not break the user
UX. The Qdrant-side write still happens in the route handler.
"""
from __future__ import annotations

import logging
from typing import Any

from psycopg_pool import AsyncConnectionPool

_log = logging.getLogger("opsrag.feedback_store")

_SCHEMA_TABLE = """
CREATE TABLE IF NOT EXISTS opsrag_feedback (
    id                BIGSERIAL PRIMARY KEY,
    investigation_id  TEXT NOT NULL,
    thread_id         TEXT,
    user_id           TEXT,
    direction         SMALLINT NOT NULL,
    note              TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    query_snippet     TEXT,
    answer_snippet    TEXT
)
"""

_SCHEMA_INDEX_NEGATIVE = (
    "CREATE INDEX IF NOT EXISTS idx_opsrag_feedback_negative "
    "ON opsrag_feedback(created_at) WHERE direction = -1"
)

_SCHEMA_INDEX_INVESTIGATION = (
    "CREATE INDEX IF NOT EXISTS idx_opsrag_feedback_investigation "
    "ON opsrag_feedback(investigation_id)"
)


class FeedbackStore:
    """Idempotent Postgres feedback persistence.

    Reuses POSTGRES_DSN (same connection string as session store + usage
    persistence). Pool is small -- feedback traffic is human-paced.
    """

    def __init__(self, dsn: str, min_pool: int = 1, max_pool: int = 3) -> None:
        self._dsn = dsn
        self._pool = AsyncConnectionPool(
            conninfo=dsn,
            min_size=min_pool,
            max_size=max_pool,
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        self._opened = False

    async def open(self) -> None:
        await self._pool.open()
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_SCHEMA_TABLE)
                await cur.execute(_SCHEMA_INDEX_NEGATIVE)
                await cur.execute(_SCHEMA_INDEX_INVESTIGATION)
        self._opened = True
        _log.info("feedback_store schema ready")

    async def close(self) -> None:
        if self._opened:
            await self._pool.close()
            self._opened = False

    async def record(
        self,
        *,
        investigation_id: str,
        direction: int,
        thread_id: str | None = None,
        user_id: str | None = None,
        note: str | None = None,
        query_snippet: str | None = None,
        answer_snippet: str | None = None,
    ) -> int | None:
        """Insert a feedback row. Returns the new row id, or ``None``
        on failure. Failures are LOGGED, not raised -- feedback insertion
        must never break the UX (route handler returns 200 either way).
        """
        if not self._opened:
            _log.warning("feedback_store.record called but store not opened")
            return None
        # Valid directions:
        #   1  = thumbs-up
        #  -1  = thumbs-down
        #   2  = user-correction -- the user submitted a
        #        corrective answer that was also stored in Qdrant. This
        #        Postgres row gives us a non-Qdrant replay path if the
        #        vector collection is rebuilt.
        if direction not in (1, -1, 2):
            _log.warning("invalid feedback direction: %r", direction)
            return None
        # Truncate snippets defensively in case caller didn't.
        if query_snippet is not None:
            query_snippet = query_snippet[:400]
        if answer_snippet is not None:
            answer_snippet = answer_snippet[:400]
        if note is not None:
            note = note[:2000]
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "INSERT INTO opsrag_feedback "
                        "(investigation_id, thread_id, user_id, direction, "
                        " note, query_snippet, answer_snippet) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
                        (
                            investigation_id,
                            thread_id,
                            user_id,
                            int(direction),
                            note,
                            query_snippet,
                            answer_snippet,
                        ),
                    )
                    row = await cur.fetchone()
                    return int(row[0]) if row else None
        except Exception as exc:
            _log.warning("feedback insert failed (graceful): %s", exc)
            return None

    async def list_recent(
        self,
        *,
        direction: int | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List recent feedback rows for SRE triage. ``direction`` filter:
        ``-1`` = thumbs-down, ``1`` = thumbs-up, ``None`` = both. Ordered
        ``created_at DESC``.
        """
        if not self._opened:
            return []
        limit = max(1, min(int(limit), 500))
        params: list[Any] = []
        where = ""
        if direction in (1, -1, 2):
            where = "WHERE direction = %s"
            params.append(int(direction))
        params.append(limit)
        sql = (
            "SELECT id, investigation_id, thread_id, user_id, direction, "
            "note, created_at, query_snippet, answer_snippet "
            f"FROM opsrag_feedback {where} "
            "ORDER BY created_at DESC LIMIT %s"
        )
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(sql, params)
                    rows = await cur.fetchall()
        except Exception as exc:
            _log.warning("feedback list query failed: %s", exc)
            return []
        out: list[dict[str, Any]] = []
        for r in rows:
            out.append({
                "id": r[0],
                "investigation_id": r[1],
                "thread_id": r[2],
                "user_id": r[3],
                "direction": int(r[4]),
                "note": r[5],
                "created_at": r[6].isoformat() if r[6] else None,
                "query_snippet": r[7],
                "answer_snippet": r[8],
            })
        return out
