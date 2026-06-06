"""Postgres-backed moderation queue for user corrections.

``POST /correction`` enqueues here (status ``pending``) instead of writing
straight into the live retrieval collection. A pending correction is invisible
to retrieval -- nothing reaches Qdrant until an operator approves it via the
admin-gated ``/corrections/{id}/approve`` endpoint, at which point
:class:`opsrag.correction_store.CorrectionStore` injects the boosted chunk and
the row is marked ``approved``. This closes the prior poisoning vector where any
caller could inject a 2.5x-boosted "fact" for everyone with no review.

Schema lives in migration 0012. Reuses POSTGRES_DSN like the other stores.
Methods are graceful on a broken pool EXCEPT ``submit`` and the approve/reject
transitions, where the route handler needs to surface failure to the operator.
"""
from __future__ import annotations

import logging
from typing import Any

from psycopg_pool import AsyncConnectionPool

_log = logging.getLogger("opsrag.pending_corrections")

_VALID_STATUS = ("pending", "approved", "rejected")


class PendingCorrectionStore:
    """Idempotent Postgres moderation queue. Pool is small -- corrections are
    human-paced."""

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
        self._opened = True
        _log.info("pending_corrections store ready")

    async def close(self) -> None:
        if self._opened:
            await self._pool.close()
            self._opened = False

    async def submit(
        self,
        *,
        question: str,
        correct_answer: str,
        wrong_answer: str = "",
        evidence_url: str | None = None,
        user_id: str | None = None,
        thread_id: str | None = None,
    ) -> int:
        """Enqueue a pending correction. Returns the new row id. Raises on
        failure -- the submitter should know the correction wasn't queued."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO opsrag_pending_corrections "
                    "(question, wrong_answer, correct_answer, evidence_url, "
                    " user_id, thread_id) "
                    "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                    (
                        question[:4000],
                        (wrong_answer or "")[:8000],
                        correct_answer[:8000],
                        evidence_url,
                        user_id,
                        thread_id,
                    ),
                )
                row = await cur.fetchone()
        return int(row[0])

    async def get(self, correction_id: int) -> dict[str, Any] | None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, question, wrong_answer, correct_answer, "
                    "evidence_url, user_id, thread_id, status, chunk_id "
                    "FROM opsrag_pending_corrections WHERE id = %s",
                    (correction_id,),
                )
                r = await cur.fetchone()
        if not r:
            return None
        return {
            "id": r[0], "question": r[1], "wrong_answer": r[2],
            "correct_answer": r[3], "evidence_url": r[4], "user_id": r[5],
            "thread_id": r[6], "status": r[7], "chunk_id": r[8],
        }

    async def list_pending(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self._opened:
            return []
        limit = max(1, min(int(limit), 500))
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, question, wrong_answer, correct_answer, "
                    "evidence_url, user_id, created_at "
                    "FROM opsrag_pending_corrections WHERE status = 'pending' "
                    "ORDER BY created_at DESC LIMIT %s",
                    (limit,),
                )
                rows = await cur.fetchall()
        return [
            {
                "id": r[0], "question": r[1], "wrong_answer": r[2],
                "correct_answer": r[3], "evidence_url": r[4], "user_id": r[5],
                "created_at": r[6].isoformat() if r[6] else None,
            }
            for r in rows
        ]

    async def resolve(
        self,
        correction_id: int,
        *,
        status: str,
        chunk_id: str | None = None,
        reviewed_by: str | None = None,
    ) -> bool:
        """Mark a pending row approved/rejected. Only transitions rows that are
        still ``pending`` (idempotent + prevents double-injection on a retry).
        Returns True if a row was transitioned."""
        if status not in ("approved", "rejected"):
            raise ValueError(f"invalid terminal status: {status!r}")
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE opsrag_pending_corrections "
                    "SET status = %s, chunk_id = %s, reviewed_by = %s, "
                    "    reviewed_at = NOW() "
                    "WHERE id = %s AND status = 'pending'",
                    (status, chunk_id, reviewed_by, correction_id),
                )
                return cur.rowcount > 0
