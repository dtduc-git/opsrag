"""InvestigationEventStore -- Postgres-backed event ledger + lifecycle row.

Layout:
  - opsrag_investigations: one row per investigation (alert_text, status,
    root_cause, completed_at). Tiny -- full state is reconstructable
    from the events table.
  - opsrag_investigation_events: append-only stream of typed events.
    GET /investigations/{id}/events?since=N tails this table.

Design notes (Option B refactor):
  emit_event() is the ONLY public way to append. It opens a fresh
  session per call (out-of-band of any LangGraph transaction) and
  commits immediately. Swallows + logs errors so an observability
  failure never breaks the supervisor.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Final

from psycopg.types.json import Json
from psycopg_pool import AsyncConnectionPool

_log = logging.getLogger("opsrag.investigations.store")


class InvestigationStatus:
    RUNNING: Final = "running"
    COMPLETED: Final = "completed"
    FAILED: Final = "failed"
    CANCELLED: Final = "cancelled"


class InvestigationEventStore:
    """All Postgres operations for the Investigate-mode event ledger.

    Construction is light. The pool is passed in (shared with the
    runbook store / qa_cache pool so we don't fan out connection counts
    across modules)."""

    def __init__(self, pg_pool: AsyncConnectionPool) -> None:
        self._pool = pg_pool

    # -- Lifecycle row ---------------------------------------------

    async def create_investigation(
        self, *, alert_text: str, incident_target: str | None = None,
    ) -> str:
        """Insert a fresh row in opsrag_investigations and return the UUID
        (string form) the caller passes to the LangGraph runner + back to
        the UI."""
        inv_id = str(uuid.uuid4())
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO opsrag_investigations
                        (id, alert_text, incident_target, status)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (inv_id, alert_text, incident_target, InvestigationStatus.RUNNING),
                )
            await conn.commit()
        return inv_id

    async def get_investigation(self, inv_id: str) -> dict[str, Any] | None:
        """Read the lifecycle row. Returns None when the id is unknown
        (caller renders a 404)."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT id, alert_text, incident_target, status,
                           root_cause, outcome, created_at, completed_at
                      FROM opsrag_investigations
                     WHERE id = %s
                    """,
                    (inv_id,),
                )
                row = await cur.fetchone()
        if row is None:
            return None
        return {
            "id": str(row[0]),
            "alert_text": row[1],
            "incident_target": row[2],
            "status": row[3],
            "root_cause": row[4],
            "outcome": row[5],
            "created_at": row[6].isoformat() if row[6] else None,
            "completed_at": row[7].isoformat() if row[7] else None,
        }

    async def list_investigations(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Most-recent-first listing for the sidebar."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT id, alert_text, incident_target, status,
                           root_cause, created_at, completed_at
                      FROM opsrag_investigations
                     ORDER BY created_at DESC
                     LIMIT %s
                    """,
                    (limit,),
                )
                rows = await cur.fetchall()
        return [
            {
                "id": str(r[0]),
                "alert_text": r[1],
                "incident_target": r[2],
                "status": r[3],
                "root_cause": r[4],
                "created_at": r[5].isoformat() if r[5] else None,
                "completed_at": r[6].isoformat() if r[6] else None,
            }
            for r in rows
        ]

    async def mark_status(
        self,
        inv_id: str,
        *,
        status: str,
        root_cause: str | None = None,
        outcome: str | None = None,
    ) -> None:
        """Update the lifecycle row. Used on INVESTIGATION_COMPLETED to
        stamp final root_cause + outcome alongside the event."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE opsrag_investigations
                       SET status      = %s,
                           root_cause  = COALESCE(%s, root_cause),
                           outcome     = COALESCE(%s, outcome),
                           completed_at = CASE WHEN %s IN ('completed','failed','cancelled')
                                                THEN NOW() ELSE completed_at END
                     WHERE id = %s
                    """,
                    (status, root_cause, outcome, status, inv_id),
                )
            await conn.commit()

    # -- Event ledger ----------------------------------------------

    async def append_event(
        self,
        *,
        investigation_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> int:
        """Append one event row. Returns the new sequence number.

        Errors are NOT swallowed here -- the public `emit_event` helper
        wraps this with try/except so node code never sees DB exceptions.
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO opsrag_investigation_events
                        (investigation_id, event_type, payload, tags)
                    VALUES (%s, %s, %s, %s)
                    RETURNING sequence
                    """,
                    (
                        investigation_id,
                        event_type,
                        Json(payload or {}),
                        tags or [],
                    ),
                )
                row = await cur.fetchone()
            await conn.commit()
        return int(row[0]) if row else 0

    async def list_events_since(
        self,
        *,
        investigation_id: str,
        since: int = 0,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Tail-cursor read for the SSE stream. Caller passes the highest
        sequence it has already rendered; we return rows strictly
        greater than that, ordered ascending."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT sequence, event_type, payload, tags, created_at
                      FROM opsrag_investigation_events
                     WHERE investigation_id = %s
                       AND sequence > %s
                     ORDER BY sequence ASC
                     LIMIT %s
                    """,
                    (investigation_id, since, limit),
                )
                rows = await cur.fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            seq, etype, payload_json, tags, created_at = r
            # psycopg returns JSONB as already-decoded dict
            payload = payload_json if isinstance(payload_json, dict) else (
                json.loads(payload_json) if payload_json else {}
            )
            out.append({
                "sequence": int(seq),
                "type": etype,
                "payload": payload,
                "tags": list(tags or []),
                "ts": created_at.isoformat() if created_at else None,
            })
        return out

    async def list_all_events(
        self, *, investigation_id: str,
    ) -> list[dict[str, Any]]:
        """Full replay for the GET /investigations/{id} snapshot endpoint
        -- same shape as list_events_since but unbounded."""
        return await self.list_events_since(
            investigation_id=investigation_id, since=0, limit=10_000,
        )


# -- Module-level convenience: emit_event(store, ...) with error-swallow --

async def emit_event(
    store: InvestigationEventStore | None,
    *,
    investigation_id: str,
    event_type: str,
    payload: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> int | None:
    """Append-and-commit one event row, swallowing all errors.

    Returns the new sequence (caller can log it) or None on failure.
    Node code calls this without try/except -- an observability failure
    must NEVER break the LangGraph supervisor.
    """
    if store is None:
        return None
    try:
        return await store.append_event(
            investigation_id=investigation_id,
            event_type=event_type,
            payload=payload,
            tags=tags,
        )
    except Exception as exc:  # noqa: BLE001
        _log.warning(
            "emit_event failed inv=%s type=%s: %s",
            investigation_id, event_type, exc,
        )
        return None
