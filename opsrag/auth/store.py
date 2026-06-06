"""M1 -- Postgres-backed UserStore for the ``opsrag_user`` identity table.

The integration layer constructs one UserStore per process, opens its
pool at startup, and calls :meth:`upsert` on every authenticated request.
:meth:`upsert` is debounced: we only run the actual INSERT/UPDATE if our
in-process cache says we haven't written this oid in the last 5 minutes.
For a worker handling 100 requests/s from 50 users that drops the
identity write load from ~100 inserts/s to ~50 inserts every 5 minutes
(~=0.17/s) -- irrelevant compared to the rest of the query load, but
avoids hot-row contention on a single high-traffic oid.

Schema lives in ``opsrag/db/migrations/0001_opsrag_user.sql``; the
migration is the source of truth. :meth:`init_schema` is a no-op kept
for parity with the other ``*Store`` classes in this codebase, so the
integration layer's startup sequence can call it uniformly.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

from psycopg_pool import AsyncConnectionPool

if TYPE_CHECKING:
    from opsrag.auth.pomerium import CurrentUser

_log = logging.getLogger("opsrag.auth.store")

# Per-oid debounce window. 5 minutes balances "we want to update
# last_seen often enough that the admin dashboard reflects activity"
# against "don't slam Postgres with duplicate writes on a hot user".
_DEBOUNCE_SECONDS = 300.0


class UserStore:
    """Async Postgres-backed wrapper for the ``opsrag_user`` table."""

    def __init__(
        self,
        dsn: str,
        *,
        min_pool: int = 1,
        max_pool: int = 3,
        debounce_seconds: float = _DEBOUNCE_SECONDS,
    ) -> None:
        self._dsn = dsn
        self._pool = AsyncConnectionPool(
            conninfo=dsn,
            min_size=min_pool,
            max_size=max_pool,
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        self._opened = False
        self._debounce_seconds = debounce_seconds
        # oid -> unix-ts of last upsert. Bounded growth is fine -- total
        # user population is in the hundreds, not millions.
        self._last_upsert: dict[str, float] = {}
        self._lock = threading.Lock()

    async def open(self) -> None:
        await self._pool.open()
        self._opened = True

    async def close(self) -> None:
        if self._opened:
            await self._pool.close()
            self._opened = False

    async def init_schema(self) -> None:
        """No-op: the migration framework owns the ``opsrag_user`` DDL.

        Kept for symmetry with the other ``*Store`` classes -- the
        lifespan boot sequence calls ``init_schema`` uniformly so adding
        a new store doesn't require a new branch in server startup.
        """
        return None

    async def upsert(self, user: CurrentUser) -> None:
        """Upsert this user into ``opsrag_user``.

        Debounced: if we wrote this oid less than ``debounce_seconds``
        ago, skip the SQL. Anonymous users (no oid) are also skipped --
        they don't have a primary key to upsert against.
        """
        if user.oid is None or user.is_anonymous:
            return
        if not self._opened:
            return

        now = time.time()
        with self._lock:
            last = self._last_upsert.get(user.oid)
            if last is not None and (now - last) < self._debounce_seconds:
                return
            self._last_upsert[user.oid] = now

        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        INSERT INTO opsrag_user
                          (oid, email, display_name, picture_url, groups, first_seen, last_seen)
                        VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                        ON CONFLICT (oid) DO UPDATE SET
                          email        = EXCLUDED.email,
                          display_name = EXCLUDED.display_name,
                          picture_url  = EXCLUDED.picture_url,
                          groups       = EXCLUDED.groups,
                          last_seen    = NOW()
                        """,
                        (
                            user.oid,
                            user.email or "",
                            user.name,
                            user.picture_url,
                            list(user.groups),
                        ),
                    )
        except Exception as exc:
            # On failure roll back the debounce-marker so the next
            # request retries the write. Identity persistence is
            # best-effort -- never bubble up into the user-facing path.
            with self._lock:
                self._last_upsert.pop(user.oid, None)
            _log.warning("opsrag_user upsert failed for oid=%s: %s", user.oid, exc)

    async def get_user(self, oid: str) -> dict | None:
        """Simple lookup by oid. Returns the row as a dict, or None."""
        if not self._opened:
            return None
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT oid, email, display_name, picture_url, groups,
                           first_seen, last_seen
                    FROM opsrag_user
                    WHERE oid = %s
                    """,
                    (oid,),
                )
                row = await cur.fetchone()
        if row is None:
            return None
        return {
            "oid": str(row[0]),
            "email": row[1],
            "display_name": row[2],
            "picture_url": row[3],
            "groups": list(row[4] or []),
            "first_seen": row[5].isoformat() if row[5] else None,
            "last_seen": row[6].isoformat() if row[6] else None,
        }
