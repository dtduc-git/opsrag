"""Postgres-backed durable persistence for the in-memory UsageTracker.

The tracker itself is fast (lock + dict) and stays the source of truth
for live aggregates. This module adds:

  - A buffered fire-and-forget event sink: `enqueue(...)` is sync and
    cheap (list append under a lock). A background task flushes
    batches every ~2s into `opsrag_usage_events`.
  - A startup aggregator: `seed_tracker(...)` rolls up historical
    rows into the in-memory tracker so `/usage` reflects all-time
    totals immediately after any restart, instead of resetting to 0.

The cost field is intentionally NOT stored -- recompute on read using
the current pricing table so model price changes apply retroactively.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import UTC
from typing import Any

from psycopg_pool import AsyncConnectionPool

_log = logging.getLogger("opsrag.usage_persistence")

_SCHEMA_TABLE = """
CREATE TABLE IF NOT EXISTS opsrag_usage_events (
    id            BIGSERIAL PRIMARY KEY,
    ts            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model         TEXT NOT NULL,
    purpose       TEXT NOT NULL,
    session_id    TEXT,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms    DOUBLE PRECISION NOT NULL DEFAULT 0
)
"""

_SCHEMA_INDEX_MODEL_PURPOSE = (
    "CREATE INDEX IF NOT EXISTS opsrag_usage_events_model_purpose "
    "ON opsrag_usage_events (model, purpose)"
)
_SCHEMA_INDEX_TS = (
    "CREATE INDEX IF NOT EXISTS opsrag_usage_events_ts "
    "ON opsrag_usage_events (ts)"
)
_SCHEMA_INDEX_SESSION = (
    "CREATE INDEX IF NOT EXISTS opsrag_usage_events_session "
    "ON opsrag_usage_events (session_id) "
    "WHERE session_id IS NOT NULL"
)


class UsagePersistence:
    def __init__(
        self,
        dsn: str,
        flush_interval_s: float = 2.0,
        batch_size: int = 200,
        min_pool: int = 1,
        max_pool: int = 3,
    ) -> None:
        self._dsn = dsn
        self._flush_interval = flush_interval_s
        self._batch_size = batch_size
        self._pool = AsyncConnectionPool(
            conninfo=dsn,
            min_size=min_pool,
            max_size=max_pool,
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        # Bounded-ish in-memory buffer. Sync access from `enqueue` (any
        # thread / coroutine) and async drain from `_flush_loop`.
        self._buf: list[tuple[Any, ...]] = []
        self._buf_lock = threading.Lock()
        self._flush_task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._opened = False

    async def open(self) -> None:
        await self._pool.open()
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_SCHEMA_TABLE)
                await cur.execute(_SCHEMA_INDEX_MODEL_PURPOSE)
                await cur.execute(_SCHEMA_INDEX_TS)
                await cur.execute(_SCHEMA_INDEX_SESSION)
        self._opened = True
        _log.info("usage persistence schema ready")

    async def close(self) -> None:
        # Stop the drain loop, flush whatever's left, close the pool.
        self._stop.set()
        if self._flush_task is not None:
            try:
                await asyncio.wait_for(self._flush_task, timeout=5.0)
            except TimeoutError:
                self._flush_task.cancel()
        try:
            await self._flush_once()
        except Exception as exc:
            _log.warning("final usage flush failed: %s", exc)
        if self._opened:
            await self._pool.close()
            self._opened = False

    def enqueue(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        session_id: str | None,
        purpose: str,
        user_oid: str | None = None,
    ) -> None:
        """Sync, non-blocking. Buffers the event for the next flush.

        M2 -- ``user_oid`` is the verified Pomerium identity for the
        request that produced this LLM call. None is fine: anonymous /
        background-job events keep working and the column is nullable
        (see migration 0002).
        """
        if not self._opened:
            return
        with self._buf_lock:
            # Cap the buffer at 50k events. If we're past that we're
            # losing the DB anyway; drop oldest to make room.
            if len(self._buf) >= 50_000:
                # Drop ~10% of the oldest events.
                del self._buf[: 5_000]
            self._buf.append((
                model, purpose, session_id,
                int(input_tokens), int(output_tokens), float(latency_ms),
                user_oid,
            ))

    def start(self) -> None:
        if self._flush_task is not None:
            return
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def _flush_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._flush_interval)
            except TimeoutError:
                pass
            try:
                await self._flush_once()
            except Exception as exc:
                _log.warning("usage flush iteration failed: %s", exc)

    async def _flush_once(self) -> None:
        if not self._opened:
            return
        with self._buf_lock:
            if not self._buf:
                return
            # Swap the buffer so writers can keep appending while we
            # write. If the DB call fails we splice the rejected batch
            # back to the FRONT so order is roughly preserved.
            batch = self._buf
            self._buf = []
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    # Chunk into INSERT statements of `batch_size` rows
                    # so a 50k flush doesn't blow the parameter limit.
                    for start in range(0, len(batch), self._batch_size):
                        chunk = batch[start : start + self._batch_size]
                        # Build a single multi-VALUES insert by hand.
                        # 7th placeholder = user_oid (M2; migration 0002
                        # added the column nullable so old buffered
                        # tuples missing the field are reshaped at
                        # enqueue time, not here).
                        values_sql = ",".join(
                            ["(%s, %s, %s, %s, %s, %s, %s)"] * len(chunk)
                        )
                        params: list[Any] = []
                        for ev in chunk:
                            params.extend(ev)
                        await cur.execute(
                            "INSERT INTO opsrag_usage_events "
                            "(model, purpose, session_id, "
                            "input_tokens, output_tokens, latency_ms, "
                            "user_oid) "
                            f"VALUES {values_sql}",
                            params,
                        )
        except Exception:
            # Re-queue the batch so we don't drop telemetry on a
            # transient DB hiccup. If the DB is broken for long
            # enough the 50k cap kicks in and oldest events get
            # culled.
            with self._buf_lock:
                self._buf[:0] = batch
            raise

    async def seed_tracker(self, tracker) -> None:
        """Roll up historical rows into the in-memory tracker.

        Called once at startup, AFTER `open()` and BEFORE
        `set_persistence_hook` so live `record()` calls don't double-
        count their own row that just got flushed.
        """
        if not self._opened:
            return
        t0 = time.time()
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT model, purpose, "
                    "  SUM(input_tokens), SUM(output_tokens), "
                    "  COUNT(*), SUM(latency_ms) "
                    "FROM opsrag_usage_events "
                    "GROUP BY model, purpose"
                )
                rows = await cur.fetchall()
        rolled = 0
        for model, purpose, in_t, out_t, n, lat in rows:
            tracker.seed_historical(
                model=model,
                purpose=purpose or "unknown",
                input_tokens=int(in_t or 0),
                output_tokens=int(out_t or 0),
                call_count=int(n or 0),
                latency_ms=float(lat or 0.0),
            )
            rolled += int(n or 0)
        _log.info(
            "usage tracker seeded from postgres: rows=%d events=%d in %.2fs",
            len(rows), rolled, time.time() - t0,
        )

    async def aggregate_by_user(
        self,
        *,
        since_iso: str | None = None,
        until_iso: str | None = None,
    ) -> list[dict]:
        """M2 -- per-user usage roll-up, ordered by cost descending.

        Joins against ``opsrag_user`` so the response carries email +
        display name. Events with ``user_oid IS NULL`` (anonymous /
        pre-Pomerium) are excluded -- they show up in the global
        ``/usage`` dashboard but have no owner to attribute to.

        ``since_iso`` / ``until_iso`` are optional ISO-8601 timestamps;
        when set we restrict ``ts`` to that range. Passed straight to
        Postgres as ``timestamptz`` literals (psycopg parses ISO).

        Cost is computed in Python via the pricing table so model
        price-sheet changes apply retroactively -- same contract as
        ``get_summary()``.
        """
        if not self._opened:
            return []
        # Local import -- pricing imports may pull SDK code in future,
        # and we want this module to stay import-safe.
        from opsrag.llms.pricing import cost_usd_micros

        where_parts: list[str] = ["e.user_oid IS NOT NULL"]
        params: list[Any] = []
        if since_iso:
            where_parts.append("e.ts >= %s")
            params.append(since_iso)
        if until_iso:
            where_parts.append("e.ts <= %s")
            params.append(until_iso)
        where_sql = " AND ".join(where_parts)

        sql = f"""
            SELECT
                e.user_oid,
                COALESCE(u.email, au.email) AS email,
                COALESCE(u.display_name, au.name) AS display_name,
                e.model,
                COUNT(*) AS query_count,
                SUM(e.input_tokens)::BIGINT AS prompt_tokens,
                SUM(e.output_tokens)::BIGINT AS completion_tokens,
                MAX(e.ts) AS last_active_at
            FROM opsrag_usage_events e
            LEFT JOIN opsrag_user u ON u.oid = e.user_oid
            -- Login-mode users live in opsrag_auth_user (id-keyed), not the
            -- oid-keyed analytics table -- join both + coalesce so emails show.
            LEFT JOIN opsrag_auth_user au ON au.id::text = e.user_oid::text
            WHERE {where_sql}
            GROUP BY e.user_oid, u.email, u.display_name, au.email, au.name, e.model
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                rows = await cur.fetchall()

        # Aggregate per (user, model) -> per user, summing token-cost
        # across all the user's models. Done in Python so the pricing
        # table stays the single source of truth and SQL doesn't need a
        # rates table.
        agg: dict[str, dict[str, Any]] = {}
        for oid, email, name, model, qcount, ptok, ctok, last_ts in rows:
            oid_s = str(oid)
            slot = agg.setdefault(oid_s, {
                "user_oid": oid_s,
                "email": email,
                "display_name": name,
                "query_count": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd_micros": 0,
                "last_active_at": None,
            })
            slot["query_count"] += int(qcount or 0)
            slot["prompt_tokens"] += int(ptok or 0)
            slot["completion_tokens"] += int(ctok or 0)
            slot["cost_usd_micros"] += cost_usd_micros(
                model, int(ptok or 0), int(ctok or 0),
            )
            if last_ts is not None:
                last_iso = last_ts.isoformat() if hasattr(last_ts, "isoformat") else str(last_ts)
                if slot["last_active_at"] is None or last_iso > slot["last_active_at"]:
                    slot["last_active_at"] = last_iso

        out = sorted(agg.values(), key=lambda r: r["cost_usd_micros"], reverse=True)
        return out

    async def aggregate_for_user(
        self,
        user_oid: str,
        *,
        since_iso: str | None = None,
    ) -> dict:
        """M2 -- single-user roll-up.

        Mirrors :meth:`aggregate_by_user` for one ``user_oid``.
        Returns a dict in the same row shape (never wrapped in a list).
        An empty result still returns a zeroed row keyed to ``user_oid``
        so the caller can render "no activity yet" without a None-check.
        """
        if not self._opened:
            return {
                "user_oid": user_oid,
                "email": None,
                "display_name": None,
                "query_count": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd_micros": 0,
                "last_active_at": None,
            }
        from opsrag.llms.pricing import cost_usd_micros

        where_parts = ["e.user_oid = %s"]
        params: list[Any] = [user_oid]
        if since_iso:
            where_parts.append("e.ts >= %s")
            params.append(since_iso)
        where_sql = " AND ".join(where_parts)

        sql = f"""
            SELECT
                e.model,
                COUNT(*) AS query_count,
                SUM(e.input_tokens)::BIGINT AS prompt_tokens,
                SUM(e.output_tokens)::BIGINT AS completion_tokens,
                MAX(e.ts) AS last_active_at,
                COALESCE(u.email, au.email) AS email,
                COALESCE(u.display_name, au.name) AS display_name
            FROM opsrag_usage_events e
            LEFT JOIN opsrag_user u ON u.oid = e.user_oid
            LEFT JOIN opsrag_auth_user au ON au.id::text = e.user_oid::text
            WHERE {where_sql}
            GROUP BY e.model, u.email, u.display_name, au.email, au.name
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                rows = await cur.fetchall()

        result = {
            "user_oid": user_oid,
            "email": None,
            "display_name": None,
            "query_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_usd_micros": 0,
            "last_active_at": None,
        }
        for model, qcount, ptok, ctok, last_ts, email, name in rows:
            result["query_count"] += int(qcount or 0)
            result["prompt_tokens"] += int(ptok or 0)
            result["completion_tokens"] += int(ctok or 0)
            result["cost_usd_micros"] += cost_usd_micros(
                model, int(ptok or 0), int(ctok or 0),
            )
            if email and not result["email"]:
                result["email"] = email
            if name and not result["display_name"]:
                result["display_name"] = name
            if last_ts is not None:
                last_iso = last_ts.isoformat() if hasattr(last_ts, "isoformat") else str(last_ts)
                if result["last_active_at"] is None or last_iso > result["last_active_at"]:
                    result["last_active_at"] = last_iso
        return result

    async def weekly_series(self, weeks: int = 6) -> list[dict] | None:
        """Per-week token + cost buckets for the last ``weeks`` ISO weeks.

        Powers the Home dashboard's "Usage this month" mini bar chart.
        Buckets ``opsrag_usage_events.ts`` by the Monday-anchored week
        start (``date_trunc('week', ts)``) and sums tokens; cost is
        recomputed per-row in Python via the pricing table so it stays
        consistent with ``get_summary()`` / ``aggregate_by_user()``.

        Always returns exactly ``weeks`` entries, oldest-first, with the
        current (most-recent) week last. Weeks with no activity are
        zero-filled so the chart axis is stable. Returns None when
        persistence isn't open (caller falls back to an empty state).
        """
        if not self._opened:
            return None
        from datetime import datetime, timedelta

        from opsrag.llms.pricing import cost_usd_micros

        # Anchor on this week's Monday (UTC) so buckets line up with
        # Postgres' ISO-week date_trunc.
        now = datetime.now(UTC)
        this_week_start = (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        oldest_start = this_week_start - timedelta(weeks=weeks - 1)

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT date_trunc('week', ts) AS wk, model, "
                    "  SUM(input_tokens)::BIGINT, SUM(output_tokens)::BIGINT, "
                    "  COUNT(*) "
                    "FROM opsrag_usage_events "
                    "WHERE ts >= %s "
                    "GROUP BY wk, model",
                    (oldest_start,),
                )
                rows = await cur.fetchall()

        # Pre-build the zero-filled week slots keyed by ISO date string.
        slots: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for i in range(weeks):
            wk_start = oldest_start + timedelta(weeks=i)
            key = wk_start.date().isoformat()
            order.append(key)
            slots[key] = {
                "week_start": key,
                "input_tokens": 0,
                "output_tokens": 0,
                "tokens": 0,
                "call_count": 0,
                "cost_usd": 0.0,
            }

        for wk, model, in_t, out_t, n in rows:
            wk_date = wk.date() if hasattr(wk, "date") else wk
            key = wk_date.isoformat()
            slot = slots.get(key)
            if slot is None:
                continue
            it, ot = int(in_t or 0), int(out_t or 0)
            slot["input_tokens"] += it
            slot["output_tokens"] += ot
            slot["tokens"] += it + ot
            slot["call_count"] += int(n or 0)
            # cost_usd_micros returns MICRO-CENTS (1 USD == 100_000_000 of them;
            # see pricing.py). Divide by 1e8, not 1e6 (the /1e6 bug inflated the
            # weekly cost ~100x, e.g. $2399 vs the real ~$24).
            slot["cost_usd"] += cost_usd_micros(model, it, ot) / 100_000_000.0

        return [slots[k] for k in order]

    async def get_summary(self) -> dict | None:
        """Build a usage summary directly from Postgres -- pod-agnostic.

        Backend + indexer pods both record into the same `opsrag_usage_events`
        table; an in-memory tracker on either pod only sees its own pod's
        post-startup work. The UI needs a unified view, so this method
        builds a transient `UsageTracker`, seeds it via the SQL aggregate,
        and returns its `.get_summary()` -- same shape the UI expects, but
        always reflecting the full cross-pod truth.

        Returns None if persistence isn't open (caller should fall back).
        """
        if not self._opened:
            return None
        # Local import to avoid a circular import at module load time
        # (usage.py doesn't depend on this module, but this module is
        # already imported by api/server.py before usage.py's tracker
        # finishes initializing).
        from opsrag.usage import UsageTracker
        transient = UsageTracker()
        await self.seed_tracker(transient)
        return transient.get_summary()
