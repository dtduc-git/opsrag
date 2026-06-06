"""Buffered Postgres audit logger for MCP-server-as-proxy tool calls.

Same pattern as `opsrag.usage_persistence`: a synchronous `log()`
enqueues a tuple into an in-memory buffer under a `threading.Lock`,
and a background asyncio task flushes batches every ~2s into
`opsrag_mcp_audit`. The hot path (an MCP `tools/call` JSON-RPC
request) never blocks on a DB INSERT.

Schema (defined in `0003_mcp_tokens.sql`):

    opsrag_mcp_audit (
      id          BIGSERIAL PRIMARY KEY,
      occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      user_oid    UUID,
      token_id    UUID,
      tool_name   TEXT NOT NULL,
      args_hash   TEXT,
      latency_ms  INTEGER,
      status      TEXT NOT NULL,
      error       TEXT
    )

`args_hash` is provided by the caller as the hex sha256 of a canonical
JSON representation of the args dict -- this lets us aggregate "how
often is the same (tool, args) pair called?" without storing literal
arguments (which can contain SQL-shaped strings or query secrets).
"""
from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime
from typing import Any

from psycopg_pool import AsyncConnectionPool

_log = logging.getLogger("opsrag.mcp_server.audit")

_BUFFER_HARD_CAP = 20_000  # drop oldest if we exceed this
_BUFFER_DROP_CHUNK = 2_000  # drop oldest 10% at once

# Tuple shape -- must match the INSERT below.
# (occurred_at, user_oid, token_id, tool_name, args_hash, latency_ms, status, error)
_Row = tuple[datetime, str | None, str | None, str, str | None, int | None, str, str | None]


class AuditLogger:
    """Buffered async audit-log writer for `opsrag_mcp_audit`.

    Lifecycle: `open()` -> opens the pool; `start()` -> launches the
    flush loop; `close()` -> drains the buffer + closes the pool.
    """

    def __init__(
        self,
        dsn: str,
        *,
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
        self._buf: list[_Row] = []
        self._buf_lock = threading.Lock()
        self._flush_task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._opened = False

    async def open(self) -> None:
        await self._pool.open()
        self._opened = True
        _log.info("mcp audit logger opened")

    def start(self) -> None:
        """Launch the background flush loop. Idempotent."""
        if self._flush_task is not None:
            return
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def close(self) -> None:
        self._stop.set()
        if self._flush_task is not None:
            try:
                await asyncio.wait_for(self._flush_task, timeout=5.0)
            except TimeoutError:
                self._flush_task.cancel()
        try:
            await self._flush_once()
        except Exception as exc:
            _log.warning("final mcp audit flush failed: %s", exc)
        if self._opened:
            await self._pool.close()
            self._opened = False

    # --- public API ------------------------------------------------

    def log(
        self,
        *,
        occurred_at: datetime,
        user_oid: str | None,
        token_id: str | None,
        tool_name: str,
        args_hash: str | None,
        latency_ms: int | None,
        status: str,
        error: str | None = None,
    ) -> None:
        """Sync, non-blocking. Buffers an audit row.

        ``status`` is one of "ok", "denied" (rate-limited / unknown
        tool), or "error" (handler raised). ``error`` carries the
        truncated exception string when status != "ok".
        """
        if not self._opened:
            # Logger isn't wired (e.g. unit test) -- silently no-op.
            return
        if error is not None:
            error = error[:4000]  # keep TOAST happy
        row: _Row = (
            occurred_at,
            user_oid,
            token_id,
            tool_name,
            args_hash,
            latency_ms,
            status,
            error,
        )
        with self._buf_lock:
            if len(self._buf) >= _BUFFER_HARD_CAP:
                # DB has been broken for a while. Drop the oldest 10%
                # to keep the in-memory footprint bounded.
                del self._buf[:_BUFFER_DROP_CHUNK]
            self._buf.append(row)

    # --- internal flush loop --------------------------------------

    async def _flush_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._flush_interval)
            except TimeoutError:
                pass
            try:
                await self._flush_once()
            except Exception as exc:
                _log.warning("mcp audit flush iteration failed: %s", exc)

    async def _flush_once(self) -> None:
        if not self._opened:
            return
        with self._buf_lock:
            if not self._buf:
                return
            batch = self._buf
            self._buf = []
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    for start in range(0, len(batch), self._batch_size):
                        chunk = batch[start : start + self._batch_size]
                        values_sql = ",".join(
                            ["(%s, %s, %s, %s, %s, %s, %s, %s)"] * len(chunk)
                        )
                        params: list[Any] = []
                        for r in chunk:
                            params.extend(r)
                        await cur.execute(
                            "INSERT INTO opsrag_mcp_audit "
                            "(occurred_at, user_oid, token_id, tool_name, "
                            " args_hash, latency_ms, status, error) "
                            f"VALUES {values_sql}",
                            params,
                        )
        except Exception:
            # Splice the rejected batch back to the FRONT so order is
            # roughly preserved across DB hiccups.
            with self._buf_lock:
                self._buf[:0] = batch
            raise
