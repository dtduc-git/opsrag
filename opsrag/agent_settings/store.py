"""Postgres-backed agent settings (schema: migration 0010).

Backs the admin "Agent Guidance" page -- deployment-wide custom instructions
that inject into the agent's answer + chat prompts and are editable live in the
UI (no redeploy). Generic key/value so other operator-tunable agent settings
can reuse it. Every method is best-effort: a DB hiccup must never break the
prompt path (callers fall back to the config seed).
"""
from __future__ import annotations

import logging
from typing import Any

from psycopg_pool import AsyncConnectionPool

_log = logging.getLogger("opsrag.agent_settings")

CUSTOM_INSTRUCTIONS_KEY = "custom_instructions"


class AgentSettingsStore:
    def __init__(self, dsn: str, *, min_size: int = 1, max_size: int = 4) -> None:
        self._dsn = dsn
        self._pool = AsyncConnectionPool(
            conninfo=dsn, min_size=min_size, max_size=max_size,
            open=False, kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        self._ready = False

    async def open(self) -> None:
        await self._pool.open()
        self._ready = True

    async def close(self) -> None:
        if self._ready:
            await self._pool.close()
            self._ready = False

    async def init_schema(self) -> None:
        """No-op: migration 0010 owns the DDL (parity with the other stores)."""
        return None

    async def get(self, key: str, default: str = "") -> str:
        if not self._ready:
            return default
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT value FROM opsrag_agent_settings WHERE key = %s", (key,)
                    )
                    row = await cur.fetchone()
            return row[0] if row is not None else default
        except Exception as exc:  # noqa: BLE001
            _log.warning("agent_settings get(%s) failed: %s", key, exc)
            return default

    async def get_meta(self, key: str) -> dict[str, Any] | None:
        """Value + updated_at/updated_by (for the admin UI). None if unset."""
        if not self._ready:
            return None
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT value, updated_at, updated_by "
                        "FROM opsrag_agent_settings WHERE key = %s", (key,)
                    )
                    row = await cur.fetchone()
            if row is None:
                return None
            ts = row[1]
            return {
                "value": row[0],
                "updated_at": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "updated_by": row[2],
            }
        except Exception as exc:  # noqa: BLE001
            _log.warning("agent_settings get_meta(%s) failed: %s", key, exc)
            return None

    async def set(self, key: str, value: str, updated_by: str | None = None) -> None:
        if not self._ready:
            raise RuntimeError("agent settings store not open")
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO opsrag_agent_settings (key, value, updated_at, updated_by)
                    VALUES (%s, %s, now(), %s)
                    ON CONFLICT (key) DO UPDATE SET
                        value = EXCLUDED.value,
                        updated_at = now(),
                        updated_by = EXCLUDED.updated_by
                    """,
                    (key, value, updated_by),
                )
