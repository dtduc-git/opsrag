"""PostgreSQL-backed long-term memory store.

Uses a simple table with JSON value columns. Semantic recall via
optional vector search can be added later.
"""
from __future__ import annotations

import json

from psycopg_pool import AsyncConnectionPool

from opsrag.interfaces.memory import Memory

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS opsrag_memory (
    namespace TEXT NOT NULL,
    key TEXT NOT NULL,
    value JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (namespace, key)
);
"""


class PostgresMemoryStore:
    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 5):
        self._dsn = dsn
        self._pool = AsyncConnectionPool(
            conninfo=dsn,
            min_size=min_size,
            max_size=max_size,
            open=False,
        )
        self._setup_done = False

    async def open(self) -> None:
        await self._pool.open()
        if not self._setup_done:
            async with self._pool.connection() as conn:
                await conn.execute(_CREATE_TABLE)
            self._setup_done = True

    async def close(self) -> None:
        await self._pool.close()

    @staticmethod
    def _encode_ns(namespace: tuple[str, ...]) -> str:
        return "/".join(namespace)

    @staticmethod
    def _decode_ns(raw: str) -> tuple[str, ...]:
        return tuple(raw.split("/"))

    async def put(
        self, namespace: tuple[str, ...], key: str, value: dict
    ) -> None:
        ns = self._encode_ns(namespace)
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO opsrag_memory (namespace, key, value, created_at, updated_at)
                VALUES (%s, %s, %s, now(), now())
                ON CONFLICT (namespace, key)
                DO UPDATE SET value = EXCLUDED.value, updated_at = now()
                """,
                (ns, key, json.dumps(value)),
            )

    async def get(
        self, namespace: tuple[str, ...], key: str
    ) -> Memory | None:
        ns = self._encode_ns(namespace)
        async with self._pool.connection() as conn:
            row = await conn.execute(
                "SELECT key, namespace, value, created_at, updated_at "
                "FROM opsrag_memory WHERE namespace = %s AND key = %s",
                (ns, key),
            )
            record = await row.fetchone()
            if not record:
                return None
            return Memory(
                key=record[0],
                namespace=self._decode_ns(record[1]),
                value=json.loads(record[2]) if isinstance(record[2], str) else record[2],
                created_at=record[3],
                updated_at=record[4],
            )

    async def search(
        self,
        namespace: tuple[str, ...],
        query: str | None = None,
        limit: int = 10,
    ) -> list[Memory]:
        ns = self._encode_ns(namespace)
        if query:
            async with self._pool.connection() as conn:
                rows = await conn.execute(
                    "SELECT key, namespace, value, created_at, updated_at "
                    "FROM opsrag_memory "
                    "WHERE namespace = %s AND (key ILIKE %s OR value::text ILIKE %s) "
                    "ORDER BY updated_at DESC LIMIT %s",
                    (ns, f"%{query}%", f"%{query}%", limit),
                )
                records = await rows.fetchall()
        else:
            async with self._pool.connection() as conn:
                rows = await conn.execute(
                    "SELECT key, namespace, value, created_at, updated_at "
                    "FROM opsrag_memory WHERE namespace = %s "
                    "ORDER BY updated_at DESC LIMIT %s",
                    (ns, limit),
                )
                records = await rows.fetchall()

        return [
            Memory(
                key=r[0],
                namespace=self._decode_ns(r[1]),
                value=json.loads(r[2]) if isinstance(r[2], str) else r[2],
                created_at=r[3],
                updated_at=r[4],
            )
            for r in records
        ]

    async def delete(self, namespace: tuple[str, ...], key: str) -> bool:
        ns = self._encode_ns(namespace)
        async with self._pool.connection() as conn:
            result = await conn.execute(
                "DELETE FROM opsrag_memory WHERE namespace = %s AND key = %s",
                (ns, key),
            )
            return result.rowcount > 0
