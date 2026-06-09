"""Postgres-backed `indexed_files` tracker.

Schema (created on first ``open()`` call -- idempotent):

    CREATE TABLE indexed_files (
      repo         TEXT NOT NULL,
      branch       TEXT NOT NULL,
      path         TEXT NOT NULL,
      source_type  TEXT NOT NULL DEFAULT 'gitlab',
      content_hash TEXT NOT NULL,
      indexed_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      chunk_count  INT NOT NULL DEFAULT 0,
      PRIMARY KEY (repo, branch, path)
    );

The ``last_seen_at`` column drives the repo-level deletion sweep: at the END
of a successful ``index_repo`` run, rows whose ``last_seen_at`` predates the
run start were not seen this pass (deleted from source) and are purged --
see ``sweep_deleted`` and ``IngestionPipeline.index_repo``.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime

from psycopg_pool import AsyncConnectionPool

_log = logging.getLogger("opsrag.indexed_files.postgres")

# Two statements run separately -- psycopg with prepare_threshold=0 rejects
# multi-statement prepared queries.
_SCHEMA_TABLE = """
CREATE TABLE IF NOT EXISTS indexed_files (
    repo         TEXT NOT NULL,
    branch       TEXT NOT NULL,
    path         TEXT NOT NULL,
    source_type  TEXT NOT NULL DEFAULT 'gitlab',
    content_hash TEXT NOT NULL,
    indexed_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    chunk_count  INT NOT NULL DEFAULT 0,
    PRIMARY KEY (repo, branch, path)
)
"""

_SCHEMA_INDEX = (
    "CREATE INDEX IF NOT EXISTS indexed_files_repo_branch "
    "ON indexed_files (repo, branch)"
)


class PostgresIndexedFilesTracker:
    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 5):
        self._dsn = dsn
        self._pool = AsyncConnectionPool(
            conninfo=dsn,
            min_size=min_size,
            max_size=max_size,
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        self._ready = False

    async def open(self) -> None:
        await self._pool.open()
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_SCHEMA_TABLE)
                await cur.execute(_SCHEMA_INDEX)
        self._ready = True
        _log.info("indexed_files schema ready")

    async def close(self) -> None:
        await self._pool.close()
        self._ready = False

    async def should_skip(
        self, repo: str, branch: str, path: str, content_hash: str
    ) -> bool:
        if not self._ready:
            return False
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT content_hash FROM indexed_files "
                    "WHERE repo = %s AND branch = %s AND path = %s",
                    (repo, branch, path),
                )
                row = await cur.fetchone()
        return bool(row and row[0] == content_hash)

    async def record(
        self,
        repo: str,
        branch: str,
        path: str,
        content_hash: str,
        chunk_count: int,
    ) -> None:
        if not self._ready:
            return
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO indexed_files
                        (repo, branch, path, content_hash, chunk_count,
                         indexed_at, last_seen_at)
                    VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                    ON CONFLICT (repo, branch, path) DO UPDATE SET
                        content_hash = EXCLUDED.content_hash,
                        chunk_count  = EXCLUDED.chunk_count,
                        indexed_at   = NOW(),
                        last_seen_at = NOW()
                    """,
                    (repo, branch, path, content_hash, chunk_count),
                )

    async def repo_branches(self) -> dict[str, str]:
        """Best-known branch per repo, from the actually-indexed files.

        The startup backfill rebuilds /indexing/status from Qdrant payloads,
        which carry `repo` but not `branch`. Without this it defaulted every
        unconfigured repo to "main", mislabeling repos indexed on `master`
        (e.g. genapp) and creating duplicate rows. We pick the branch with the
        most indexed files per repo as canonical. Empty on any failure."""
        if not self._ready:
            return {}
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT DISTINCT ON (repo) repo, branch "
                        "FROM indexed_files "
                        "GROUP BY repo, branch "
                        "ORDER BY repo, count(*) DESC"
                    )
                    rows = await cur.fetchall()
            return {r[0]: r[1] for r in rows}
        except Exception as exc:
            _log.warning("repo_branches lookup failed: %s", exc)
            return {}

    async def mark_seen(
        self, repo: str, branch: str, paths: Iterable[str]
    ) -> None:
        if not self._ready:
            return
        path_list = list(paths)
        if not path_list:
            return
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                # Single bulk update -- psycopg expands the list to ANY ($1).
                await cur.execute(
                    "UPDATE indexed_files SET last_seen_at = NOW() "
                    "WHERE repo = %s AND branch = %s AND path = ANY(%s)",
                    (repo, branch, path_list),
                )

    async def sweep_deleted(
        self, repo: str, branch: str, run_started_at: datetime
    ) -> list[str]:
        """Purge tracker rows for files that vanished from source this run.

        A file present last run but NOT seen this run keeps its older
        ``last_seen_at`` (every seen file -- re-indexed via ``record`` or
        skipped via ``mark_seen`` -- gets bumped to NOW()). So rows with
        ``last_seen_at < run_started_at`` are exactly the deleted files.
        DELETE...RETURNING gives us their paths in one round-trip so the
        caller can drop the matching chunks from the vector store.
        """
        if not self._ready:
            return []
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM indexed_files "
                    "WHERE repo = %s AND branch = %s AND last_seen_at < %s "
                    "RETURNING path",
                    (repo, branch, run_started_at),
                )
                rows = await cur.fetchall()
        return [r[0] for r in rows]
