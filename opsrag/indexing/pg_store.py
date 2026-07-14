"""Postgres-backed indexing job-state store (schema: migration 0009).

Durable mirror of the in-memory ``IndexingTracker``. The single indexing
*writer* at a time (the ephemeral job-indexer Job, the legacy ``indexer`` role,
or local dev) keeps the fast in-memory tracker and FLUSHES it here on a
throttle; backend pods only READ via :meth:`read_summary` / :meth:`read_jobs`.
That makes ``/indexing/status`` consistent across replicas (the original
in-memory tracker gave each pod its own copy) and durable across restarts.

Design points:
  - Per-row UPSERT keyed on ``(repo, branch)`` -> concurrent Jobs for different
    repos never clobber each other.
  - Run rows are keyed on a process-unique ``run_key`` so a writer can flip its
    own run from ``running`` -> ``success``/``failed`` idempotently. ``restored``
    runs (rebuilt from Qdrant at boot) use a STABLE key so repeated restarts
    don't duplicate them.
  - :meth:`backfill_upsert` is guarded: it never overwrites a row that a live
    Job is actively writing (status ``listing``/``indexing``).
  - Every method is non-fatal -- a DB hiccup must never break indexing or the
    status endpoint (callers fall back to the in-memory tracker).
"""
from __future__ import annotations

import asyncio
import logging
import time

from psycopg_pool import AsyncConnectionPool

_log = logging.getLogger("opsrag.indexing.pg_store")

_UPSERT_PROGRESS = """
INSERT INTO opsrag_index_progress
    (repo, branch, status, source_type, display_name, total_files,
     indexed_files, skipped_files, total_chunks, entities_found, error,
     started_at, finished_at, updated_at)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now())
ON CONFLICT (repo, branch) DO UPDATE SET
    status         = EXCLUDED.status,
    source_type    = EXCLUDED.source_type,
    display_name   = COALESCE(EXCLUDED.display_name, opsrag_index_progress.display_name),
    total_files    = EXCLUDED.total_files,
    indexed_files  = EXCLUDED.indexed_files,
    skipped_files  = EXCLUDED.skipped_files,
    total_chunks   = EXCLUDED.total_chunks,
    entities_found = EXCLUDED.entities_found,
    error          = EXCLUDED.error,
    started_at     = EXCLUDED.started_at,
    finished_at    = EXCLUDED.finished_at,
    updated_at     = now()
"""

# Guarded variant: fill missing rows and refresh stale terminal rows, but NEVER
# stomp a row a live Job owns. Used both by the boot-time restore-from-Qdrant and
# by the serving pod's steady-state flush_loop(guarded=True) -- so it must keep
# ALL columns consistent (error/skipped_files/entities_found included), else a
# guarded UPDATE that flips status to 'done' would leave stale error text behind.
_BACKFILL_PROGRESS = """
INSERT INTO opsrag_index_progress
    (repo, branch, status, source_type, display_name, total_files,
     indexed_files, skipped_files, total_chunks, entities_found, error,
     started_at, finished_at, updated_at)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now())
ON CONFLICT (repo, branch) DO UPDATE SET
    status         = EXCLUDED.status,
    source_type    = EXCLUDED.source_type,
    display_name   = COALESCE(EXCLUDED.display_name, opsrag_index_progress.display_name),
    total_files    = EXCLUDED.total_files,
    indexed_files  = EXCLUDED.indexed_files,
    skipped_files  = EXCLUDED.skipped_files,
    total_chunks   = EXCLUDED.total_chunks,
    entities_found = EXCLUDED.entities_found,
    error          = EXCLUDED.error,
    started_at     = EXCLUDED.started_at,
    finished_at    = EXCLUDED.finished_at,
    updated_at     = now()
WHERE opsrag_index_progress.status NOT IN ('listing', 'indexing')
"""

_UPSERT_RUN = """
INSERT INTO opsrag_index_runs
    (run_key, repo, branch, source_type, display_name, status, started_at,
     finished_at, chunks_indexed, files_indexed, error, kind)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
ON CONFLICT (run_key) DO UPDATE SET
    status         = EXCLUDED.status,
    display_name   = COALESCE(EXCLUDED.display_name, opsrag_index_runs.display_name),
    finished_at    = EXCLUDED.finished_at,
    chunks_indexed = EXCLUDED.chunks_indexed,
    files_indexed  = EXCLUDED.files_indexed,
    error          = EXCLUDED.error
"""


def _run_key(proc_token: str, job: dict) -> str:
    """Process-unique key for a run row. ``restored`` runs get a STABLE key so
    re-running the boot backfill doesn't duplicate them every restart."""
    if job.get("kind") == "restored":
        return f"restored:{job['repo']}@{job['branch']}"
    return f"{proc_token}:{job['id']}"


class PostgresIndexStore:
    def __init__(self, dsn: str, *, proc_token: str, min_size: int = 1, max_size: int = 4) -> None:
        self._dsn = dsn
        self._proc_token = proc_token
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
        """No-op: migration 0009 owns the DDL (parity with the other *Store
        classes so the boot sequence can call it uniformly)."""
        return None

    # -- writer side ---------------------------------------------------------
    async def flush(self, summary: dict, jobs: dict, *, guarded: bool = False) -> None:
        """UPSERT the writer's current per-repo state + run history. Non-fatal.

        ``guarded=True`` uses the WHERE-guarded PROGRESS upsert so this writer
        never stomps a row a live (ephemeral Job) writer owns (status
        ``listing``/``indexing``). The serving pod passes this when a Job
        launcher is active (real concurrency); dev/in-process (no launcher)
        leaves it False so it can still advance its own ``indexing`` -> ``done``.
        The run-row upsert is key-isolated by ``run_key`` and stays unchanged."""
        if not self._ready:
            return
        repos = (summary or {}).get("repos", []) or []
        runs = (jobs or {}).get("jobs", []) or []
        progress_sql = _BACKFILL_PROGRESS if guarded else _UPSERT_PROGRESS
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    for r in repos:
                        await cur.execute(progress_sql, _progress_row(r))
                    for j in runs:
                        await cur.execute(_UPSERT_RUN, (
                            _run_key(self._proc_token, j),
                            j["repo"], j["branch"], j.get("source_type", "git"),
                            j.get("display_name"), j["status"],
                            j.get("started_at") or 0, j.get("finished_at") or 0,
                            j.get("chunks_indexed", 0), j.get("files_indexed", 0),
                            j.get("error"), j.get("kind", "run"),
                        ))
        except Exception as exc:
            _log.warning("index-state flush failed: %s -- continuing", exc)

    async def backfill_upsert(self, repos: list[dict], runs: list[dict]) -> None:
        """Boot-time restore-from-Qdrant. Guarded so a live Job's row is never
        clobbered; restored runs are idempotent across restarts. Non-fatal."""
        if not self._ready:
            return
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    for r in repos:
                        await cur.execute(_BACKFILL_PROGRESS, _progress_row(r))
                    for j in runs:
                        await cur.execute(_UPSERT_RUN, (
                            _run_key(self._proc_token, j),
                            j["repo"], j["branch"], j.get("source_type", "git"),
                            j.get("display_name"), j.get("status", "success"),
                            j.get("started_at") or 0, j.get("finished_at") or 0,
                            j.get("chunks_indexed", 0), j.get("files_indexed", 0),
                            j.get("error"), j.get("kind", "restored"),
                        ))
        except Exception as exc:
            _log.warning("index-state backfill_upsert failed: %s -- continuing", exc)

    # -- reader side (backend pods) -----------------------------------------
    async def read_summary(self) -> dict:
        """Same shape as ``IndexingTracker.get_summary`` -- percent/elapsed are
        recomputed from the stored epochs so all pods agree.

        ``total_chunks`` is sourced from ``indexed_files.chunk_count`` (the
        persistent per-file tally that mirrors what is actually in the vector
        store) rather than ``opsrag_index_progress.total_chunks`` (which only
        counts chunks *produced this run*). On an incremental re-index every
        unchanged file is skipped and produces 0 chunks, so the per-run column
        collapses to 0 for skipped repos and the "In vector store" stat would
        undercount by ~8x even though Qdrant is untouched. We fall back to the
        per-run value for sources not tracked in ``indexed_files``
        (slack/confluence/rootly, or a repo mid first-index)."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT repo, branch, status, source_type, display_name, "
                    "total_files, indexed_files, skipped_files, total_chunks, "
                    "entities_found, error, started_at, finished_at "
                    "FROM opsrag_index_progress ORDER BY repo, branch"
                )
                rows = await cur.fetchall()
                # Source-of-truth chunk counts per (repo, branch). Guarded: a
                # dev/no-postgres-dedup setup may lack the table -- degrade to
                # the per-run counts instead of failing the whole summary.
                store_chunks: dict[tuple[str, str], int] = {}
                try:
                    await cur.execute(
                        "SELECT repo, branch, SUM(chunk_count) "
                        "FROM indexed_files GROUP BY repo, branch"
                    )
                    for r_repo, r_branch, r_chunks in await cur.fetchall():
                        store_chunks[(r_repo, r_branch)] = int(r_chunks or 0)
                except Exception as exc:
                    _log.warning(
                        "indexed_files chunk rollup failed: %s -- using per-run counts",
                        exc,
                    )
        now = time.time()
        repos: list[dict] = []
        t_files = t_indexed = t_chunks = 0
        for row in rows:
            (repo, branch, status, st, dn, tf, idx, skp, tc, ent, err, sa, fa) = row
            # Prefer the persistent vector-store tally; keep the per-run value
            # for repos absent from indexed_files (non-git sources / first run).
            tc = store_chunks.get((repo, branch), tc)
            processed = idx + skp
            percent = round(processed / tf * 100, 1) if tf else 0.0
            elapsed = round((fa or now) - sa, 1) if sa else 0.0
            repos.append({
                "repo": repo, "branch": branch, "status": status,
                "source_type": st, "display_name": dn,
                "total_files": tf, "indexed_files": idx, "skipped_files": skp,
                "processed_files": processed, "total_chunks": tc,
                "entities_found": ent, "percent": percent,
                "elapsed_seconds": elapsed, "error": err,
                "started_at": sa, "finished_at": fa,
            })
            t_files += tf
            t_indexed += idx
            t_chunks += tc
        return {
            "total_repos": len(repos),
            "total_files": t_files,
            "total_indexed": t_indexed,
            "total_chunks": t_chunks,
            "repos": repos,
        }

    async def read_jobs(self, limit: int = 200) -> dict:
        """Same shape as ``IndexingTracker.get_jobs`` (newest-first)."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, repo, branch, source_type, display_name, status, "
                    "started_at, finished_at, chunks_indexed, files_indexed, "
                    "error, kind FROM opsrag_index_runs "
                    "ORDER BY created_at DESC LIMIT %s",
                    (limit,),
                )
                rows = await cur.fetchall()
        now = time.time()
        jobs: list[dict] = []
        running = failed = 0
        for row in rows:
            (jid, repo, branch, st, dn, status, sa, fa, ci, fi, err, kind) = row
            dur = round((fa or now) - sa, 1) if sa else 0.0
            jobs.append({
                "id": jid, "repo": repo, "branch": branch, "source_type": st,
                "display_name": dn, "status": status, "started_at": sa,
                "finished_at": fa or None, "duration_seconds": dur,
                "chunks_indexed": ci, "files_indexed": fi, "error": err, "kind": kind,
            })
            if status == "running":
                running += 1
            elif status == "failed":
                failed += 1
        return {"jobs": jobs, "total": len(jobs), "running": running, "failed": failed}


async def flush_loop(store: PostgresIndexStore, tracker, *, interval: float = 2.0,
                     stop_event: asyncio.Event, guarded: bool = False) -> None:
    """Periodically flush the in-memory tracker to Postgres until ``stop_event``
    is set, then do one final flush. Shared by the writer roles (server.py
    lifespan) and the ephemeral job-indexer entrypoint so the dashboard tracks
    progress in near-real-time without per-file DB writes.

    ``guarded`` is forwarded to :meth:`PostgresIndexStore.flush`: the serving
    pod sets it when a Job launcher is active so its flush never reverts a live
    Job's ``indexing`` row back to ``done`` with stale counts. The job-indexer
    and dev/in-process paths leave it False (they own the row they advance)."""
    try:
        while not stop_event.is_set():
            await store.flush(tracker.get_summary(), tracker.get_jobs(), guarded=guarded)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except TimeoutError:
                pass
    finally:
        # Final flush so the terminal state lands even on a fast exit.
        await store.flush(tracker.get_summary(), tracker.get_jobs(), guarded=guarded)


def _progress_row(r: dict) -> tuple:
    return (
        r["repo"], r["branch"], r["status"], r.get("source_type", "git"),
        r.get("display_name"), r.get("total_files", 0), r.get("indexed_files", 0),
        r.get("skipped_files", 0), r.get("total_chunks", 0), r.get("entities_found", 0),
        r.get("error"), r.get("started_at") or 0, r.get("finished_at") or 0,
    )
