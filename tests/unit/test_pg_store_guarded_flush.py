"""Unit test (F1): the serving pod's *guarded* flush must not revert a row a
live ephemeral Job owns (status ``listing``/``indexing``) back to ``done`` with
stale counts, while the dev/in-process *unguarded* flush still advances its own
``indexing`` -> ``done``.

We don't need a live Postgres: the guard lives entirely in WHICH SQL statement
``flush`` chooses (the WHERE-guarded ``_BACKFILL_PROGRESS`` vs the unguarded
``_UPSERT_PROGRESS``). We capture the SQL each PROGRESS row is executed with via
a fake connection/cursor, and separately assert the guarded SQL actually carries
the ``status NOT IN ('listing','indexing')`` clause that blocks the stomp.
"""
from __future__ import annotations

import asyncio

import pytest

from opsrag.indexing.pg_store import (
    _BACKFILL_PROGRESS,
    _UPSERT_PROGRESS,
    PostgresIndexStore,
    flush_loop,
)


class _FakeCursor:
    def __init__(self, calls: list) -> None:
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, sql, params):
        self._calls.append((sql, params))


class _FakeConn:
    def __init__(self, calls: list) -> None:
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def cursor(self):
        return _FakeCursor(self._calls)


class _FakePool:
    def __init__(self, calls: list) -> None:
        self._calls = calls

    def connection(self):
        return _FakeConn(self._calls)


def _store_with_fake_pool(calls: list) -> PostgresIndexStore:
    # Build the store object without opening a real pool, then swap in the fake.
    store = PostgresIndexStore.__new__(PostgresIndexStore)
    store._proc_token = "proc-test"
    store._pool = _FakePool(calls)
    store._ready = True
    return store


# A live Job owns this repo and is mid-index; the serving pod restored a stale
# "done" snapshot for the same (repo, branch) from Qdrant.
_LIVE_INDEXING_SUMMARY = {
    "repos": [{
        "repo": "git@example:acme/app", "branch": "main", "status": "done",
        "source_type": "git", "display_name": "acme/app",
        "total_files": 100, "indexed_files": 100, "skipped_files": 0,
        "total_chunks": 500, "entities_found": 0, "error": None,
        "started_at": 1, "finished_at": 2,
    }],
}
_NO_JOBS = {"jobs": []}


def test_guard_clause_present_only_in_backfill_sql():
    """Sanity-check the two PROGRESS statements: the guarded one carries the
    WHERE clause that prevents stomping a live Job row; the plain one does not."""
    guard = "status NOT IN ('listing', 'indexing')"
    assert guard in _BACKFILL_PROGRESS
    assert guard not in _UPSERT_PROGRESS


def test_guarded_progress_update_refreshes_all_columns():
    """The guarded statement is now the serving pod's steady-state writer
    (flush_loop(guarded=True)), not just boot backfill. Its ON CONFLICT UPDATE
    must refresh ALL mutable columns -- especially error, skipped_files and
    entities_found -- so a guarded UPDATE that passes the WHERE guard (e.g. flips
    a 'failed' row to 'done') doesn't leave stale error text / counts behind."""
    # Only inspect the ON CONFLICT ... DO UPDATE SET clause (the VALUES list
    # naturally names every column).
    update_clause = _BACKFILL_PROGRESS.split("DO UPDATE SET", 1)[1]
    for col in ("error", "skipped_files", "entities_found"):
        assert f"{col}" in update_clause, f"guarded UPDATE drops {col}"
        assert f"EXCLUDED.{col}" in update_clause, f"guarded UPDATE must set {col} from EXCLUDED"
    # Guarded and plain upserts now refresh the same column set so a writer
    # switching paths can't silently lose columns.
    plain_update = _UPSERT_PROGRESS.split("DO UPDATE SET", 1)[1]
    for col in ("status", "source_type", "display_name", "total_files",
                "indexed_files", "skipped_files", "total_chunks",
                "entities_found", "error", "started_at", "finished_at"):
        assert f"EXCLUDED.{col}" in update_clause, f"guarded missing {col}"
        assert f"EXCLUDED.{col}" in plain_update, f"plain missing {col}"


@pytest.mark.asyncio
async def test_guarded_flush_passes_error_and_skipped_in_params():
    """End-to-end through flush(guarded=True): the params bound for the PROGRESS
    UPSERT carry the refreshed error / skipped_files / entities_found values so
    the (now-extended) ON CONFLICT UPDATE has something current to write."""
    calls: list = []
    store = _store_with_fake_pool(calls)

    summary = {
        "repos": [{
            "repo": "git@example:acme/app", "branch": "main", "status": "done",
            "source_type": "git", "display_name": "acme/app",
            "total_files": 100, "indexed_files": 90, "skipped_files": 10,
            "total_chunks": 500, "entities_found": 42, "error": "prior boom",
            "started_at": 1, "finished_at": 2,
        }],
    }
    await store.flush(summary, _NO_JOBS, guarded=True)

    progress_calls = [c for c in calls if "opsrag_index_progress" in c[0]]
    assert progress_calls, "expected a PROGRESS upsert"
    _sql, params = progress_calls[0]
    # _progress_row order: repo, branch, status, source_type, display_name,
    # total_files, indexed_files, skipped_files, total_chunks, entities_found,
    # error, started_at, finished_at
    assert params[7] == 10, "skipped_files must be bound"
    assert params[9] == 42, "entities_found must be bound"
    assert params[10] == "prior boom", "error must be bound"


@pytest.mark.asyncio
async def test_guarded_flush_uses_where_guarded_progress_sql():
    calls: list = []
    store = _store_with_fake_pool(calls)

    await store.flush(_LIVE_INDEXING_SUMMARY, _NO_JOBS, guarded=True)

    progress_calls = [c for c in calls if "opsrag_index_progress" in c[0]]
    assert progress_calls, "expected a PROGRESS upsert"
    for sql, _params in progress_calls:
        # Guarded: must be the WHERE-guarded variant so it cannot move a live
        # 'indexing'/'listing' row back to 'done'.
        assert sql == _BACKFILL_PROGRESS
        assert "status NOT IN ('listing', 'indexing')" in sql


@pytest.mark.asyncio
async def test_unguarded_flush_uses_plain_progress_sql():
    calls: list = []
    store = _store_with_fake_pool(calls)

    await store.flush(_LIVE_INDEXING_SUMMARY, _NO_JOBS)  # guarded defaults False

    progress_calls = [c for c in calls if "opsrag_index_progress" in c[0]]
    assert progress_calls, "expected a PROGRESS upsert"
    for sql, _params in progress_calls:
        # Unguarded (dev/in-process): plain upsert so it CAN advance its own
        # indexing -> done.
        assert sql == _UPSERT_PROGRESS
        assert "status NOT IN" not in sql


@pytest.mark.asyncio
async def test_flush_loop_forwards_guarded_flag():
    """flush_loop must forward ``guarded`` to store.flush so the serving pod's
    loop stays guarded for every periodic + final flush."""
    seen: list[bool] = []

    class _Store:
        async def flush(self, summary, jobs, *, guarded=False):
            seen.append(guarded)

    class _Tracker:
        def get_summary(self):
            return {"repos": []}

        def get_jobs(self):
            return {"jobs": []}

    stop = asyncio.Event()
    stop.set()  # pre-set: skip the periodic body, run only the final flush
    await flush_loop(_Store(), _Tracker(), stop_event=stop, guarded=True)
    # With stop pre-set, only the final flush runs; it must be guarded.
    assert seen == [True]
