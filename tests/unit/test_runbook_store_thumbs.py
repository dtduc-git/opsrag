"""Unit tests: RunbookStore.record_thumbs — thumbs-only counter bump.

`record_use` bumps used_count (+last_used_at) in EVERY branch, which is
correct for its one real caller (runbook_load counting a use at load time)
but wrong for answer feedback: thumbing an answer that loaded a runbook
must bump ONLY thumbs_up_count/thumbs_down_count, or USED stops meaning
"times loaded". record_thumbs is that separate path.
"""
from __future__ import annotations

from opsrag.runbooks.store import RunbookStore


class _Cursor:
    def __init__(self, log):
        self._log = log

    async def execute(self, sql, params=None):
        self._log.append((" ".join(sql.split()), params))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _Conn:
    def __init__(self, log):
        self._log = log

    def cursor(self):
        return _Cursor(self._log)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _Pool:
    def __init__(self, log, *, raise_on_connect=False):
        self._log = log
        self._raise = raise_on_connect

    def connection(self):
        if self._raise:
            raise RuntimeError("pg down")
        return _Conn(self._log)


def _store(log, **pool_kw) -> RunbookStore:
    store = RunbookStore.__new__(RunbookStore)  # skip __init__ (needs qdrant)
    store._pg = _Pool(log, **pool_kw)
    return store


async def test_thumbs_up_bumps_only_thumbs_counter():
    log: list = []
    await _store(log).record_thumbs("rb1", thumbs="up")

    assert len(log) == 1
    sql, params = log[0]
    assert "thumbs_up_count = thumbs_up_count + 1" in sql
    assert "used_count" not in sql
    assert "last_used_at" not in sql
    assert params == ("rb1",)


async def test_thumbs_down_symmetric():
    log: list = []
    await _store(log).record_thumbs("rb2", thumbs="down")

    sql, _ = log[0]
    assert "thumbs_down_count = thumbs_down_count + 1" in sql
    assert "used_count" not in sql


async def test_invalid_thumbs_executes_nothing():
    log: list = []
    await _store(log).record_thumbs("rb3", thumbs="sideways")
    assert log == []


async def test_pg_failure_never_raises():
    await _store([], raise_on_connect=True).record_thumbs("rb4", thumbs="up")
