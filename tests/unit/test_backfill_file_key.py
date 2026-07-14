"""Unit tests: safety gates of the file_key backfill tool.

The tool's whole job is refusing to declare success while ANY point could
still be missing file_key -- a single missed point makes the flag-ON delete
path silently skip it (stale chunks keep matching queries). Pin the three
gates: durable-retry failure aggregation, the refuse-success exit code, and
--create-index gating on the exhaustive missing==0 verify.
"""
from __future__ import annotations

import argparse

from qdrant_client import models as qm

from opsrag.tools import backfill_file_key as bf


class _FakeClient:
    """Minimal AsyncQdrantClient stand-in: one scroll page of `points`,
    scripted set_payload failures, scripted missing-count."""

    def __init__(self, points, *, fail_set_payload=0, missing=0):
        self._points = points
        self._fail_remaining = fail_set_payload
        self._missing = missing
        self.set_payload_calls: list[tuple[str, list]] = []
        self.created_indexes: list[str] = []

    async def scroll(self, collection_name, limit, offset, with_payload, with_vectors):
        return (self._points, None) if offset is None else ([], None)

    async def set_payload(self, collection_name, payload, points, wait):
        assert wait is True, "backfill writes must be durable (wait=True)"
        if self._fail_remaining:
            self._fail_remaining -= 1
            raise RuntimeError("write executor saturated")
        self.set_payload_calls.append((payload["file_key"], points))

    async def count(self, collection_name, count_filter, exact):
        assert exact is True
        return qm.CountResult(count=self._missing)

    async def create_payload_index(self, collection_name, field_name, field_schema, wait):
        self.created_indexes.append(field_name)

    async def close(self):
        pass


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        url="http://fake:6333", collection=["c1"], api_key_env=None,
        page_size=512, sleep_ms=0, timeout=5.0,
        dry_run=False, verify_only=False, create_index=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _point(pid, repo, path):
    from types import SimpleNamespace
    return SimpleNamespace(id=pid, payload={"repo": repo, "source_path": path})


async def test_retry_exhaustion_reports_failure_not_success(monkeypatch):
    monkeypatch.setattr(bf, "_BACKOFF_BASE_S", 0.0)  # no real sleeping in tests
    client = _FakeClient([_point("p1", "r", "a.md")], fail_set_payload=bf._RETRIES)
    monkeypatch.setattr(bf, "AsyncQdrantClient", lambda **kw: client)

    exit_code = await bf.run(_args())

    assert exit_code == 1, "residual failed points MUST refuse success"
    assert client.set_payload_calls == []


async def test_transient_failure_retried_to_success(monkeypatch):
    monkeypatch.setattr(bf, "_BACKOFF_BASE_S", 0.0)
    client = _FakeClient([_point("p1", "r", "a.md")], fail_set_payload=2)
    monkeypatch.setattr(bf, "AsyncQdrantClient", lambda **kw: client)

    exit_code = await bf.run(_args())

    assert exit_code == 0
    assert client.set_payload_calls == [("r\x00a.md", ["p1"])]


async def test_create_index_gated_on_exhaustive_verify(monkeypatch):
    # Backfill "succeeds" but the server-side exhaustive count still reports a
    # missing point (e.g. a concurrent writer without the new code) -> the
    # index must NOT be created and the run must fail.
    monkeypatch.setattr(bf, "_BACKOFF_BASE_S", 0.0)
    client = _FakeClient([_point("p1", "r", "a.md")], missing=1)
    monkeypatch.setattr(bf, "AsyncQdrantClient", lambda **kw: client)

    exit_code = await bf.run(_args(create_index=True))

    assert exit_code == 1
    assert client.created_indexes == []


async def test_create_index_runs_when_verify_clean(monkeypatch):
    monkeypatch.setattr(bf, "_BACKOFF_BASE_S", 0.0)
    client = _FakeClient([_point("p1", "r", "a.md")], missing=0)
    monkeypatch.setattr(bf, "AsyncQdrantClient", lambda **kw: client)

    exit_code = await bf.run(_args(create_index=True))

    assert exit_code == 0
    assert client.created_indexes == ["file_key"]


async def test_dry_run_writes_nothing(monkeypatch):
    client = _FakeClient([_point("p1", "r", "a.md")])
    monkeypatch.setattr(bf, "AsyncQdrantClient", lambda **kw: client)

    exit_code = await bf.run(_args(dry_run=True, create_index=True))

    assert exit_code == 0
    assert client.set_payload_calls == []
    assert client.created_indexes == []
