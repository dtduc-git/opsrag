"""Unit test (FINDING #2): repo-level deletion sweep.

Files removed from source between index runs must be purged from the vector
store and dropped from the indexed_files tracker at the END of a successful
``index_repo`` pass. Earlier code tracked ``last_seen_at`` but never acted on
it, so decommissioned services / revoked runbooks stayed retrievable forever.

The test wires a fake tracker that faithfully mimics the Postgres
``last_seen_at`` semantics (a bumped timestamp on every seen file, a stale one
on files not seen this run) and a fake vector store recording delete-by-filter
calls, then drives the real ``IngestionPipeline.index_repo``:

  run 1: repo has [a.md, b.md]  -> both indexed, tracker has 2 rows
  run 2: repo has [a.md] only   -> b.md is swept: chunks deleted + row dropped
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest

from opsrag.ingestion.pipeline import IngestionPipeline
from opsrag.interfaces.scm import RepoFile


class _FakeTracker:
    """In-memory stand-in for PostgresIndexedFilesTracker that reproduces the
    last_seen_at semantics the sweep relies on. ``record`` and ``mark_seen``
    bump last_seen_at to "now"; ``sweep_deleted`` drops + returns rows older
    than the run start."""

    def __init__(self) -> None:
        # (repo, branch, path) -> last_seen_at
        self.rows: dict[tuple[str, str, str], datetime] = {}
        self._last = datetime.now(UTC)

    def _now(self) -> datetime:
        # Real wall-clock NOW (like the DB's NOW()), so a bump always lands
        # AFTER the run_started_at = datetime.now(utc) captured at the top of
        # index_repo. Monotonic-guarded so two bumps in the same microsecond
        # still order strictly.
        now = datetime.now(UTC)
        if now <= self._last:
            now = self._last + timedelta(microseconds=1)
        self._last = now
        return now

    async def should_skip(self, repo, branch, path, content_hash) -> bool:
        return False

    async def record(self, repo, branch, path, content_hash, chunk_count) -> None:
        self.rows[(repo, branch, path)] = self._now()

    async def mark_seen(self, repo, branch, paths) -> None:
        for p in paths:
            if (repo, branch, p) in self.rows:
                self.rows[(repo, branch, p)] = self._now()

    async def sweep_deleted(self, repo, branch, run_started_at) -> list[str]:
        stale = [
            path
            for (r, b, path), seen in self.rows.items()
            if r == repo and b == branch and seen < run_started_at
        ]
        for path in stale:
            del self.rows[(repo, branch, path)]
        return stale


class _FakeVectorStore:
    """Records delete_by_filter calls; upsert is a no-op."""

    def __init__(self) -> None:
        self.deleted: list[dict] = []

    async def delete_by_filter(self, flt: dict) -> None:
        self.deleted.append(flt)

    async def upsert(self, chunks, embeddings) -> None:  # pragma: no cover
        return None


class _FakeSCM:
    """Returns a configurable set of files per run."""

    def __init__(self, files: dict[str, str]) -> None:
        self.files = files

    async def list_files(self, repo, branch="main", patterns=None) -> list[str]:
        return list(self.files)

    async def get_files_batch(
        self, repo, paths, branch="main"
    ) -> AsyncIterator[RepoFile]:
        for p in paths:
            yield RepoFile(
                path=p,
                content=self.files[p],
                sha="0" * 64,
                last_modified=datetime(2026, 1, 1, tzinfo=UTC),
                repo=repo,
                branch=branch,
            )


def _make_pipeline(scm, tracker, store) -> IngestionPipeline:
    pipe = IngestionPipeline(
        scm=scm,
        parsers=[],
        chunker=None,
        embedder=None,
        vector_store=store,
        indexed_files=tracker,
    )

    # Stub the heavy parse/chunk/embed body: record the file in the tracker
    # exactly as the real _process_file does (record on index), so the
    # last_seen_at semantics that the sweep depends on are exercised end to
    # end through index_repo.
    async def _fake_process_file(file: RepoFile) -> int:
        await tracker.record(file.repo, file.branch, file.path, "h", 1)
        return 1

    pipe._process_file = _fake_process_file  # type: ignore[method-assign]
    return pipe


@pytest.mark.asyncio
async def test_deleted_file_is_swept_on_reindex():
    tracker = _FakeTracker()
    store = _FakeVectorStore()
    repo, branch = "svc", "main"

    # Run 1: two files present -> both tracked, none swept.
    scm = _FakeSCM({"a.md": "alpha", "b.md": "bravo"})
    pipe = _make_pipeline(scm, tracker, store)
    await pipe.index_repo(repo, branch)

    assert (repo, branch, "a.md") in tracker.rows
    assert (repo, branch, "b.md") in tracker.rows
    assert store.deleted == []  # nothing removed on a first clean index

    # Real-time gap so run 2's run_started_at (datetime.now) is provably after
    # run 1's last_seen_at bumps regardless of clock resolution.
    await asyncio.sleep(0.005)

    # Run 2: b.md deleted from source -> only a.md streamed.
    scm.files = {"a.md": "alpha"}
    await pipe.index_repo(repo, branch)

    # b.md's chunks purged from the vector store via the orphan-delete path...
    assert {"repo": repo, "source_path": "b.md"} in store.deleted
    # ...and never a.md (still present).
    assert {"repo": repo, "source_path": "a.md"} not in store.deleted
    # ...and its tracker row dropped (marked invalid / removed).
    assert (repo, branch, "b.md") not in tracker.rows
    assert (repo, branch, "a.md") in tracker.rows


@pytest.mark.asyncio
async def test_no_sweep_when_nothing_deleted():
    """A re-index with the same file set must not purge anything."""
    tracker = _FakeTracker()
    store = _FakeVectorStore()
    repo, branch = "svc", "main"
    scm = _FakeSCM({"a.md": "alpha"})
    pipe = _make_pipeline(scm, tracker, store)

    await pipe.index_repo(repo, branch)
    await asyncio.sleep(0.005)
    await pipe.index_repo(repo, branch)

    assert store.deleted == []
    assert (repo, branch, "a.md") in tracker.rows
