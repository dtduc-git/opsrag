"""Protocol for tracking which (repo, branch, path) blobs are already indexed.

Used to skip re-embedding unchanged files on subsequent indexing runs, which
makes daily reindex cheap (only changed files cost Vertex tokens) and turns
container restarts into ~no-op for already-indexed repos.

A no-op implementation is provided for environments without Postgres so the
ingestion path stays unchanged when dedup is disabled.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Protocol


class IndexedFilesTracker(Protocol):
    async def open(self) -> None: ...
    async def close(self) -> None: ...

    async def should_skip(
        self, repo: str, branch: str, path: str, content_hash: str
    ) -> bool:
        """True if (repo, branch, path) is already recorded with this exact
        content_hash. The caller should still call ``mark_seen`` to keep
        last_seen_at fresh for retention policies."""
        ...

    async def record(
        self,
        repo: str,
        branch: str,
        path: str,
        content_hash: str,
        chunk_count: int,
    ) -> None:
        """Upsert a row after a successful indexing pass. Sets indexed_at
        and last_seen_at to now."""
        ...

    async def mark_seen(
        self, repo: str, branch: str, paths: Iterable[str]
    ) -> None:
        """Bulk-update last_seen_at for paths still present in the repo
        (whether or not they were re-indexed). Used so a future deletion
        sweep can identify files no longer present in source."""
        ...

    async def sweep_deleted(
        self, repo: str, branch: str, run_started_at: datetime
    ) -> list[str]:
        """Repo-level deletion sweep run at the END of a successful index pass.

        Returns the paths of rows for ``(repo, branch)`` whose ``last_seen_at``
        predates ``run_started_at`` -- i.e. files that existed before but were
        NOT seen during this run (deleted from source) -- and removes their
        tracker rows. The caller is responsible for purging the corresponding
        chunks from the vector store. Returns an empty list if the tracker is
        not ready or nothing is stale."""
        ...
