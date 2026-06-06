"""No-op tracker -- every file is treated as new. Used when dedup is
disabled (no Postgres DSN, or session.provider != postgres)."""
from __future__ import annotations

from collections.abc import Iterable


class NoopIndexedFilesTracker:
    async def open(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def should_skip(
        self, repo: str, branch: str, path: str, content_hash: str
    ) -> bool:
        return False

    async def record(
        self,
        repo: str,
        branch: str,
        path: str,
        content_hash: str,
        chunk_count: int,
    ) -> None:
        return None

    async def mark_seen(
        self, repo: str, branch: str, paths: Iterable[str]
    ) -> None:
        return None
