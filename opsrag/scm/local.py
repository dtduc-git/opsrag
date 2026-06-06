"""Local filesystem SCM provider -- reads files from cloned git repos.

Used by GitCloneSCM as the read layer after a shallow clone, and can
also be used standalone for dev/testing against local directories.
"""
from __future__ import annotations

import asyncio
import fnmatch
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

from opsrag.interfaces.scm import RepoFile, WebhookEvent


def _matches_any(path: str, patterns: list[str]) -> bool:
    for pat in patterns:
        if fnmatch.fnmatch(path, pat):
            return True
        if pat.startswith("**/") and fnmatch.fnmatch(path, pat[3:]):
            return True
    return False


class LocalFSSCM:
    """Reads files from a directory on the local filesystem."""

    def __init__(self, base_dir: str | Path):
        self._base = Path(base_dir)

    async def list_repos(self, org: str) -> list[str]:
        return [
            d.name for d in self._base.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

    async def list_files(
        self,
        repo: str,
        branch: str = "main",
        patterns: list[str] | None = None,
    ) -> list[str]:
        # os.walk over a large repo (gitops-monorepo has 1000s of files)
        # blocks the event loop for hundreds of ms. Push to a thread.
        return await asyncio.to_thread(self._list_files_sync, repo, patterns)

    def _list_files_sync(
        self, repo: str, patterns: list[str] | None
    ) -> list[str]:
        repo_dir = self._base / repo.replace("/", "__")
        if not repo_dir.exists():
            repo_dir = self._base / repo
        if not repo_dir.exists():
            return []

        all_paths: list[str] = []
        for root, _dirs, files in os.walk(repo_dir):
            # Skip .git directory
            if "/.git" in root or root.endswith("/.git"):
                continue
            for f in files:
                full = Path(root) / f
                rel = str(full.relative_to(repo_dir))
                all_paths.append(rel)

        if patterns:
            return [p for p in all_paths if _matches_any(p, patterns)]
        return all_paths

    async def get_file(self, repo: str, path: str, branch: str = "main") -> RepoFile:
        # full.read_text() and stat() are sync disk I/O -- wrap in a thread
        # so concurrent file pipelines don't pin the event loop.
        return await asyncio.to_thread(self._get_file_sync, repo, path, branch)

    def _get_file_sync(self, repo: str, path: str, branch: str) -> RepoFile:
        repo_dir = self._base / repo.replace("/", "__")
        if not repo_dir.exists():
            repo_dir = self._base / repo
        full = repo_dir / path
        if not full.exists():
            raise FileNotFoundError(f"{repo}/{path}")

        content = full.read_text(errors="replace")
        stat = full.stat()

        return RepoFile(
            path=path,
            content=content,
            sha="",
            last_modified=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
            repo=repo,
            branch=branch,
            metadata={"size": len(content)},
        )

    async def get_files_batch(
        self,
        repo: str,
        paths: list[str],
        branch: str = "main",
    ) -> AsyncIterator[RepoFile]:
        for p in paths:
            try:
                yield await self.get_file(repo, p, branch)
            except (FileNotFoundError, UnicodeDecodeError):
                continue

    async def get_changed_files_since(
        self,
        repo: str,
        since: datetime,
        branch: str = "main",
    ) -> list[RepoFile]:
        return []

    def parse_webhook(self, payload: dict, headers: dict) -> WebhookEvent:
        return WebhookEvent(
            event_type="manual",
            repo="",
            branch="main",
            changed_files=[],
            timestamp=datetime.now(UTC),
        )
