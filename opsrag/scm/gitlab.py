"""GitLab SCM provider.

Implements SCMProvider protocol against the GitLab REST API v4 using httpx.
Supports self-hosted and gitlab.com. Auth via personal access token.
"""
from __future__ import annotations

import asyncio
import base64
import fnmatch
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from urllib.parse import quote

import httpx

from opsrag.interfaces.scm import RepoFile, WebhookEvent


def _matches_any(path: str, patterns: list[str]) -> bool:
    """fnmatch-based glob matching that also treats leading ``**/`` as optional
    so patterns like ``**/*.tf`` match ``main.tf`` at the repo root."""
    for pat in patterns:
        if fnmatch.fnmatch(path, pat):
            return True
        if pat.startswith("**/") and fnmatch.fnmatch(path, pat[3:]):
            return True
    return False


class GitLabSCM:
    """GitLab REST API v4 SCM provider."""

    def __init__(
        self,
        token: str,
        base_url: str = "https://gitlab.com",
        timeout: float = 30.0,
        max_concurrent: int = 10,
    ):
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._api = f"{self._base_url}/api/v4"
        self._client = httpx.AsyncClient(
            headers={"PRIVATE-TOKEN": token},
            timeout=timeout,
        )
        self._sem = asyncio.Semaphore(max_concurrent)

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> GitLabSCM:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    @staticmethod
    def _encode_project(repo: str) -> str:
        return quote(repo, safe="")

    async def list_repos(self, org: str) -> list[str]:
        """List projects under a group (org), recursively."""
        repos: list[str] = []
        page = 1
        while True:
            resp = await self._client.get(
                f"{self._api}/groups/{self._encode_project(org)}/projects",
                params={"per_page": 100, "page": page, "include_subgroups": "true"},
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            repos.extend(p["path_with_namespace"] for p in batch)
            if len(batch) < 100:
                break
            page += 1
        return repos

    async def list_files(
        self,
        repo: str,
        branch: str = "main",
        patterns: list[str] | None = None,
    ) -> list[str]:
        """List all file paths in a repo tree, filtered by glob patterns."""
        proj = self._encode_project(repo)
        all_paths: list[str] = []
        page = 1
        while True:
            resp = await self._client.get(
                f"{self._api}/projects/{proj}/repository/tree",
                params={
                    "ref": branch,
                    "recursive": "true",
                    "per_page": 100,
                    "page": page,
                },
            )
            resp.raise_for_status()
            entries = resp.json()
            if not entries:
                break
            all_paths.extend(e["path"] for e in entries if e["type"] == "blob")
            if len(entries) < 100:
                break
            page += 1

        if patterns:
            return [p for p in all_paths if _matches_any(p, patterns)]
        return all_paths

    async def get_file(self, repo: str, path: str, branch: str = "main") -> RepoFile:
        proj = self._encode_project(repo)
        encoded_path = quote(path, safe="")
        async with self._sem:
            resp = await self._client.get(
                f"{self._api}/projects/{proj}/repository/files/{encoded_path}",
                params={"ref": branch},
            )
            resp.raise_for_status()
            data = resp.json()

        content_b64 = data.get("content", "")
        try:
            content = base64.b64decode(content_b64).decode("utf-8", errors="replace")
        except Exception:
            content = ""

        last_modified = datetime.now(UTC)

        return RepoFile(
            path=path,
            content=content,
            sha=data.get("blob_id", ""),
            last_modified=last_modified,
            repo=repo,
            branch=branch,
            metadata={"size": data.get("size", len(content))},
        )

    async def get_files_batch(
        self,
        repo: str,
        paths: list[str],
        branch: str = "main",
    ) -> AsyncIterator[RepoFile]:
        """Fetch files in batches, yielding as each batch completes."""
        batch_size = self._sem._value  # match concurrency limit
        for i in range(0, len(paths), batch_size):
            batch = paths[i : i + batch_size]
            tasks = [asyncio.create_task(self.get_file(repo, p, branch)) for p in batch]
            for coro in asyncio.as_completed(tasks):
                try:
                    yield await coro
                except Exception:
                    continue

    async def get_changed_files_since(
        self,
        repo: str,
        since: datetime,
        branch: str = "main",
    ) -> list[RepoFile]:
        proj = self._encode_project(repo)
        resp = await self._client.get(
            f"{self._api}/projects/{proj}/repository/commits",
            params={
                "ref_name": branch,
                "since": since.isoformat(),
                "per_page": 100,
            },
        )
        resp.raise_for_status()
        commits = resp.json()

        changed: set[str] = set()
        for c in commits:
            diff_resp = await self._client.get(
                f"{self._api}/projects/{proj}/repository/commits/{c['id']}/diff"
            )
            if diff_resp.status_code == 200:
                for d in diff_resp.json():
                    if not d.get("deleted_file"):
                        changed.add(d["new_path"])

        files: list[RepoFile] = []
        async for f in self.get_files_batch(repo, list(changed), branch):
            files.append(f)
        return files

    def parse_webhook(self, payload: dict, headers: dict) -> WebhookEvent:
        event_type = headers.get("X-Gitlab-Event", "unknown")
        repo = payload.get("project", {}).get("path_with_namespace", "")
        ref = payload.get("ref", "refs/heads/main")
        branch = ref.rsplit("/", 1)[-1]

        changed: set[str] = set()
        for commit in payload.get("commits", []):
            changed.update(commit.get("added", []))
            changed.update(commit.get("modified", []))

        ts_str = payload.get("commits", [{}])[-1].get("timestamp") if payload.get("commits") else None
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")) if ts_str else datetime.now(UTC)

        return WebhookEvent(
            event_type=event_type,
            repo=repo,
            branch=branch,
            changed_files=sorted(changed),
            timestamp=ts,
            raw_payload=payload,
        )
