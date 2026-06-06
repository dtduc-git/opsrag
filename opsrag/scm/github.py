"""GitHub SCM provider.

Implements SCMProvider protocol against the GitHub REST API v3 using httpx.
Auth via personal access token (PAT) or GitHub Apps installation token.
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
    for pat in patterns:
        if fnmatch.fnmatch(path, pat):
            return True
        if pat.startswith("**/") and fnmatch.fnmatch(path, pat[3:]):
            return True
    return False


class GitHubSCM:
    """GitHub REST API v3 SCM provider."""

    def __init__(
        self,
        token: str,
        base_url: str = "https://api.github.com",
        timeout: float = 30.0,
        max_concurrent: int = 10,
    ):
        self._token = token
        self._api = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github.v3+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=timeout,
        )
        self._sem = asyncio.Semaphore(max_concurrent)

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> GitHubSCM:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def list_repos(self, org: str) -> list[str]:
        repos: list[str] = []
        page = 1
        while True:
            resp = await self._client.get(
                f"{self._api}/orgs/{quote(org)}/repos",
                params={"per_page": 100, "page": page, "type": "all"},
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            repos.extend(r["full_name"] for r in batch)
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
        all_paths: list[str] = []
        resp = await self._client.get(
            f"{self._api}/repos/{repo}/git/trees/{quote(branch)}",
            params={"recursive": "1"},
        )
        resp.raise_for_status()
        tree = resp.json().get("tree", [])
        all_paths = [e["path"] for e in tree if e["type"] == "blob"]

        if patterns:
            return [p for p in all_paths if _matches_any(p, patterns)]
        return all_paths

    async def get_file(self, repo: str, path: str, branch: str = "main") -> RepoFile:
        encoded_path = quote(path, safe="")
        async with self._sem:
            resp = await self._client.get(
                f"{self._api}/repos/{repo}/contents/{encoded_path}",
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
        # Fetch last commit date for this file
        try:
            commits_resp = await self._client.get(
                f"{self._api}/repos/{repo}/commits",
                params={"path": path, "sha": branch, "per_page": 1},
            )
            if commits_resp.status_code == 200:
                commits = commits_resp.json()
                if commits:
                    ts = commits[0].get("commit", {}).get("committer", {}).get("date")
                    if ts:
                        last_modified = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            pass

        return RepoFile(
            path=path,
            content=content,
            sha=data.get("sha", ""),
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
        tasks = [asyncio.create_task(self.get_file(repo, p, branch)) for p in paths]
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
        resp = await self._client.get(
            f"{self._api}/repos/{repo}/commits",
            params={
                "sha": branch,
                "since": since.isoformat(),
                "per_page": 100,
            },
        )
        resp.raise_for_status()
        commits = resp.json()

        changed: set[str] = set()
        for c in commits:
            detail_resp = await self._client.get(
                f"{self._api}/repos/{repo}/commits/{c['sha']}"
            )
            if detail_resp.status_code == 200:
                for f in detail_resp.json().get("files", []):
                    if f.get("status") != "removed":
                        changed.add(f["filename"])

        files: list[RepoFile] = []
        async for f in self.get_files_batch(repo, list(changed), branch):
            files.append(f)
        return files

    def parse_webhook(self, payload: dict, headers: dict) -> WebhookEvent:
        event_type = headers.get("X-GitHub-Event", "push")
        repo = payload.get("repository", {}).get("full_name", "")
        ref = payload.get("ref", "refs/heads/main")
        branch = ref.rsplit("/", 1)[-1]

        changed: set[str] = set()
        for commit in payload.get("commits", []):
            changed.update(commit.get("added", []))
            changed.update(commit.get("modified", []))

        ts_str = payload.get("head_commit", {}).get("timestamp") if payload.get("head_commit") else None
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")) if ts_str else datetime.now(UTC)

        return WebhookEvent(
            event_type=event_type,
            repo=repo,
            branch=branch,
            changed_files=sorted(changed),
            timestamp=ts,
            raw_payload=payload,
        )
