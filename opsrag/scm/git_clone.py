"""Git-clone SCM provider -- shallow clones repos, then reads from disk.

Clones once per repo into a local cache directory, then uses LocalFSSCM
for all file reads. Much faster than API-per-file fetching, and avoids
GitLab/GitHub rate limits.

Supports both GitLab and GitHub. Auth via token embedded in the clone URL,
or via SSH (set use_ssh=True) when an HTTPS proxy blocks git clone.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import shutil
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from opsrag.interfaces.scm import RepoFile, WebhookEvent
from opsrag.scm.local import LocalFSSCM

_log = logging.getLogger("opsrag.scm.git_clone")


class GitCloneSCM:
    def __init__(
        self,
        base_url: str,
        token: str,
        cache_dir: str = "/tmp/opsrag-repos",
        provider: str = "gitlab",
        use_ssh: bool = False,
        ssh_host: str | None = None,
        ssh_user: str = "git",
    ):
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._cache = Path(cache_dir)
        self._cache.mkdir(parents=True, exist_ok=True)
        self._provider = provider
        self._use_ssh = use_ssh
        self._ssh_user = ssh_user
        self._ssh_host = ssh_host or urlparse(self._base_url).hostname or ""
        self._cloned: set[str] = set()
        self._local = LocalFSSCM(self._cache)

    def _clone_url(self, repo: str) -> str:
        if self._use_ssh:
            return f"{self._ssh_user}@{self._ssh_host}:{repo}.git"
        return f"{self._base_url}/{repo}.git"

    def _auth_header(self) -> str:
        if self._provider == "github":
            return f"Authorization: Bearer {self._token}"
        # GitLab: `PRIVATE-TOKEN` is for the REST API only -- git-http-backend
        # (gitlab-workhorse) ignores it, so the clone falls back to a
        # username prompt and fails with `could not read Username` in non-TTY
        # contexts (pods, CI). Use HTTP Basic auth with `oauth2:<pat>` instead;
        # this is GitLab's documented HTTPS-with-PAT pattern and works on
        # every version.
        encoded = base64.b64encode(f"oauth2:{self._token}".encode()).decode()
        return f"Authorization: Basic {encoded}"

    def _git_env(self) -> dict[str, str]:
        """Env for git subprocesses; in SSH mode, relax host key checks.

        Operators can fully override the ssh invocation via
        ``OPSRAG_GIT_SSH_COMMAND`` -- needed when the key is host-mounted into
        a container running as a different uid (point ``-i`` at the key and add
        ``-o StrictModes=no``), or to pin a custom known_hosts / identity.
        """
        env = dict(os.environ)
        if self._use_ssh:
            env["GIT_SSH_COMMAND"] = os.environ.get("OPSRAG_GIT_SSH_COMMAND") or (
                # Default: non-interactive, accept new host keys on first sight.
                "ssh -o StrictHostKeyChecking=accept-new "
                "-o BatchMode=yes "
                "-o UserKnownHostsFile=/root/.ssh/known_hosts"
            )
        return env

    def _repo_dir(self, repo: str) -> Path:
        return self._cache / repo.replace("/", "__")

    async def _ensure_cloned(self, repo: str, branch: str) -> None:
        key = f"{repo}@{branch}"
        if key in self._cloned:
            return

        repo_dir = self._repo_dir(repo)
        env = self._git_env()
        auth_args = (
            ["-c", f"http.extraHeader={self._auth_header()}"]
            if not self._use_ssh
            else []
        )

        if repo_dir.exists():
            _log.info("git pull repo=%s branch=%s", repo, branch)
            proc = await asyncio.create_subprocess_exec(
                "git", *auth_args,
                "-C", str(repo_dir), "pull", "--ff-only",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            await proc.wait()
        else:
            _log.info("git clone repo=%s branch=%s ssh=%s", repo, branch, self._use_ssh)
            url = self._clone_url(repo)
            proc = await asyncio.create_subprocess_exec(
                "git", *auth_args,
                "clone", "--depth", "1", "--branch", branch,
                "--single-branch", url, str(repo_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(
                    f"git clone failed for {repo}@{branch}: {stderr.decode()}"
                )

        self._cloned.add(key)
        _log.info("repo ready repo=%s path=%s", repo, repo_dir)

    async def list_repos(self, org: str) -> list[str]:
        return []

    async def list_files(
        self,
        repo: str,
        branch: str = "main",
        patterns: list[str] | None = None,
    ) -> list[str]:
        await self._ensure_cloned(repo, branch)
        return await self._local.list_files(repo, branch, patterns)

    async def get_file(self, repo: str, path: str, branch: str = "main") -> RepoFile:
        await self._ensure_cloned(repo, branch)
        return await self._local.get_file(repo, path, branch)

    async def get_files_batch(
        self,
        repo: str,
        paths: list[str],
        branch: str = "main",
    ) -> AsyncIterator[RepoFile]:
        await self._ensure_cloned(repo, branch)
        async for f in self._local.get_files_batch(repo, paths, branch):
            yield f

    async def get_changed_files_since(
        self,
        repo: str,
        since: datetime,
        branch: str = "main",
    ) -> list[RepoFile]:
        await self._ensure_cloned(repo, branch)
        return []

    async def warm_repo_cache(
        self,
        repos_with_branch: list[tuple[str, str]],
        *,
        concurrency: int = 6,
    ) -> dict:
        """Shallow-clone every (repo, branch) pair concurrently.

        Idempotent: a repo already present in the cache is skipped.
        Bounded concurrency (default 6) keeps GitLab happy. Per-repo
        failures are collected, not raised -- pod startup must not block
        on one bad repo.

        Returns a summary dict::

            {
              "total": int,    # repos attempted
              "ok": int,       # cloned or already-cached
              "failed": int,
              "elapsed_s": float,
              "failures": list[(repo, error_message)],
            }

        The same logic powers both the lifespan auto-warm hook AND the
        ``scripts/bootstrap_code_cache.py`` manual entry point so they
        can't drift apart.
        """
        import time

        sem = asyncio.Semaphore(max(1, int(concurrency)))

        async def _one(repo: str, branch: str) -> tuple[str, bool, str]:
            async with sem:
                repo_dir = self._repo_dir(repo)
                if repo_dir.is_dir() and (repo_dir / ".git").exists():
                    return repo, True, "already cached"
                t0 = time.time()
                try:
                    await self._ensure_cloned(repo, branch=branch)
                    return repo, True, f"cloned @ {branch} in {time.time() - t0:.1f}s"
                except Exception as exc:  # noqa: BLE001
                    return repo, False, f"{type(exc).__name__}: {exc}"

        t_start = time.time()
        results = await asyncio.gather(
            *(_one(r, b) for r, b in repos_with_branch),
            return_exceptions=False,
        )
        elapsed = time.time() - t_start

        ok = sum(1 for _, success, _ in results if success)
        failures = [(r, msg) for r, success, msg in results if not success]
        for r, success, msg in results:
            level = _log.info if success else _log.warning
            level("warm_repo_cache  %s: %s", r, msg)
        _log.info(
            "warm_repo_cache DONE -- %d/%d cached in %.1fs (%d failures)",
            ok, len(repos_with_branch), elapsed, len(failures),
        )
        return {
            "total": len(repos_with_branch),
            "ok": ok,
            "failed": len(failures),
            "elapsed_s": elapsed,
            "failures": failures,
        }

    def parse_webhook(self, payload: dict, headers: dict) -> WebhookEvent:
        event_type = headers.get("X-Gitlab-Event") or headers.get("X-GitHub-Event", "push")
        repo = (
            payload.get("project", {}).get("path_with_namespace")
            or payload.get("repository", {}).get("full_name", "")
        )
        ref = payload.get("ref", "refs/heads/main")
        branch = ref.rsplit("/", 1)[-1]

        changed: set[str] = set()
        for commit in payload.get("commits", []):
            changed.update(commit.get("added", []))
            changed.update(commit.get("modified", []))

        return WebhookEvent(
            event_type=event_type,
            repo=repo,
            branch=branch,
            changed_files=sorted(changed),
            timestamp=datetime.now(UTC),
            raw_payload=payload,
        )

    async def invalidate(self, repo: str) -> None:
        """Remove cached clone so next access re-clones."""
        repo_dir = self._repo_dir(repo)
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        self._cloned.discard(repo)

    async def close(self) -> None:
        pass
