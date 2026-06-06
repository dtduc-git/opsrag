"""SCM provider interface -- abstraction over git hosting platforms."""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable


@dataclass
class RepoFile:
    path: str
    content: str
    sha: str
    last_modified: datetime
    repo: str
    branch: str
    metadata: dict = field(default_factory=dict)


@dataclass
class WebhookEvent:
    event_type: str
    repo: str
    branch: str
    changed_files: list[str]
    timestamp: datetime
    raw_payload: dict = field(default_factory=dict)


@runtime_checkable
class SCMProvider(Protocol):
    async def list_repos(self, org: str) -> list[str]: ...

    async def list_files(
        self,
        repo: str,
        branch: str = "main",
        patterns: list[str] | None = None,
    ) -> list[str]: ...

    async def get_file(self, repo: str, path: str, branch: str = "main") -> RepoFile: ...

    def get_files_batch(
        self,
        repo: str,
        paths: list[str],
        branch: str = "main",
    ) -> AsyncIterator[RepoFile]: ...

    async def get_changed_files_since(
        self,
        repo: str,
        since: datetime,
        branch: str = "main",
    ) -> list[RepoFile]: ...

    def parse_webhook(self, payload: dict, headers: dict) -> WebhookEvent: ...
