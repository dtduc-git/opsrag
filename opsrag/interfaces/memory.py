"""Long-term memory store interface."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable


@dataclass
class Memory:
    key: str
    namespace: tuple[str, ...]
    value: dict
    created_at: datetime
    updated_at: datetime


@runtime_checkable
class MemoryStore(Protocol):
    async def put(
        self, namespace: tuple[str, ...], key: str, value: dict
    ) -> None: ...

    async def get(
        self, namespace: tuple[str, ...], key: str
    ) -> Memory | None: ...

    async def search(
        self,
        namespace: tuple[str, ...],
        query: str | None = None,
        limit: int = 10,
    ) -> list[Memory]: ...

    async def delete(self, namespace: tuple[str, ...], key: str) -> bool: ...
