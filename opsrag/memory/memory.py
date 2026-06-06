"""In-memory long-term memory store -- dev and testing only.

Data is lost on process restart.
"""
from __future__ import annotations

from datetime import UTC, datetime

from opsrag.interfaces.memory import Memory


class InMemoryMemoryStore:
    def __init__(self) -> None:
        self._store: dict[tuple[tuple[str, ...], str], Memory] = {}

    async def put(
        self, namespace: tuple[str, ...], key: str, value: dict
    ) -> None:
        now = datetime.now(UTC)
        lookup = (namespace, key)
        existing = self._store.get(lookup)
        self._store[lookup] = Memory(
            key=key,
            namespace=namespace,
            value=value,
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )

    async def get(
        self, namespace: tuple[str, ...], key: str
    ) -> Memory | None:
        return self._store.get((namespace, key))

    async def search(
        self,
        namespace: tuple[str, ...],
        query: str | None = None,
        limit: int = 10,
    ) -> list[Memory]:
        matches = [
            m for (ns, _), m in self._store.items()
            if ns == namespace
        ]
        if query:
            q_lower = query.lower()
            matches = [
                m for m in matches
                if q_lower in m.key.lower() or q_lower in str(m.value).lower()
            ]
        matches.sort(key=lambda m: m.updated_at, reverse=True)
        return matches[:limit]

    async def delete(self, namespace: tuple[str, ...], key: str) -> bool:
        lookup = (namespace, key)
        if lookup in self._store:
            del self._store[lookup]
            return True
        return False
