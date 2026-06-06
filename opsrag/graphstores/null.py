"""Null knowledge-graph backend.

A first-class implementation of ``KnowledgeGraphStore`` that satisfies the
interface with empty results. Lets a minimal opsrag deployment ship with
no Neo4j (or equivalent) running -- the agent graph topology stays
identical; the graph node simply returns no entities, no relationships,
no paths.

Per FR-019 + Constitution Principle II clarification: graph store is
provider-selected (not MCP-flagged), and the null backend is the default.

Behaviour guarantees:

- Every read returns an empty ``GraphSearchResult`` or ``[]`` / ``{}``.
- Every write returns ``0`` (rows-affected style; the writes are
  intentional no-ops, not errors).
- ``get_schema()`` returns an empty schema dict.

The store is async to match the protocol, even though every operation is
synchronous-in-effect.
"""
from __future__ import annotations

from opsrag.interfaces.graphstore import (
    Entity,
    GraphSearchResult,
    Relationship,
)


class NullGraphStore:
    """No-op ``KnowledgeGraphStore`` implementation."""

    async def close(self) -> None:
        return None

    async def __aenter__(self) -> NullGraphStore:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def ensure_indexes(self) -> None:
        return None

    async def upsert_entities(self, entities: list[Entity]) -> int:
        return 0

    async def upsert_relationships(self, relationships: list[Relationship]) -> int:
        return 0

    async def search_entities(
        self,
        query: str,
        labels: list[str] | None = None,
        limit: int = 10,
    ) -> list[Entity]:
        return []

    async def get_subgraph(
        self,
        entity_ids: list[str],
        include_neighbors: bool = True,
        neighbor_depth: int = 1,
    ) -> GraphSearchResult:
        return GraphSearchResult()

    async def delete_by_source(self, source_chunk_ids: list[str]) -> int:
        return 0

    async def get_schema(self) -> dict:
        return {}

    async def view_subgraph(
        self,
        node_labels: list[str],
        rel_types: list[str],
        limit: int = 300,
    ) -> tuple[list[dict], list[dict]]:
        return ([], [])
