"""Knowledge graph store interface."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Entity:
    id: str
    label: str
    name: str
    properties: dict = field(default_factory=dict)
    source_chunk_id: str | None = None


@dataclass
class Relationship:
    source_id: str
    target_id: str
    rel_type: str
    properties: dict = field(default_factory=dict)


@dataclass
class GraphSearchResult:
    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    paths: list[list[str]] = field(default_factory=list)
    context_text: str = ""


@runtime_checkable
class KnowledgeGraphStore(Protocol):
    async def upsert_entities(self, entities: list[Entity]) -> int: ...
    async def upsert_relationships(self, relationships: list[Relationship]) -> int: ...

    async def search_entities(
        self,
        query: str,
        labels: list[str] | None = None,
        limit: int = 10,
    ) -> list[Entity]: ...

    # NOTE: free-form Cypher methods (traverse/query_raw/find_paths) were
    # removed -- they had zero retrieval callers (retrieval rides the Postgres
    # light graph, see opsrag.light_graph) and traverse/find_paths interpolated
    # into the Cypher string. Re-add as parameterized + wired before relying on
    # multi-hop Neo4j traversal in the agent.

    async def get_subgraph(
        self,
        entity_ids: list[str],
        include_neighbors: bool = True,
        neighbor_depth: int = 1,
    ) -> GraphSearchResult: ...

    async def delete_by_source(self, source_chunk_ids: list[str]) -> int: ...
    async def get_schema(self) -> dict: ...

    async def view_subgraph(
        self,
        node_labels: list[str],
        rel_types: list[str],
        limit: int = 300,
    ) -> tuple[list[dict], list[dict]]: ...
