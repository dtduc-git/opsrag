"""Vector store interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from opsrag.interfaces.chunker import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    distance_metric: str = "cosine"


@runtime_checkable
class VectorStore(Protocol):
    async def upsert(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> int: ...

    async def search(
        self,
        embedding: list[float],
        top_k: int = 10,
        filters: dict | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]: ...

    async def delete(self, chunk_ids: list[str]) -> int: ...
    async def delete_by_filter(self, filters: dict) -> int: ...
    async def get_collection_stats(self) -> dict: ...

    async def hybrid_search(
        self,
        embedding: list[float],
        query_text: str,
        top_k: int = 10,
        alpha: float = 0.7,
        filters: dict | None = None,
    ) -> list[SearchResult]: ...

    async def search_by_text(
        self,
        text: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Text-match-only search on the indexed `content` field.

        Used to surface chunks whose embedding doesn't semantically match
        the query but whose content (and usually source_path) literally
        contains the term -- e.g. service names like "acme-notes-be" appearing
        in YAML across many repos. Score is binary (1.0 for matches);
        callers should rely on a downstream reranker for ordering.
        """
        ...
