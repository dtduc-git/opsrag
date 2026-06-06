"""Reranker interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from opsrag.interfaces.chunker import Chunk
from opsrag.interfaces.vectorstore import SearchResult


@dataclass
class RerankResult:
    chunk: Chunk
    relevance_score: float


@runtime_checkable
class Reranker(Protocol):
    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[RerankResult]: ...
