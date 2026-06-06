"""No-op reranker -- trusts the vector search ordering.

Useful as a default when no reranker is configured, and as a fallback
when the configured reranker is unavailable.
"""
from __future__ import annotations

from opsrag.interfaces.reranker import RerankResult
from opsrag.interfaces.vectorstore import SearchResult


class NoOpReranker:
    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[RerankResult]:
        return [
            RerankResult(chunk=r.chunk, relevance_score=r.score)
            for r in results[:top_k]
        ]
