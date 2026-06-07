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
        # Emit a NEUTRAL relevance_score (1.0), not the passthrough vector score.
        # The upstream score is an RRF micro-score (~0.01-0.03) on a totally
        # different scale than a cross-encoder's [0,1]; surfacing it as
        # `best_rerank_score` made the weak-retrieval gate
        # (rerank_decision: best < 0.05) fire UNCONDITIONALLY for any
        # unmatched-anchor query -> spurious "insufficient information". A no-op
        # reranker has no relevance opinion, so it must not trip a [0,1]-scaled
        # gate; ordering (the vector order) is preserved by the stable slice.
        return [
            RerankResult(chunk=r.chunk, relevance_score=1.0)
            for r in results[:top_k]
        ]
