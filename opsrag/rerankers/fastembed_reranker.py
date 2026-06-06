"""FastEmbed cross-encoder reranker -- local ONNX model, no API key.

Uses fastembed's TextCrossEncoder to re-score query-document pairs.
Much more precise than bi-encoder similarity -- a cross-encoder sees
query and document together, not separately.

Default model: Xenova/ms-marco-MiniLM-L-6-v2 (~90MB, CPU-friendly).
First call downloads the model; subsequent calls are instant.

Requires: fastembed >= 0.4.0
"""
from __future__ import annotations

from fastembed.rerank.cross_encoder import TextCrossEncoder

from opsrag.interfaces.reranker import RerankResult
from opsrag.interfaces.vectorstore import SearchResult


class FastEmbedReranker:
    def __init__(self, model: str = "Xenova/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model
        self._encoder = TextCrossEncoder(model_name=model)

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 3,
    ) -> list[RerankResult]:
        if not results:
            return []

        documents = [r.chunk.content for r in results]
        scores = list(self._encoder.rerank(query, documents))

        scored = sorted(
            zip(results, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            RerankResult(chunk=sr.chunk, relevance_score=float(score))
            for sr, score in scored[:top_k]
        ]
