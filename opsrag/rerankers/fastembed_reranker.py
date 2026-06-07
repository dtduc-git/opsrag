"""FastEmbed cross-encoder reranker -- local ONNX model, no API key.

Uses fastembed's TextCrossEncoder to re-score query-document pairs.
Much more precise than bi-encoder similarity -- a cross-encoder sees
query and document together, not separately.

Default model: Xenova/ms-marco-MiniLM-L-6-v2 (~90MB, CPU-friendly).
First call downloads the model; subsequent calls are instant.

Requires: fastembed >= 0.4.0
"""
from __future__ import annotations

import asyncio
import math

from fastembed.rerank.cross_encoder import TextCrossEncoder

from opsrag.interfaces.reranker import RerankResult
from opsrag.interfaces.vectorstore import SearchResult


def _sigmoid(x: float) -> float:
    # Numerically stable logistic, maps a raw cross-encoder logit to (0, 1).
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class FastEmbedReranker:
    _DEFAULT_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model: str | None = None):
        # The shared config default `reranker.model` is "rerank-v3.5" (a Cohere
        # model name), which is NOT a valid fastembed cross-encoder. Since
        # fastembed is now the default provider, blindly forwarding that name
        # would crash model load and silently fall back to no-op -- defeating
        # the point. Fall back to the local default for any non-fastembed name.
        name = model or self._DEFAULT_MODEL
        try:
            supported = {m["model"] for m in TextCrossEncoder.list_supported_models()}
        except Exception:
            supported = set()
        if supported and name not in supported:
            name = self._DEFAULT_MODEL
        self._model_name = name
        self._encoder = TextCrossEncoder(model_name=name)

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 3,
    ) -> list[RerankResult]:
        if not results:
            return []

        documents = [r.chunk.content for r in results]
        # TextCrossEncoder.rerank is a blocking CPU-bound ONNX pass. FastEmbed is
        # the DEFAULT reranker, so running it inline would serialize the FastAPI
        # event loop on every query (~40-120 docs per pass). Offload to a thread.
        raw = await asyncio.to_thread(
            lambda: list(self._encoder.rerank(query, documents))
        )
        # ms-marco-MiniLM (the default model) emits RAW LOGITS (~-11..+11), not
        # probabilities. The rerank node + weak-retrieval gate downstream assume
        # scores in [0,1]: a `*1.5` anchor boost on a negative logit would push a
        # relevant doc DOWN, and the 0.05 floor would be meaningless. Map logits
        # to [0,1] with a sigmoid so those callers behave as documented.
        scores = [_sigmoid(s) for s in raw]

        scored = sorted(
            zip(results, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            RerankResult(chunk=sr.chunk, relevance_score=float(score))
            for sr, score in scored[:top_k]
        ]
