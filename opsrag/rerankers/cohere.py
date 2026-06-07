"""Cohere reranker -- direct httpx call against the v2 rerank endpoint.

The cohere SDK is an optional extra; we avoid importing it here so that
CohereReranker works with only httpx installed.
"""
from __future__ import annotations

import httpx

from opsrag.interfaces.reranker import RerankResult
from opsrag.interfaces.vectorstore import SearchResult

_ENDPOINT = "https://api.cohere.com/v2/rerank"


class CohereReranker:
    # Cohere rerank-v3.5 returns [0,1] relevance, but the distribution is
    # COMPRESSED LOW vs FastEmbed's sigmoid: genuinely relevant docs often score
    # ~0.1-0.4, so the FastEmbed 0.05 floor would false-trip weak-retrieval and
    # 0.65 trust would essentially never fire (burning the whole CRAG budget).
    # Lower both. (Tune against your corpus -- these are conservative defaults.)
    score_floor = 0.02
    trust_score = 0.5

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-v3.5",
        timeout: float = 20.0,
    ):
        if not api_key:
            raise ValueError("Cohere API key is required")
        self._api_key = api_key
        self._model = model
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[RerankResult]:
        if not results:
            return []
        documents = [r.chunk.content for r in results]
        resp = await self._client.post(
            _ENDPOINT,
            json={
                "model": self._model,
                "query": query,
                "documents": documents,
                "top_n": min(top_k, len(documents)),
            },
        )
        resp.raise_for_status()
        data = resp.json()

        out: list[RerankResult] = []
        for item in data.get("results", []):
            idx = item.get("index")
            if idx is None or idx >= len(results):
                continue
            out.append(
                RerankResult(
                    chunk=results[idx].chunk,
                    relevance_score=float(item.get("relevance_score", 0.0)),
                )
            )
        return out
