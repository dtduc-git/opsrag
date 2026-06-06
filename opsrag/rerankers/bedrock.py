"""Bedrock reranker -- uses the Amazon Bedrock Rerank API (no separate
Cohere SaaS key). Default model is Cohere Rerank 3.5 hosted ON Bedrock
(`cohere.rerank-v3-5:0`); `amazon.rerank-v1:0` also works. Same AWS
credentials/region as the Bedrock LLM + Titan embedder.
"""
from __future__ import annotations

import asyncio
import time

import boto3

from opsrag.interfaces.reranker import RerankResult
from opsrag.interfaces.vectorstore import SearchResult
from opsrag.usage import tracker as _usage_tracker


class BedrockReranker:
    def __init__(
        self,
        model: str = "cohere.rerank-v3-5:0",
        region: str | None = None,
        profile: str | None = None,
    ):
        session = boto3.Session(region_name=region, profile_name=profile)
        # The Rerank API lives on the bedrock-agent-runtime client.
        self._client = session.client("bedrock-agent-runtime")
        self._region = region or session.region_name or "us-west-2"
        self._model = model
        self._model_arn = (
            f"arn:aws:bedrock:{self._region}::foundation-model/{model}"
        )

    async def close(self) -> None:  # parity with CohereReranker
        return None

    def _rerank_sync(self, query: str, documents: list[str], top_n: int) -> list[dict]:
        resp = self._client.rerank(
            queries=[{"type": "TEXT", "textQuery": {"text": query}}],
            sources=[
                {
                    "type": "INLINE",
                    "inlineDocumentSource": {
                        "type": "TEXT",
                        "textDocument": {"text": doc},
                    },
                }
                for doc in documents
            ],
            rerankingConfiguration={
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "modelConfiguration": {"modelArn": self._model_arn},
                    "numberOfResults": top_n,
                },
            },
        )
        return resp.get("results", [])

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[RerankResult]:
        if not results:
            return []
        documents = [r.chunk.content for r in results]
        top_n = min(top_k, len(documents))
        # boto3 is sync -> off the event loop.
        t0 = time.perf_counter()
        items = await asyncio.to_thread(self._rerank_sync, query, documents, top_n)
        # The Bedrock Rerank API is priced per request (not per token), so
        # record call_count=1 with zero tokens -- the tracker's per-call
        # pricing table converts this to USD (purpose='rerank' -> query cost).
        _usage_tracker.record(
            model=self._model,
            input_tokens=0,
            output_tokens=0,
            latency_ms=(time.perf_counter() - t0) * 1000,
            purpose="rerank",
        )

        out: list[RerankResult] = []
        for item in items:
            idx = item.get("index")
            if idx is None or idx >= len(results):
                continue
            out.append(
                RerankResult(
                    chunk=results[idx].chunk,
                    relevance_score=float(item.get("relevanceScore", 0.0)),
                )
            )
        return out
