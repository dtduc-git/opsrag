"""LiteLLM embeddings provider -- routes embeddings through LiteLLM so any
LiteLLM-supported backend (Voyage, Gemini, Cohere, OpenAI, Bedrock, or a
self-hosted OpenAI-compatible / TEI endpoint via ``api_base``) becomes a
config flip.

LiteLLM provider-string convention: the ``model`` is passed verbatim to
``litellm.aembedding`` using LiteLLM's ``<provider>/<model>`` convention,
e.g. ``voyage/voyage-code-3``, ``gemini/text-embedding-004``,
``cohere/embed-english-v3.0``, or ``openai/<model>`` (the latter also covers
self-hosted OpenAI-compatible servers when paired with ``api_base``).

``litellm`` is imported lazily inside the methods so this module imports
cleanly without the optional dependency installed (and so tests can
monkeypatch the import target).
"""
from __future__ import annotations

import asyncio
import os
import time

from opsrag.tokenization import estimate_tokens
from opsrag.usage import tracker as _usage_tracker


class LiteLLMEmbeddings:
    def __init__(
        self,
        model: str,
        dimension: int | None = None,
        api_base: str | None = None,
        api_key_env: str | None = None,
        batch_size: int = 96,
    ):
        self._model = model
        # dimension is config-driven; LiteLLM doesn't reliably report it, so
        # the vector-store schema relies on the configured value. Fail closed
        # if unset -- a silent 1536 guess bakes a wrong-dim collection that only
        # surfaces as a cryptic upsert error later. Consistent with the cloud
        # embedders' unknown-model guard (LiteLLM has no model->dim map at all).
        if dimension is None:
            raise ValueError(
                "LiteLLM embedder requires an explicit embedding.dimension "
                "(the proxy does not reliably report it)."
            )
        self._dimension = dimension
        self._api_base = api_base
        self._api_key = os.environ.get(api_key_env) if api_key_env else None
        self._batch_size = batch_size

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model

    def _call_kwargs(self, inputs: list[str]) -> dict:
        kwargs: dict = {"model": self._model, "input": inputs}
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if self._api_key:
            kwargs["api_key"] = self._api_key
        return kwargs

    async def _aembed(self, inputs: list[str]) -> list[list[float]]:
        import litellm

        resp = await litellm.aembedding(**self._call_kwargs(inputs))
        # LiteLLM normalises to an OpenAI-shaped response: r.data[i]["embedding"].
        return [row["embedding"] for row in resp.data]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        t0 = time.perf_counter()
        bs = self._batch_size
        batches = [texts[i : i + bs] for i in range(0, len(texts), bs)]
        batch_vecs = await asyncio.gather(*(self._aembed(b) for b in batches))
        vectors = [v for bv in batch_vecs for v in bv]
        latency_ms = (time.perf_counter() - t0) * 1000
        _usage_tracker.record(
            model=self._model,
            input_tokens=sum(estimate_tokens(t) for t in texts),
            output_tokens=0,
            latency_ms=latency_ms,
            purpose="embed-index",
        )
        return vectors

    async def embed_query(self, query: str) -> list[float]:
        t0 = time.perf_counter()
        vecs = await self._aembed([query])
        latency_ms = (time.perf_counter() - t0) * 1000
        _usage_tracker.record(
            model=self._model,
            input_tokens=estimate_tokens(query),
            output_tokens=0,
            latency_ms=latency_ms,
            purpose="embed-query",
        )
        return vecs[0]
