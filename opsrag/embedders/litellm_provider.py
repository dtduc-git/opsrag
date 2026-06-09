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

# Cap on in-flight embedding batches during a bulk index. Without it an
# unbounded ``asyncio.gather`` over the whole corpus fires every batch at
# once -> a provider rate-limit storm (429) that the per-call ``num_retries``
# can't fully absorb, dropping vectors. 8 keeps the pipe busy without melting
# the provider's per-minute quota.
_MAX_CONCURRENT_BATCHES = 8


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

    def _input_type(self, role: str) -> str | None:
        """Provider-specific ``input_type`` for query/document asymmetry.

        Cohere and Voyage embed queries and documents into DIFFERENT spaces and
        REQUIRE this hint -- without it (the prior behaviour) queries were
        embedded as documents, a measurable recall loss (an entirely different
        vector space for Voyage). OpenAI-compatible endpoints don't take an
        input_type, so we omit it there. ``role`` is 'query' or 'document'.
        """
        m = self._model.lower()
        if "voyage" in m:
            return "query" if role == "query" else "document"
        if "cohere" in m or "/embed-" in m or m.startswith("embed-"):
            return "search_query" if role == "query" else "search_document"
        return None  # OpenAI / generic OpenAI-compatible: no input_type param

    def _call_kwargs(self, inputs: list[str], role: str = "document") -> dict:
        kwargs: dict = {"model": self._model, "input": inputs}
        # Request the CONFIGURED output dimension. Without it, a Matryoshka model
        # (Cohere v4, OpenAI-3) returns its NATIVE dim, which can diverge from the
        # collection's dim -> a wrong-dim upsert. LiteLLM maps `dimensions` to the
        # provider-specific param (output_dimension for Cohere/Voyage).
        if self._dimension:
            kwargs["dimensions"] = self._dimension
        input_type = self._input_type(role)
        if input_type:
            kwargs["input_type"] = input_type
        # Cohere/Voyage reject inputs over their token window with a hard error;
        # ask the provider to trim server-side so an over-long (contextual-
        # prefixed) chunk degrades gracefully instead of failing the batch.
        m = self._model.lower()
        if "cohere" in m or "voyage" in m or "/embed-" in m or m.startswith("embed-"):
            kwargs["truncate"] = "END"
        # LiteLLM-level retries with backoff -- bulk indexing hits provider rate
        # limits; without this a single 429 drops a whole batch's vectors.
        kwargs["num_retries"] = 4
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if self._api_key:
            kwargs["api_key"] = self._api_key
        return kwargs

    async def _aembed(self, inputs: list[str], role: str = "document") -> list[list[float]]:
        import litellm

        resp = await litellm.aembedding(**self._call_kwargs(inputs, role))
        # LiteLLM normalises to an OpenAI-shaped response: r.data[i]["embedding"].
        return [row["embedding"] for row in resp.data]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        t0 = time.perf_counter()
        bs = self._batch_size
        batches = [texts[i : i + bs] for i in range(0, len(texts), bs)]
        # Bound concurrency: a raw gather over every batch fans the whole
        # corpus at the provider at once -> rate-limit storm. The semaphore
        # caps in-flight requests so bulk indexing degrades gracefully.
        sem = asyncio.Semaphore(_MAX_CONCURRENT_BATCHES)

        async def _bounded(batch: list[str]) -> list[list[float]]:
            async with sem:
                return await self._aembed(batch, "document")

        batch_vecs = await asyncio.gather(*(_bounded(b) for b in batches))
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
        vecs = await self._aembed([query], "query")
        latency_ms = (time.perf_counter() - t0) * 1000
        _usage_tracker.record(
            model=self._model,
            input_tokens=estimate_tokens(query),
            output_tokens=0,
            latency_ms=latency_ms,
            purpose="embed-query",
        )
        return vecs[0]
