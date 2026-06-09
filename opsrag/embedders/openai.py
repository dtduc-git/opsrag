"""OpenAI embeddings provider -- direct openai SDK, no LangChain."""
from __future__ import annotations

import asyncio
import logging
import random

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

_log = logging.getLogger("opsrag.embedders.openai")

_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}

# Mirror vertex.py / bedrock.py: bounded exponential backoff with jitter so
# parallel bulk indexing doesn't crater on 429 (rate-limit) or transient 5xx.
_MAX_RETRIES = 6
_BASE_BACKOFF = 1.5  # seconds; 1.5, 3, 6, 12, 24, 48 -> ~94s worst case

# OpenAI SDK exception types that are worth retrying: per-minute rate limits,
# request timeouts, connection blips, and 5xx server errors.
_RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


class OpenAIEmbeddings:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-large",
        dimension: int | None = None,
        batch_size: int = 128,
    ):
        self._api_key = api_key
        self._client: AsyncOpenAI | None = None
        self._model = model
        self._batch_size = batch_size
        _known_dim = _MODEL_DIMENSIONS.get(model)
        if dimension is None and _known_dim is None:
            # Fail closed -- see vertex.py: a silent 1536 fallback for an
            # unknown model bakes a wrong-dim collection on first boot.
            raise ValueError(
                f"Unknown OpenAI embedding model {model!r} and no dimension set. "
                f"Set embedding.dimension explicitly or use a known model: "
                f"{sorted(_MODEL_DIMENSIONS)}"
            )
        self._dimension = dimension or _known_dim

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model

    def _call_kwargs(self, inputs: list[str]) -> dict:
        kwargs: dict = {"model": self._model, "input": inputs}
        if self._model.startswith("text-embedding-3"):
            kwargs["dimensions"] = self._dimension
        return kwargs

    async def _create_with_retry(self, inputs: list[str]):
        """Call embeddings.create with bounded exponential backoff + jitter on
        429/5xx/timeouts -- mirrors vertex.py's `_embed_with_retry`."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return await self._get_client().embeddings.create(
                    **self._call_kwargs(inputs)
                )
            except _RETRYABLE as exc:
                last_exc = exc
                if attempt == _MAX_RETRIES - 1:
                    raise
                delay = _BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 1.0)
                _log.warning(
                    "openai embed retry %d/%d in %.1fs: %s",
                    attempt + 1, _MAX_RETRIES, delay, str(exc)[:160],
                )
                await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            resp = await self._create_with_retry(batch)
            vectors.extend(d.embedding for d in resp.data)
        return vectors

    async def embed_query(self, query: str) -> list[float]:
        resp = await self._create_with_retry([query])
        return resp.data[0].embedding
