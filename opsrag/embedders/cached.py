"""In-memory LRU + TTL embedder decorator.

Same query gets re-embedded 3-5x per chat turn -- cache lookup, classifier,
semantic router, retrieval, qa_cache. Wrapping the embedder with a small
in-process cache cuts ~50-150ms latency and ~$0.0002/turn at Vertex
pricing.

Design notes:
- Wraps any object satisfying `EmbeddingProvider` (Protocol). No subclass
  required; we delegate `dimension` and `model_name`.
- Cache key = `(query.strip(), embedder.model_name, embedder.dimension)`.
  Case is PRESERVED: the embedder produces different vectors for "KafkaConsumer"
  vs "kafkaconsumer", and case is semantically significant in this code/identifier
  -heavy workload -- lower-casing the key returned the wrong cached vector for a
  differently-cased query. Strip drops leading/trailing whitespace. Model name
  AND dimension are part of the key so swapping models -- or changing
  `output_dimension` on a provider that keeps the same model id (Cohere v4,
  OpenAI-3, Gemini) -- doesn't serve stale-dim vectors.
- LRU eviction: `OrderedDict.move_to_end` on hit, `popitem(last=False)`
  on overflow. O(1) per access.
- TTL is per-entry -- set at insert time, checked at hit time. Expired
  entries are evicted lazily on lookup (not via background task).
- `embed_texts` (batch) is intentionally NOT cached. Batches have
  per-call composition; caching would require keying on the joined
  text and would mostly miss anyway during bulk indexing.
"""
from __future__ import annotations

import logging
import time
from collections import OrderedDict

from opsrag.interfaces.embedder import EmbeddingProvider

_log = logging.getLogger("opsrag.embedders.cached")


class CachedEmbedder:
    """Decorator that wraps any `EmbeddingProvider` with an in-memory LRU+TTL.

    Only `embed_query` is cached; `embed_texts` passes through unchanged.
    """

    def __init__(
        self,
        inner: EmbeddingProvider,
        max_size: int = 1000,
        ttl_seconds: float = 60.0,
    ):
        self._inner = inner
        self._max_size = max_size
        self._ttl = ttl_seconds
        # key -> (vector, expires_at). OrderedDict preserves insertion / access
        # order for cheap LRU semantics.
        self._cache: OrderedDict[tuple[str, str, int], tuple[list[float], float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    def _key(self, query: str) -> tuple[str, str, int]:
        # NB: case-preserving on purpose -- see module docstring.
        return (query.strip(), self._inner.model_name, self._inner.dimension)

    async def embed_query(self, query: str) -> list[float]:
        key = self._key(query)
        now = time.monotonic()

        # Hit path -- must check TTL before returning.
        entry = self._cache.get(key)
        if entry is not None:
            vector, expires_at = entry
            if now < expires_at:
                self._cache.move_to_end(key)  # LRU touch
                self._hits += 1
                return vector
            # Expired -- drop and fall through to re-embed.
            del self._cache[key]

        # Miss path.
        self._misses += 1
        vector = await self._inner.embed_query(query)
        self._cache[key] = (vector, now + self._ttl)
        self._cache.move_to_end(key)

        # Bound size -- evict LRU. Keep at most `max_size` entries.
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

        return vector

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # Batch calls bypass the cache entirely -- different semantics.
        return await self._inner.embed_texts(texts)

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
            "hit_rate": (self._hits / total) if total else 0.0,
        }
