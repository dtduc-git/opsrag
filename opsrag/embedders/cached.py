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
        # One-shot runtime dimension check -- see `_verify_dim`. Flips True
        # after the first non-empty result is checked so the assertion costs
        # nothing on the hot path thereafter.
        self._dim_verified = False

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    def _key(self, query: str) -> tuple[str, str, int]:
        # NB: case-preserving on purpose -- see module docstring.
        #
        # Key is (text, model, dim) -- deliberately QUERY-ONLY. It omits
        # input_type and api_base/endpoint, which is correct here because this
        # cache wraps embed_query exclusively (always input_type=query, one
        # endpoint per process). Do NOT reuse this cache for document embeddings
        # or across endpoints: queries and documents embed into different spaces
        # for Cohere/Voyage, so a shared key would serve a query vector for a
        # document (silent recall loss). Add input_type/endpoint to the key first
        # if that ever changes.
        return (query.strip(), self._inner.model_name, self._inner.dimension)

    def _verify_dim(self, vector: list[float]) -> None:
        """One-time runtime assertion that the provider's actual vector length
        matches its DECLARED `.dimension`.

        Construction-time guards (per-embedder None+unknown -> ValueError) plus
        the H4 collection guard catch most dim drift, but a stale
        `_MODEL_DIMENSIONS` entry or a Matryoshka model that silently truncates
        can still declare one dim while emitting another -- baking a wrong-dim
        FRESH collection with no error. This wraps EVERY provider centrally:
        the first non-empty vector to flow through is length-checked against the
        declared dim, then `_dim_verified` is set so this runs exactly ONCE
        (negligible cost). Empty vectors are ignored -- nothing to measure.
        """
        if self._dim_verified or not vector:
            return
        declared = self._inner.dimension
        actual = len(vector)
        if actual != declared:
            raise ValueError(
                f"Embedding dimension mismatch for model "
                f"{self._inner.model_name!r}: declared dimension {declared} "
                f"but provider returned a vector of length {actual}. A stale "
                f"_MODEL_DIMENSIONS entry or a truncating (Matryoshka) model "
                f"would silently bake a wrong-dimension collection -- refusing."
            )
        # Mark verified only after a successful (matching) check.
        self._dim_verified = True

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
        # Length-check ONCE before caching -- never store a wrong-dim vector.
        self._verify_dim(vector)
        self._cache[key] = (vector, now + self._ttl)
        self._cache.move_to_end(key)

        # Bound size -- evict LRU. Keep at most `max_size` entries.
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

        return vector

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # Batch calls bypass the cache entirely -- different semantics.
        vectors = await self._inner.embed_texts(texts)
        # Bulk indexing flows through here -- this is exactly where a wrong-dim
        # collection gets baked, so length-check the first non-empty vector ONCE.
        for vector in vectors:
            if vector:
                self._verify_dim(vector)
                break
        return vectors

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
