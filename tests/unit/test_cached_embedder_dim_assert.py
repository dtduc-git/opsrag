"""Unit tests for CachedEmbedder's one-shot runtime dimension assertion (R13).

Construction-time guards (per-embedder None+unknown -> ValueError) and the H4
collection guard catch most dimension drift, but a stale ``_MODEL_DIMENSIONS``
entry or a truncating (Matryoshka) model can DECLARE one dim while EMITTING
another -- silently baking a wrong-dimension fresh collection. ``CachedEmbedder``
wraps every provider centrally, so it length-checks the first non-empty vector
exactly once and refuses on mismatch.
"""
from __future__ import annotations

import pytest

from opsrag.embedders.cached import CachedEmbedder


class _FakeEmbedder:
    """Minimal EmbeddingProvider that DECLARES `dimension` independently of the
    vector length it actually returns -- so we can simulate dim drift."""

    def __init__(self, declared_dim: int, returned_len: int) -> None:
        self._declared = declared_dim
        self._returned_len = returned_len
        self.query_calls = 0
        self.texts_calls = 0

    @property
    def dimension(self) -> int:
        return self._declared

    @property
    def model_name(self) -> str:
        return "fake-embed-v1"

    def _vec(self) -> list[float]:
        return [0.1] * self._returned_len

    async def embed_query(self, query: str) -> list[float]:
        self.query_calls += 1
        return self._vec()

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.texts_calls += 1
        return [self._vec() for _ in texts]


def _cached(inner: _FakeEmbedder) -> CachedEmbedder:
    # ttl high so cache hits don't re-embed during multi-call tests.
    return CachedEmbedder(inner, max_size=100, ttl_seconds=600.0)


def test_forwards_dimension() -> None:
    """CachedEmbedder must expose/forward the wrapped embedder's .dimension."""
    cached = _cached(_FakeEmbedder(declared_dim=1024, returned_len=1024))
    assert cached.dimension == 1024


async def test_embed_query_wrong_dim_trips_assertion() -> None:
    inner = _FakeEmbedder(declared_dim=1024, returned_len=512)
    cached = _cached(inner)

    with pytest.raises(ValueError) as exc:
        await cached.embed_query("how do i restart the pod")

    msg = str(exc.value)
    assert "fake-embed-v1" in msg  # names the model
    assert "1024" in msg  # declared dim
    assert "512" in msg  # actual dim
    # A wrong-dim vector must NEVER be cached.
    assert cached.stats()["size"] == 0


async def test_embed_texts_wrong_dim_trips_assertion() -> None:
    inner = _FakeEmbedder(declared_dim=1024, returned_len=256)
    cached = _cached(inner)

    with pytest.raises(ValueError) as exc:
        await cached.embed_texts(["alpha", "beta"])

    msg = str(exc.value)
    assert "fake-embed-v1" in msg
    assert "1024" in msg
    assert "256" in msg


async def test_assertion_runs_only_once() -> None:
    """Once a matching vector verifies the dim, the check is skipped forever --
    even if a later (impossible-in-practice) call would mismatch. Proves the
    one-shot `_dim_verified` flag short-circuits the hot path."""
    inner = _FakeEmbedder(declared_dim=8, returned_len=8)
    cached = _cached(inner)

    # First call: correct dim -> verifies and flips the flag.
    vec = await cached.embed_query("alpha")
    assert len(vec) == 8
    assert cached._dim_verified is True

    # Mutate the fake to start returning a WRONG length. Because verification
    # already ran once, subsequent calls must NOT re-check (no raise).
    inner._returned_len = 3
    later = await cached.embed_query("beta")  # distinct key -> miss -> re-embed
    assert len(later) == 3  # served despite mismatch: check is one-shot


async def test_correct_dim_passes_query_and_texts() -> None:
    inner = _FakeEmbedder(declared_dim=384, returned_len=384)
    cached = _cached(inner)

    q = await cached.embed_query("how do i restart the pod")
    assert len(q) == 384

    batch = await cached.embed_texts(["a", "b", "c"])
    assert [len(v) for v in batch] == [384, 384, 384]


async def test_empty_vectors_do_not_falsely_verify() -> None:
    """An empty batch / empty vector carries no length signal, so it must
    neither raise nor mark the dim verified -- the FIRST real (non-empty)
    vector is what gets checked."""
    inner = _FakeEmbedder(declared_dim=1024, returned_len=512)
    cached = _cached(inner)

    # Empty batch -> nothing to measure -> still unverified.
    assert await cached.embed_texts([]) == []
    assert cached._dim_verified is False

    # The next real call (wrong dim) must still trip the assertion.
    with pytest.raises(ValueError):
        await cached.embed_query("alpha")
