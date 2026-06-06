"""Unit tests for the fail-closed dimension guard (models feature, F2)."""
from __future__ import annotations

import pytest

from opsrag.vectorstore_guard import (
    DimensionMismatchError,
    assert_dimension_compatible,
)


class _VP:
    """Minimal stand-in for qdrant VectorParams."""

    def __init__(self, size: int):
        self.size = size


class _Params:
    def __init__(self, vectors):
        self.vectors = vectors


class _Config:
    def __init__(self, vectors):
        self.params = _Params(vectors)


class _CollectionInfo:
    def __init__(self, vectors):
        self.config = _Config(vectors)


class _FakeQdrant:
    """Mock AsyncQdrantClient: existence + named-vector get_collection."""

    def __init__(self, *, exists: bool, dim: int | None = None, named: bool = True):
        self._exists = exists
        self._dim = dim
        self._named = named

    async def collection_exists(self, collection):
        return self._exists

    async def get_collection(self, collection):
        assert self._dim is not None
        vp = _VP(self._dim)
        vectors = {"dense": vp} if self._named else vp
        return _CollectionInfo(vectors)


@pytest.mark.asyncio
async def test_raises_on_mismatch_when_allow_change_false():
    client = _FakeQdrant(exists=True, dim=768)
    with pytest.raises(DimensionMismatchError) as exc:
        await assert_dimension_compatible(
            client, "opsrag", expected_dim=1024, allow_change=False,
        )
    assert "DIMENSION_MISMATCH" in str(exc.value)
    assert "768" in str(exc.value) and "1024" in str(exc.value)


@pytest.mark.asyncio
async def test_passes_when_equal():
    client = _FakeQdrant(exists=True, dim=1024)
    # No raise.
    await assert_dimension_compatible(
        client, "opsrag", expected_dim=1024, allow_change=False,
    )


@pytest.mark.asyncio
async def test_noop_on_missing_collection():
    client = _FakeQdrant(exists=False)
    # Collection absent -> no-op, no get_collection call, no raise.
    await assert_dimension_compatible(
        client, "opsrag", expected_dim=1024, allow_change=False,
    )


@pytest.mark.asyncio
async def test_allow_change_true_does_not_raise_on_mismatch():
    client = _FakeQdrant(exists=True, dim=768)
    # Operator opted into a reindex -> warn + continue, no raise.
    await assert_dimension_compatible(
        client, "opsrag", expected_dim=1024, allow_change=True,
    )


@pytest.mark.asyncio
async def test_single_vector_shape_supported():
    # Some collections use an unnamed single VectorParams.
    client = _FakeQdrant(exists=True, dim=768, named=False)
    with pytest.raises(DimensionMismatchError):
        await assert_dimension_compatible(
            client, "opsrag", expected_dim=1024, allow_change=False,
        )


@pytest.mark.asyncio
async def test_fallback_existence_via_get_collections():
    class _Col:
        def __init__(self, name):
            self.name = name

    class _Cols:
        def __init__(self, names):
            self.collections = [_Col(n) for n in names]

    class _NoCheckerClient:
        async def get_collections(self):
            return _Cols(["other"])

    # collection_exists absent -> falls back to get_collections; "opsrag"
    # not present -> no-op.
    await assert_dimension_compatible(
        _NoCheckerClient(), "opsrag", expected_dim=1024, allow_change=False,
    )
