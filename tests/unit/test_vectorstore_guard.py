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
        self.deleted: list[str] = []

    async def collection_exists(self, collection):
        return self._exists

    async def get_collection(self, collection):
        assert self._dim is not None
        vp = _VP(self._dim)
        vectors = {"dense": vp} if self._named else vp
        return _CollectionInfo(vectors)

    async def delete_collection(self, collection):
        self.deleted.append(collection)


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
async def test_allow_change_true_drops_collection_on_mismatch():
    client = _FakeQdrant(exists=True, dim=768)
    # Operator opted into a reindex -> TRUTHFUL: drop the mismatched collection
    # (no raise) so the caller's create path rebuilds it at expected_dim.
    await assert_dimension_compatible(
        client, "opsrag", expected_dim=1024, allow_change=True,
    )
    assert client.deleted == ["opsrag"]


@pytest.mark.asyncio
async def test_allow_change_true_drop_failure_raises():
    class _DropFails(_FakeQdrant):
        async def delete_collection(self, collection):
            raise RuntimeError("qdrant down")

    client = _DropFails(exists=True, dim=768)
    # A failed drop must NOT silently continue (the create path would skip and
    # leave the mismatch in place) -- surface it as the fail-closed error.
    with pytest.raises(DimensionMismatchError):
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


# --- QdrantVectorStore.ensure_collection() in-store guard -------------------
# The API server runs assert_dimension_compatible in its lifespan, but the
# ingestion/indexer Job builds its own providers and writes WITHOUT that
# lifespan. These tests prove ensure_collection() now fails closed on a
# dimension mismatch (and that allow_dimension_change bypasses it), so a
# 3072 -> 768 embedder swap no longer surfaces as a cryptic upsert error.


class _Col:
    def __init__(self, name):
        self.name = name


class _Cols:
    def __init__(self, names):
        self.collections = [_Col(n) for n in names]


class _StoreFakeQdrant:
    """Mock AsyncQdrantClient covering both ensure_collection() paths:
    get_collections() (name listing) and get_collection() (named-vector
    size for the guard). Tracks live collections so a drop is reflected by
    the post-guard existence re-check, and records create/delete calls."""

    def __init__(self, *, existing_name: str, existing_dim: int):
        self._existing_name = existing_name
        self._existing_dim = existing_dim
        self._live = {existing_name}
        self.create_called = False
        self.created_dim: int | None = None
        self.deleted: list[str] = []

    async def get_collections(self):
        return _Cols(sorted(self._live))

    async def collection_exists(self, collection):
        return collection in self._live

    async def get_collection(self, collection):
        return _CollectionInfo({"dense": _VP(self._existing_dim)})

    async def delete_collection(self, collection):
        self.deleted.append(collection)
        self._live.discard(collection)

    async def create_collection(self, **kwargs):
        self.create_called = True
        # Record the requested dense vector size so a test can assert the
        # collection was recreated at the NEW embedder dimension (not the old).
        vectors_config = kwargs.get("vectors_config") or {}
        dense = vectors_config.get("dense")
        if dense is not None:
            self.created_dim = dense.size
        self._live.add(self._existing_name)

    async def create_payload_index(self, **kwargs):
        pass


@pytest.mark.asyncio
async def test_ensure_collection_raises_on_dim_mismatch(monkeypatch):
    from opsrag.vectorstores import qdrant as qdrant_mod

    # Don't construct a real AsyncQdrantClient (no network) -- swap our fake in.
    fake = _StoreFakeQdrant(existing_name="opsrag", existing_dim=3072)
    monkeypatch.setattr(
        qdrant_mod, "AsyncQdrantClient", lambda *a, **k: fake
    )
    store = qdrant_mod.QdrantVectorStore(collection_name="opsrag", dimension=768)
    with pytest.raises(DimensionMismatchError) as exc:
        await store.ensure_collection()
    assert "DIMENSION_MISMATCH" in str(exc.value)
    assert "3072" in str(exc.value) and "768" in str(exc.value)
    assert fake.create_called is False  # never recreate an existing collection


@pytest.mark.asyncio
async def test_ensure_collection_allow_change_drops_and_recreates(monkeypatch):
    from opsrag.vectorstores import qdrant as qdrant_mod

    fake = _StoreFakeQdrant(existing_name="opsrag", existing_dim=3072)
    monkeypatch.setattr(
        qdrant_mod, "AsyncQdrantClient", lambda *a, **k: fake
    )
    store = qdrant_mod.QdrantVectorStore(
        collection_name="opsrag", dimension=768, allow_dimension_change=True,
    )
    # Operator opted into a reindex -> TRUTHFUL: the guard drops the mismatched
    # collection and ensure_collection's create path rebuilds it at the new dim.
    await store.ensure_collection()
    assert store._ensured is True
    assert fake.deleted == ["opsrag"]   # mismatched collection was dropped
    assert fake.create_called is True   # ...and recreated at dim=768
    assert fake.created_dim == 768      # recreated at the NEW dim, not 3072


# --- New-collection payload-index coverage ----------------------------------
# Every search lane carries a `must_not chunk_type == "parent"` filter, so
# `chunk_type` MUST be KEYWORD-indexed at create time -- otherwise Qdrant
# scans the payload for that exclusion on every query (latency + recall
# degradation at scale). This fake exercises the NEW-collection branch and
# records each create_payload_index(field_name=...) call.


class _NewCollectionFakeQdrant:
    """Mock AsyncQdrantClient for ensure_collection()'s create path: the
    target collection does NOT exist yet, so create_collection + the
    create_payload_index loop run. Records every payload-index field name."""

    def __init__(self):
        self.create_called = False
        self.payload_index_fields: list[str] = []

    async def get_collections(self):
        return _Cols(["some_other_collection"])

    async def collection_exists(self, collection):
        return False

    async def create_collection(self, **kwargs):
        self.create_called = True

    async def create_payload_index(self, **kwargs):
        self.payload_index_fields.append(kwargs.get("field_name"))


@pytest.mark.asyncio
async def test_ensure_collection_indexes_chunk_type(monkeypatch):
    from opsrag.vectorstores import qdrant as qdrant_mod

    fake = _NewCollectionFakeQdrant()
    monkeypatch.setattr(
        qdrant_mod, "AsyncQdrantClient", lambda *a, **k: fake
    )
    store = qdrant_mod.QdrantVectorStore(collection_name="opsrag", dimension=768)
    await store.ensure_collection()

    assert fake.create_called is True
    # chunk_type must be among the KEYWORD payload indexes created so the
    # per-search must_not parent exclusion hits an index, not a scan.
    assert "chunk_type" in fake.payload_index_fields
    # The pre-existing KEYWORD indexes are still created (no regression).
    for field in ("repo", "source_path", "doc_type", "entity_ids"):
        assert field in fake.payload_index_fields
