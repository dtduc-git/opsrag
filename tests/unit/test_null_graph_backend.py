"""Unit test (T070): the null knowledge-graph backend conforms to the
KnowledgeGraphStore interface and returns empty results for every read while
no-op'ing every write (FR-019).
"""
from __future__ import annotations

import pytest

from opsrag.graphstores.null import NullGraphStore
from opsrag.interfaces.graphstore import (
    GraphSearchResult,
    KnowledgeGraphStore,
)


def test_implements_every_interface_method() -> None:
    store = NullGraphStore()
    for method in (
        "close", "ensure_indexes", "upsert_entities", "upsert_relationships",
        "search_entities", "get_subgraph", "delete_by_source", "get_schema",
    ):
        assert callable(getattr(store, method)), f"missing {method}"


def test_structural_conformance_to_protocol() -> None:
    # KnowledgeGraphStore is a typing.Protocol; NullGraphStore must satisfy it
    # structurally. Skip gracefully if the protocol isn't runtime-checkable.
    try:
        assert isinstance(NullGraphStore(), KnowledgeGraphStore)
    except TypeError:
        pytest.skip("KnowledgeGraphStore is not @runtime_checkable")


@pytest.mark.asyncio
async def test_reads_return_empty() -> None:
    store = NullGraphStore()

    assert await store.search_entities("anything") == []

    result = await store.get_subgraph(["a", "b"])
    assert isinstance(result, GraphSearchResult)
    assert result.entities == []
    assert result.relationships == []
    assert result.paths == []
    assert result.context_text == ""

    assert await store.get_schema() == {}


@pytest.mark.asyncio
async def test_writes_are_noops_returning_zero() -> None:
    store = NullGraphStore()
    assert await store.upsert_entities([]) == 0
    assert await store.upsert_relationships([]) == 0
    assert await store.delete_by_source(["chunk-1"]) == 0
    # Lifecycle hooks never raise.
    assert await store.ensure_indexes() is None
    assert await store.close() is None


@pytest.mark.asyncio
async def test_async_context_manager() -> None:
    async with NullGraphStore() as store:
        assert isinstance(store, NullGraphStore)
