"""H4 (Qdrant distance wiring) + D1-alpha (signature prune) coverage.

Qdrant always supported the distance metric in its VectorParams; H4 just wires
config.vector_store.distance THROUGH the constructor. These assertions are pure
(no network): the in-memory client is only constructed when a real upsert runs,
and the distance mapping is computed in __init__.
"""
from __future__ import annotations

import inspect

import pytest
from qdrant_client import models as qm

from opsrag.vectorstores.qdrant import QdrantVectorStore


def test_distance_default_is_cosine():
    store = QdrantVectorStore(url=":memory:", collection_name="t", dimension=8)
    assert store._distance == qm.Distance.COSINE


def test_distance_dot_maps_to_qdrant_dot():
    store = QdrantVectorStore(
        url=":memory:", collection_name="t", dimension=8, distance="dot"
    )
    assert store._distance == qm.Distance.DOT


def test_distance_euclid_maps_to_qdrant_euclid():
    store = QdrantVectorStore(
        url=":memory:", collection_name="t", dimension=8, distance="euclid"
    )
    assert store._distance == qm.Distance.EUCLID


def test_unknown_distance_raises_keyerror():
    with pytest.raises(KeyError):
        QdrantVectorStore(
            url=":memory:", collection_name="t", dimension=8, distance="manhattan"
        )


def test_hybrid_search_signature_drops_alpha_and_graph_anchored_paths():
    params = inspect.signature(QdrantVectorStore.hybrid_search).parameters
    assert "alpha" not in params, "alpha must be removed (vestigial, RRF is parameter-free)"
    assert "graph_anchored_paths" not in params, "graph_anchored_paths must be removed"
    # The live kwargs (including the optional code lane) survive.
    assert {"embedding", "query_text", "top_k", "filters",
            "code_embedding", "code_store"} <= set(params)
