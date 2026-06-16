"""In-memory QdrantVectorStore smoke test (offline, no server).

Proves the `url=":memory:"` enablement: an in-process Qdrant can ensure a
collection, upsert a chunk with a hand-built embedding, and return it from a
search. This is the foundation the offline retrieval eval (samples/) builds on.
"""
from __future__ import annotations

import pytest

# upsert() auto-computes BM25 sparse vectors via fastembed; skip cleanly in jobs
# that don't install the `fastembed` extra (e.g. the plain `unit` job). The
# in-memory + upsert + search path is also exercised by the offline eval gate.
pytest.importorskip("fastembed")

from opsrag.interfaces.chunker import Chunk  # noqa: E402
from opsrag.interfaces.parser import DocType  # noqa: E402
from opsrag.vectorstores.qdrant import QdrantVectorStore  # noqa: E402


@pytest.mark.asyncio
async def test_inmemory_upsert_and_search():
    dim = 8
    store = QdrantVectorStore(
        url=":memory:", collection_name="eval_inmem", dimension=dim
    )
    await store.ensure_collection()

    chunk = Chunk(
        id="c1",
        content="acme-notes deploy rollback runbook",
        doc_type=DocType.RUNBOOK,
        source_path="runbooks/001-acme-notes-deploy.md",
        repo="samples",
        chunk_type="child",
    )
    # A trivial, normalized-ish fake embedding -- the search query reuses it so
    # cosine similarity is maximal and the chunk is the top hit.
    embedding = [1.0] + [0.0] * (dim - 1)
    n = await store.upsert([chunk], [embedding])
    assert n == 1

    results = await store.search(embedding=embedding, top_k=5)
    assert results, "in-memory search returned no results"
    assert results[0].chunk.source_path == "runbooks/001-acme-notes-deploy.md"
    assert results[0].chunk.id == "c1"
