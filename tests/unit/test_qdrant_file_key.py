"""Unit tests: keyword-indexed `file_key` fast-delete path (Qdrant).

On Qdrant 1.12.4 the `repo`/`source_path` payload fields carry a TEXT index
(required by the MatchText lanes: filename fanout, repo-slug fanout,
find_repo_by_substring, enumerate_paths) and a field cannot hold BOTH a text
and a keyword index -- so the exact-match `delete_by_filter({repo,
source_path})` used by every per-file orphan sweep has NO usable index and
full-scans the collection (~7s per delete at ~700K points, measured).

Fix under test: a NEW payload field `file_key = repo + "\\x00" + source_path`,
written at upsert, KEYWORD-indexed at collection creation, and -- behind the
`use_file_key_delete` flag (default OFF until the live collection is
backfilled) -- used by `delete_by_filter` in place of the two text-indexed
fields. The text indexes are never touched, so no search lane can regress.

The store runs against a real in-process Qdrant (`url=":memory:"`); only the
BM25 sparse encoder is stubbed (fastembed model -- an external download -- is
not installed in the unit env, and BM25 vectors are irrelevant to payload
shape and delete filtering).
"""
from __future__ import annotations

from types import SimpleNamespace

from qdrant_client import models as qm

from opsrag.interfaces.chunker import Chunk
from opsrag.interfaces.parser import DocType

DIM = 8
EMB = [1.0] + [0.0] * (DIM - 1)


def _fake_sparse(texts):
    """Stand-in for bm25_sparse.encode_documents -- one trivial vector per text."""
    return [qm.SparseVector(indices=[0], values=[1.0]) for _ in texts]


def _mk_store(monkeypatch, **kwargs):
    from opsrag.vectorstores.qdrant import QdrantVectorStore

    monkeypatch.setattr(
        "opsrag.vectorstores.bm25_sparse.encode_documents", _fake_sparse
    )
    return QdrantVectorStore(
        url=":memory:", collection_name="fk_test", dimension=DIM, **kwargs
    )


async def _seed(store):
    """3 chunks across 2 files of the same repo: a.md (c1, c2) + b.md (c3)."""
    chunks = [
        Chunk(id="c1", content="alpha", doc_type=DocType.RUNBOOK,
              source_path="a.md", repo="r1"),
        Chunk(id="c2", content="beta", doc_type=DocType.RUNBOOK,
              source_path="a.md", repo="r1"),
        Chunk(id="c3", content="gamma", doc_type=DocType.RUNBOOK,
              source_path="b.md", repo="r1"),
    ]
    n = await store.upsert(chunks, [EMB] * 3)
    assert n == 3
    return chunks


def _spy_delete(store, captured: list):
    """Record the points_selector of every client.delete, then run it for real."""
    orig = store._client.delete

    async def spy(*args, **kwargs):
        captured.append(kwargs.get("points_selector"))
        return await orig(*args, **kwargs)

    store._client.delete = spy


def _must_keys(selector) -> list[str]:
    return [c.key for c in selector.filter.must]


async def _remaining_chunk_ids(store) -> set[str]:
    points, _ = await store._client.scroll(
        collection_name="fk_test", limit=100, with_payload=True,
    )
    return {p.payload["chunk_id"] for p in points}


# ---------------------------------------------------------------- helper --

def test_make_file_key_joins_repo_and_path_with_nul():
    from opsrag.vectorstores.qdrant import _FILE_KEY_SEP, _make_file_key

    assert _FILE_KEY_SEP == "\x00"
    assert _make_file_key("r1", "a/b.md") == "r1\x00a/b.md"


def test_make_file_key_coalesces_none_to_empty():
    from opsrag.vectorstores.qdrant import _make_file_key

    # Byte-identical None handling on the write and delete sides is what makes
    # the delete filter match the written key -- pin it.
    assert _make_file_key(None, "a.md") == "\x00a.md"
    assert _make_file_key("r1", None) == "r1\x00"
    assert _make_file_key(None, None) == "\x00"


# ----------------------------------------------------------------- config --

def test_config_flag_defaults_off():
    from opsrag.config import VectorStoreConfig

    assert VectorStoreConfig().use_file_key_delete is False
    assert VectorStoreConfig(use_file_key_delete=True).use_file_key_delete is True


# ------------------------------------------------------------------ write --

async def test_upsert_writes_file_key_payload(monkeypatch):
    store = _mk_store(monkeypatch)
    await _seed(store)
    points, _ = await store._client.scroll(
        collection_name="fk_test", limit=10, with_payload=True,
    )
    by_id = {p.payload["chunk_id"]: p.payload for p in points}
    assert by_id["c1"]["file_key"] == "r1\x00a.md"
    assert by_id["c3"]["file_key"] == "r1\x00b.md"


async def test_ensure_collection_requests_file_key_keyword_index(monkeypatch):
    store = _mk_store(monkeypatch)
    calls: list[tuple] = []

    async def spy_index(collection_name, field_name, field_schema, **kwargs):
        calls.append((field_name, field_schema))

    store._client.create_payload_index = spy_index
    await store.ensure_collection()
    keyword_fields = [f for f, s in calls if s == qm.PayloadSchemaType.KEYWORD]
    assert "file_key" in keyword_fields
    # The text indexes the search lanes depend on must still be requested.
    text_fields = [f for f, s in calls if not isinstance(s, qm.PayloadSchemaType)]
    assert {"content", "source_path", "repo"} <= set(text_fields)


# ----------------------------------------------------------------- delete --

async def test_delete_flag_off_uses_repo_and_source_path_conditions(monkeypatch):
    store = _mk_store(monkeypatch)  # default: flag OFF
    await _seed(store)
    captured: list = []
    _spy_delete(store, captured)

    await store.delete_by_filter({"repo": "r1", "source_path": "a.md"})

    assert sorted(_must_keys(captured[0])) == ["repo", "source_path"]
    assert await _remaining_chunk_ids(store) == {"c3"}


async def test_delete_flag_on_translates_to_file_key(monkeypatch):
    store = _mk_store(monkeypatch, use_file_key_delete=True)
    await _seed(store)
    captured: list = []
    _spy_delete(store, captured)

    await store.delete_by_filter({"repo": "r1", "source_path": "a.md"})

    assert _must_keys(captured[0]) == ["file_key"]
    cond = captured[0].filter.must[0]
    assert cond.match.value == "r1\x00a.md"
    assert await _remaining_chunk_ids(store) == {"c3"}


async def test_delete_flag_on_carries_extra_filter_keys(monkeypatch):
    store = _mk_store(monkeypatch, use_file_key_delete=True)
    await _seed(store)
    captured: list = []
    _spy_delete(store, captured)

    await store.delete_by_filter(
        {"repo": "r1", "source_path": "a.md", "chunk_type": "child"}
    )

    assert sorted(_must_keys(captured[0])) == ["chunk_type", "file_key"]
    assert await _remaining_chunk_ids(store) == {"c3"}


async def test_delete_flag_on_repo_only_falls_back(monkeypatch):
    store = _mk_store(monkeypatch, use_file_key_delete=True)
    await _seed(store)
    captured: list = []
    _spy_delete(store, captured)

    await store.delete_by_filter({"repo": "r1"})

    assert _must_keys(captured[0]) == ["repo"]
    assert await _remaining_chunk_ids(store) == set()


async def test_delete_flag_on_list_value_falls_back(monkeypatch):
    # A list repo means MatchAny semantics -- file_key cannot represent it.
    store = _mk_store(monkeypatch, use_file_key_delete=True)
    await _seed(store)
    captured: list = []
    _spy_delete(store, captured)

    await store.delete_by_filter({"repo": ["r1", "rX"], "source_path": "a.md"})

    assert sorted(_must_keys(captured[0])) == ["repo", "source_path"]
    assert await _remaining_chunk_ids(store) == {"c3"}


async def test_delete_flag_on_silently_misses_points_lacking_file_key(monkeypatch):
    """Pin the DOCUMENTED mixed-state hazard the rollout order exists for:
    with the flag ON, a translated delete matches ONLY points that carry
    file_key -- pre-backfill points survive untouched (stale chunks keep
    matching queries). This is why use_file_key_delete must stay OFF until
    the backfill's exhaustive verify reports 0 missing."""
    store = _mk_store(monkeypatch, use_file_key_delete=True)
    await store.ensure_collection()
    # Simulate a pre-backfill point: written WITHOUT file_key (as every point
    # predating this change is), bypassing the new upsert path.
    await store._client.upsert(
        collection_name="fk_test",
        points=[
            qm.PointStruct(
                id="00000000-0000-0000-0000-000000000001",
                vector={"dense": EMB},
                payload={"chunk_id": "old1", "repo": "r1", "source_path": "a.md"},
            )
        ],
        wait=True,
    )

    await store.delete_by_filter({"repo": "r1", "source_path": "a.md"})

    # The old point SURVIVES the translated delete -- the hazard, pinned.
    assert await _remaining_chunk_ids(store) == {"old1"}


# --------------------------------------------------------------- backfill --

def test_backfill_groups_points_missing_file_key():
    from opsrag.tools.backfill_file_key import group_missing_file_key

    points = [
        # Missing file_key -> needs backfill.
        SimpleNamespace(id="p1", payload={"repo": "r1", "source_path": "a.md"}),
        # Same file -> grouped under the same key (one set_payload call).
        SimpleNamespace(id="p2", payload={"repo": "r1", "source_path": "a.md"}),
        # Already correct -> skipped (idempotent re-runs stay cheap).
        SimpleNamespace(
            id="p3",
            payload={"repo": "r1", "source_path": "b.md",
                     "file_key": "r1\x00b.md"},
        ),
        # Present but WRONG -> repaired.
        SimpleNamespace(
            id="p4",
            payload={"repo": "r2", "source_path": "c.md", "file_key": "stale"},
        ),
        # None source_path -> coalesced exactly like the upsert write side.
        SimpleNamespace(id="p5", payload={"repo": "r3", "source_path": None}),
    ]
    groups = group_missing_file_key(points)
    assert groups == {
        "r1\x00a.md": ["p1", "p2"],
        "r2\x00c.md": ["p4"],
        "r3\x00": ["p5"],
    }
