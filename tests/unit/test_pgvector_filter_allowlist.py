"""R5 / R11 coverage for the pgvector vector store.

Fully OFFLINE (no Postgres). Proves:

  * R5 -- _build_where pins the filter KEY (which is interpolated straight into
    the WHERE clause, not a $N placeholder) to the known column allowlist and
    FAILS CLOSED on an unknown key. This is the guardrail that keeps
    delete_by_filter (a DESTRUCTIVE statement) from ever routing an
    attacker-influenced key into raw SQL.
  * R5 -- known keys still build the expected parameterized clause (no
    regression for the hardcoded repo / source_path callers).
  * R11 -- a one-time WARNING is emitted noting pgvector lacks the BM25 + code
    lanes Qdrant provides (hybrid quality reduced).
"""
from __future__ import annotations

import logging

import pytest

from opsrag.vectorstores import pgvector as pgmod
from opsrag.vectorstores.pgvector import (
    _FILTER_KEY_ALLOWLIST,
    PgVectorStore,
)


# --------------------------------------------------------------------------
# R5 -- _build_where key allowlist.
# --------------------------------------------------------------------------
def test_build_where_rejects_unknown_key():
    with pytest.raises(ValueError, match="not an allowed column"):
        PgVectorStore._build_where({"id": "x"})


def test_build_where_rejects_injection_shaped_key():
    # The exact threat R5 guards: a key that is raw SQL, not a column. Must
    # raise BEFORE it can be interpolated into a (potentially DELETE) WHERE.
    with pytest.raises(ValueError, match="not an allowed column"):
        PgVectorStore._build_where({"repo = repo OR 1=1; --": "x"})


def test_build_where_rejects_unknown_among_known():
    # A single bad key in an otherwise-valid filter dict still fails closed.
    with pytest.raises(ValueError, match="not an allowed column"):
        PgVectorStore._build_where({"repo": "samples", "entity_ids": ["e1"]})


def test_build_where_accepts_known_scalar_key():
    where, params = PgVectorStore._build_where({"repo": "samples"}, start_idx=2)
    assert where == " AND repo = $2"
    assert params == ["samples"]


def test_build_where_accepts_known_list_key_as_any():
    where, params = PgVectorStore._build_where(
        {"source_path": ["a.md", "b.md"]}, start_idx=1
    )
    assert where == " AND source_path = ANY($1)"
    assert params == [["a.md", "b.md"]]


def test_build_where_multi_known_keys_preserved():
    # The real delete_by_filter call shape: {repo, source_path}.
    where, params = PgVectorStore._build_where(
        {"repo": "samples", "source_path": "runbooks/001.md"}, start_idx=1
    )
    assert " AND repo = $1" in where
    assert " AND source_path = $2" in where
    assert params == ["samples", "runbooks/001.md"]


def test_build_where_empty_is_noop():
    assert PgVectorStore._build_where(None) == ("", [])
    assert PgVectorStore._build_where({}) == ("", [])


def test_allowlist_is_subset_of_real_columns():
    # Guard against the allowlist drifting to a name that isn't a real column
    # (which would re-open a path to a column-does-not-exist error or worse).
    real_columns = {
        "id", "chunk_id", "content", "doc_type", "source_path", "repo",
        "parent_chunk_id", "chunk_type", "token_count", "metadata",
        "priority", "embedding",
    }
    assert _FILTER_KEY_ALLOWLIST <= real_columns


# --------------------------------------------------------------------------
# R11 -- one-time hybrid-parity warning.
# --------------------------------------------------------------------------
def test_parity_warning_emitted_once(caplog):
    pgmod._parity_warned = False  # reset the module-level latch for the test
    with caplog.at_level(logging.WARNING, logger="opsrag.vectorstores.pgvector"):
        PgVectorStore._warn_hybrid_parity()
        PgVectorStore._warn_hybrid_parity()  # second call must be a no-op
    parity_msgs = [
        r.getMessage() for r in caplog.records
        if "BM25" in r.getMessage()
    ]
    assert len(parity_msgs) == 1, "parity warning must fire exactly once"
    assert "use Qdrant for full hybrid" in parity_msgs[0]
