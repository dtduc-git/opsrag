"""Offline retrieval eval gate over the shipped `samples/` corpus.

This is the always-on, NO-SECRETS proof that opsrag retrieval works: it
indexes `samples/` into an in-process Qdrant with a local FastEmbed ONNX
embedder, loads the public golden set, runs retrieval for every scored
golden, and asserts aggregate Recall@5, MRR, Precision@5 and NDCG@5 all
stay above their thresholds.

The only network it needs is the one-time FastEmbed model fetch (cached
thereafter). No API keys, no running services.

The full answer-quality eval (LLM + judge) lives in
`python -m opsrag.eval run` and needs credentials -- that's the other tier.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# FastEmbed is an optional extra (`uv sync --extra fastembed`). Skip cleanly
# rather than error when it (or qdrant-client) isn't installed.
pytest.importorskip("fastembed")
pytest.importorskip("qdrant_client")

from opsrag.eval.loaders import load_golden  # noqa: E402
from opsrag.eval.retrieval_offline import (  # noqa: E402
    build_offline_index,
    retrieval_scores,
)

_SAMPLES_DIR = Path(__file__).resolve().parents[2] / "samples"

# Observed aggregate Recall@5 over the public golden set was 1.0 on the
# validated recipe (BAAI/bge-small-en-v1.5 @384, in-process Qdrant). Threshold
# set well below that so embedder/chunker noise can't flake the gate while a
# real retrieval regression (a golden's doc dropping out of top-5) still trips.
_RECALL_THRESHOLD = 0.85

# Additional ranking-quality floors, all set conservatively below observed
# values so embedder/chunker noise can't flake the gate while a real regression
# (relevant docs sinking down the ranking) still trips. MRR/Precision/NDCG are
# rank-sensitive, so they sit lower than Recall@5: a golden's doc can be in the
# top-5 (recall stays high) yet ranked 3rd-5th (MRR/NDCG drop). Precision@5 and
# NDCG@5 are graded at DOCUMENT granularity (the per-chunk retrieved list is
# deduped to distinct docs first, like Recall@5), so Precision@5 is inherently
# <= (relevant docs in top-5) / (distinct docs in top-5) -- with most goldens
# having a single relevant doc, that lands well under 1.0, so its floor is the
# lowest of the three.
# Precision@5 is the structurally-capped metric (dominated by 1/(distinct docs
# in top-5) for the many single-relevant-doc goldens), so its floor sits well
# below the observed mean (~0.43) -- a real "relevant docs fall out of top-5"
# regression craters Recall/MRR/NDCG first, which carry the protective signal.
_MRR_THRESHOLD = 0.5
_PRECISION_THRESHOLD = 0.3
_NDCG_THRESHOLD = 0.5


@pytest.mark.asyncio
async def test_samples_retrieval_recall_at_5():
    assert _SAMPLES_DIR.is_dir(), f"samples corpus missing: {_SAMPLES_DIR}"

    goldens = load_golden()
    assert goldens, "no goldens loaded"

    embedder, vector_store = await build_offline_index(_SAMPLES_DIR)
    agg = await retrieval_scores(embedder, vector_store, goldens, k=5)

    assert agg["num_scored"] > 0, "no scored goldens (all skipped?)"

    # Surface per-golden detail on failure so a regression names the offender.
    below = [
        p for p in agg["per_golden"] if p["recall_at_k"] < 1.0
    ]
    detail = "\n".join(
        f"  {p['id']} R@5={p['recall_at_k']:.2f} retrieved={p['retrieved']}"
        for p in below
    )
    recall = agg["mean_recall_at_k"]
    mrr = agg["mean_mrr"]
    precision = agg["mean_precision_at_k"]
    ndcg = agg["mean_ndcg_at_k"]
    print(
        f"\nOffline retrieval eval over samples/: "
        f"mean Recall@5={recall:.4f}, mean MRR={mrr:.4f}, "
        f"mean Precision@5={precision:.4f}, mean NDCG@5={ndcg:.4f}, "
        f"scored={agg['num_scored']}, skipped={agg['num_skipped']}"
    )
    assert recall >= _RECALL_THRESHOLD, (
        f"aggregate Recall@5 {recall:.4f} < threshold {_RECALL_THRESHOLD}\n"
        f"goldens below 1.0:\n{detail}"
    )
    assert mrr >= _MRR_THRESHOLD, (
        f"aggregate MRR {mrr:.4f} < threshold {_MRR_THRESHOLD}\n"
        f"goldens below 1.0:\n{detail}"
    )
    assert precision >= _PRECISION_THRESHOLD, (
        f"aggregate Precision@5 {precision:.4f} < threshold "
        f"{_PRECISION_THRESHOLD}\ngoldens below 1.0:\n{detail}"
    )
    assert ndcg >= _NDCG_THRESHOLD, (
        f"aggregate NDCG@5 {ndcg:.4f} < threshold {_NDCG_THRESHOLD}\n"
        f"goldens below 1.0:\n{detail}"
    )


# --- document-vs-chunk granularity regression guard -------------------------
#
# retrieval_scores receives a PER-CHUNK retrieved list (the same source doc
# surfaces as several chunks). Precision@K/NDCG@K must grade DISTINCT docs --
# like Recall@K -- not chunk positions, otherwise one relevant doc split into 5
# chunks reads as P@5=1.0 and is incomparable to the document-level Recall@K.
# These stubs feed a controlled retrieved list straight into retrieval_scores
# (no FastEmbed/Qdrant) to assert the chunk->document dedupe holds.

from dataclasses import dataclass  # noqa: E402

from opsrag.eval.loaders import GoldenQuery  # noqa: E402
from opsrag.eval.retrieval_offline import _dedupe_to_documents  # noqa: E402


@dataclass
class _StubChunk:
    source_path: str


@dataclass
class _StubResult:
    chunk: _StubChunk


class _StubEmbedder:
    async def embed_query(self, _query: str):
        return [0.0]


class _StubStore:
    """Returns a fixed retrieved list regardless of the query embedding."""

    def __init__(self, paths: list[str]):
        self._paths = paths

    async def search(self, embedding, top_k: int):  # noqa: ARG002
        return [_StubResult(_StubChunk(p)) for p in self._paths[:top_k]]


def test_dedupe_to_documents_keeps_first_per_canonical_path():
    # Same doc as several chunks (incl. a repo-prefixed alias of one) collapses
    # to first-occurrence; a genuinely different doc survives.
    retrieved = [
        "runbooks/db-failover.md",
        "runbooks/db-failover.md",
        "runbooks/db-failover.md",
        "helm/values.yaml",
        "runbooks/db-failover.md",
    ]
    assert _dedupe_to_documents(retrieved) == [
        "runbooks/db-failover.md",
        "helm/values.yaml",
    ]


@pytest.mark.asyncio
async def test_precision_ndcg_are_document_level_not_chunk_inflated():
    # One relevant doc retrieved as 5 chunks. At CHUNK granularity this would be
    # P@5=1.0 / NDCG=1.0 (every position "relevant"); at DOCUMENT granularity
    # there is exactly one distinct doc, so P@5 = 1/1 = 1.0 but it is NOT 5/5 of
    # five padding positions -- prove the deduped denominator, then prove a doc
    # with irrelevant siblings is correctly penalised.
    golden = GoldenQuery(
        id="dedupe_001",
        category="factual_lookup",
        query="anything",
        expected_sources=["runbooks/db-failover.md"],
    )
    # 5 chunks of the SAME relevant doc -> 1 distinct doc, all relevant.
    same_doc = ["runbooks/db-failover.md"] * 5
    agg = await retrieval_scores(
        _StubEmbedder(), _StubStore(same_doc), [golden], k=5
    )
    p = agg["per_golden"][0]
    # Recall is 1.0 (the expected doc is present) and -- because there is only
    # one DISTINCT doc -- precision is 1/1, not the chunk-inflated 5/5-of-5.
    assert p["recall_at_k"] == 1.0
    assert p["precision_at_k"] == pytest.approx(1.0)

    # Now 1 relevant doc (as 3 chunks) + 2 distinct irrelevant docs. Chunk-level
    # precision would be 3/5=0.6; document-level it is 1 relevant of 3 distinct
    # docs = 1/3.
    mixed = [
        "runbooks/db-failover.md",
        "runbooks/db-failover.md",
        "runbooks/db-failover.md",
        "helm/values.yaml",
        "terraform/main.tf",
    ]
    agg2 = await retrieval_scores(
        _StubEmbedder(), _StubStore(mixed), [golden], k=5
    )
    p2 = agg2["per_golden"][0]
    assert p2["recall_at_k"] == 1.0
    assert p2["precision_at_k"] == pytest.approx(1.0 / 3.0)
    # NDCG: the single relevant doc is at deduped rank 1 -> DCG=IDCG -> 1.0.
    assert p2["ndcg_at_k"] == pytest.approx(1.0)
