"""Offline retrieval eval harness over the shipped ``samples/`` corpus.

No API keys, no running services: a local FastEmbed ONNX embedder
(``BAAI/bge-small-en-v1.5``, 384-dim) + an in-process Qdrant
(``url=":memory:"``) index the synthetic ``samples/`` corpus, then we score
the public golden set with pure-arithmetic ranking metrics (Recall@K, MRR,
Precision@K, NDCG@K).

Retrieval exercises the SAME core pipeline the live app uses by default --
**Dense + BM25 RRF fusion** (``hybrid_search``) followed by a **local
FastEmbed cross-encoder rerank** over the fused candidates -- all of which
run locally (the Qdrant ``Qdrant/bm25`` sparse model and the
``Xenova/ms-marco-MiniLM-L-6-v2`` cross-encoder are ONNX models cached on
disk, no network at score time). It does NOT exercise MMR diversification or
the live LLM faithfulness judge -- those stay in the live-server tier.

The only network the harness needs is the one-time FastEmbed model fetch
(cached after the first run). This is the always-on, secret-free proof that
retrieval actually works -- the answer-quality tier (LLM + judge) lives in
``python -m opsrag.eval run``.

Public API
----------
``build_offline_index(samples_dir)`` -> ``(embedder, vector_store)``
``retrieval_scores(embedder, vector_store, goldens, k=5)`` -> aggregate dict

``retrieval_scores`` returns aggregate ``mean_recall_at_k``, ``mean_mrr``,
``mean_precision_at_k`` and ``mean_ndcg_at_k`` (all computed inline with
binary relevance -- no deepeval/vertexai import).

``retrieval_scores`` runs ``hybrid_search`` (Dense+BM25 RRF) then a local
cross-encoder rerank for each golden; pass ``reranker=`` to inject one,
otherwise the app's default local ``FastEmbedReranker`` is built once.

Both reuse the production ingestion pipeline and the shared ``match_path`` /
``_expected_hits_in_topk`` logic so the eval grades against the same matching
rules SourceRecall uses in the live gate.
"""
from __future__ import annotations

from math import log2
from pathlib import Path
from typing import TYPE_CHECKING

from opsrag.eval.loaders import canonical_path, match_path

if TYPE_CHECKING:
    from opsrag.eval.loaders import GoldenQuery


def _expected_hits_in_topk(
    expected: list[str], retrieved: list[str], k: int
) -> list[str]:
    """Subset of ``expected`` that has a path-match in ``retrieved[:k]``.

    Inlined (instead of importing from ``opsrag.eval.metrics.ranking``) so this
    offline gate imports NO deepeval/vertexai: importing that metrics submodule
    triggers ``metrics/__init__`` -> faithfulness -> vertex_judge -> vertexai,
    none of which the secret-free retrieval gate needs. Same logic as ranking's.
    """
    top = retrieved[:k]
    return [e for e in expected if any(match_path(e, r) for r in top)]


def _dedupe_to_documents(retrieved: list[str]) -> list[str]:
    """Collapse a per-CHUNK retrieved list to DOCUMENT granularity.

    Retrieval returns one entry per chunk, so a single source doc surfaces
    several times. Precision@K/NDCG@K must grade distinct docs (like Recall@K),
    not chunk positions -- otherwise one relevant doc split into 5 chunks reads
    as P@5=1.0. Keep first occurrence per ``canonical_path`` (the same
    chunker-stable key Recall@K matches on), preserving rank order.
    """
    seen: set[str] = set()
    out: list[str] = []
    for r in retrieved:
        key = canonical_path(r)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

# 384-dim local ONNX embedder; matches the validated offline recipe.
_OFFLINE_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


async def build_offline_index(samples_dir: str | Path):
    """Index ``samples_dir`` into an in-process Qdrant with a local embedder.

    Returns ``(embedder, vector_store)``. The vector store holds every chunk
    of the corpus; ``embedder.embed_query`` produces query vectors compatible
    with it. Lazy imports keep the heavy ingestion stack out of module import.
    """
    from qdrant_client import AsyncQdrantClient  # noqa: F401  (ensures dep present)

    from opsrag.chunkers.parent_child import ParentChildChunker
    from opsrag.embedders.fastembed import FastEmbedEmbeddings
    from opsrag.ingestion.indexer import index_local_path
    from opsrag.ingestion.pipeline import IngestionPipeline
    from opsrag.parsers.generic import GenericConfigParser
    from opsrag.parsers.helm import HelmParser
    from opsrag.parsers.k8s import K8sManifestParser
    from opsrag.parsers.markdown import GenericMarkdownParser
    from opsrag.parsers.postmortem import PostmortemParser
    from opsrag.parsers.runbook import RunbookParser
    from opsrag.parsers.terraform import TerraformParser
    from opsrag.vectorstores.qdrant import QdrantVectorStore

    embedder = FastEmbedEmbeddings(model=_OFFLINE_EMBED_MODEL)
    vector_store = QdrantVectorStore(
        url=":memory:",
        collection_name="eval_samples",
        dimension=embedder.dimension,
    )
    parsers = [
        RunbookParser(),
        PostmortemParser(),
        K8sManifestParser(),
        HelmParser(),
        TerraformParser(),
        GenericMarkdownParser(),
        GenericConfigParser(),
    ]
    pipeline = IngestionPipeline(
        scm=None,
        parsers=parsers,
        chunker=ParentChildChunker(),
        embedder=embedder,
        vector_store=vector_store,
    )
    await index_local_path(
        pipeline, str(samples_dir), repo="samples", branch="local"
    )
    return embedder, vector_store


def _scored(g: GoldenQuery) -> bool:
    """A golden is scored only if it defines a relevant set (expected OR
    acceptable). Negative goldens (both empty) are skipped for ranking -- the
    score is meaningless without a relevant doc and they exist for the
    answer-quality tier."""
    return bool(g.expected_sources or g.acceptable_sources)


# Depth of the fused candidate pool handed to the cross-encoder reranker. The
# pool must be deep enough that, after the per-chunk -> per-document dedupe,
# there are still >= K distinct docs to rank -- so the rerank actually decides
# the top-K rather than rubber-stamping whatever fusion surfaced. Matches the
# spirit of the live app, which reranks a multiples-of-top_k pool.
_RERANK_POOL = 50


def _build_default_reranker():
    """Construct the SAME local cross-encoder reranker the app uses by default.

    The shipped default ``reranker.provider`` is ``fastembed`` ->
    ``FastEmbedReranker`` (``Xenova/ms-marco-MiniLM-L-6-v2``), a local ONNX
    cross-encoder -- no API key, no network at score time (only the one-time
    model fetch, like the embedder). Constructed lazily so module import stays
    cheap and the stub-driven unit tests can inject their own reranker.
    """
    from opsrag.rerankers.fastembed_reranker import FastEmbedReranker

    return FastEmbedReranker()


async def retrieval_scores(
    embedder,
    vector_store,
    goldens: list[GoldenQuery],
    k: int = 5,
    reranker=None,
) -> dict:
    """Run retrieval for every scored golden; return per-golden + aggregate.

    Retrieval mirrors the app's default core path: ``hybrid_search`` (Dense +
    BM25 RRF fusion, both local) returns a deep fused candidate pool, which a
    local FastEmbed cross-encoder reranker re-orders; the reranked order is the
    scored ``retrieved`` list. MMR diversification and the live LLM judge are
    intentionally NOT exercised here (they live in the live-server tier). When
    ``reranker`` is ``None`` the default local cross-encoder is constructed once
    and reused across all goldens.

    Recall@K (per golden):
      - ``expected_sources`` non-empty -> AND-recall: fraction of expected
        sources found in the top-K (reuses ``_expected_hits_in_topk``).
      - else (``acceptable_sources`` only) -> OR-recall: 1.0 if ANY acceptable
        source is in the top-K, else 0.0.

    MRR (per golden): reciprocal rank of the first relevant DOCUMENT in the
    top-K deduped list (the per-chunk retrieved list is collapsed to distinct
    docs and bounded by K first, matching the Precision@K/NDCG@K basis below);
    0.0 if no expected/acceptable source appears.

    Precision@K and NDCG@K are computed at DOCUMENT granularity: the raw
    retrieved list is per-CHUNK (the same source doc surfaces as several
    chunks), so before scoring it is deduped to first-occurrence per canonical
    doc id/path -- mirroring how Recall@K above counts distinct expected docs.
    This keeps both metrics document-level and directly comparable to Recall
    (one relevant doc retrieved as 5 chunks no longer reads as P@5=1.0).

    Precision@K (per golden): fraction of the top-K deduped (document-level)
    retrieved positions that match a relevant (expected OR acceptable) source.
    Binary relevance; the denominator is ``min(k, len(deduped))`` so a short
    result list isn't penalised for positions that never existed.

    NDCG@K (per golden): binary-gain DCG over the top-K deduped (document-level)
    positions (gain 1 for a relevant doc, discount ``1/log2(rank+1)``)
    normalised by the ideal DCG -- all relevant docs ranked first, capped at K.
    0.0 when nothing relevant exists in range.

    Aggregate keys: ``mean_recall_at_k``, ``mean_mrr``,
    ``mean_precision_at_k``, ``mean_ndcg_at_k``, ``k``, ``num_scored``,
    ``num_skipped``, ``per_golden`` (list of dicts).
    """
    per_golden: list[dict] = []
    recall_sum = 0.0
    mrr_sum = 0.0
    precision_sum = 0.0
    ndcg_sum = 0.0
    scored = 0
    skipped = 0

    if reranker is None:
        reranker = _build_default_reranker()

    for g in goldens:
        if not _scored(g):
            skipped += 1
            continue
        scored += 1
        embedding = await embedder.embed_query(g.query)
        # Dense + BM25 RRF fusion (both local) -> a deep fused candidate pool.
        candidates = await vector_store.hybrid_search(
            embedding=embedding, query_text=g.query, top_k=_RERANK_POOL
        )
        # Local FastEmbed cross-encoder rerank over the fused pool. Keep the
        # whole pool (top_k=len) so the per-chunk -> per-document dedupe below
        # still has depth; the reranked order IS the scored retrieved list, so
        # the gate exercises Dense+BM25+RRF+rerank end to end (no MMR, no judge).
        reranked = await reranker.rerank(
            g.query, candidates, top_k=len(candidates)
        )
        retrieved = [r.chunk.source_path for r in reranked]

        if g.expected_sources:
            hits = _expected_hits_in_topk(g.expected_sources, retrieved, k)
            recall = len(hits) / len(g.expected_sources)
        else:  # acceptable-only -> OR-recall
            top = retrieved[:k]
            recall = (
                1.0
                if any(
                    any(match_path(a, r) for r in top)
                    for a in g.acceptable_sources
                )
                else 0.0
            )

        relevant = list(g.expected_sources) + list(g.acceptable_sources)
        # Dedupe the per-chunk list to DOCUMENT granularity first (first
        # occurrence per canonical path), then take the top-K. This makes
        # Precision@K/NDCG@K document-level and comparable to Recall@K -- a doc
        # split into N chunks no longer counts as N relevant positions.
        documents = _dedupe_to_documents(retrieved)
        top = documents[:k]
        # Per-position binary relevance over the top-K docs (a retrieved doc is
        # relevant if it matches ANY expected OR acceptable source). Shared by
        # MRR, Precision@K and NDCG@K so they grade identical relevance.
        rels = [any(match_path(e, r) for e in relevant) for r in top]

        mrr = 0.0
        for rank, is_rel in enumerate(rels, start=1):
            if is_rel:
                mrr = 1.0 / rank
                break

        # Precision@K: relevant positions / K (denominator clamped to the
        # number of positions that actually exist so a short list isn't
        # under-credited for ranks it never filled).
        num_relevant_positions = sum(rels)
        denom = min(k, len(top)) or 1
        precision = num_relevant_positions / denom

        # NDCG@K with binary gains. DCG sums 1/log2(rank+1) over relevant
        # positions; IDCG is the same sum with the relevant positions packed at
        # the front (count capped at K). Normalising by IDCG keeps it in [0, 1].
        dcg = sum(
            1.0 / log2(rank + 1)
            for rank, is_rel in enumerate(rels, start=1)
            if is_rel
        )
        ideal_count = min(num_relevant_positions, k)
        idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_count + 1))
        ndcg = (dcg / idcg) if idcg else 0.0

        recall_sum += recall
        mrr_sum += mrr
        precision_sum += precision
        ndcg_sum += ndcg
        per_golden.append(
            {
                "id": g.id,
                "category": g.category,
                "recall_at_k": recall,
                "mrr": mrr,
                "precision_at_k": precision,
                "ndcg_at_k": ndcg,
                "retrieved": retrieved,
            }
        )

    return {
        "k": k,
        "num_scored": scored,
        "num_skipped": skipped,
        "mean_recall_at_k": (recall_sum / scored) if scored else 0.0,
        "mean_mrr": (mrr_sum / scored) if scored else 0.0,
        "mean_precision_at_k": (precision_sum / scored) if scored else 0.0,
        "mean_ndcg_at_k": (ndcg_sum / scored) if scored else 0.0,
        "per_golden": per_golden,
    }
