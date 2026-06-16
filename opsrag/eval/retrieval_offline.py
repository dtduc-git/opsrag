"""Offline retrieval eval harness over the shipped ``samples/`` corpus.

No API keys, no running services: a local FastEmbed ONNX embedder
(``BAAI/bge-small-en-v1.5``, 384-dim) + an in-process Qdrant
(``url=":memory:"``) index the synthetic ``samples/`` corpus, then we score
the public golden set with pure-arithmetic ranking metrics (Recall@K, MRR).

The only network the harness needs is the one-time FastEmbed model fetch
(cached after the first run). This is the always-on, secret-free proof that
retrieval actually works -- the answer-quality tier (LLM + judge) lives in
``python -m opsrag.eval run``.

Public API
----------
``build_offline_index(samples_dir)`` -> ``(embedder, vector_store)``
``retrieval_scores(embedder, vector_store, goldens, k=5)`` -> aggregate dict

Both reuse the production ingestion pipeline and the shared ``match_path`` /
``_expected_hits_in_topk`` logic so the eval grades against the same matching
rules SourceRecall uses in the live gate.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from opsrag.eval.loaders import match_path

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


async def retrieval_scores(
    embedder,
    vector_store,
    goldens: list[GoldenQuery],
    k: int = 5,
) -> dict:
    """Run retrieval for every scored golden; return per-golden + aggregate.

    Recall@K (per golden):
      - ``expected_sources`` non-empty -> AND-recall: fraction of expected
        sources found in the top-K (reuses ``_expected_hits_in_topk``).
      - else (``acceptable_sources`` only) -> OR-recall: 1.0 if ANY acceptable
        source is in the top-K, else 0.0.

    MRR (per golden): reciprocal rank of the first retrieved source matching
    any expected OR acceptable source; 0.0 if none.

    Aggregate keys: ``mean_recall_at_k``, ``mean_mrr``, ``k``,
    ``num_scored``, ``num_skipped``, ``per_golden`` (list of dicts).
    """
    per_golden: list[dict] = []
    recall_sum = 0.0
    mrr_sum = 0.0
    scored = 0
    skipped = 0

    for g in goldens:
        if not _scored(g):
            skipped += 1
            continue
        scored += 1
        embedding = await embedder.embed_query(g.query)
        results = await vector_store.search(embedding=embedding, top_k=k)
        retrieved = [r.chunk.source_path for r in results]

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
        mrr = 0.0
        for rank, r in enumerate(retrieved, start=1):
            if any(match_path(e, r) for e in relevant):
                mrr = 1.0 / rank
                break

        recall_sum += recall
        mrr_sum += mrr
        per_golden.append(
            {
                "id": g.id,
                "category": g.category,
                "recall_at_k": recall,
                "mrr": mrr,
                "retrieved": retrieved,
            }
        )

    return {
        "k": k,
        "num_scored": scored,
        "num_skipped": skipped,
        "mean_recall_at_k": (recall_sum / scored) if scored else 0.0,
        "mean_mrr": (mrr_sum / scored) if scored else 0.0,
        "per_golden": per_golden,
    }
