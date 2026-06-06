"""Rerank node -- wraps a Reranker provider in state-dict plumbing.

Also applies a **path-aware boost** to chunks whose `source_path` or
`repo` literally contains an anchor entity extracted from the query
(see `opsrag.agent.anchors`). Without this, queries that name a
specific repo / module / service slug get drowned out by Confluence
pages that merely link to the same repo -- both score similarly under
pure dense+cross-encoder ranking because both mention the slug in text.

The boost is multiplicative and capped to keep the cross-encoder's
relative ordering meaningful when boosts collide (multiple chunks all
anchor-match).
"""
from __future__ import annotations

import logging

from opsrag.agent.anchors import extract_anchors, path_matches_any_anchor
from opsrag.interfaces.chunker import Chunk
from opsrag.interfaces.observability import ObservabilityProvider
from opsrag.interfaces.reranker import Reranker
from opsrag.interfaces.vectorstore import SearchResult

_log = logging.getLogger("opsrag.agent.reranker")

# Boost multiplier when chunk's source_path or repo contains an anchor
# token literally. Cross-encoder scores are in [0, 1] so 1.5x pushes a
# moderate match (0.4) above a strong-but-irrelevant match (0.55).
_PATH_ANCHOR_BOOST = 1.5

# Min rerank score considered "real signal" from the cross-encoder.
# `semantic-ranker-default-004` returns 0..1; scores below this are
# typically noise. Configurable via state["min_rerank_score"] for evals.
_DEFAULT_MIN_RERANK_SCORE = 0.05


def rerank_node(reranker: Reranker, observability: ObservabilityProvider, top_k: int = 5):
    async def _rerank(state: dict) -> dict:
        query = state["query"]
        chunks: list[Chunk] = state.get("merged_results") or state.get("retrieved_chunks") or []
        if not chunks:
            return {
                "merged_results": [],
                "best_rerank_score": 0.0,
                "anchors": [],
                "anchors_matched_in_results": False,
                "current_step": "reranked",
            }

        anchors = extract_anchors(query)

        # Listing-intent ("what's in repo X") and plural-repo intent
        # ("all repos with config of X") both want broad coverage --
        # config files don't semantically match the meta-question, so
        # narrow reranking throws the answer away. Skip rerank and take
        # a wider slice instead.
        if state.get("listing_intent") or state.get("plural_repo_intent"):
            wide_top_k = min(len(chunks), 30)
            kept = chunks[:wide_top_k]
            anchors_in_kept = any(
                path_matches_any_anchor(c.source_path, c.repo, anchors) for c in kept
            )
            return {
                "merged_results": kept,
                "best_rerank_score": 1.0,  # bypass min-gate for listing-intent
                "anchors": anchors,
                "anchors_matched_in_results": anchors_in_kept,
                "current_step": "reranked",
            }

        # Synthesis intent ("compare A vs B", "relationship between X and Y")
        # needs BOTH anchor docs in the final context. Default top_k=5 was
        # dropping one of the two when they had similar but not identical
        # semantic embeddings. Bump to 10 for synthesis only -- costs
        # roughly one extra rerank request per query, fixes recall on
        # multi_doc_synthesis from 0.45 -> ~0.7+ in eval.
        effective_top_k = top_k
        if state.get("synthesis_intent"):
            effective_top_k = max(top_k, 10)

        results = [SearchResult(chunk=c, score=0.0) for c in chunks]
        # Pull a wider candidate pool from the cross-encoder so the
        # path-anchor boost has room to re-order. We still cap output at
        # effective_top_k after boost.
        wide_top_k = max(effective_top_k * 3, effective_top_k)
        reranked = await reranker.rerank(query, results, top_k=wide_top_k)

        # Apply path-anchor boost. Order-preserving multiplicative boost.
        boosted = []
        for r in reranked:
            chunk = r.chunk
            base = float(r.relevance_score)
            if anchors and path_matches_any_anchor(chunk.source_path, chunk.repo, anchors):
                score = base * _PATH_ANCHOR_BOOST
            else:
                score = base
            boosted.append((score, base, chunk))
        boosted.sort(key=lambda t: t[0], reverse=True)

        kept = [c for _s, _b, c in boosted[:effective_top_k]]
        best_base_score = max((b for _s, b, _c in boosted), default=0.0)
        anchors_in_kept = any(
            path_matches_any_anchor(c.source_path, c.repo, anchors) for c in kept
        )

        if anchors and not anchors_in_kept:
            _log.info(
                "rerank: anchors=%s present in query but NO chunk path/repo matches; "
                "best_score=%.3f kept=%d",
                anchors, best_base_score, len(kept),
            )

        return {
            "merged_results": kept,
            "best_rerank_score": best_base_score,
            "anchors": anchors,
            "anchors_matched_in_results": anchors_in_kept,
            "current_step": "reranked",
        }

    return _rerank


def rerank_decision(state: dict) -> str:
    """Conditional edge AFTER rerank.

    If the user named specific entities (anchors) but NONE of the kept
    chunks' source_path/repo match any anchor, AND the cross-encoder's
    best score is below the noise floor, the retrieval has nothing
    relevant to the actually-asked-about entity. Skip generation and
    emit insufficient_info immediately -- rewriting the query and
    re-retrieving will not help because the entity is either not in
    the corpus or is named differently than the user wrote it.

    Returns:
      "weak_retrieval"  -> route to insufficient_info node
      "ok"              -> continue to grader / generator
    """
    min_score = float(state.get("min_rerank_score", _DEFAULT_MIN_RERANK_SCORE))
    best = float(state.get("best_rerank_score", 0.0))
    anchors = state.get("anchors") or []
    matched = bool(state.get("anchors_matched_in_results", False))
    kept = state.get("merged_results") or []

    if not kept:
        return "weak_retrieval"
    if anchors and not matched and best < min_score:
        return "weak_retrieval"
    return "ok"
