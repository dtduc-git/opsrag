"""Rerank node -- wraps a Reranker provider in state-dict plumbing.

Also applies a **path-aware boost** to chunks whose `source_path` or
`repo` literally contains an anchor entity extracted from the query
(see `opsrag.agent.anchors`). Without this, queries that name a
specific repo / module / service slug get drowned out by Confluence
pages that merely link to the same repo -- both score similarly under
pure dense+cross-encoder ranking because both mention the slug in text.

The boost is additive and capped (see ``_PATH_ANCHOR_BONUS``) to keep
the cross-encoder's relative ordering meaningful when boosts collide
(multiple chunks all anchor-match) -- it only overturns CLOSE calls, not
large quality gaps.

The dedup -> rerank -> anchor-boost -> MMR pipeline lives in the shared,
pure ``apply_rerank_enrichments`` helper so the ``knowledge_search`` MCP
tool path (opsrag/mcp/knowledge.py) reuses the EXACT same logic and the
two retrieval paths can't re-diverge.
"""
from __future__ import annotations

import logging

from opsrag.agent.anchors import extract_anchors, path_matches_any_anchor
from opsrag.interfaces.chunker import Chunk
from opsrag.interfaces.observability import ObservabilityProvider
from opsrag.interfaces.reranker import Reranker
from opsrag.interfaces.vectorstore import SearchResult
from opsrag.rerankers.mmr import _tokens, mmr_reorder

_log = logging.getLogger("opsrag.agent.reranker")

# Additive, CAPPED tie-breaker bonus when a chunk's source_path/repo contains an
# anchor token literally. Cross-encoder scores are in [0, 1]. A multiplicative
# 1.5x (the old value) was NOT capped despite the docstring -- it let a path-
# matching 0.40 doc (->0.60) leapfrog a genuinely-better 0.55 doc. An additive
# bonus only overturns CLOSE calls (gaps < the bonus), not large quality gaps.
_PATH_ANCHOR_BONUS = 0.15

# Min rerank score considered "real signal" from the cross-encoder.
# `semantic-ranker-default-004` returns 0..1; scores below this are
# typically noise. Configurable via state["min_rerank_score"] for evals.
_DEFAULT_MIN_RERANK_SCORE = 0.05
# Fallback grader trust floor when a reranker doesn't declare trust_score.
_DEFAULT_TRUST_RERANK_SCORE = 0.65


def _content_dedup_pool(
    chunks: list[Chunk],
    *,
    content_dedup: bool,
    content_dedup_threshold: float,
    skip_near_dup: bool,
) -> list[Chunk]:
    """Drop exact (and optionally near-) duplicate-content chunks BEFORE the
    rerank call -- so the cross-encoder isn't billed for duplicate bytes and a
    copied values.yaml can't crowd out distinct docs.

    Keeps the FIRST occurrence to preserve the RRF/merge order. With a >0
    threshold AND ``skip_near_dup`` False, additionally drops a near-duplicate
    whose max token-set Jaccard vs an already-kept chunk exceeds the threshold.

    ``skip_near_dup`` is set for synthesis intent ("compare A vs B"), which
    deliberately wants both distinct-but-similar anchor docs; only exact
    byte-for-byte copies are collapsed there.

    Pure + side-effect free (no state, no closures) so BOTH rerank_node and
    the knowledge_search MCP path can share it and never re-diverge.
    """
    if not content_dedup:
        return list(chunks)

    threshold = float(content_dedup_threshold or 0.0)
    apply_near_dup = threshold > 0.0 and not skip_near_dup
    # Key exact-dedup on the content STRING itself, not hash(content): hashing
    # risks a silent drop of a DISTINCT chunk on a hash collision. Membership
    # on the string is exact.
    seen: set[str] = set()
    kept_token_sets: list[frozenset[str]] = []
    deduped: list[Chunk] = []
    for c in chunks:
        content = c.content or ""
        if content in seen:
            continue
        if apply_near_dup:
            toks = _tokens(content)
            is_near_dup = any(
                (
                    1.0 if (not toks and not kt)
                    else 0.0 if (not toks or not kt)
                    else len(toks & kt) / len(toks | kt)
                ) > threshold
                for kt in kept_token_sets
            )
            if is_near_dup:
                continue
            kept_token_sets.append(toks)
        seen.add(content)
        deduped.append(c)
    if len(deduped) < len(chunks):
        _log.info(
            "rerank: content-dedup collapsed pool %d -> %d (threshold=%.2f)",
            len(chunks), len(deduped), threshold,
        )
    return deduped


async def apply_rerank_enrichments(
    query: str,
    pool: list[Chunk],
    reranker: Reranker,
    *,
    anchors: list[str] | None = None,
    top_k: int = 5,
    diversity: float = 0.0,
    content_dedup: bool = True,
    content_dedup_threshold: float = 0.0,
    skip_near_dup: bool = False,
) -> tuple[list[Chunk], float, bool]:
    """Shared rerank-enrichment pipeline used by BOTH the LangGraph rerank node
    and the ``knowledge_search`` MCP tool, so the two paths can't re-diverge.

    Given a candidate ``pool`` of chunks (already merged / RRF-fused), this:

      1. content-deduplicates the pool BEFORE the rerank call
         (gated by ``content_dedup`` + ``content_dedup_threshold``);
      2. cross-encoder reranks the FULL deduped pool;
      3. applies the additive, capped path-anchor boost
         (``+_PATH_ANCHOR_BONUS`` when a chunk's source_path/repo literally
         contains an anchor) and re-sorts;
      4. optionally re-orders for MMR diversity (``diversity`` > 0).

    Returns ``(kept, best_score, anchors_matched)`` where ``best_score`` is the
    max RAW cross-encoder score (pre-boost) -- the weak-retrieval signal -- and
    ``anchors_matched`` is True iff a kept chunk's path/repo matches an anchor.

    The reranker is NOT invoked on an empty pool. This helper does NOT swallow
    reranker exceptions -- the caller decides the fallback (the node uses a
    neutral mid-band score; the MCP path falls back to pre-rerank order),
    because the two paths have different outage semantics.
    """
    anchors = anchors or []
    if not pool:
        return [], 0.0, False

    chunks = _content_dedup_pool(
        pool,
        content_dedup=content_dedup,
        content_dedup_threshold=content_dedup_threshold,
        skip_near_dup=skip_near_dup,
    )

    # Score the FULL candidate pool so the anchor boost can rescue an
    # anchor-matching doc the cross-encoder ranked deep.
    results = [SearchResult(chunk=c, score=0.0) for c in chunks]
    reranked = await reranker.rerank(query, results, top_k=len(results))

    # Path-anchor boost -- additive + capped (see _PATH_ANCHOR_BONUS).
    boosted: list[tuple[float, float, Chunk]] = []
    for r in reranked:
        chunk = r.chunk
        base = float(r.relevance_score)
        if anchors and path_matches_any_anchor(chunk.source_path, chunk.repo, anchors):
            score = min(1.0, base + _PATH_ANCHOR_BONUS)
        else:
            score = base
        boosted.append((score, base, chunk))
    boosted.sort(key=lambda t: t[0], reverse=True)

    active_diversity = float(diversity or 0.0)
    if active_diversity > 0.0 and len(boosted) > 1:
        reordered = mmr_reorder(
            boosted,
            relevance=[s for s, _b, _c in boosted],
            diversity=active_diversity,
            text_of=lambda t: t[2].content,
            top_k=top_k,
        )
        kept = [c for _s, _b, c in reordered]
    else:
        kept = [c for _s, _b, c in boosted[:top_k]]

    best_base_score = max((b for _s, b, _c in boosted), default=0.0)
    anchors_matched = any(
        path_matches_any_anchor(c.source_path, c.repo, anchors) for c in kept
    )
    return kept, best_base_score, anchors_matched


def rerank_node(
    reranker: Reranker,
    observability: ObservabilityProvider,
    top_k: int = 5,
    diversity: float = 0.0,
    content_dedup: bool = True,
    content_dedup_threshold: float = 0.0,
):
    async def _rerank(state: dict) -> dict:
        query = state["query"]
        chunks: list[Chunk] = state.get("merged_results") or state.get("retrieved_chunks") or []
        # Per-reranker calibration: the weak-retrieval floor and the grader's
        # trust-rerank floor are scale-dependent (FastEmbed sigmoid vs Cohere's
        # compressed-low [0,1]). Surface the active reranker's values into state
        # so rerank_decision + the grader use the right thresholds for THIS
        # provider instead of one hard-coded constant. An explicit eval override
        # of min_rerank_score still wins.
        rerank_floor = state.get(
            "min_rerank_score",
            getattr(reranker, "score_floor", _DEFAULT_MIN_RERANK_SCORE),
        )
        rerank_trust = getattr(reranker, "trust_score", _DEFAULT_TRUST_RERANK_SCORE)
        if not chunks:
            return {
                "merged_results": [],
                "best_rerank_score": 0.0,
                "anchors": [],
                "anchors_matched_in_results": False,
                "min_rerank_score": rerank_floor,
                "rerank_trust_score": rerank_trust,
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
                "min_rerank_score": rerank_floor,
                "rerank_trust_score": rerank_trust,
                "current_step": "reranked",
            }

        # Synthesis intent ("compare A vs B", "relationship between X and Y")
        # needs BOTH anchor docs in the final context. Default top_k=5 was
        # dropping one of the two when they had similar but not identical
        # semantic embeddings. Bump to 10 for synthesis only -- costs
        # roughly one extra rerank request per query, fixes recall on
        # multi_doc_synthesis from 0.45 -> ~0.7+ in eval.
        effective_top_k = top_k
        synthesis = bool(state.get("synthesis_intent"))
        if synthesis:
            effective_top_k = max(top_k, 10)

        # Content dedup + path-anchor boost + MMR run via the SHARED
        # apply_rerank_enrichments helper (also used by the knowledge_search
        # MCP tool) so the two retrieval paths can't re-diverge. State keys win
        # over the node's closure args (forwarded from cfg.agent.*), so both
        # per-request/eval overrides AND config-only settings take effect.
        #   * content_dedup gate (mirrors rerank_diversity): state -> closure.
        #   * synthesis intent skips the near-dup (Jaccard) pass so two
        #     distinct-but-similar anchor docs both survive; exact-dup collapse
        #     still runs.
        active_dedup = state.get("rerank_content_dedup", content_dedup)
        dedup_threshold = float(
            state.get("rerank_content_dedup_threshold", content_dedup_threshold) or 0.0
        )
        active_diversity = float(state.get("rerank_diversity", diversity) or 0.0)

        # The node owns reranker-outage semantics (the shared helper does NOT
        # swallow exceptions -- the MCP path falls back to pre-rerank order
        # instead). Wrap the reranker so an outage yields a NEUTRAL mid-band
        # score that DECOUPLES the two things a reranker score drives:
        #   * bypass the weak-retrieval gate (>= floor) -- an outage must not
        #     fake a spurious "insufficient information", AND
        #   * do NOT signal max confidence (< trust) to the grader's
        #     trust-rerank-over-CRAG floor -- an outage is when retrieval is
        #     LEAST verifiable, so it must not suppress CRAG correction.
        outage_flag = {"hit": False}
        _neutral = (rerank_floor + rerank_trust) / 2.0

        class _OutageGuardedReranker:
            async def rerank(self, q, results, top_k=5):  # noqa: ANN001
                try:
                    return await reranker.rerank(q, results, top_k=top_k)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("reranker failed (%s) -- falling back to vector order", exc)
                    from opsrag.interfaces.reranker import RerankResult
                    outage_flag["hit"] = True
                    return [
                        RerankResult(chunk=r.chunk, relevance_score=_neutral)
                        for r in results
                    ]

        kept, best_base_score, anchors_in_kept = await apply_rerank_enrichments(
            query,
            chunks,
            _OutageGuardedReranker(),
            anchors=anchors,
            top_k=effective_top_k,
            diversity=active_diversity,
            content_dedup=bool(active_dedup),
            content_dedup_threshold=dedup_threshold,
            skip_near_dup=synthesis,
        )
        reranker_outage = outage_flag["hit"]

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
            "min_rerank_score": rerank_floor,
            "rerank_trust_score": rerank_trust,
            "reranker_outage": reranker_outage,
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
