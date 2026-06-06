"""T1.1 -- Cross-pool Reciprocal Rank Fusion.

When multi-query decomposition fans out into N parallel
`hybrid_search` calls, each returning its own ranked list of
`SearchResult`s, we need to merge the N pools into a single ranked
list. RRF (Cormack et al. 2009) is the standard approach -- same
algorithm `qdrant.py:hybrid_search` already uses internally to fuse
dense + BM25 + graph + code-dense lanes; here we apply it ONE LEVEL
UP across whole result lists from parallel sub-queries.

Why a separate helper instead of cramming this into hybrid_search:
the hybrid_search internal RRF fuses *raw qdrant points* from
different lanes of a SINGLE query. Cross-pool RRF fuses *already-
ranked SearchResult lists* from MULTIPLE queries. Same math,
different inputs. Keeping them separate keeps each function
single-purpose and unit-testable.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable

from opsrag.interfaces.vectorstore import SearchResult

_log = logging.getLogger("opsrag.vectorstores.rrf")

_RRF_K = 60  # Cormack 2009 default -- match hybrid_search.py constant.


def _chunk_key(r: SearchResult) -> str:
    """Stable identity for a chunk across pools. `chunk.id` is set by
    the chunker and persists across embeddings."""
    return getattr(r.chunk, "id", "") or f"{r.chunk.repo}:{r.chunk.source_path}:{hash(r.chunk.content[:200])}"


def rrf_merge_pools(
    pools: Iterable[list[SearchResult]],
    top_k: int = 10,
    pool_weights: list[float] | None = None,
) -> list[SearchResult]:
    """Merge N already-ranked SearchResult lists via RRF.

    For each chunk, RRF score = sum across pools of
        pool_weight[i] / (k + rank_in_pool_i + 1)
    Chunks that appear in MULTIPLE pools accumulate score -- strong
    consensus signal. Chunks in only one pool still contribute
    (single-source evidence).

    `pool_weights` defaults to 1.0 per pool. The FIRST pool (which
    by convention is the original query's own retrieval) can be
    weighted higher to bias toward "what the user literally asked"
    if needed; for now we treat all pools equally so a strong
    secondary-pool hit isn't suppressed.

    Preserves SearchResult shape; the returned objects are the SAME
    instances seen first across pools (first-occurrence wins for
    payload/score metadata, fused RRF score overwrites `.score`).
    """
    pools_list = list(pools)
    if not pools_list:
        return []
    if pool_weights is None:
        pool_weights = [1.0] * len(pools_list)
    elif len(pool_weights) != len(pools_list):
        raise ValueError(
            f"pool_weights length {len(pool_weights)} != pools length {len(pools_list)}"
        )

    rrf_score: dict[str, float] = {}
    first_seen: dict[str, SearchResult] = {}
    for pool_idx, (pool, weight) in enumerate(zip(pools_list, pool_weights)):
        for rank, result in enumerate(pool):
            key = _chunk_key(result)
            rrf_score[key] = rrf_score.get(key, 0.0) + weight / (_RRF_K + rank + 1)
            if key not in first_seen:
                first_seen[key] = result

    ranked = sorted(rrf_score.items(), key=lambda x: x[1], reverse=True)[:top_k]
    out: list[SearchResult] = []
    for key, score in ranked:
        sr = first_seen[key]
        # Replace score with the cross-pool RRF score so downstream
        # callers (reranker, generator) see a consistent fused signal
        # rather than per-pool scores.
        out.append(SearchResult(
            chunk=sr.chunk,
            score=float(score),
            distance_metric="rrf-cross-pool",
        ))
    _log.info(
        "rrf_merge_pools: %d pools -> %d unique chunks -> top %d",
        len(pools_list), len(rrf_score), len(out),
    )
    return out
