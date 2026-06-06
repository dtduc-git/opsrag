"""entity_expand node -- the 1-hop entity-graph augmentation.

Runs AFTER vector_retrieve, BEFORE rerank. It is NOT the retrieval main line:
it only ADDS a few related chunks (fail-safe -- empty graph => no change). Flow:

  1. take the top vector chunks (already the authoritative result)
  2. gather their `entity_ids` (stamped on the Qdrant payload at index time)
  3. 1-hop neighbors from the Postgres light graph (get_neighbors)
  4. fetch a few chunks whose `entity_ids` intersect the neighbors, ranked by
     relevance to the query (Qdrant MatchAny metadata filter + vector score)
  5. merge (dedup by chunk id) -> rerank picks the final set

This is the industry-recommended "lightweight entity graph + entity-id metadata
+ 1-hop" pattern: multi-hop reach without a graph engine and without the graph
ever driving the answer.
"""
from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger("opsrag.agent.entity_expand")


def entity_expand_node(
    vector_store: Any,
    light_graph: Any,
    embedder: Any,
    *,
    seed_chunks: int = 6,
    max_neighbors: int = 40,
    expand_top_k: int = 4,
):
    async def _expand(state: dict) -> dict:
        chunks = state.get("merged_results") or state.get("retrieved_chunks") or []
        if not chunks or light_graph is None:
            return {"merged_results": chunks}

        # 1-2. seed entity ids from the top chunks.
        seeds: set[str] = set()
        for c in chunks[:seed_chunks]:
            md = getattr(c, "metadata", None) or {}
            for eid in (md.get("entity_ids") or []):
                if eid:
                    seeds.add(eid)
        if not seeds:
            return {"merged_results": chunks}

        # 3. 1-hop neighbors (Postgres adjacency). Fail-safe on any error.
        try:
            neighbors = await light_graph.get_neighbors(seeds, limit=max_neighbors)
        except Exception as exc:
            _log.debug("entity_expand: get_neighbors failed (%s) -- skipping", exc)
            return {"merged_results": chunks}
        if not neighbors:
            return {"merged_results": chunks}

        # 4. fetch related chunks via the entity_ids metadata filter, ranked by
        #    relevance to the query (cheap: query embedding is cached).
        try:
            q = state.get("query") or ""
            emb = await embedder.embed_query(q)
            extra = await vector_store.search(
                embedding=emb,
                top_k=expand_top_k,
                filters={"entity_ids": list(neighbors)},
            )
        except Exception as exc:
            _log.debug("entity_expand: filtered fetch failed (%s) -- skipping", exc)
            return {"merged_results": chunks}

        # 5. merge, dedup by chunk id (originals first, so vector order wins).
        seen = {getattr(c, "id", None) for c in chunks}
        merged = list(chunks)
        added = 0
        for r in extra:
            ch = getattr(r, "chunk", r)
            cid = getattr(ch, "id", None)
            if cid is not None and cid not in seen:
                merged.append(ch)
                seen.add(cid)
                added += 1
        if added:
            _log.info("entity_expand: +%d related chunk(s) via %d neighbor entities", added, len(neighbors))
        return {"merged_results": merged}

    return _expand
