"""Qdrant vector store provider.

Uses AsyncQdrantClient. Chunks are stored as points with their content
and metadata in the payload, keyed by a stable UUID derived from chunk.id.

Schema:
  - Named dense vector "dense" -- Vertex text-embedding-005, 768d, cosine
  - Named sparse vector "bm25" -- FastEmbed Qdrant/bm25 with IDF modifier

Hybrid search uses Reciprocal Rank Fusion (RRF, Cormack et al. 2009)
with k=60 (industry default, parameter-free fusion) over both vector
indices.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from opsrag.interfaces.chunker import Chunk
from opsrag.interfaces.parser import DocType
from opsrag.interfaces.vectorstore import SearchResult
from opsrag.vectorstores import bm25_sparse
from opsrag.vectorstores.lane_weights import compute_lane_weights

_log = logging.getLogger("opsrag.vectorstores.qdrant")

# Named vector field constants -- keep in sync with collection schema.
_DENSE = "dense"
_BM25 = "bm25"
# HNSW search-time ef. The Qdrant default (~100, tied to construction ef) caps
# recall even when we over-fetch candidate_k; raising it lets the ANN actually
# return the deep candidates the reranker is meant to rescue. Cost is sublinear.
_HNSW_EF = 192
# RRF fusion constant -- Cormack 2009 default. Don't tune without strong reason.
_RRF_K = 60

# Identifier-aware RRF lane weights now live in vectorstores/lane_weights.py
# (single source, shared with pgvector so both backends boost symbol queries
# identically). Imported as compute_lane_weights below.

# -- Priority boost (Layer 3 SRE-KB authoritative ranking) -----------------
# Chunks whose `repo` matches one of these substrings get `priority: high`
# stamped on their payload at upsert time. At search time both `search()`
# and `hybrid_search()` multiply the score of high-priority hits by
# `_HIGH_PRIORITY_BOOST`, so SRE-authored canonical docs out-rank
# Confluence/Slack on overlapping queries. The list is a substring match,
# not exact, so e.g. "devops/sre/sre-knowledge-base" and
# "sre-knowledge-base" both trigger.
_HIGH_PRIORITY_REPO_PATTERNS: tuple[str, ...] = (
    "sre-knowledge-base",
)
_HIGH_PRIORITY_BOOST: float = 1.5  # 50% lift -- enough to dominate ties, not steamroll

# User-correction tier. An OPERATOR-APPROVED correction (see the moderation
# queue in opsrag.pending_corrections -- corrections no longer go live without
# review) is stored in the main collection with `priority: user-correction` so
# it competes in the normal retrieval lane with a modest lift above SRE-KB.
# 1.8x is enough to win ties over generic canonical content without steamrolling
# every overlapping query the way the old un-moderated 2.5x did.
_USER_CORRECTION_BOOST: float = 1.8

# Canonical architecture docs (`docs/architecture/*` in
# the SRE-KB) describe the platform topology. Generic SRE-KB docs are
# already boosted at 1.5x, but for ambiguous queries like "show me the
# whole architecture" the architecture overview was being out-ranked by
# service-specific READMEs (which mention the organization name more often).
# Bump architecture docs to 2.0x -- above the SRE-KB baseline, below
# user-corrections -- so they reliably win when relevant.
_ARCHITECTURE_CANONICAL_BOOST: float = 2.0
_ARCHITECTURE_PATH_PREFIX: str = "docs/architecture/"


def _chunk_priority(
    repo: str | None, source_path: str | None = None,
) -> str | None:
    """Return a priority tag for the chunk, stamped into the payload at
    upsert time so search-time scoring is a cheap lookup.

    Tiers (highest -> lowest):
      - "architecture-canonical": SRE-KB chunks whose path begins with
        `docs/architecture/` -- the platform's canonical topology docs.
      - "high":                   any other SRE-KB chunk.
      - None:                     everything else (Confluence, Slack,
        service repos, ...) -- they compete at 1.0x.

    User-corrections set `priority` directly at upsert time (see
    `opsrag.correction_store.CorrectionStore`) -- this helper only
    assigns the SRE-KB and architecture-canonical tiers.
    """
    if not repo:
        return None
    low = repo.lower()
    if not any(p in low for p in _HIGH_PRIORITY_REPO_PATTERNS):
        return None
    if (
        source_path
        and source_path.startswith(_ARCHITECTURE_PATH_PREFIX)
    ):
        return "architecture-canonical"
    return "high"


def _priority_multiplier(payload: dict | None) -> float:
    """Score multiplier derived from the payload's priority tag. Defaults
    to 1.0 (no boost). Used by both dense `search()` and `hybrid_search()`
    so the boost is consistent regardless of retrieval path.

    Priorities:
      - "architecture-canonical"  -> 2.0x (SRE-KB docs/architecture/*)
      - "user-correction"         -> 1.8x (operator-APPROVED correction)
      - "high"                    -> 1.5x (other SRE-KB canonical content)
      - anything else             -> 1.0x (no boost)
    """
    # Single source of truth for the tier->multiplier mapping (also used by the
    # pgvector store + the agent fanout re-merge).
    from opsrag.vectorstores.priority import priority_multiplier
    return priority_multiplier((payload or {}).get("priority"))


def _chunk_point_id(chunk_id: str) -> str:
    """Derive a deterministic UUID from a chunk id so upserts are idempotent."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantVectorStore:
    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        collection_name: str = "opsrag",
        dimension: int = 3072,
        distance: str = "cosine",
    ):
        self._client = AsyncQdrantClient(url=url, api_key=api_key)
        self._collection = collection_name
        self._dimension = dimension
        self._distance = {
            "cosine": qm.Distance.COSINE,
            "dot": qm.Distance.DOT,
            "euclid": qm.Distance.EUCLID,
        }[distance]
        self._ensured = False

    async def ensure_collection(self) -> None:
        if self._ensured:
            return
        existing = await self._client.get_collections()
        names = {c.name for c in existing.collections}
        if self._collection not in names:
            # New collection: named dense + named sparse with IDF modifier.
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config={
                    _DENSE: qm.VectorParams(size=self._dimension, distance=self._distance),
                },
                sparse_vectors_config={
                    _BM25: qm.SparseVectorParams(modifier=qm.Modifier.IDF),
                },
            )
            for field in ("repo", "source_path", "doc_type", "entity_ids"):
                try:
                    await self._client.create_payload_index(
                        collection_name=self._collection,
                        field_name=field,
                        field_schema=qm.PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass
            # Full-text indexes. `content` powers BM25-style slug fanout, and
            # `source_path` / `repo` are REQUIRED for the MatchText filters in
            # search_by_path / find_repo_by_substring / enumerate_paths -- a
            # field with only a KEYWORD index makes those MatchText calls raise
            # "Index required ... text index", which the callers swallow and
            # return []. That silently killed filename fanout + path-tree
            # listing. A field can carry BOTH a keyword and a text index.
            for _text_field in ("content", "source_path", "repo"):
                try:
                    await self._client.create_payload_index(
                        collection_name=self._collection,
                        field_name=_text_field,
                        field_schema=qm.TextIndexParams(
                            type="text",
                            tokenizer=qm.TokenizerType.WORD,
                            min_token_len=2,
                            max_token_len=20,
                            lowercase=True,
                        ),
                    )
                except Exception:
                    pass
        self._ensured = True

    async def upsert(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        await self.ensure_collection()

        # Compute BM25 sparse vectors for each chunk's content. Local CPU,
        # no LLM cost. Batched encode handles fastembed's internal batching.
        sparse_vectors = bm25_sparse.encode_documents([c.content for c in chunks])
        # FastEmbed contracts one vector per input, but a partial/short return
        # would silently truncate the zip below and misalign sparse vectors
        # against the wrong payloads. Fail closed instead.
        if len(sparse_vectors) != len(chunks):
            raise ValueError(
                f"sparse vector count {len(sparse_vectors)} != chunk count {len(chunks)}"
            )

        points = [
            qm.PointStruct(
                id=_chunk_point_id(c.id),
                vector={
                    _DENSE: v,
                    _BM25: sv,
                },
                payload={
                    "chunk_id": c.id,
                    "content": c.content,
                    "doc_type": c.doc_type.value,
                    "source_path": c.source_path,
                    "repo": c.repo,
                    "parent_chunk_id": c.parent_chunk_id,
                    "chunk_type": c.chunk_type,
                    "token_count": c.token_count,
                    "metadata": c.metadata,
                    # Light-graph lane: deterministic entity ids this chunk
                    # mentions (top-level + KEYWORD-indexed so entity_expand can
                    # MatchAny-filter on them). Empty list when the lane is off.
                    "entity_ids": (c.metadata.get("entity_ids", []) if isinstance(c.metadata, dict) else []),
                    # Authoritative-content boost tag -- see _chunk_priority().
                    # `None` for ordinary content; "high" for SRE-KB so
                    # search-time scoring can rank it above Confluence/Slack
                    # on overlapping queries. Architecture canonical docs
                    # (`docs/architecture/*` in SRE-KB) get the higher
                    # "architecture-canonical" tier.
                    "priority": _chunk_priority(c.repo, c.source_path),
                },
            )
            for c, v, sv in zip(chunks, embeddings, sparse_vectors)
        ]
        # wait=True: the ingest pipeline records the per-file dedup hash
        # immediately after this returns. With wait=False a crash in that
        # window leaves the file hash-recorded (so it's skipped on the next
        # run) but its vectors never durably landed -> the file is permanently
        # absent from the index. Block until Qdrant acks the write instead.
        await self._client.upsert(collection_name=self._collection, points=points, wait=True)
        return len(points)

    async def search(
        self,
        embedding: list[float],
        top_k: int = 10,
        filters: dict | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Dense-only search (named "dense" vector). Kept for backwards compat.

        For new code: prefer hybrid_search() which uses BM25+dense RRF fusion.
        """
        await self.ensure_collection()
        qfilter = self._search_filter(filters)
        # Over-fetch so the post-boost re-rank doesn't drop relevant hits.
        # We boost high-priority chunks (SRE-KB) by ~50% which can shuffle
        # the top-K significantly -- without over-fetch, a boosted chunk at
        # rank K+1 would never make the cut.
        fetch_k = top_k * 2 if top_k > 0 else top_k
        result = await self._client.query_points(
            collection_name=self._collection,
            query=embedding,
            using=_DENSE,
            query_filter=qfilter,
            search_params=qm.SearchParams(hnsw_ef=_HNSW_EF),
            limit=fetch_k,
            # Do NOT pass score_threshold to Qdrant: it would filter on the RAW
            # cosine score, dropping a high-priority (SRE-KB / user-correction)
            # chunk whose raw score is just under threshold BEFORE the boost
            # below could lift it over. Apply the threshold post-boost instead.
            with_payload=True,
        )
        # Apply priority boost + re-sort, threshold on the BOOSTED score, trim.
        boosted: list[tuple[float, object]] = []
        for h in result.points:
            mult = _priority_multiplier(getattr(h, "payload", None))
            boosted.append((float(h.score) * mult, h))
        boosted.sort(key=lambda x: -x[0])
        if score_threshold is not None:
            boosted = [b for b in boosted if b[0] >= score_threshold]
        out: list[SearchResult] = []
        for score, h in boosted[:top_k]:
            sr = self._hit_to_result(h)
            out.append(SearchResult(chunk=sr.chunk, score=score, distance_metric=sr.distance_metric))
        return out

    async def delete(self, chunk_ids: list[str]) -> int:
        if not chunk_ids:
            return 0
        await self.ensure_collection()
        point_ids = [_chunk_point_id(cid) for cid in chunk_ids]
        await self._client.delete(
            collection_name=self._collection,
            points_selector=qm.PointIdsList(points=point_ids),
            wait=True,
        )
        return len(point_ids)

    async def delete_by_filter(self, filters: dict) -> int:
        await self.ensure_collection()
        qfilter = self._build_filter(filters)
        if qfilter is None:
            return 0
        await self._client.delete(
            collection_name=self._collection,
            points_selector=qm.FilterSelector(filter=qfilter),
            wait=True,
        )
        return -1  # Qdrant delete_by_filter doesn't return count

    async def list_files(
        self,
        repo: str | None = None,
        path_prefix: str | None = None,
        limit: int = 200,
    ) -> tuple[list[str], int]:
        """Return (paths_capped_at_limit, total_distinct_count) in scope.

        Used for listing-intent queries where vector search can't enumerate
        files structurally. Filters by repo (Qdrant payload index, fast)
        and path prefix (Python-side prefix match on the scrolled payloads).
        """
        await self.ensure_collection()
        filter_conds: list = []
        if repo:
            filter_conds.append(
                qm.FieldCondition(key="repo", match=qm.MatchValue(value=repo))
            )
        qfilter = qm.Filter(must=filter_conds) if filter_conds else None

        seen: set[str] = set()
        offset = None
        for _ in range(50):
            points, offset = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=qfilter,
                limit=1000,
                with_payload=["source_path"],
                with_vectors=False,
                offset=offset,
            )
            for p in points:
                sp = (p.payload or {}).get("source_path", "")
                if not sp:
                    continue
                if path_prefix and not sp.startswith(path_prefix):
                    continue
                seen.add(sp)
            if offset is None:
                break
        total = len(seen)
        return sorted(seen)[:limit], total

    async def get_chunks_by_chunk_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        """Fetch chunks by their stable string chunk_id (not Qdrant UUID)."""
        if not chunk_ids:
            return []
        await self.ensure_collection()
        point_ids = [_chunk_point_id(cid) for cid in chunk_ids]
        points = await self._client.retrieve(
            collection_name=self._collection,
            ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )
        by_chunk_id: dict[str, Chunk] = {}
        for p in points:
            payload = p.payload or {}
            cid = payload.get("chunk_id", str(p.id))
            by_chunk_id[cid] = Chunk(
                id=cid,
                content=payload.get("content", ""),
                doc_type=DocType(payload.get("doc_type", DocType.GENERIC_MARKDOWN.value)),
                source_path=payload.get("source_path", ""),
                repo=payload.get("repo", ""),
                metadata=payload.get("metadata", {}) or {},
                parent_chunk_id=payload.get("parent_chunk_id"),
                chunk_type=payload.get("chunk_type", "child"),
                token_count=payload.get("token_count", 0),
            )
        return [by_chunk_id[cid] for cid in chunk_ids if cid in by_chunk_id]

    async def get_collection_stats(self) -> dict:
        await self.ensure_collection()
        info = await self._client.get_collection(self._collection)
        return {
            "name": self._collection,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": str(info.status),
        }

    async def hybrid_search(
        self,
        embedding: list[float],
        query_text: str,
        top_k: int = 10,
        alpha: float = 0.7,  # ignored -- kept for backwards compat with callers
        filters: dict | None = None,
        graph_anchored_paths: list[tuple[str, str]] | None = None,
        code_embedding: list[float] | None = None,
        code_store: QdrantVectorStore | None = None,
    ) -> list[SearchResult]:
        """Hybrid search via Reciprocal Rank Fusion (BM25 + dense + optional code).

        Industry-standard RRF with k=60 replaces the prior
        linear alpha combination. Equation:

            score(d) = sum over each result-list of: 1 / (k + rank_in_list(d))

        where rank_in_list is 1-indexed. Documents appearing in multiple
        lists accumulate scores from each. Documents in only one list
        still get their single contribution.

        Lanes
        -----
        1. Dense (semantic, Vertex text-embedding-005)
        2. Sparse (BM25)
        3. Code lane (optional): code-specific embedder against the
           `opsrag_code` collection.

        The `alpha` parameter is kept in the signature for backwards
        compatibility (callers that pass it are not broken) but is
        IGNORED -- RRF is parameter-free.

        The `graph_anchored_paths` parameter is also kept for backwards
        compat but is IGNORED -- the entity-anchored Neo4j lane was
        removed (replacement Cartography integration in progress).
        """
        await self.ensure_collection()
        qfilter = self._search_filter(filters)  # excludes parent chunks
        # Fetch a larger candidate pool from each lane; RRF benefits from depth.
        # max(top_k*8, 50) matches the pgvector backend so the two stores fuse
        # over comparable depth and the reranker sees similar pools.
        candidate_k = max(top_k * 8, 50)

        # Dense (semantic) results -- skipped if caller passed zero-vec
        # (keyword_retriever's BM25-only intent, signalled today via alpha=0
        # but here we detect the zero-vec directly so RRF stays parameter-free).
        dense_hits: list = []
        if embedding and any(abs(x) > 1e-9 for x in embedding):
            dense_result = await self._client.query_points(
                collection_name=self._collection,
                query=embedding,
                using=_DENSE,
                query_filter=qfilter,
                search_params=qm.SearchParams(hnsw_ef=_HNSW_EF),
                limit=candidate_k,
                with_payload=True,
            )
            dense_hits = list(dense_result.points)

        # BM25 (lexical) results -- only if query text is non-empty.
        sparse_hits: list = []
        if query_text and query_text.strip():
            try:
                sparse_query = bm25_sparse.encode_query(query_text)
                if sparse_query.indices:
                    sparse_result = await self._client.query_points(
                        collection_name=self._collection,
                        query=sparse_query,
                        using=_BM25,
                        query_filter=qfilter,
                        limit=candidate_k,
                        with_payload=True,
                    )
                    sparse_hits = list(sparse_result.points)
            except Exception as exc:
                _log.warning("bm25 sparse query failed; falling back to dense-only: %s", exc)

        # Graph-anchored Neo4j lane removed. The
        # `graph_anchored_paths` kwarg is still accepted for backwards
        # compat with older callers but does nothing -- entity-extractor
        # wiring is gone and the Neo4j driver is reserved for the new
        # Cartography integration.
        _ = graph_anchored_paths  # silence unused-var lint
        graph_hits: list = []

        # Code lane: when caller passes a code-specific query
        # embedding AND a code vector store (the `opsrag_code` collection
        # populated by ingestion's dual-write), fetch a dense-only lane
        # from that store. Skipped silently when either is None or when
        # the code embedding is zero-vec -- keeps behavior identical when
        # the code lane is unused.
        code_hits: list = []
        if code_embedding and code_store is not None and any(abs(x) > 1e-9 for x in code_embedding):
            try:
                code_qfilter = code_store._search_filter(filters) if hasattr(code_store, "_search_filter") else None
                code_result = await code_store._client.query_points(
                    collection_name=code_store._collection,
                    query=code_embedding,
                    using=_DENSE,
                    query_filter=code_qfilter,
                    search_params=qm.SearchParams(hnsw_ef=_HNSW_EF),
                    limit=candidate_k,
                    with_payload=True,
                )
                code_hits = list(code_result.points)
            except Exception as exc:
                _log.warning("code-collection query failed (%s) -- proceeding without code lane", exc)
                code_hits = []

        # RRF fusion over all active lanes.
        # rrf_score[point_id] = sum(lane_weight / (k + rank))
        # `lane_weight` defaults to 1.0 per lane (vanilla RRF) but
        # is boosted to 1.5 on the sparse lane when the query looks
        # identifier-heavy (snake_case, dotted paths, kebab service
        # names, backticked tokens, route paths). See
        # `_compute_lane_weights` docstring above.
        lane_weights = compute_lane_weights(query_text)
        rrf_score: dict[str, float] = {}
        seen_hits: dict[str, object] = {}
        for hit_list, lane_weight in (
            (dense_hits, lane_weights["dense"]),
            (sparse_hits, lane_weights["sparse"]),
            (graph_hits, lane_weights["graph"]),
            # Code lane is a SEMANTIC lane -- its own gentler boost, not BM25's
            # full identifier weight (which suppressed dense neighbors 3:1).
            (code_hits, lane_weights["code"]),
        ):
            for rank, h in enumerate(hit_list, start=1):
                key = str(h.id)
                rrf_score[key] = rrf_score.get(key, 0.0) + lane_weight / (_RRF_K + rank)
                if key not in seen_hits:
                    seen_hits[key] = h

        # Priority boost -- multiply the fused RRF score by the payload's
        # priority multiplier. SRE-KB chunks land 50% higher in the
        # ranking so canonical answers out-rank Confluence/Slack on
        # overlapping queries. Applied AFTER RRF so the multiplier
        # composes cleanly with both lexical and semantic contributions.
        for key in list(rrf_score.keys()):
            h = seen_hits.get(key)
            mult = _priority_multiplier(getattr(h, "payload", None) if h else None)
            if mult != 1.0:
                rrf_score[key] *= mult

        # Order by boosted RRF score descending
        ranked = sorted(rrf_score.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results: list[SearchResult] = []
        for key, score in ranked:
            h = seen_hits[key]
            sr = self._hit_to_result(h)
            results.append(SearchResult(chunk=sr.chunk, score=float(score), distance_metric="rrf"))
        return results

    @staticmethod
    def _scroll_hit_to_result(point) -> SearchResult:
        """Convert a scroll result (no score) to a SearchResult."""
        payload = point.payload or {}
        chunk = Chunk(
            id=payload.get("chunk_id", str(point.id)),
            content=payload.get("content", ""),
            doc_type=DocType(payload.get("doc_type", DocType.GENERIC_MARKDOWN.value)),
            source_path=payload.get("source_path", ""),
            repo=payload.get("repo", ""),
            metadata=payload.get("metadata", {}) or {},
            parent_chunk_id=payload.get("parent_chunk_id"),
            chunk_type=payload.get("chunk_type", "child"),
            token_count=payload.get("token_count", 0),
        )
        return SearchResult(chunk=chunk, score=0.0, distance_metric="text")

    async def search_by_text(
        self,
        text: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Pure keyword search via Qdrant payload text index. Kept for
        backwards compat -- newer keyword retriever should use hybrid_search
        with empty embedding (dense fallback) or BM25-only path.
        """
        await self.ensure_collection()
        text_filter_conditions = [qm.FieldCondition(key="content", match=qm.MatchText(text=text))]
        qfilter = self._build_filter(filters)
        if qfilter and qfilter.must:
            text_filter_conditions.extend(qfilter.must)
        # Exclude parents (like the main search lanes). Without this the
        # slug/filename fanout reintroduces the parent chunks _search_filter
        # deliberately removes, flooding the rerank pool with parent+child
        # near-duplicates.
        text_filter = qm.Filter(
            must=text_filter_conditions,
            must_not=[qm.FieldCondition(key="chunk_type", match=qm.MatchValue(value="parent"))],
        )
        try:
            text_hits, _offset = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=text_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            return [self._scroll_hit_to_result(h) for h in text_hits]
        except Exception as exc:
            _log.warning("search_by_text failed: %s", exc)
            return []

    async def search_by_path(
        self,
        path_text: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Filename / path / repo substring filter search.

        Matches `path_text` as a substring against EITHER `source_path` OR
        `repo`. The repo arm catches queries that name a repository slug
        (e.g. "acme-tf-state") whose `.tf` files live under paths
        like `modules/<name>/variables.tf` -- those file paths don't contain
        the repo slug, so a source_path-only search misses them entirely.
        """
        await self.ensure_collection()
        # OR group: source_path contains text  OR  repo contains text.
        path_or_repo = qm.Filter(should=[
            qm.FieldCondition(key="source_path", match=qm.MatchText(text=path_text)),
            qm.FieldCondition(key="repo",        match=qm.MatchText(text=path_text)),
        ])
        must_conds: list[Any] = [path_or_repo]
        qfilter = self._build_filter(filters)
        if qfilter and qfilter.must:
            must_conds.extend(qfilter.must)
        # Exclude parents (parity with the main lanes / search_by_text).
        f = qm.Filter(
            must=must_conds,
            must_not=[qm.FieldCondition(key="chunk_type", match=qm.MatchValue(value="parent"))],
        )
        try:
            hits, _offset = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=f,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            return [self._scroll_hit_to_result(h) for h in hits]
        except Exception as exc:
            _log.warning("search_by_path failed: %s", exc)
            return []

    async def find_repo_by_substring(self, needle: str) -> str | None:
        """Return any indexed `repo` value whose name contains `needle`
        (case-insensitive substring match), or None. Used to resolve an
        anchor entity from a user query (e.g. `acme-tf-state`)
        to a concrete indexed repo (`devops/terraform2.0/acme-tf-
        state/terraform`) when the retriever happened to return chunks
        from OTHER repos that merely mention the same string in prose.
        """
        await self.ensure_collection()
        if not needle:
            return None
        f = qm.Filter(must=[
            qm.FieldCondition(key="repo", match=qm.MatchText(text=needle)),
        ])
        try:
            hits, _ = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=f,
                limit=1,
                with_payload=["repo"],
                with_vectors=False,
            )
            if hits:
                return (hits[0].payload or {}).get("repo") or None
        except Exception as exc:
            _log.warning("find_repo_by_substring failed: %s", exc)
        return None

    async def enumerate_paths(
        self,
        repo: str,
        path_prefix: str | None = None,
        max_paths: int = 5000,
    ) -> list[str]:
        """Enumerate all distinct `source_path` values for a given repo,
        optionally restricted to paths whose `source_path` contains
        `path_prefix` as a substring.

        Returns up to `max_paths` distinct paths. Used by the directory-
        tree summarizer to enumerate the COMPLETE set of subdirs under a
        repo (not just the 10 chunks the retriever happened to pick).

        Implementation: scrolls all matching points with only
        `source_path` projected, dedupes, and stops early once
        `max_paths` distinct values are seen. Each scroll page returns
        up to 1024 points.
        """
        await self.ensure_collection()
        must_conds: list[Any] = [
            qm.FieldCondition(key="repo", match=qm.MatchValue(value=repo)),
        ]
        if path_prefix:
            must_conds.append(
                qm.FieldCondition(key="source_path", match=qm.MatchText(text=path_prefix))
            )
        f = qm.Filter(must=must_conds)
        seen: set[str] = set()
        offset: Any = None
        page_size = 1024
        try:
            while len(seen) < max_paths:
                hits, next_offset = await self._client.scroll(
                    collection_name=self._collection,
                    scroll_filter=f,
                    limit=page_size,
                    offset=offset,
                    with_payload=["source_path"],
                    with_vectors=False,
                )
                if not hits:
                    break
                for h in hits:
                    sp = (h.payload or {}).get("source_path") or ""
                    if sp:
                        seen.add(sp)
                        if len(seen) >= max_paths:
                            break
                if next_offset is None:
                    break
                offset = next_offset
        except Exception as exc:
            _log.warning("enumerate_paths failed: %s", exc)
        return sorted(seen)

    @staticmethod
    def _hit_to_result(point) -> SearchResult:
        payload = point.payload or {}
        metadata = payload.get("metadata", {}) or {}
        # Carry the priority tag onto the chunk so the agent's fanout RRF
        # re-merge can RE-APPLY the authoritative-content boost. rrf_merge_pools
        # re-derives scores from rank and discards the boosted score this store
        # computed, so without this the SRE-KB / architecture / user-correction
        # boost silently vanished on any query that fired a slug/filename fanout.
        if payload.get("priority") is not None and "priority" not in metadata:
            metadata = {**metadata, "priority": payload.get("priority")}
        chunk = Chunk(
            id=payload.get("chunk_id", str(point.id)),
            content=payload.get("content", ""),
            doc_type=DocType(payload.get("doc_type", DocType.GENERIC_MARKDOWN.value)),
            source_path=payload.get("source_path", ""),
            repo=payload.get("repo", ""),
            metadata=metadata,
            parent_chunk_id=payload.get("parent_chunk_id"),
            chunk_type=payload.get("chunk_type", "child"),
            token_count=payload.get("token_count", 0),
        )
        return SearchResult(chunk=chunk, score=float(getattr(point, "score", 0.0) or 0.0), distance_metric="cosine")

    @staticmethod
    def _build_filter(filters: dict | None) -> qm.Filter | None:
        if not filters:
            return None
        conds: list = []
        for key, value in filters.items():
            if isinstance(value, list):
                conds.append(qm.FieldCondition(key=key, match=qm.MatchAny(any=value)))
            else:
                conds.append(qm.FieldCondition(key=key, match=qm.MatchValue(value=value)))
        return qm.Filter(must=conds) if conds else None

    @classmethod
    def _search_filter(cls, filters: dict | None) -> qm.Filter | None:
        """Like _build_filter but EXCLUDES parent chunks from the search lanes.

        Parents and their children are both indexed with dense+sparse vectors;
        without this, a parent and its overlapping children co-occur in top-K,
        waste candidate slots, and inflate BM25/RRF stats (the same phrase
        indexed 2-5x). The generator fetches parents BY ID for context
        (get_chunks_by_chunk_ids / retrieve), which bypasses this filter, so
        parent-substitution is unaffected -- search returns children, the LLM
        gets parents. (Every parent has >=1 child covering its content, so no
        content becomes unreachable.)"""
        base = cls._build_filter(filters)
        must = list(base.must) if base and base.must else None
        return qm.Filter(
            must=must,
            must_not=[qm.FieldCondition(key="chunk_type", match=qm.MatchValue(value="parent"))],
        )
