"""pgvector vector store -- PostgreSQL with the pgvector extension.

Uses asyncpg for async access. Suitable for deployments already running
PostgreSQL (CloudSQL, AlloyDB, RDS) where adding Qdrant is undesirable.

Requires: asyncpg, pgvector (pip install asyncpg pgvector)
"""
from __future__ import annotations

import json
import logging
import uuid

import asyncpg

from opsrag.interfaces.chunker import Chunk
from opsrag.interfaces.parser import DocType
from opsrag.interfaces.vectorstore import SearchResult
from opsrag.vectorstores.lane_weights import compute_lane_weights, extract_identifiers
from opsrag.vectorstores.priority import chunk_priority, priority_rrf_bonus
from opsrag.vectorstores.rrf import rrf_merge_pools

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS opsrag_chunks (
    id UUID PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    content TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    source_path TEXT NOT NULL,
    repo TEXT NOT NULL,
    parent_chunk_id TEXT,
    chunk_type TEXT NOT NULL DEFAULT 'child',
    token_count INT NOT NULL DEFAULT 0,
    metadata JSONB NOT NULL DEFAULT '{}',
    priority TEXT,
    embedding vector({dim})
);
"""

# Bring pre-priority tables up to schema (CREATE TABLE IF NOT EXISTS won't add a
# column to an existing table). Idempotent.
_ALTER_ADD_PRIORITY = (
    "ALTER TABLE opsrag_chunks ADD COLUMN IF NOT EXISTS priority TEXT"
)

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_chunks_repo ON opsrag_chunks (repo)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_source ON opsrag_chunks (source_path)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_doc_type ON opsrag_chunks (doc_type)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id ON opsrag_chunks (chunk_id)",
    # GIN full-text index over content -- powers the real lexical lane in
    # hybrid_search (replaces the old binary ILIKE substring match).
    "CREATE INDEX IF NOT EXISTS idx_chunks_content_fts "
    "ON opsrag_chunks USING gin (to_tsvector('simple', content))",
]

# HNSW, not IVFFlat: IVFFlat must be built AFTER the table has rows (and needs
# `ivfflat.probes` tuned, else it scans a single list -> severe recall loss),
# and the old code built it on an empty table inside a swallowed try/except so
# it silently never existed. HNSW builds fine on an empty table, grows
# incrementally, and gives better recall; tune recall at query time via
# `hnsw.ef_search`.
# Explicit build params (pgvector defaults m=16, ef_construction=64 are low for
# a code/identifier corpus -- bake in higher recall at index time).
_CREATE_VECTOR_INDEX = """
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
ON opsrag_chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);
"""

# Per-session recall knob for HNSW search (higher = better recall, slower).
# Matches the Qdrant store's hnsw_ef so the two backends have comparable recall.
_HNSW_EF_SEARCH = 192

_log = logging.getLogger("opsrag.vectorstores.pgvector")


def _deterministic_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class PgVectorStore:
    def __init__(
        self,
        dsn: str,
        dimension: int = 3072,
        table: str = "opsrag_chunks",
        min_pool: int = 2,
        max_pool: int = 10,
        allow_dimension_change: bool = False,
    ):
        self._dsn = dsn
        self._dimension = dimension
        self._table = table
        self._min_pool = min_pool
        self._max_pool = max_pool
        self._allow_dimension_change = allow_dimension_change
        self._pool: asyncpg.Pool | None = None
        self._ensured = False
        # Set by ensure_table: True if pg_trgm + the trgm index are available,
        # gating the substring identifier lane in hybrid_search.
        self._trgm_available = False

    async def _assert_dimension_compatible(self, conn) -> None:
        """Fail closed if an existing table's vector dimension differs from the
        embedder's. Parity with the Qdrant dimension guard, which pgvector
        previously lacked entirely -- a silent embedder swap would otherwise
        fail deep in an upsert with a cryptic pgvector error (or worse, on a
        fresh CREATE the wrong dim would bake in). No-op on a missing table."""
        existing = await conn.fetchval(
            "SELECT format_type(a.atttypid, a.atttypmod) "
            "FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
            "WHERE c.relname = $1 AND a.attname = 'embedding' "
            "AND NOT a.attisdropped",
            self._table,
        )
        if not existing:
            return  # table (or column) absent -- first boot, will be created
        # format_type returns e.g. 'vector(3072)'.
        try:
            current = int(existing.split("(")[1].rstrip(")"))
        except (IndexError, ValueError):
            return  # unparseable (untyped vector) -- nothing to compare
        if current != self._dimension and not self._allow_dimension_change:
            raise RuntimeError(
                f"pgvector dimension mismatch: table {self._table!r} is "
                f"vector({current}) but the embedder produces {self._dimension}-d "
                f"vectors. Re-index into a fresh table or set "
                f"allow_dimension_change=true to override."
            )

    async def open(self) -> None:
        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._min_pool,
            max_size=self._max_pool,
        )

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    async def ensure_table(self) -> None:
        if self._ensured or not self._pool:
            return
        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await self._assert_dimension_compatible(conn)
            await conn.execute(_CREATE_TABLE.format(dim=self._dimension))
            await conn.execute(_ALTER_ADD_PRIORITY)
            for idx in _CREATE_INDEXES:
                await conn.execute(idx)
            try:
                await conn.execute(_CREATE_VECTOR_INDEX)
            except Exception:
                pass  # best-effort; HNSW builds on an empty table but tolerate
                      # races / older pgvector without HNSW support
            # Best-effort trigram lane: pg_trgm + a GIN trgm index make the
            # substring identifier lane (in hybrid_search) fast. Skipped silently
            # if the extension isn't available (managed PG without it / no
            # privilege) -- the lane then just doesn't fire, no error.
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chunks_content_trgm "
                    "ON opsrag_chunks USING gin (content gin_trgm_ops)"
                )
                self._trgm_available = True
            except Exception as exc:
                # Loud, not silent: without pg_trgm the substring identifier lane
                # can't fire, so exact-symbol recall quietly collapses to
                # dense-only -- an env-dependent cliff on managed Postgres where
                # CREATE EXTENSION isn't grantable. Surface it so operators know
                # symbol queries are degraded rather than discovering it via
                # missing results.
                self._trgm_available = False
                _log.warning(
                    "pg_trgm unavailable (%s) -- the trigram identifier lane is "
                    "DISABLED; exact-symbol recall falls back to FTS + dense only. "
                    "Grant `CREATE EXTENSION pg_trgm` for full symbol parity.",
                    exc,
                )
        self._ensured = True

    async def upsert(
        self, chunks: list[Chunk], embeddings: list[list[float] | None]
    ) -> int:
        # `embeddings[i]` may be None for non-searchable chunks (parents) -> the
        # embedding column is stored NULL (search filters embedding IS NOT NULL).
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        await self.ensure_table()

        records = [
            (
                _deterministic_uuid(c.id),
                c.id,
                c.content,
                c.doc_type.value,
                c.source_path,
                c.repo,
                c.parent_chunk_id,
                c.chunk_type,
                c.token_count,
                json.dumps(c.metadata),
                self._priority_for(c),
                # `v is None` -> caller skipped the dense embedding (parents are
                # stored for parent-substitution but never searched, so the
                # column is left NULL; search/HNSW already filter on
                # `embedding IS NOT NULL`).
                str(v) if v is not None else None,
            )
            for c, v in zip(chunks, embeddings)
        ]

        async with self._pool.acquire() as conn:
            await conn.executemany(
                f"""
                INSERT INTO {self._table}
                    (id, chunk_id, content, doc_type, source_path, repo,
                     parent_chunk_id, chunk_type, token_count, metadata,
                     priority, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb,
                        $11, $12::vector)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    doc_type = EXCLUDED.doc_type,
                    metadata = EXCLUDED.metadata,
                    priority = EXCLUDED.priority,
                    embedding = EXCLUDED.embedding
                """,
                records,
            )
        return len(records)

    @staticmethod
    def _priority_for(c: Chunk) -> str | None:
        """Priority tier stamped at upsert (parity with the Qdrant payload tag).

        A chunk may carry an explicit ``priority`` in its metadata (e.g. an
        operator-approved ``user-correction``); honor that first. Otherwise
        derive the SRE-KB / architecture tier from repo + source_path.
        """
        if isinstance(c.metadata, dict):
            tag = c.metadata.get("priority")
            if tag:
                return str(tag)
        return chunk_priority(c.repo, c.source_path)

    async def search(
        self,
        embedding: list[float],
        top_k: int = 10,
        filters: dict | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        await self.ensure_table()
        where, params = self._build_where(filters, start_idx=2)
        if score_threshold is not None:
            where += f" AND 1 - (embedding <=> $1::vector) >= ${len(params) + 2}"
            params.append(score_threshold)

        query = f"""
        SELECT chunk_id, content, doc_type, source_path, repo,
               parent_chunk_id, chunk_type, token_count, metadata, priority,
               1 - (embedding <=> $1::vector) AS score
        FROM {self._table}
        WHERE embedding IS NOT NULL
          AND chunk_type IS DISTINCT FROM 'parent'{where}
        ORDER BY embedding <=> $1::vector
        LIMIT ${len(params) + 2}
        """
        params.append(top_k)

        async with self._pool.acquire() as conn:
            await conn.execute(f"SET hnsw.ef_search = {_HNSW_EF_SEARCH}")
            rows = await conn.fetch(query, str(embedding), *params)

        return [self._row_to_result(r) for r in rows]

    async def get_chunks_by_chunk_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        """Fetch full chunks by stable chunk_id. Without this the generator's
        parent-substitution (children retrieved -> parents fed to the LLM)
        silently no-ops on pgvector, handing the LLM 256-tok child slices
        instead of 1024-tok parents -- defeating the parent-child design."""
        if not chunk_ids:
            return []
        await self.ensure_table()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT chunk_id, content, doc_type, source_path, repo,
                       parent_chunk_id, chunk_type, token_count, metadata, priority
                FROM {self._table}
                WHERE chunk_id = ANY($1)
                """,
                chunk_ids,
            )
        return [self._row_to_chunk(r) for r in rows]

    async def delete(self, chunk_ids: list[str]) -> int:
        if not chunk_ids:
            return 0
        await self.ensure_table()
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {self._table} WHERE chunk_id = ANY($1)",
                chunk_ids,
            )
            return int(result.split()[-1])

    async def delete_by_filter(self, filters: dict) -> int:
        await self.ensure_table()
        where, params = self._build_where(filters, start_idx=1)
        if not where:
            return 0
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {self._table} WHERE true{where}",
                *params,
            )
            return int(result.split()[-1])

    async def get_collection_stats(self) -> dict:
        await self.ensure_table()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT count(*) AS cnt FROM {self._table}"
            )
            return {
                "name": self._table,
                "points_count": row["cnt"] if row else 0,
            }

    async def hybrid_search(
        self,
        embedding: list[float],
        query_text: str,
        top_k: int = 10,
        alpha: float = 0.7,  # kept for interface compat; RRF replaces the blend
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Hybrid RRF fusion. Lanes: dense ANN (HNSW), lexical FTS (GIN,
        ts_rank_cd), and -- when the query has identifiers + pg_trgm is present --
        a trigram substring lane.

        The previous single-query `ORDER BY (alpha*cosine + (1-alpha)*ts_rank)`
        ordered on a COMPUTED column, so Postgres could not use the HNSW index
        and scanned the whole table per query. Each lane now runs as its own
        index-using query and they're fused with the same RRF the Qdrant store
        uses; the authoritative-content priority boost is then applied.

        PARITY CAVEAT -- this is NOT full Qdrant parity:
          * No CODE lane. The retriever's 4th lane (code-specific embedder over
            the code collection) is Qdrant-only; the retriever feature-detects
            and does NOT pass code_embedding/code_store here. Symbol *recall* is
            partly recovered by the trgm lane, but code-semantic ranking is absent.
          * Sparse math differs. Qdrant = subword BM25 (true IDF); here = ts_rank_cd
            over `simple` FTS + a trgm substring lane. This recovers exact-symbol
            recall, not BM25-ranked lexical relevance.
          * The trgm lane is best-effort and silently absent without pg_trgm (a
            WARNING is logged at ensure_table time).
        Treat pgvector as "strong dense + recall of exact symbols", not a
        drop-in lexical-ranking equivalent of Qdrant.
        """
        await self.ensure_table()
        candidate_k = max(top_k * 8, 50)

        # Lane 1 -- dense ANN. ORDER BY the raw distance (NOT a blended score)
        # so the HNSW index is actually used.
        dwhere, dparams = self._build_where(filters, start_idx=2)
        dense_sql = f"""
        SELECT chunk_id, content, doc_type, source_path, repo,
               parent_chunk_id, chunk_type, token_count, metadata, priority,
               1 - (embedding <=> $1::vector) AS score
        FROM {self._table}
        WHERE embedding IS NOT NULL
          AND chunk_type IS DISTINCT FROM 'parent'{dwhere}
        ORDER BY embedding <=> $1::vector
        LIMIT {candidate_k}
        """
        # Lane 2 -- lexical FTS over the GIN index. websearch_to_tsquery
        # tokenizes/stems the query (handles multi-word, phrases, negation),
        # unlike the old whole-query ILIKE substring that never fired.
        lwhere, lparams = self._build_where(filters, start_idx=2)
        lex_sql = f"""
        SELECT chunk_id, content, doc_type, source_path, repo,
               parent_chunk_id, chunk_type, token_count, metadata, priority,
               ts_rank_cd(to_tsvector('simple', content),
                          websearch_to_tsquery('simple', $1), 32) AS score
        FROM {self._table}
        WHERE to_tsvector('simple', content)
              @@ websearch_to_tsquery('simple', $1)
          AND chunk_type IS DISTINCT FROM 'parent'{lwhere}
        ORDER BY score DESC
        LIMIT {candidate_k}
        """

        # Lane 3 (optional, identifier queries only) -- trigram substring lane.
        # The 'simple' FTS lexer keys on whole tokens, so a partial / camelCase /
        # dotted identifier query can @@-match ZERO rows even when the symbol is
        # in the corpus -- the exact case the Qdrant subword-BM25 lane handles
        # and pgvector previously could not. A pg_trgm GIN index makes an ILIKE
        # substring per identifier fast, recovering that exact-symbol recall.
        # Fires only when the query has identifier tokens AND pg_trgm is present,
        # so prose queries are byte-identical to the two-lane path.
        idents = extract_identifiers(query_text) if self._trgm_available else []
        trgm_sql = ""
        trgm_params: list = []
        if idents:
            iwhere, iparams = self._build_where(filters, start_idx=len(idents) + 1)
            ilike_clauses = " OR ".join(
                f"content ILIKE ${i + 1} ESCAPE '\\'" for i in range(len(idents))
            )
            trgm_params = [f"%{self._ilike_escape(t)}%" for t in idents]
            trgm_sql = f"""
            SELECT chunk_id, content, doc_type, source_path, repo,
                   parent_chunk_id, chunk_type, token_count, metadata, priority,
                   1.0 AS score
            FROM {self._table}
            WHERE ({ilike_clauses})
              AND chunk_type IS DISTINCT FROM 'parent'{iwhere}
            LIMIT {candidate_k}
            """

        async with self._pool.acquire() as conn:
            await conn.execute(f"SET hnsw.ef_search = {_HNSW_EF_SEARCH}")
            dense_rows = await conn.fetch(dense_sql, str(embedding), *dparams)
            lex_rows = await conn.fetch(lex_sql, query_text, *lparams)
            trgm_rows = []
            if trgm_sql:
                try:
                    trgm_rows = await conn.fetch(trgm_sql, *trgm_params, *iparams)
                except Exception:
                    trgm_rows = []  # best-effort -- degrade to the two-lane path

        dense_results = [self._row_to_result(r) for r in dense_rows]
        lex_results = [self._row_to_result(r) for r in lex_rows]
        trgm_results = [self._row_to_result(r) for r in trgm_rows]

        # Identifier-aware lane weighting (parity with Qdrant): boost the lexical
        # lane on symbol queries (handle_webhook_callback, acme-notes-be-api) so
        # pgvector doesn't retrieve measurably worse than Qdrant on exact-symbol
        # queries. dense=1.0, lexical=sparse weight; the trgm lane (exact symbol
        # substring) rides the same sparse weight.
        lw = compute_lane_weights(query_text)
        pools = [dense_results, lex_results]
        pool_weights = [lw["dense"], lw["sparse"]]
        if trgm_results:
            pools.append(trgm_results)
            pool_weights.append(lw["sparse"])
        fused = rrf_merge_pools(
            pools,
            top_k=max(candidate_k, top_k),
            pool_weights=pool_weights,
        )
        # Authoritative-content priority boost (parity with Qdrant). Prefer the
        # tag STORED at upsert (carried into chunk.metadata by _row_to_chunk) so
        # user-correction / explicit tiers are honored; fall back to deriving
        # from repo/source_path for rows indexed before the priority column.
        # ADDITIVE in RRF units (not a multiplier): rrf_merge_pools scores live
        # in the same compressed ~0.01-0.016 band as Qdrant's, where a x2.0 would
        # vault a weakly-ranked SRE-KB chunk past a genuine #1. See priority.py.
        boosted: list[SearchResult] = []
        for sr in fused:
            tag = (sr.chunk.metadata or {}).get("priority") or chunk_priority(
                sr.chunk.repo, sr.chunk.source_path
            )
            boosted.append(
                SearchResult(chunk=sr.chunk, score=sr.score + priority_rrf_bonus(tag),
                             distance_metric="rrf+priority")
            )
        boosted.sort(key=lambda s: s.score, reverse=True)
        return boosted[:top_k]

    # ---------------- Fanout / listing parity with the Qdrant store ----------
    # The retriever feature-detects these via hasattr(); without them pgvector
    # silently loses cross-repo slug fanout, filename fanout, repo-slug
    # resolution, directory enumeration, and listing-intent answers -- a real
    # capability gap vs Qdrant, not just a perf difference.

    @staticmethod
    def _ilike_escape(s: str) -> str:
        """Escape LIKE/ILIKE wildcards so user text matches literally."""
        return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    async def search_by_text(
        self, text: str, top_k: int = 10, filters: dict | None = None,
    ) -> list[SearchResult]:
        """Lexical keyword search (parity with Qdrant.search_by_text). Surfaces
        chunks mentioning a term/slug across repos. FTS over the GIN index;
        parents excluded."""
        await self.ensure_table()
        if not text:
            return []
        where, params = self._build_where(filters, start_idx=2)
        sql = f"""
        SELECT chunk_id, content, doc_type, source_path, repo,
               parent_chunk_id, chunk_type, token_count, metadata, priority,
               ts_rank(to_tsvector('simple', content),
                       websearch_to_tsquery('simple', $1), 32) AS score
        FROM {self._table}
        WHERE to_tsvector('simple', content)
              @@ websearch_to_tsquery('simple', $1)
          AND chunk_type IS DISTINCT FROM 'parent'{where}
        ORDER BY score DESC
        LIMIT {int(top_k)}
        """
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, text, *params)
            return [self._row_to_result(r) for r in rows]
        except Exception:
            return []

    async def search_by_path(
        self, path_text: str, top_k: int = 10, filters: dict | None = None,
    ) -> list[SearchResult]:
        """Filename / path / repo substring search (parity with Qdrant). Matches
        `path_text` against source_path OR repo -- the repo arm catches slugs
        whose files live under paths that don't contain the slug."""
        await self.ensure_table()
        if not path_text:
            return []
        pat = f"%{self._ilike_escape(path_text)}%"
        where, params = self._build_where(filters, start_idx=2)
        sql = f"""
        SELECT chunk_id, content, doc_type, source_path, repo,
               parent_chunk_id, chunk_type, token_count, metadata, priority,
               1.0 AS score
        FROM {self._table}
        WHERE (source_path ILIKE $1 ESCAPE '\\' OR repo ILIKE $1 ESCAPE '\\')
          AND chunk_type IS DISTINCT FROM 'parent'{where}
        LIMIT {int(top_k)}
        """
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, pat, *params)
            return [self._row_to_result(r) for r in rows]
        except Exception:
            return []

    async def find_repo_by_substring(self, needle: str) -> str | None:
        """Resolve an anchor entity to a concrete indexed repo whose name
        contains `needle` (parity with Qdrant). None when nothing matches."""
        if not needle:
            return None
        await self.ensure_table()
        pat = f"%{self._ilike_escape(needle)}%"
        try:
            async with self._pool.acquire() as conn:
                return await conn.fetchval(
                    f"SELECT repo FROM {self._table} "
                    f"WHERE repo ILIKE $1 ESCAPE '\\' LIMIT 1",
                    pat,
                )
        except Exception:
            return None

    async def enumerate_paths(
        self, repo: str, path_prefix: str | None = None, max_paths: int = 5000,
    ) -> list[str]:
        """Distinct source_path values for a repo (parity with Qdrant), used by
        the directory-tree summarizer. `path_prefix` is a substring filter."""
        await self.ensure_table()
        params: list = [repo]
        extra = ""
        if path_prefix:
            params.append(f"%{self._ilike_escape(path_prefix)}%")
            extra = " AND source_path ILIKE $2 ESCAPE '\\'"
        sql = (
            f"SELECT DISTINCT source_path FROM {self._table} "
            f"WHERE repo = $1{extra} LIMIT {int(max_paths)}"
        )
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)
            return [r["source_path"] for r in rows if r["source_path"]]
        except Exception:
            return []

    async def list_files(
        self,
        repo: str | None = None,
        path_prefix: str | None = None,
        limit: int = 200,
    ) -> tuple[list[str], int]:
        """(paths_capped_at_limit, total_distinct_count) in scope (parity with
        Qdrant.list_files). `path_prefix` is a PREFIX match (like Qdrant's)."""
        await self.ensure_table()
        clauses: list[str] = []
        params: list = []
        idx = 1
        if repo:
            clauses.append(f"repo = ${idx}")
            params.append(repo)
            idx += 1
        if path_prefix:
            clauses.append(f"source_path LIKE ${idx} ESCAPE '\\'")
            params.append(f"{self._ilike_escape(path_prefix)}%")
            idx += 1
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        try:
            async with self._pool.acquire() as conn:
                total = await conn.fetchval(
                    f"SELECT count(DISTINCT source_path) FROM {self._table}{where}",
                    *params,
                )
                rows = await conn.fetch(
                    f"SELECT DISTINCT source_path FROM {self._table}{where} "
                    f"ORDER BY source_path LIMIT {int(limit)}",
                    *params,
                )
            return [r["source_path"] for r in rows if r["source_path"]], int(total or 0)
        except Exception:
            return [], 0

    @staticmethod
    def _build_where(
        filters: dict | None, start_idx: int = 1
    ) -> tuple[str, list]:
        if not filters:
            return "", []
        clauses: list[str] = []
        params: list = []
        idx = start_idx
        for key, value in filters.items():
            if isinstance(value, (list, tuple, set)):
                clauses.append(f" AND {key} = ANY(${idx})")
                params.append(list(value))
            else:
                clauses.append(f" AND {key} = ${idx}")
                params.append(value)
            idx += 1
        return "".join(clauses), params

    @staticmethod
    def _row_to_chunk(row) -> Chunk:
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        metadata = metadata or {}
        # Carry the stored priority tag onto the chunk (parity with Qdrant), so
        # the fanout RRF re-merge in the agent can re-apply the boost too. Only
        # when the SELECT projected it and it isn't already in metadata.
        # asyncpg Record (and the test fakes) support .get().
        row_priority = row.get("priority") if hasattr(row, "get") else None
        if row_priority and "priority" not in metadata:
            metadata = {**metadata, "priority": row_priority}
        return Chunk(
            id=row["chunk_id"],
            content=row["content"],
            doc_type=DocType(row["doc_type"]),
            source_path=row["source_path"],
            repo=row["repo"],
            metadata=metadata or {},
            parent_chunk_id=row["parent_chunk_id"],
            chunk_type=row["chunk_type"],
            token_count=row["token_count"],
        )

    @classmethod
    def _row_to_result(cls, row) -> SearchResult:
        return SearchResult(
            chunk=cls._row_to_chunk(row),
            score=float(row["score"]),
            distance_metric="cosine",
        )
