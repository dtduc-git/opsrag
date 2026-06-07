"""pgvector vector store -- PostgreSQL with the pgvector extension.

Uses asyncpg for async access. Suitable for deployments already running
PostgreSQL (CloudSQL, AlloyDB, RDS) where adding Qdrant is undesirable.

Requires: asyncpg, pgvector (pip install asyncpg pgvector)
"""
from __future__ import annotations

import json
import uuid

import asyncpg

from opsrag.interfaces.chunker import Chunk
from opsrag.interfaces.parser import DocType
from opsrag.interfaces.vectorstore import SearchResult
from opsrag.vectorstores.lane_weights import compute_lane_weights
from opsrag.vectorstores.priority import chunk_priority, priority_multiplier
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
    embedding vector({dim})
);
"""

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
            for idx in _CREATE_INDEXES:
                await conn.execute(idx)
            try:
                await conn.execute(_CREATE_VECTOR_INDEX)
            except Exception:
                pass  # best-effort; HNSW builds on an empty table but tolerate
                      # races / older pgvector without HNSW support
        self._ensured = True

    async def upsert(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> int:
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
                str(v),  # pgvector accepts text representation
            )
            for c, v in zip(chunks, embeddings)
        ]

        async with self._pool.acquire() as conn:
            await conn.executemany(
                f"""
                INSERT INTO {self._table}
                    (id, chunk_id, content, doc_type, source_path, repo,
                     parent_chunk_id, chunk_type, token_count, metadata, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11::vector)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    doc_type = EXCLUDED.doc_type,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                records,
            )
        return len(records)

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
               parent_chunk_id, chunk_type, token_count, metadata,
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
                       parent_chunk_id, chunk_type, token_count, metadata
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
        """Two-lane hybrid with RRF fusion -- parity with the Qdrant path.

        The previous single-query `ORDER BY (alpha*cosine + (1-alpha)*ts_rank)`
        ordered on a COMPUTED column, so Postgres could not use the HNSW index
        and scanned the whole table per query (an O(N) recall/latency cliff at
        scale). Instead we run TWO index-using queries -- a dense ANN lane
        (`ORDER BY embedding <=> $1`, HNSW) and a lexical FTS lane (GIN) -- and
        fuse them with Reciprocal Rank Fusion, the same algorithm the Qdrant
        store uses. The authoritative-content priority boost (SRE-KB /
        architecture tiers) is then applied so it takes effect on pgvector too.
        """
        await self.ensure_table()
        candidate_k = max(top_k * 8, 50)

        # Lane 1 -- dense ANN. ORDER BY the raw distance (NOT a blended score)
        # so the HNSW index is actually used.
        dwhere, dparams = self._build_where(filters, start_idx=2)
        dense_sql = f"""
        SELECT chunk_id, content, doc_type, source_path, repo,
               parent_chunk_id, chunk_type, token_count, metadata,
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
               parent_chunk_id, chunk_type, token_count, metadata,
               ts_rank(to_tsvector('simple', content),
                       websearch_to_tsquery('simple', $1), 32) AS score
        FROM {self._table}
        WHERE to_tsvector('simple', content)
              @@ websearch_to_tsquery('simple', $1)
          AND chunk_type IS DISTINCT FROM 'parent'{lwhere}
        ORDER BY score DESC
        LIMIT {candidate_k}
        """

        async with self._pool.acquire() as conn:
            await conn.execute(f"SET hnsw.ef_search = {_HNSW_EF_SEARCH}")
            dense_rows = await conn.fetch(dense_sql, str(embedding), *dparams)
            lex_rows = await conn.fetch(lex_sql, query_text, *lparams)

        dense_results = [self._row_to_result(r) for r in dense_rows]
        lex_results = [self._row_to_result(r) for r in lex_rows]

        # Identifier-aware lane weighting (parity with Qdrant): boost the lexical
        # lane on symbol queries (handle_webhook_callback, acme-notes-be-api) so
        # pgvector doesn't retrieve measurably worse than Qdrant on exact-symbol
        # queries. dense=1.0, lexical=sparse weight.
        lw = compute_lane_weights(query_text)
        fused = rrf_merge_pools(
            [dense_results, lex_results],
            top_k=max(candidate_k, top_k),
            pool_weights=[lw["dense"], lw["sparse"]],
        )
        # Authoritative-content priority boost (parity with Qdrant). pgvector
        # has no `priority` column, so derive the tier from repo/source_path.
        boosted: list[SearchResult] = []
        for sr in fused:
            mult = priority_multiplier(chunk_priority(sr.chunk.repo, sr.chunk.source_path))
            boosted.append(
                SearchResult(chunk=sr.chunk, score=sr.score * mult,
                             distance_metric="rrf+priority")
            )
        boosted.sort(key=lambda s: s.score, reverse=True)
        return boosted[:top_k]

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
