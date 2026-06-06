"""RunbookStore -- CRUD + hybrid retrieval over `opsrag_runbooks` table.

Storage layout (split-store, same pattern as investigation_cache):
  - Postgres  -> metadata + body + tsvector (`opsrag_runbooks`)
  - Postgres  -> version history (`opsrag_runbook_versions`)
  - Qdrant    -> embedding vectors (`opsrag_runbooks_vec` collection)

Why split: Zalando's spilo Postgres image doesn't ship pgvector. Qdrant
already runs in-cluster, has the right cosine semantics, and is the
same vector backend the investigation_cache uses.

Retrieval is hybrid:
  1. Embed the query, pull top-K from Qdrant (cosine).
  2. Parallel: tsv_rank on the query against the `tsv` GENERATED column
     (Postgres BM25-like).
  3. Merge by reciprocal-rank fusion (RRF) into a unified score.
  4. Apply hard filters (`service`, `issue_kind`) when supplied.
  5. Boost by `priority` column (hand-authored dial; default 100).

Hand-authored vs RAG ordering is enforced ONE LAYER UP, in Lane A's
retrieval node, not here -- this store only returns hand-authored hits.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from psycopg_pool import AsyncConnectionPool
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from opsrag.runbooks.models import (
    Runbook,
    RunbookCreate,
    RunbookHit,
    RunbookUpdate,
    RunbookVersion,
)

_log = logging.getLogger("opsrag.runbooks.store")

DEFAULT_VECTOR_COLLECTION = "opsrag_runbooks_vec"
DEFAULT_VECTOR_SIZE = 768  # text-embedding-005, same as investigation_cache


class RunbookStore:
    """All Postgres-side CRUD + the Qdrant-side embedding dual-write.

    Construction is light; `_ensure_qdrant_collection()` is called
    lazily on first write so a cluster without Qdrant connectivity
    doesn't crash the API.

    Thread-safety: the underlying pools are async-safe; this class
    owns no mutable state of its own.
    """

    def __init__(
        self,
        *,
        pg_pool: AsyncConnectionPool,
        qdrant: AsyncQdrantClient | None = None,
        embedder: Any = None,
        vector_collection: str = DEFAULT_VECTOR_COLLECTION,
        vector_size: int = DEFAULT_VECTOR_SIZE,
    ) -> None:
        self._pg = pg_pool
        self._qdrant = qdrant
        self._embedder = embedder
        self._collection = vector_collection
        self._vector_size = vector_size
        self._qdrant_ensured = False

    # -- lifecycle -------------------------------------------------

    async def _ensure_qdrant_collection(self) -> None:
        """Idempotent: create the Qdrant collection if missing.

        Called at first write -- by then the Qdrant server is up. We do
        NOT call this in `__init__` so a stale qdrant URL doesn't
        crash boot.
        """
        if self._qdrant_ensured or self._qdrant is None:
            return
        try:
            collections = await self._qdrant.get_collections()
            existing = {c.name for c in collections.collections}
            if self._collection not in existing:
                await self._qdrant.create_collection(
                    collection_name=self._collection,
                    vectors_config=qm.VectorParams(
                        size=self._vector_size,
                        distance=qm.Distance.COSINE,
                    ),
                )
                _log.info(
                    "runbooks: created Qdrant collection %r (size=%d)",
                    self._collection, self._vector_size,
                )
            self._qdrant_ensured = True
        except Exception as exc:
            _log.warning(
                "runbooks: ensure Qdrant collection failed (will retry "
                "next write): %s", exc,
            )

    # -- CRUD ------------------------------------------------------

    async def create(
        self,
        body: RunbookCreate,
        *,
        author_email: str | None = None,
        source: str = "hand",
        source_investigation_id: str | None = None,
    ) -> Runbook:
        """Insert a new runbook row + write version 1 + embed to Qdrant.

        The trigger keeps `updated_at` accurate so we don't pass it.
        """
        new_id = str(uuid.uuid4())
        embedding = await self._embed(body.title, body.body_markdown)

        async with self._pg.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        INSERT INTO opsrag_runbooks (
                          id, title, body_markdown, service, issue_kind,
                          severity_min, priority, tags, source,
                          author_email, source_investigation_id, enabled
                        ) VALUES (
                          %s, %s, %s, %s, %s,
                          %s, %s, %s, %s,
                          %s, %s, TRUE
                        )
                        RETURNING id, created_at, updated_at
                        """,
                        (
                            new_id, body.title, body.body_markdown,
                            body.service, body.issue_kind,
                            body.severity_min, body.priority, list(body.tags),
                            source, author_email, source_investigation_id,
                        ),
                    )
                    row = await cur.fetchone()
                    created_at = row[1]
                    updated_at = row[2]
                    # Version 1 -- snapshot of the initial state.
                    await cur.execute(
                        """
                        INSERT INTO opsrag_runbook_versions (
                          runbook_id, version_num, title, body_markdown,
                          service, issue_kind, severity_min, priority,
                          tags, edited_by, change_note
                        ) VALUES (
                          %s, 1, %s, %s,
                          %s, %s, %s, %s,
                          %s, %s, 'initial'
                        )
                        """,
                        (
                            new_id, body.title, body.body_markdown,
                            body.service, body.issue_kind, body.severity_min,
                            body.priority, list(body.tags), author_email,
                        ),
                    )

        # Best-effort Qdrant upsert. Failure here doesn't roll back the
        # Postgres row -- the runbook still exists, just won't be found
        # by embedding lookup until we re-sync. The TSV index covers
        # keyword retrieval as a fallback.
        if embedding is not None:
            await self._upsert_vector(new_id, embedding, body)

        return await self.get(new_id)

    async def get(self, runbook_id: str) -> Runbook:
        async with self._pg.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    _SELECT_RUNBOOK_SQL + " WHERE id = %s",
                    (runbook_id,),
                )
                row = await cur.fetchone()
        if row is None:
            raise KeyError(f"runbook not found: {runbook_id}")
        return _row_to_runbook(row)

    async def list(
        self,
        *,
        service: str | None = None,
        issue_kind: str | None = None,
        enabled_only: bool = True,
        limit: int = 100,
    ) -> list[Runbook]:
        clauses = []
        params: list[Any] = []
        if enabled_only:
            clauses.append("enabled = TRUE")
        if service:
            clauses.append("service = %s")
            params.append(service)
        if issue_kind:
            clauses.append("issue_kind = %s")
            params.append(issue_kind)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        async with self._pg.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    _SELECT_RUNBOOK_SQL + where +
                    " ORDER BY priority DESC, updated_at DESC LIMIT %s",
                    tuple(params),
                )
                rows = await cur.fetchall()
        return [_row_to_runbook(r) for r in rows]

    async def update(
        self,
        runbook_id: str,
        body: RunbookUpdate,
        *,
        editor_email: str | None = None,
    ) -> Runbook:
        # Read the current state first so we can set unchanged fields
        # AND snapshot it into a new version row.
        current = await self.get(runbook_id)
        # Apply patch (only present-non-None fields).
        new_title = body.title or current.title
        new_body = body.body_markdown or current.body_markdown
        new_service = body.service if body.service is not None else current.service
        new_issue_kind = body.issue_kind if body.issue_kind is not None else current.issue_kind
        new_severity = body.severity_min if body.severity_min is not None else current.severity_min
        new_priority = body.priority if body.priority is not None else current.priority
        new_tags = body.tags if body.tags is not None else current.tags
        new_enabled = body.enabled if body.enabled is not None else current.enabled

        # If title or body changed, re-embed.
        re_embed = (
            new_title != current.title
            or new_body != current.body_markdown
        )

        async with self._pg.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    # Find current max version_num atomically.
                    await cur.execute(
                        """
                        SELECT COALESCE(MAX(version_num), 0)
                        FROM opsrag_runbook_versions
                        WHERE runbook_id = %s
                        """,
                        (runbook_id,),
                    )
                    next_version = (await cur.fetchone())[0] + 1

                    await cur.execute(
                        """
                        UPDATE opsrag_runbooks
                        SET title = %s,
                            body_markdown = %s,
                            service = %s,
                            issue_kind = %s,
                            severity_min = %s,
                            priority = %s,
                            tags = %s,
                            enabled = %s
                        WHERE id = %s
                        """,
                        (
                            new_title, new_body, new_service, new_issue_kind,
                            new_severity, new_priority, list(new_tags),
                            new_enabled, runbook_id,
                        ),
                    )
                    await cur.execute(
                        """
                        INSERT INTO opsrag_runbook_versions (
                          runbook_id, version_num, title, body_markdown,
                          service, issue_kind, severity_min, priority,
                          tags, edited_by, change_note
                        ) VALUES (
                          %s, %s, %s, %s,
                          %s, %s, %s, %s,
                          %s, %s, %s
                        )
                        """,
                        (
                            runbook_id, next_version, new_title, new_body,
                            new_service, new_issue_kind, new_severity,
                            new_priority, list(new_tags), editor_email,
                            body.change_note,
                        ),
                    )

        if re_embed:
            embedding = await self._embed(new_title, new_body)
            if embedding is not None:
                # Build a synthetic RunbookCreate-shaped record for the
                # vector upserter (it only needs service/issue_kind/tags).
                upsert_body = RunbookCreate(
                    title=new_title,
                    body_markdown=new_body,
                    service=new_service,
                    issue_kind=new_issue_kind,
                    severity_min=new_severity,
                    priority=new_priority,
                    tags=list(new_tags),
                )
                await self._upsert_vector(runbook_id, embedding, upsert_body)

        return await self.get(runbook_id)

    async def delete(self, runbook_id: str, *, hard: bool = False) -> bool:
        """Soft-delete (set enabled=FALSE) by default; hard-delete drops
        the row + cascades versions + drops the Qdrant point."""
        async with self._pg.connection() as conn:
            async with conn.cursor() as cur:
                if hard:
                    await cur.execute(
                        "DELETE FROM opsrag_runbooks WHERE id = %s",
                        (runbook_id,),
                    )
                    rc = cur.rowcount
                else:
                    await cur.execute(
                        "UPDATE opsrag_runbooks SET enabled = FALSE WHERE id = %s",
                        (runbook_id,),
                    )
                    rc = cur.rowcount
        if hard and rc and self._qdrant is not None:
            try:
                await self._qdrant.delete(
                    collection_name=self._collection,
                    points_selector=qm.PointIdsList(points=[runbook_id]),
                )
            except Exception as exc:
                _log.warning("runbooks: qdrant delete failed for %s: %s", runbook_id, exc)
        return bool(rc)

    async def versions(self, runbook_id: str, *, limit: int = 50) -> list[RunbookVersion]:
        async with self._pg.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT id, runbook_id, version_num, title, body_markdown,
                           service, issue_kind, severity_min, priority,
                           tags, edited_by, edited_at, change_note
                    FROM opsrag_runbook_versions
                    WHERE runbook_id = %s
                    ORDER BY version_num DESC
                    LIMIT %s
                    """,
                    (runbook_id, limit),
                )
                rows = await cur.fetchall()
        return [
            RunbookVersion(
                id=r[0], runbook_id=str(r[1]), version_num=r[2],
                title=r[3], body_markdown=r[4],
                service=r[5], issue_kind=r[6], severity_min=r[7],
                priority=r[8], tags=list(r[9] or []),
                edited_by=r[10], edited_at=r[11], change_note=r[12],
            )
            for r in rows
        ]

    # -- Retrieval ------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        query_embedding: list[float] | None = None,
        service: str | None = None,
        issue_kind: str | None = None,
        top_k: int = 5,
    ) -> list[RunbookHit]:
        """Hybrid retrieval: Qdrant (embedding) + Postgres tsv, merged
        by reciprocal-rank fusion, then boosted by `priority`.

        Returns hand-authored runbooks only (this store owns no RAG
        hits -- Lane A does the RAG concat outside).

        When `query_embedding` is not supplied, we embed via the
        configured embedder. If the embedder is None too, falls back
        to tsv-only ranking.
        """
        if not query.strip():
            return []

        # -- 1. Embedding lookup (best-effort) --------------------
        embed_hits: dict[str, float] = {}
        if self._qdrant is not None:
            if query_embedding is None and self._embedder is not None:
                try:
                    query_embedding = await self._embedder.embed_query(query)
                except Exception as exc:
                    _log.warning("runbooks: query embed failed: %s", exc)
            if query_embedding:
                try:
                    result = await self._qdrant.query_points(
                        collection_name=self._collection,
                        query=list(query_embedding),
                        limit=top_k * 3,
                        with_payload=False,
                    )
                    for p in result.points:
                        embed_hits[str(p.id)] = float(p.score)
                except Exception as exc:
                    _log.warning("runbooks: qdrant search failed: %s", exc)

        # -- 2. TSV lookup ---------------------------------------
        clauses = ["enabled = TRUE", "tsv @@ plainto_tsquery('english', %s)"]
        params: list[Any] = [query]
        if service:
            clauses.append("service = %s")
            params.append(service)
        if issue_kind:
            clauses.append("issue_kind = %s")
            params.append(issue_kind)
        where_sql = " AND ".join(clauses)
        params.append(top_k * 3)
        tsv_rows: list[tuple[str, float]] = []
        async with self._pg.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    SELECT id, ts_rank(tsv, plainto_tsquery('english', %s)) AS rank
                    FROM opsrag_runbooks
                    WHERE {where_sql}
                    ORDER BY rank DESC
                    LIMIT %s
                    """,
                    (query, *params),
                )
                for r in await cur.fetchall():
                    tsv_rows.append((str(r[0]), float(r[1])))
        tsv_hits = dict(tsv_rows)

        # -- 3. Reciprocal-rank fusion --------------------------
        # RRF with k=60 (the classical default from the Cormack paper).
        # Score = sum( 1 / (k + rank_in_each_list) ).
        K_RRF = 60
        embed_sorted = sorted(embed_hits.items(), key=lambda x: -x[1])
        tsv_sorted = sorted(tsv_hits.items(), key=lambda x: -x[1])
        rrf: dict[str, float] = {}
        breakdown: dict[str, dict] = {}
        for rank, (rid, score) in enumerate(embed_sorted):
            rrf[rid] = rrf.get(rid, 0.0) + 1.0 / (K_RRF + rank)
            breakdown.setdefault(rid, {})["embed"] = {"rank": rank, "cosine": score}
        for rank, (rid, score) in enumerate(tsv_sorted):
            rrf[rid] = rrf.get(rid, 0.0) + 1.0 / (K_RRF + rank)
            breakdown.setdefault(rid, {})["tsv"] = {"rank": rank, "score": score}

        if not rrf:
            return []

        # -- 4. Hydrate rows (filtered by service/issue_kind already
        # for tsv path; embedding path may have returned excluded ids
        # -- re-filter at hydration).
        ids = list(rrf.keys())
        async with self._pg.connection() as conn:
            async with conn.cursor() as cur:
                # Note: we always re-filter by enabled+service+issue_kind
                # here so embedding-only hits respect the same filter.
                hclauses = ["enabled = TRUE", "id = ANY(%s)"]
                hparams: list[Any] = [ids]
                if service:
                    hclauses.append("service = %s")
                    hparams.append(service)
                if issue_kind:
                    hclauses.append("issue_kind = %s")
                    hparams.append(issue_kind)
                await cur.execute(
                    _SELECT_RUNBOOK_SQL + " WHERE " + " AND ".join(hclauses),
                    tuple(hparams),
                )
                rows = await cur.fetchall()
        by_id = {str(r[0]): r for r in rows}

        # -- 5. Build hits, apply priority boost ----------------
        # Score = base_rrf * (priority / 100). priority=100 -> *1.0;
        # 200 -> *2.0; 500 -> *5.0. Clamps to [0, 10].
        hits: list[RunbookHit] = []
        for rid, base_score in rrf.items():
            row = by_id.get(rid)
            if row is None:
                continue
            rb = _row_to_runbook(row)
            multiplier = max(0.1, min(10.0, rb.priority / 100.0))
            final = base_score * multiplier
            bd = breakdown.get(rid, {})
            bd["priority_multiplier"] = multiplier
            bd["rrf_base"] = base_score
            hits.append(RunbookHit(
                runbook=rb, score=final,
                score_breakdown=bd, origin="hand",
            ))
        hits.sort(key=lambda h: -h.score)
        return hits[:top_k]

    # -- Usage telemetry -----------------------------------------

    async def record_use(
        self,
        runbook_id: str,
        *,
        thumbs: str | None = None,
    ) -> None:
        """Bump used_count + last_used_at; if `thumbs` is 'up'/'down',
        also bump the corresponding counter. Best-effort, doesn't raise."""
        try:
            async with self._pg.connection() as conn:
                async with conn.cursor() as cur:
                    if thumbs == "up":
                        await cur.execute(
                            """
                            UPDATE opsrag_runbooks
                            SET used_count = used_count + 1,
                                thumbs_up_count = thumbs_up_count + 1,
                                last_used_at = NOW()
                            WHERE id = %s
                            """,
                            (runbook_id,),
                        )
                    elif thumbs == "down":
                        await cur.execute(
                            """
                            UPDATE opsrag_runbooks
                            SET used_count = used_count + 1,
                                thumbs_down_count = thumbs_down_count + 1,
                                last_used_at = NOW()
                            WHERE id = %s
                            """,
                            (runbook_id,),
                        )
                    else:
                        await cur.execute(
                            """
                            UPDATE opsrag_runbooks
                            SET used_count = used_count + 1,
                                last_used_at = NOW()
                            WHERE id = %s
                            """,
                            (runbook_id,),
                        )
        except Exception as exc:
            _log.warning("runbooks: record_use failed for %s: %s", runbook_id, exc)

    # -- Internals -----------------------------------------------

    async def _embed(self, title: str, body: str) -> list[float] | None:
        """Embed title + body_markdown for Qdrant. Returns None when no
        embedder configured (the store stays usable, retrieval falls
        back to tsv-only)."""
        if self._embedder is None:
            return None
        text = (title or "") + "\n\n" + (body or "")
        try:
            vec = await self._embedder.embed_query(text[:8000])
            return list(vec) if vec else None
        except Exception as exc:
            _log.warning("runbooks: embed failed: %s", exc)
            return None

    async def _upsert_vector(
        self,
        runbook_id: str,
        embedding: list[float],
        body: RunbookCreate,
    ) -> None:
        if self._qdrant is None:
            return
        await self._ensure_qdrant_collection()
        if not self._qdrant_ensured:
            return
        payload = {
            "runbook_id": runbook_id,
            "title": body.title,
            "service": body.service,
            "issue_kind": body.issue_kind,
            "severity_min": body.severity_min,
            "priority": body.priority,
            "tags": list(body.tags),
            "updated_at": time.time(),
        }
        try:
            await self._qdrant.upsert(
                collection_name=self._collection,
                points=[qm.PointStruct(
                    id=runbook_id, vector=embedding, payload=payload,
                )],
            )
        except Exception as exc:
            _log.warning(
                "runbooks: qdrant upsert failed for %s: %s",
                runbook_id, exc,
            )


# -- Row helpers ------------------------------------------------------

_SELECT_RUNBOOK_SQL = """
SELECT id, title, body_markdown, service, issue_kind, severity_min,
       priority, tags, source, author_email, source_investigation_id,
       enabled, created_at, updated_at, used_count,
       thumbs_up_count, thumbs_down_count, last_used_at
FROM opsrag_runbooks
"""


def _row_to_runbook(row: tuple) -> Runbook:
    return Runbook(
        id=str(row[0]),
        title=row[1],
        body_markdown=row[2],
        service=row[3],
        issue_kind=row[4],
        severity_min=row[5],
        priority=row[6],
        tags=list(row[7] or []),
        source=row[8],
        author_email=row[9],
        source_investigation_id=row[10],
        enabled=row[11],
        created_at=row[12],
        updated_at=row[13],
        used_count=row[14],
        thumbs_up_count=row[15],
        thumbs_down_count=row[16],
        last_used_at=row[17],
    )
