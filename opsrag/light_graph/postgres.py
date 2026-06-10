"""Postgres-backed lightweight entity adjacency for entity-expansion.

A tiny edges table (``opsrag_entity_edges``) populated from the structured /
metadata entity extraction (the same deterministic ``label:name:hash`` ids that
land in each chunk's Qdrant ``entity_ids`` payload). The retriever does a 1-hop
``get_neighbors`` here AFTER vector search, then pulls related chunks via a
Qdrant metadata filter. It never drives the answer (fail-safe: empty graph ->
no extra chunks). Independent of Neo4j / knowledge_graph.provider.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from opsrag.interfaces.graphstore import Relationship

# Canonical-case lookup so a parsed `label:name:hash` id (the label segment is
# lower-cased by make_entity_id) maps back to the display label the UI palette
# keys on ("service" -> "Service", "repository" -> "Repository", ...).
try:  # pragma: no cover - import guard keeps the store usable in minimal envs
    from opsrag.extractors.schema import ALLOWED_LABELS

    _LABEL_CASE: dict[str, str] = {lbl.lower(): lbl for lbl in ALLOWED_LABELS}
except Exception:  # pragma: no cover
    _LABEL_CASE = {}


def _parse_entity_id(eid: str) -> tuple[str, str]:
    """Split a deterministic ``label:name:hash`` entity id into (type, name).

    The name segment may itself contain ``/`` (repo paths) but never ``:``;
    we still split defensively from both ends so an unexpected colon in a name
    can't corrupt the type or drop the name.
    """
    head, _, tail = eid.partition(":")  # head = label segment
    label = _LABEL_CASE.get(head.lower(), head.capitalize() or head)
    name = tail.rsplit(":", 1)[0] if ":" in tail else (tail or eid)
    return label, name


class LightGraphStore:
    """Async Postgres store. Schema lives in migration 0008 (source of truth)."""

    def __init__(self, dsn: str, *, min_pool: int = 1, max_pool: int = 4) -> None:
        from psycopg_pool import AsyncConnectionPool

        self._dsn = dsn
        self._pool = AsyncConnectionPool(
            conninfo=dsn, min_size=min_pool, max_size=max_pool,
            open=False, kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        self._opened = False

    async def open(self) -> None:
        await self._pool.open()
        self._opened = True

    async def close(self) -> None:
        if self._opened:
            await self._pool.close()
            self._opened = False

    async def health_check(self) -> None:
        """Lightweight readiness probe (``SELECT 1``). Raises if the
        light-graph Postgres pool is unreachable so /readyz reports the
        light-graph lane as down rather than discovering it weeks later."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")

    async def init_schema(self) -> None:
        """No-op: the migration framework owns the DDL (parity with the other
        *Store classes so the boot sequence can call it uniformly)."""
        return None

    async def upsert_edges(
        self, edges: list[Relationship], repo: str = "", source_path: str = ""
    ) -> int:
        """Insert structured edges (dedup on the composite PK). Returns the
        number of rows attempted.

        ``source_path`` attributes each edge to the file that produced it so a
        later re-index can drop just that file's edges (delete_by_source). The
        full-repo rebuild path passes ``source_path=""`` and pairs the upsert
        with delete_by_repo instead.
        """
        rows = [
            (e.source_id, e.rel_type, e.target_id, repo, source_path)
            for e in (edges or [])
            if getattr(e, "source_id", None) and getattr(e, "target_id", None)
        ]
        if not rows:
            return 0
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(
                    "INSERT INTO opsrag_entity_edges (src_id, rel, dst_id, repo, source_path) "
                    "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                    rows,
                )
        return len(rows)

    async def get_neighbors(self, entity_ids: Iterable[str], *, limit: int = 60) -> set[str]:
        """1-hop neighbors (both directions) of the given entity ids, excluding
        the inputs themselves."""
        ids = [i for i in {x for x in entity_ids} if i]
        if not ids:
            return set()
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT dst_id AS n FROM opsrag_entity_edges WHERE src_id = ANY(%s) "
                    "UNION "
                    "SELECT src_id AS n FROM opsrag_entity_edges WHERE dst_id = ANY(%s) "
                    "LIMIT %s",
                    (ids, ids, limit),
                )
                rows = await cur.fetchall()
        return {r[0] for r in rows} - set(ids)

    async def delete_by_repo(self, repo: str) -> None:
        """Drop a repo's edges before a full rebuild (clean refresh)."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM opsrag_entity_edges WHERE repo = %s", (repo,))

    async def delete_by_source(self, repo: str, source_path: str) -> None:
        """Drop the edges a single file previously contributed, before that file
        is re-ingested. Mirrors the Qdrant per-file orphan sweep so renamed /
        removed entities don't linger and skew entity_expand. Pre-0011 rows
        (source_path='') are not touched here -- they age out on a full rebuild."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM opsrag_entity_edges WHERE repo = %s AND source_path = %s",
                    (repo, source_path),
                )

    async def count(self) -> int:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT count(*) FROM opsrag_entity_edges")
                row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def subgraph(
        self, *, rel_types: Iterable[str] | None = None, limit: int = 300
    ) -> tuple[list[dict], list[dict], bool]:
        """Render the entity graph for the Knowledge Graph UI.

        Returns ``(nodes, edges, truncated)`` shaped exactly like the Neo4j
        ``view_subgraph`` output so the route + frontend are provider-agnostic:
        ``nodes`` = ``[{id, name, type}]`` (type derived from the id label
        segment), ``edges`` = ``[{source, target, type}]`` (type = relation).
        ``rel_types`` optionally filters by relation; ``None`` returns all.
        """
        rels = [r for r in (rel_types or []) if r]
        sql = "SELECT src_id, rel, dst_id FROM opsrag_entity_edges"
        params: list[Any] = []
        if rels:
            sql += " WHERE rel = ANY(%s)"
            params.append(rels)
        sql += " LIMIT %s"
        params.append(limit)
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                rows = await cur.fetchall()
        nodes: dict[str, dict] = {}
        edges: list[dict] = []
        for src, rel, dst in rows:
            for eid in (src, dst):
                if eid not in nodes:
                    label, name = _parse_entity_id(eid)
                    nodes[eid] = {"id": eid, "name": name, "type": label}
            edges.append({"source": src, "target": dst, "type": rel})
        return list(nodes.values()), edges, len(edges) >= limit

    async def stats(self) -> dict:
        """Summary of the entity graph for the status header: edge count plus
        the distinct entity labels + relation types actually present."""
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT src_id, rel, dst_id FROM opsrag_entity_edges")
                rows = await cur.fetchall()
        labels: set[str] = set()
        rels: set[str] = set()
        for src, rel, dst in rows:
            rels.add(rel)
            labels.add(_parse_entity_id(src)[0])
            labels.add(_parse_entity_id(dst)[0])
        return {
            "edge_count": len(rows),
            "labels": sorted(labels),
            "relationship_types": sorted(rels),
        }
