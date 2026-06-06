"""Neo4j knowledge graph store -- Cypher-based implementation.

Uses the async neo4j driver directly. The SRE schema (node labels,
relationship types) is not enforced here -- callers supply arbitrary
Entity/Relationship objects, letting the schema emerge from the
entity extractor config.
"""
from __future__ import annotations

import logging
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase

from opsrag.interfaces.graphstore import Entity, GraphSearchResult, Relationship

_log = logging.getLogger("opsrag.graphstore.neo4j")


class APOCUnavailableError(RuntimeError):
    """Raised when the Neo4j instance is missing the APOC procedure library.

    The graph store's upsert/traverse Cypher relies on ``apoc.create.addLabels``,
    ``apoc.merge.relationship`` and ``apoc.path.subgraphAll``. Without APOC
    those calls fail at query time and -- historically (2026-05-23 prod
    incident) -- the failures were swallowed, leaving a silently-empty graph
    that ran dead in production for three months. We therefore probe for APOC
    eagerly and FAIL FAST when ``provider=neo4j`` is explicitly selected,
    rather than warn.
    """


class Neo4jGraphStore:
    def __init__(
        self,
        url: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "opsrag-password",
        database: str = "neo4j",
    ):
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(url, auth=(username, password))
        self._db = database

    async def close(self) -> None:
        await self._driver.close()

    async def __aenter__(self) -> Neo4jGraphStore:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def ensure_indexes(self) -> None:
        async with self._driver.session(database=self._db) as session:
            await session.run(
                "CREATE INDEX entity_id IF NOT EXISTS FOR (n:Entity) ON (n.id)"
            )
            await session.run(
                "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)"
            )
            await session.run(
                "CREATE INDEX source_chunk IF NOT EXISTS FOR (n:Entity) ON (n.source_chunk_id)"
            )
            # T1.2 (P5) -- code-symbol + Route graph indexes. Property
            # indexes are label-specific, so they only consume space
            # for nodes that actually carry that label -- zero overhead
            # for existing `:Entity`-only nodes. Each is `IF NOT EXISTS`
            # so re-running is safe.
            await session.run(
                "CREATE INDEX route_path IF NOT EXISTS FOR (r:Route) ON (r.path)"
            )
            await session.run(
                "CREATE INDEX route_chart IF NOT EXISTS FOR (r:Route) ON (r.chart_key)"
            )
            await session.run(
                "CREATE INDEX appservice_name IF NOT EXISTS FOR (s:K8sAppService) ON (s.name)"
            )
            await session.run(
                "CREATE INDEX appservice_chart IF NOT EXISTS FOR (s:K8sAppService) ON (s.chart_key)"
            )
            await session.run(
                "CREATE INDEX function_name IF NOT EXISTS FOR (f:Function) ON (f.name)"
            )
            await session.run(
                "CREATE INDEX class_name IF NOT EXISTS FOR (c:Class) ON (c.name)"
            )
            await session.run(
                "CREATE INDEX method_name IF NOT EXISTS FOR (m:Method) ON (m.name)"
            )
            await session.run(
                "CREATE INDEX file_path IF NOT EXISTS FOR (f:File) ON (f.path)"
            )

    async def check_apoc(self) -> None:
        """Probe for the APOC library and FAIL FAST if it is missing.

        Queries ``dbms.procedures()`` (or the 5.x ``SHOW PROCEDURES``) for any
        ``apoc.*`` procedure. Raises :class:`APOCUnavailableError` when none
        are present. Call this at factory build time when the operator has
        explicitly selected ``provider=neo4j`` -- a missing APOC must surface
        loudly at startup, never as a silently-empty graph at runtime.
        """
        # `SHOW PROCEDURES` is the 5.x form; fall back to the legacy
        # `dbms.procedures()` for older servers. Either way we just need to
        # know whether *any* apoc procedure is registered.
        cyphers = (
            "SHOW PROCEDURES YIELD name WHERE name STARTS WITH 'apoc.' "
            "RETURN count(name) AS cnt",
            "CALL dbms.procedures() YIELD name WHERE name STARTS WITH 'apoc.' "
            "RETURN count(name) AS cnt",
        )
        last_exc: Exception | None = None
        async with self._driver.session(database=self._db) as session:
            for cypher in cyphers:
                try:
                    result = await session.run(cypher)
                    record = await result.single()
                    cnt = (record["cnt"] if record else 0) or 0
                    if cnt > 0:
                        return
                    # Query succeeded but found zero apoc procedures.
                    raise APOCUnavailableError(
                        "Neo4j is reachable but the APOC procedure library is "
                        "not installed (0 apoc.* procedures found). The knowledge-"
                        "graph lane requires APOC (apoc.create.addLabels, "
                        "apoc.merge.relationship, apoc.path.subgraphAll). Install "
                        "the APOC plugin on the Neo4j instance (e.g. set "
                        "NEO4J_PLUGINS='[\"apoc\"]' on neo4j:5) or set "
                        "knowledge_graph.provider to 'none'."
                    )
                except APOCUnavailableError:
                    raise
                except Exception as exc:  # noqa: BLE001 -- try next probe form
                    last_exc = exc
                    continue
        # Neither probe form worked -- treat as unavailable (fail fast).
        raise APOCUnavailableError(
            "Could not verify APOC availability on the Neo4j instance "
            f"({last_exc!r}). Refusing to start the neo4j graph lane to avoid "
            "the silent-empty-graph failure mode. Install APOC or set "
            "knowledge_graph.provider to 'none'."
        )

    async def upsert_entities(self, entities: list[Entity]) -> int:
        if not entities:
            return 0
        # Reference-counting: entity IDs are deterministic + shared across
        # source chunks/files, so we track which sources reference each node
        # in a `sources` list property. On upsert we add this source to the
        # set; `delete_by_source` removes it and only DETACH DELETEs the node
        # when no live source remains. `source_chunk_id` is kept (last writer)
        # for back-compat with the `source_chunk` index + reads.
        query = """
        UNWIND $entities AS e
        MERGE (n:Entity {id: e.id})
        SET n.name = e.name,
            n.label = e.label,
            n.source_chunk_id = e.source_chunk_id,
            n += e.properties,
            n.sources = CASE
                WHEN e.source_chunk_id IS NULL THEN coalesce(n.sources, [])
                WHEN n.sources IS NULL THEN [e.source_chunk_id]
                WHEN e.source_chunk_id IN n.sources THEN n.sources
                ELSE n.sources + e.source_chunk_id
            END
        WITH n, e
        CALL apoc.create.addLabels(n, [e.label]) YIELD node
        RETURN count(node) AS cnt
        """
        params = [
            {
                "id": e.id,
                "name": e.name,
                "label": e.label,
                "source_chunk_id": e.source_chunk_id,
                "properties": e.properties,
            }
            for e in entities
        ]
        async with self._driver.session(database=self._db) as session:
            result = await session.run(query, entities=params)
            record = await result.single()
            return record["cnt"] if record else 0

    async def upsert_relationships(self, relationships: list[Relationship]) -> int:
        if not relationships:
            return 0
        # Same reference-counting posture for edges: stamp the source onto a
        # `sources` list on the relationship so a single-source reindex can
        # remove only the edges that source contributed, leaving cross-file
        # edges (still referenced elsewhere) intact.
        query = """
        UNWIND $rels AS r
        MATCH (a:Entity {id: r.source_id})
        MATCH (b:Entity {id: r.target_id})
        CALL apoc.merge.relationship(a, r.rel_type, {}, r.properties, b, {}) YIELD rel
        SET rel.sources = CASE
            WHEN r.source_chunk_id IS NULL THEN coalesce(rel.sources, [])
            WHEN rel.sources IS NULL THEN [r.source_chunk_id]
            WHEN r.source_chunk_id IN rel.sources THEN rel.sources
            ELSE rel.sources + r.source_chunk_id
        END
        RETURN count(rel) AS cnt
        """
        params = [
            {
                "source_id": r.source_id,
                "target_id": r.target_id,
                "rel_type": r.rel_type,
                "source_chunk_id": r.properties.get("source_chunk_id"),
                "properties": {
                    k: v for k, v in r.properties.items() if k != "source_chunk_id"
                },
            }
            for r in relationships
        ]
        async with self._driver.session(database=self._db) as session:
            result = await session.run(query, rels=params)
            record = await result.single()
            return record["cnt"] if record else 0

    async def search_entities(
        self,
        query: str,
        labels: list[str] | None = None,
        limit: int = 10,
    ) -> list[Entity]:
        if labels:
            label_filter = " AND any(l IN labels(n) WHERE l IN $labels)"
        else:
            label_filter = ""
        cypher = f"""
        MATCH (n:Entity)
        WHERE (n.name CONTAINS $search_term OR n.id CONTAINS $search_term){label_filter}
        RETURN n LIMIT $result_limit
        """
        params: dict[str, Any] = {"search_term": query, "result_limit": limit}
        if labels:
            params["labels"] = labels
        async with self._driver.session(database=self._db) as session:
            result = await session.run(cypher, **params)
            return [self._record_to_entity(r) async for r in result]

    async def get_subgraph(
        self,
        entity_ids: list[str],
        include_neighbors: bool = True,
        neighbor_depth: int = 1,
    ) -> GraphSearchResult:
        if include_neighbors:
            cypher = """
            MATCH (n:Entity) WHERE n.id IN $ids
            CALL apoc.path.subgraphAll(n, {maxLevel: $depth})
            YIELD nodes, relationships
            RETURN nodes, relationships
            """
        else:
            cypher = """
            MATCH (n:Entity) WHERE n.id IN $ids
            OPTIONAL MATCH (n)-[r]-(m:Entity) WHERE m.id IN $ids
            RETURN collect(distinct n) AS nodes, collect(distinct r) AS relationships
            """
        async with self._driver.session(database=self._db) as session:
            result = await session.run(cypher, ids=entity_ids, depth=neighbor_depth)
            record = await result.single()
            if not record:
                return GraphSearchResult()
            entities = [self._node_to_entity(n) for n in (record["nodes"] or [])]
            rels = [self._rel_to_relationship(r) for r in (record["relationships"] or []) if r]
            return GraphSearchResult(
                entities=entities,
                relationships=rels,
                context_text=self._render_context(entities, rels),
            )

    async def delete_by_source(self, source_chunk_ids: list[str]) -> int:
        """Reference-counted delete.

        Entity IDs are deterministic and SHARED across source chunks/files
        (the same ``Service:checkout`` appears in many files). A naive
        per-source ``DETACH DELETE`` would destroy cross-file edges still
        referenced by other live sources -- this was an explicit data-loss
        blocker. Instead we:

          1. Remove the given source(s) from each node's/edge's ``sources``
             list (and clear the matching ``source_chunk_id`` pointer).
          2. DETACH DELETE a node only when its ``sources`` list becomes
             empty (no live source remains).
          3. DELETE a relationship only when its ``sources`` becomes empty.

        Returns the count of nodes actually deleted (matching the previous
        contract's "rows removed" semantics).
        """
        if not source_chunk_ids:
            return 0

        # 1a. Decrement relationship reference counts and delete orphaned edges.
        rel_cypher = """
        MATCH ()-[r]->()
        WHERE r.sources IS NOT NULL AND any(s IN $ids WHERE s IN r.sources)
        SET r.sources = [s IN r.sources WHERE NOT s IN $ids]
        WITH r
        WHERE size(r.sources) = 0
        DELETE r
        """
        # 1b. Decrement node reference counts; clear stale source_chunk_id.
        node_decr_cypher = """
        MATCH (n:Entity)
        WHERE n.sources IS NOT NULL AND any(s IN $ids WHERE s IN n.sources)
        SET n.sources = [s IN n.sources WHERE NOT s IN $ids]
        SET n.source_chunk_id = CASE
            WHEN n.source_chunk_id IN $ids
                THEN head([s IN n.sources WHERE NOT s IN $ids] + [null])
            ELSE n.source_chunk_id
        END
        """
        # 2. Delete nodes whose source set is now empty. Also handle legacy
        #    nodes that never carried a `sources` list but match the old
        #    single-pointer `source_chunk_id` (so pre-refcount data still
        #    cleans up).
        node_del_cypher = """
        MATCH (n:Entity)
        WHERE (n.sources IS NOT NULL AND size(n.sources) = 0)
           OR (n.sources IS NULL AND n.source_chunk_id IN $ids)
        DETACH DELETE n
        RETURN count(n) AS cnt
        """
        async with self._driver.session(database=self._db) as session:
            await session.run(rel_cypher, ids=source_chunk_ids)
            await session.run(node_decr_cypher, ids=source_chunk_ids)
            result = await session.run(node_del_cypher, ids=source_chunk_ids)
            record = await result.single()
            return record["cnt"] if record else 0

    async def get_schema(self) -> dict:
        async with self._driver.session(database=self._db) as session:
            labels_r = await session.run("CALL db.labels()")
            labels = [r["label"] async for r in labels_r]
            types_r = await session.run("CALL db.relationshipTypes()")
            types = [r["relationshipType"] async for r in types_r]
            return {"labels": labels, "relationship_types": types}

    async def view_subgraph(
        self,
        node_labels: list[str],
        rel_types: list[str],
        limit: int = 300,
    ) -> tuple[list[dict], list[dict]]:
        """Return a filtered subgraph for the Knowledge Graph UI views.

        Matches every ``(a:Entity)-[r]->(b:Entity)`` edge where both
        endpoints carry a ``label`` in ``node_labels`` and the relationship
        type is in ``rel_types``. Returns ``(nodes, edges)`` where nodes are
        deduped ``{id, name, type}`` dicts (``type`` is the entity ``label``)
        and edges are ``{source, target, type}`` dicts (``type`` is the
        relationship type). Capped at ``limit`` edges.
        """
        cypher = """
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE a.label IN $labels AND b.label IN $labels AND type(r) IN $rels
        RETURN a.id AS aid, a.name AS aname, a.label AS alabel,
               b.id AS bid, b.name AS bname, b.label AS blabel,
               type(r) AS rt
        LIMIT $limit
        """
        nodes: dict[str, dict] = {}
        edges: list[dict] = []
        async with self._driver.session(database=self._db) as session:
            result = await session.run(
                cypher, labels=node_labels, rels=rel_types, limit=limit
            )
            async for record in result:
                aid = record["aid"]
                bid = record["bid"]
                if aid not in nodes:
                    nodes[aid] = {
                        "id": aid,
                        "name": record["aname"],
                        "type": record["alabel"],
                    }
                if bid not in nodes:
                    nodes[bid] = {
                        "id": bid,
                        "name": record["bname"],
                        "type": record["blabel"],
                    }
                edges.append(
                    {"source": aid, "target": bid, "type": record["rt"]}
                )
        return (list(nodes.values()), edges)

    @staticmethod
    def _node_to_entity(node) -> Entity:
        props = dict(node)
        return Entity(
            id=props.pop("id", ""),
            label=props.pop("label", next(iter(node.labels - {"Entity"}), "Entity")),
            name=props.pop("name", ""),
            source_chunk_id=props.pop("source_chunk_id", None),
            properties=props,
        )

    @staticmethod
    def _record_to_entity(record) -> Entity:
        node = record["n"]
        return Neo4jGraphStore._node_to_entity(node)

    @staticmethod
    def _rel_to_relationship(rel) -> Relationship:
        props = dict(rel)
        return Relationship(
            source_id=rel.start_node["id"] if hasattr(rel, "start_node") else props.pop("source_id", ""),
            target_id=rel.end_node["id"] if hasattr(rel, "end_node") else props.pop("target_id", ""),
            rel_type=rel.type if hasattr(rel, "type") else props.pop("rel_type", ""),
            properties=props,
        )

    @staticmethod
    def _render_context(entities: list[Entity], rels: list[Relationship]) -> str:
        lines: list[str] = []
        for e in entities:
            lines.append(f"[{e.label}] {e.name} (id={e.id})")
        for r in rels:
            lines.append(f"  ({r.source_id}) -[{r.rel_type}]-> ({r.target_id})")
        return "\n".join(lines)
