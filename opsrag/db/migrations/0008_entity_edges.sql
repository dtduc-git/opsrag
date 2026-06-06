-- 0008 — lightweight entity-edge table for the entity-expansion retrieval lane.
--
-- This is the "light graph" (NOT Neo4j): a tiny adjacency table populated from
-- the SAME rule-based / metadata entity extraction used elsewhere, holding only
-- structured, reliable edges (service DEPENDS_ON service, OWNED_BY team,
-- USES_DATABASE, ...). At query time the retriever does a 1-hop lookup here to
-- pull in related chunks via a Qdrant `entity_ids` metadata filter. It NEVER
-- drives the answer -- it only augments AFTER vector search (see the
-- entity_expand node). Independent of knowledge_graph.provider (works with
-- Neo4j off).
--
-- Edge ids use the deterministic `label:normalized-name:hash` scheme
-- (opsrag.extractors.schema.make_entity_id) so they match the chunk payload
-- `entity_ids`. `repo` scopes a re-index refresh (delete_by_repo).
CREATE TABLE IF NOT EXISTS opsrag_entity_edges (
  src_id  TEXT NOT NULL,
  rel     TEXT NOT NULL,
  dst_id  TEXT NOT NULL,
  repo    TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (src_id, rel, dst_id, repo)
);
CREATE INDEX IF NOT EXISTS opsrag_entity_edges_src ON opsrag_entity_edges (src_id);
CREATE INDEX IF NOT EXISTS opsrag_entity_edges_dst ON opsrag_entity_edges (dst_id);
CREATE INDEX IF NOT EXISTS opsrag_entity_edges_repo ON opsrag_entity_edges (repo);
