-- Reverse 0011. Narrowing the PK back to (src_id, rel, dst_id, repo) requires
-- collapsing any rows that differ only by source_path first, or the ADD
-- CONSTRAINT would fail on duplicate keys.
DELETE FROM opsrag_entity_edges a
  USING opsrag_entity_edges b
  WHERE a.ctid < b.ctid
    AND a.src_id = b.src_id AND a.rel = b.rel
    AND a.dst_id = b.dst_id AND a.repo = b.repo;

ALTER TABLE opsrag_entity_edges DROP CONSTRAINT IF EXISTS opsrag_entity_edges_pkey;
ALTER TABLE opsrag_entity_edges
  ADD CONSTRAINT opsrag_entity_edges_pkey
  PRIMARY KEY (src_id, rel, dst_id, repo);

DROP INDEX IF EXISTS opsrag_entity_edges_repo_path;
ALTER TABLE opsrag_entity_edges DROP COLUMN IF EXISTS source_path;
