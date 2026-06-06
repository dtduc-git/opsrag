-- 0011 — add per-file attribution to the light-graph edge table.
--
-- The edge table (0008) was keyed only by (src_id, rel, dst_id, repo), so an
-- edge contributed by a file could never be removed when that file was edited,
-- renamed, or deleted -- only a full repo wipe (delete_by_repo) could. Indexing
-- is per-file + incremental, so edges from stale/renamed entities accreted
-- forever and degraded the entity_expand retrieval lane over time.
--
-- Adding `source_path` to the row + PRIMARY KEY lets the ingest pipeline do a
-- per-file delete-before-reingest (delete_by_source), mirroring the existing
-- Qdrant orphan sweep. The same edge contributed by two files is now tracked as
-- two rows; get_neighbors UNIONs + dedups, so retrieval is unaffected.
--
-- Pre-existing edges keep source_path='' (the column default). A per-file
-- delete won't match them; they age out on the next full repo rebuild.
ALTER TABLE opsrag_entity_edges
  ADD COLUMN IF NOT EXISTS source_path TEXT NOT NULL DEFAULT '';

-- DROP IF EXISTS then ADD is idempotent: a re-run drops the just-added key and
-- re-adds it identically. All PK columns are NOT NULL, so ADD PRIMARY KEY is valid.
ALTER TABLE opsrag_entity_edges DROP CONSTRAINT IF EXISTS opsrag_entity_edges_pkey;
ALTER TABLE opsrag_entity_edges
  ADD CONSTRAINT opsrag_entity_edges_pkey
  PRIMARY KEY (src_id, rel, dst_id, repo, source_path);

CREATE INDEX IF NOT EXISTS opsrag_entity_edges_repo_path
  ON opsrag_entity_edges (repo, source_path);
