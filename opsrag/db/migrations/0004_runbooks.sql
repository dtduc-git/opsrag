-- 0004 -- Hand-authored runbook store + version history.
--
-- Runbooks are written by SREs through the OpsRAG UI ("+ New runbook").
-- They are retrieved during Investigate-mode root-cause analysis with
-- HIGHER priority than RAG-indexed runbooks (Confluence / docs). The
-- agent fuses them into the "Initial finding" Insight card.
--
-- Layout:
--   - opsrag_runbooks: main row per runbook (latest content)
--   - opsrag_runbook_versions: append-only history of every edit
--   - tsvector index for keyword fallback when embedding lookup is cold
--   - Embedding lives in Qdrant collection `opsrag_runbooks_vec` (not
--     here) because Zalando's spilo image doesn't include pgvector.

CREATE TABLE IF NOT EXISTS opsrag_runbooks (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  title           TEXT NOT NULL,
  body_markdown   TEXT NOT NULL,

  -- Classification facets (optional; help the retriever filter).
  service         TEXT,          -- e.g. "acme-notes-auth-be", "kafka", "kong-gateway"
  issue_kind      TEXT,          -- one of the 8 failure_class values, see opsrag.runbooks.taxonomy
  severity_min    TEXT,          -- "SEV1" | "SEV2" | "SEV3" | "SEV4"
  tags            TEXT[] NOT NULL DEFAULT '{}',  -- free-text labels

  -- Retrieval weight. Hand-authored runbooks ALWAYS rank above RAG hits;
  -- this dial breaks ties WITHIN the hand-authored set. Default 100.
  priority        INTEGER NOT NULL DEFAULT 100,

  -- Provenance.
  source          TEXT NOT NULL DEFAULT 'hand',     -- 'hand' | 'imported' | 'auto'
  author_email    TEXT,                              -- whoever clicked Save
  source_investigation_id  TEXT,                     -- when 'auto' (promoted from investigation)

  -- Lifecycle.
  enabled         BOOLEAN NOT NULL DEFAULT TRUE,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  -- Usage telemetry.
  used_count          INTEGER NOT NULL DEFAULT 0,
  thumbs_up_count     INTEGER NOT NULL DEFAULT 0,
  thumbs_down_count   INTEGER NOT NULL DEFAULT 0,
  last_used_at        TIMESTAMPTZ,

  -- Generated full-text search vector (English stemming). Covers title
  -- + body for the BM25-like keyword fallback path. Always present --
  -- ts_rank is roughly free when we already have to read the row.
  tsv TSVECTOR
       GENERATED ALWAYS AS (
         setweight(to_tsvector('english', coalesce(title, '')),         'A') ||
         setweight(to_tsvector('english', coalesce(service, '')),       'B') ||
         setweight(to_tsvector('english', coalesce(issue_kind, '')),    'B') ||
         setweight(to_tsvector('english', coalesce(body_markdown, '')), 'C')
       ) STORED
);

CREATE INDEX IF NOT EXISTS opsrag_runbooks_service_enabled
  ON opsrag_runbooks(service)
  WHERE enabled;

CREATE INDEX IF NOT EXISTS opsrag_runbooks_issue_kind_enabled
  ON opsrag_runbooks(issue_kind)
  WHERE enabled;

CREATE INDEX IF NOT EXISTS opsrag_runbooks_priority_enabled
  ON opsrag_runbooks(priority DESC, updated_at DESC)
  WHERE enabled;

CREATE INDEX IF NOT EXISTS opsrag_runbooks_tsv
  ON opsrag_runbooks USING GIN (tsv);

-- ------------------------------------------------------------------
-- Version history. Append-only, written by the API on every edit.
-- Keep ~20 most-recent versions per runbook; older ones can be pruned
-- by a maintenance job (out of scope for v1).
CREATE TABLE IF NOT EXISTS opsrag_runbook_versions (
  id              BIGSERIAL PRIMARY KEY,
  runbook_id      UUID NOT NULL REFERENCES opsrag_runbooks(id) ON DELETE CASCADE,
  version_num     INTEGER NOT NULL,                  -- 1, 2, 3, ... per runbook
  title           TEXT NOT NULL,
  body_markdown   TEXT NOT NULL,
  service         TEXT,
  issue_kind      TEXT,
  severity_min    TEXT,
  priority        INTEGER,
  tags            TEXT[],
  edited_by       TEXT,                              -- email of editor
  edited_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  change_note     TEXT,                              -- optional "why did I edit"

  UNIQUE (runbook_id, version_num)
);
CREATE INDEX IF NOT EXISTS opsrag_runbook_versions_by_runbook
  ON opsrag_runbook_versions(runbook_id, version_num DESC);

-- updated_at trigger so the column stays accurate without app-side care.
CREATE OR REPLACE FUNCTION opsrag_runbooks_touch_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS opsrag_runbooks_touch ON opsrag_runbooks;
CREATE TRIGGER opsrag_runbooks_touch
  BEFORE UPDATE ON opsrag_runbooks
  FOR EACH ROW
  EXECUTE FUNCTION opsrag_runbooks_touch_updated_at();
