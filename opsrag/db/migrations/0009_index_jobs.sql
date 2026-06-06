-- 0009 — durable indexing job-state (replaces the in-memory IndexingTracker
-- as the source of truth for the dashboard).
--
-- Why: indexing progress used to live only in a per-process in-memory tracker
-- (opsrag.indexing_tracker.IndexingTracker). With >1 backend replica each pod
-- held its OWN copy, so /indexing/status returned whichever pod answered the
-- request -> the UI flickered between inconsistent states. Moving the state to
-- Postgres means every backend pod reads the same truth, and it survives
-- restarts (no more rebuild-from-Qdrant on every boot).
--
-- Writer model: a single indexing writer at a time per source (the ephemeral
-- job-indexer Job, or the legacy `indexer` role, or local dev). The writer
-- keeps the fast in-memory tracker and FLUSHES it here on a throttle; backend
-- pods only READ. Per-row UPSERT keys mean concurrent Jobs for different repos
-- never clobber each other.

-- Current per-source state: one row per (repo, branch). Mirrors RepoProgress.
CREATE TABLE IF NOT EXISTS opsrag_index_progress (
  repo            TEXT NOT NULL,
  branch          TEXT NOT NULL,
  status          TEXT NOT NULL DEFAULT 'queued',  -- queued|listing|indexing|done|failed
  source_type     TEXT NOT NULL DEFAULT 'git',
  display_name    TEXT,
  total_files     INTEGER NOT NULL DEFAULT 0,
  indexed_files   INTEGER NOT NULL DEFAULT 0,
  skipped_files   INTEGER NOT NULL DEFAULT 0,
  total_chunks    INTEGER NOT NULL DEFAULT 0,
  entities_found  INTEGER NOT NULL DEFAULT 0,
  error           TEXT,
  started_at      DOUBLE PRECISION NOT NULL DEFAULT 0,
  finished_at     DOUBLE PRECISION NOT NULL DEFAULT 0,
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (repo, branch)
);

-- Append-only run history (the Operations -> Indexing Jobs view). One row per
-- run. `run_key` is process-unique ("<proc_token>:<local_id>") so the writer
-- can UPSERT its own running->terminal transition without colliding with rows
-- written by other Jobs/processes.
CREATE TABLE IF NOT EXISTS opsrag_index_runs (
  id              BIGSERIAL PRIMARY KEY,
  run_key         TEXT NOT NULL UNIQUE,
  repo            TEXT NOT NULL,
  branch          TEXT NOT NULL,
  source_type     TEXT NOT NULL DEFAULT 'git',
  display_name    TEXT,
  status          TEXT NOT NULL,                   -- running|success|failed|restored
  started_at      DOUBLE PRECISION NOT NULL DEFAULT 0,
  finished_at     DOUBLE PRECISION NOT NULL DEFAULT 0,
  chunks_indexed  INTEGER NOT NULL DEFAULT 0,
  files_indexed   INTEGER NOT NULL DEFAULT 0,
  error           TEXT,
  kind            TEXT NOT NULL DEFAULT 'run',      -- run|restored
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS opsrag_index_runs_created ON opsrag_index_runs (created_at DESC);
