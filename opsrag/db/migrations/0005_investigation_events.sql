-- 0005 — Investigation event ledger (DB-as-SoR).
--
-- Every Investigate-mode pipeline step (lane probes, insight, hypothesis
-- evaluation, tool calls, conclusion) appends a row HERE. The SSE stream
-- at GET /investigations/{id}/events is a thin tail-cursor over this
-- table — that way a browser reconnect with `?since=<seq>` replays
-- exactly what's persisted, with no in-memory state to lose.
--
-- Design notes (Option B refactor, 2026-05-27):
--   - sequence is a BIGSERIAL GLOBAL across investigations so SSE can
--     resume from a single number on the client.
--   - Each node calls emit_event_via_ctx() which opens its own session
--     OUT-OF-BAND of the LangGraph transaction. Events are durable even
--     if the graph super-step rolls back (this is the side-effect fix
--     that avoided the "47× event amplification" bug the reference
--     gn-agentic-platform repo hit when events lived in graph state).
--   - payload is JSONB so different event_types carry different shapes
--     (hypothesis verdict vs tool result vs lane completion).

CREATE TABLE IF NOT EXISTS opsrag_investigation_events (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  investigation_id  UUID NOT NULL,
  sequence          BIGSERIAL UNIQUE NOT NULL,
  event_type        TEXT NOT NULL,
  payload           JSONB NOT NULL DEFAULT '{}'::jsonb,
  tags              TEXT[] NOT NULL DEFAULT '{}',
  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Tail-cursor reads filter by investigation_id + sequence > N. Composite
-- index covers both axes so the SSE poll is a fast range scan.
CREATE INDEX IF NOT EXISTS idx_inv_events_inv_seq
  ON opsrag_investigation_events (investigation_id, sequence);

-- Snapshot reads (GET /investigations/{id}) order by sequence too, so
-- the same index serves them.

-- Investigation lifecycle metadata — kept tiny on purpose. The full
-- state is reconstructable from the events table; this row exists so
-- the listing / sidebar can query without scanning every event.
CREATE TABLE IF NOT EXISTS opsrag_investigations (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  alert_text        TEXT NOT NULL,
  incident_target   TEXT,                 -- populated by hypothesizer
  status            TEXT NOT NULL DEFAULT 'running',  -- 'running' | 'completed' | 'failed' | 'cancelled'
  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  completed_at      TIMESTAMPTZ,
  -- Final root cause + outcome (populated on INVESTIGATION_COMPLETED event).
  root_cause        TEXT,
  outcome           TEXT
);

CREATE INDEX IF NOT EXISTS idx_inv_created_at
  ON opsrag_investigations (created_at DESC);
