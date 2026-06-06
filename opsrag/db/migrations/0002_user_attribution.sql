-- 0002 — attach user_oid to existing usage/session/feedback rows for per-user analytics.
-- Nullable on purpose: anonymous / pre-Pomerium events keep working, and we
-- intentionally avoid a FK to opsrag_user so usage telemetry never blocks
-- on user-row presence.
ALTER TABLE opsrag_usage_events  ADD COLUMN IF NOT EXISTS user_oid UUID;
CREATE INDEX IF NOT EXISTS opsrag_usage_events_user_ts
  ON opsrag_usage_events(user_oid, ts DESC)
  WHERE user_oid IS NOT NULL;

-- Same column on opsrag_feedback / opsrag_corrections if those tables
-- exist yet. They're created lazily by their owning modules on first
-- write, so a fresh DB may not have them when this migration runs.
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_name = 'opsrag_feedback'
  ) THEN
    ALTER TABLE opsrag_feedback ADD COLUMN IF NOT EXISTS user_oid UUID;
  END IF;
  IF EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_name = 'opsrag_corrections'
  ) THEN
    ALTER TABLE opsrag_corrections ADD COLUMN IF NOT EXISTS user_oid UUID;
  END IF;
END $$;
