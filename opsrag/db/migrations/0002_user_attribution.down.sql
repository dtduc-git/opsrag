-- 0002 DOWN — strip the per-user attribution columns from existing
-- tables. Reversible because user_oid was nullable from day one, so
-- pre-2026-05 rows already have NULL there and post-2026-05 rows lose
-- only the attribution (the actual usage data is untouched).
DROP INDEX IF EXISTS opsrag_usage_events_user_ts;
ALTER TABLE opsrag_usage_events DROP COLUMN IF EXISTS user_oid;

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'opsrag_feedback') THEN
    ALTER TABLE opsrag_feedback DROP COLUMN IF EXISTS user_oid;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'opsrag_corrections') THEN
    ALTER TABLE opsrag_corrections DROP COLUMN IF EXISTS user_oid;
  END IF;
END $$;
