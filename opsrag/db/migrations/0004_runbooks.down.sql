-- 0004 down — drop runbook tables + helper trigger.
DROP TRIGGER IF EXISTS opsrag_runbooks_touch ON opsrag_runbooks;
DROP FUNCTION IF EXISTS opsrag_runbooks_touch_updated_at();
DROP TABLE IF EXISTS opsrag_runbook_versions;
DROP TABLE IF EXISTS opsrag_runbooks;
