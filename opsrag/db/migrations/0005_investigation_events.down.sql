-- 0005 down — drop investigation event ledger.
DROP INDEX IF EXISTS idx_inv_created_at;
DROP TABLE IF EXISTS opsrag_investigations;
DROP INDEX IF EXISTS idx_inv_events_inv_seq;
DROP TABLE IF EXISTS opsrag_investigation_events;
