-- 0001 DOWN — drop the opsrag_user table.
-- Safe because 0001 is purely additive. Any opsrag_mcp_token rows that
-- reference opsrag_user via FK will be cascade-deleted by 0003's down
-- (which is applied first when rolling back), so the FK is gone by the
-- time we get here.
DROP INDEX IF EXISTS opsrag_user_last_seen;
DROP INDEX IF EXISTS opsrag_user_email;
DROP TABLE IF EXISTS opsrag_user;
