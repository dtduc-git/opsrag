-- 0003 DOWN — drop the MCP token + audit tables.
-- DROP order matters: audit references no other tables, but token has
-- a FK to opsrag_user (ON DELETE CASCADE). DROP audit first then
-- token so a stray DELETE during rollback isn't accidentally noisy.
DROP INDEX IF EXISTS opsrag_mcp_audit_tool_ts;
DROP INDEX IF EXISTS opsrag_mcp_audit_user_ts;
DROP TABLE IF EXISTS opsrag_mcp_audit;

DROP INDEX IF EXISTS opsrag_mcp_token_user_recent;
DROP INDEX IF EXISTS opsrag_mcp_token_active;
DROP TABLE IF EXISTS opsrag_mcp_token;
