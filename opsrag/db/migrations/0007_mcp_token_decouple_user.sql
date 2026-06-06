-- 0007 — decouple MCP token ownership from the legacy opsrag_user table.
--
-- MCP tokens were attributed to opsrag_user(oid) (the Pomerium-era user
-- table) via a FK with ON DELETE CASCADE. With first-party login, users live
-- in opsrag_auth_user instead, so a session user's id is not present in
-- opsrag_user and the FK insert fails. The token store only needs an OPAQUE
-- owner id (a UUID) that is stable per identity -- it works the same for a
-- Pomerium oid and a login user id. Drop the FK; keep the column, NOT NULL,
-- and the per-user index. (opsrag_mcp_audit.user_oid already has no FK.)
ALTER TABLE opsrag_mcp_token
  DROP CONSTRAINT IF EXISTS opsrag_mcp_token_user_oid_fkey;
