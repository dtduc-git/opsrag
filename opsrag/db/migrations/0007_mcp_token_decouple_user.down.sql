-- 0007 down — re-attach the FK to opsrag_user(oid).
-- NOTE: this will FAIL if any token rows reference owners that don't exist in
-- opsrag_user (e.g. login-mode users). Clean up such rows before rolling back.
ALTER TABLE opsrag_mcp_token
  ADD CONSTRAINT opsrag_mcp_token_user_oid_fkey
  FOREIGN KEY (user_oid) REFERENCES opsrag_user(oid) ON DELETE CASCADE;
