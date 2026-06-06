-- 0003 — MCP token store + per-call audit.
CREATE TABLE IF NOT EXISTS opsrag_mcp_token (
  id           UUID PRIMARY KEY,
  user_oid     UUID NOT NULL REFERENCES opsrag_user(oid) ON DELETE CASCADE,
  token_sha256 BYTEA NOT NULL UNIQUE,
  name         TEXT NOT NULL,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at   TIMESTAMPTZ,
  revoked_at   TIMESTAMPTZ,
  last_used_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS opsrag_mcp_token_active
  ON opsrag_mcp_token(token_sha256)
  WHERE revoked_at IS NULL;
CREATE INDEX IF NOT EXISTS opsrag_mcp_token_user_recent
  ON opsrag_mcp_token(user_oid, created_at DESC);

CREATE TABLE IF NOT EXISTS opsrag_mcp_audit (
  id          BIGSERIAL PRIMARY KEY,
  occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  user_oid    UUID,
  token_id    UUID,
  tool_name   TEXT NOT NULL,
  args_hash   TEXT,
  latency_ms  INTEGER,
  status      TEXT NOT NULL,
  error       TEXT
);
CREATE INDEX IF NOT EXISTS opsrag_mcp_audit_user_ts
  ON opsrag_mcp_audit(user_oid, occurred_at DESC);
CREATE INDEX IF NOT EXISTS opsrag_mcp_audit_tool_ts
  ON opsrag_mcp_audit(tool_name, occurred_at DESC);
