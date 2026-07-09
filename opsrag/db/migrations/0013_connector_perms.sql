-- Per-connector RBAC overrides on the first-party auth user.
--
-- connectors_allow[] / connectors_deny[] are per-user grants/denials layered on
-- top of role-derived connector access (auth.role_connectors) and the
-- `restricted` flag on each mcp.<name> block. See opsrag.auth.connector_perms:
--   * allow  -> grant this connector to the user (even a restricted one).
--   * deny   -> forbid this connector for the user; wins over role grants,
--               default-allow, and admin.
-- Empty (the default) means "no per-user override" -> access follows roles.
ALTER TABLE opsrag_auth_user
  ADD COLUMN IF NOT EXISTS connectors_allow TEXT[] NOT NULL DEFAULT '{}',
  ADD COLUMN IF NOT EXISTS connectors_deny  TEXT[] NOT NULL DEFAULT '{}';
