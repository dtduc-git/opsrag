ALTER TABLE opsrag_auth_user
  DROP COLUMN IF EXISTS connectors_allow,
  DROP COLUMN IF EXISTS connectors_deny;
