-- 0006 — first-party login: users, federated identities, refresh sessions.
-- Separate from opsrag_user (oid-keyed analytics/usage table): this holds
-- CREDENTIALS and login identities for auth.mode='login' (password + SSO).

-- Local login accounts. password_hash is NULL for SSO-only accounts.
-- roles[] is the operator-assigned role set (layered on top of any
-- IdP-group-derived roles resolved at request time via opsrag.auth.scopes).
CREATE TABLE IF NOT EXISTS opsrag_auth_user (
  id             UUID PRIMARY KEY,
  email          TEXT NOT NULL UNIQUE,
  email_verified BOOLEAN NOT NULL DEFAULT FALSE,
  password_hash  TEXT,
  roles          TEXT[] NOT NULL DEFAULT '{}',
  name           TEXT,
  picture_url    TEXT,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Federated (SSO) identity links. (provider, subject) is the external
-- IdP account; unique so one IdP account maps to exactly one local user.
-- email/email_verified capture what the IdP asserted at link time (the
-- account-takeover guard requires email_verified before linking by email).
CREATE TABLE IF NOT EXISTS opsrag_auth_identity (
  provider       TEXT NOT NULL,
  subject        TEXT NOT NULL,
  user_id        UUID NOT NULL REFERENCES opsrag_auth_user(id) ON DELETE CASCADE,
  email          TEXT,
  email_verified BOOLEAN NOT NULL DEFAULT FALSE,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (provider, subject)
);
CREATE INDEX IF NOT EXISTS opsrag_auth_identity_user
  ON opsrag_auth_identity(user_id);

-- Rotating refresh sessions. token_hash is the SHA-256 of the opaque
-- refresh token (the raw token is NEVER stored). revoked_at tombstones a
-- rotated/logged-out token; expires_at bounds its lifetime.
CREATE TABLE IF NOT EXISTS opsrag_auth_refresh_session (
  id          UUID PRIMARY KEY,
  user_id     UUID NOT NULL REFERENCES opsrag_auth_user(id) ON DELETE CASCADE,
  token_hash  TEXT NOT NULL UNIQUE,
  expires_at  TIMESTAMPTZ NOT NULL,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  revoked_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS opsrag_auth_refresh_active
  ON opsrag_auth_refresh_session(token_hash)
  WHERE revoked_at IS NULL;
CREATE INDEX IF NOT EXISTS opsrag_auth_refresh_user
  ON opsrag_auth_refresh_session(user_id, created_at DESC);
