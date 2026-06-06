-- 0001 — opsrag_user: identity table populated from Pomerium-forwarded claims.
CREATE TABLE IF NOT EXISTS opsrag_user (
  oid          UUID PRIMARY KEY,
  email        TEXT NOT NULL,
  display_name TEXT,
  picture_url  TEXT,
  groups       TEXT[] NOT NULL DEFAULT '{}',
  first_seen   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  last_seen    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS opsrag_user_email ON opsrag_user(email);
CREATE INDEX IF NOT EXISTS opsrag_user_last_seen ON opsrag_user(last_seen DESC);
