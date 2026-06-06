-- 0010 — live, operator-editable agent settings (key/value).
--
-- Backs the "Agent Guidance" admin page: deployment-wide custom instructions
-- (think CLAUDE.md / custom instructions) that are injected into the agent's
-- answer + chat system prompts and editable in the UI WITHOUT a redeploy. The
-- value here overrides the optional `deployment.custom_instructions` config
-- seed; reads are cached in-process and refreshed on a short interval so edits
-- take effect on the next query across all replicas.
--
-- Generic key/value so future operator-tunable agent settings can reuse it.
CREATE TABLE IF NOT EXISTS opsrag_agent_settings (
  key        TEXT PRIMARY KEY,
  value      TEXT NOT NULL DEFAULT '',
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_by TEXT
);
