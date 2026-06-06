-- 0012 — moderation queue for user corrections.
--
-- Before this, POST /correction injected a 2.5x-boosted synthetic chunk
-- straight into the live retrieval collection with user_id='anonymous' and no
-- approval step -- any caller could poison answers for everyone, instantly and
-- globally. Corrections now land here as `pending` and are invisible to
-- retrieval; an operator must approve one before it is injected into Qdrant
-- (with a reduced boost). Reject/approve transitions are terminal.
CREATE TABLE IF NOT EXISTS opsrag_pending_corrections (
    id              BIGSERIAL PRIMARY KEY,
    question        TEXT NOT NULL,
    wrong_answer    TEXT NOT NULL DEFAULT '',
    correct_answer  TEXT NOT NULL,
    evidence_url    TEXT,
    user_id         TEXT,
    thread_id       TEXT,
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending | approved | rejected
    chunk_id        TEXT,                             -- set once approved+injected
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reviewed_at     TIMESTAMPTZ,
    reviewed_by     TEXT
);

-- Cheap "what's awaiting review" view -- the moderation list only ever wants
-- pending rows, newest first.
CREATE INDEX IF NOT EXISTS idx_opsrag_pending_corrections_pending
    ON opsrag_pending_corrections(created_at)
    WHERE status = 'pending';
