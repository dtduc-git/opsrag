import { useState, useEffect, useCallback } from "react";
import {
  fetchRecentFeedback,
  fetchCorrections,
  type FeedbackListItem,
  type CorrectionListItem,
  type MeResponse,
} from "../api";

// NOTE: This page surfaces every team member's feedback + corrections for
// now. Authorization (admin-gating) is a separate upcoming task — once it
// lands this page should be restricted to admins via `me.is_admin`.

interface Props {
  me?: MeResponse | null;
}

type Tab = "attention" | "positive" | "corrections";

// created_at arrives as an ISO string for feedback rows and as an epoch
// number (seconds or millis) for correction chunks. Handle both defensively
// and never throw on null / malformed input.
function fmtTimestamp(value: string | number | null): string {
  if (value == null) return "—";
  try {
    let d: Date;
    if (typeof value === "number") {
      // Heuristic: epoch seconds are ~1e9–1e10; millis are ~1e12+.
      const ms = value < 1e12 ? value * 1000 : value;
      d = new Date(ms);
    } else {
      d = new Date(value);
    }
    if (isNaN(d.getTime())) return typeof value === "string" ? value : "—";
    return d.toLocaleString();
  } catch {
    return typeof value === "string" ? value : "—";
  }
}

// evidence_url is user-submitted (via the correction form), so it must not be
// rendered into an href without scheme validation -- a `javascript:` URL would
// execute on click (stored XSS). Returns the URL only when it's http(s).
function safeHttpUrl(raw: string | null | undefined): string | null {
  if (!raw) return null;
  try {
    const u = new URL(raw);
    return u.protocol === "http:" || u.protocol === "https:" ? raw : null;
  } catch {
    return null;
  }
}

export default function RetrievalQualityPage({ me = null }: Props) {
  // `me` is accepted for future admin-gating; intentionally not yet used to
  // restrict rendering. Reference it so it isn't flagged as unused.
  void me;

  const [up, setUp] = useState<FeedbackListItem[]>([]);
  const [down, setDown] = useState<FeedbackListItem[]>([]);
  const [corrections, setCorrections] = useState<CorrectionListItem[]>([]);
  const [tab, setTab] = useState<Tab>("attention");
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    // Each fetch tolerates its own failure (returns []) at the api layer,
    // but we guard again here so one rejected promise never blanks the page.
    const [upRows, downRows, corrRows] = await Promise.all([
      fetchRecentFeedback(1, 50).catch(() => [] as FeedbackListItem[]),
      fetchRecentFeedback(-1, 50).catch(() => [] as FeedbackListItem[]),
      fetchCorrections(50).catch(() => [] as CorrectionListItem[]),
    ]);
    setUp(upRows);
    setDown(downRows);
    setCorrections(corrRows);
    setLoading(false);
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const upCount = up.length;
  const downCount = down.length;
  const denom = upCount + downCount;
  const satisfaction = denom > 0 ? `${Math.round((upCount / denom) * 100)}%` : "—";

  return (
    <div className="page">
      <div className="rq-grid">
        <div className="cache-stat-card rq-stat">
          <div className="cache-stat-label">Thumbs up</div>
          <div className="cache-stat-value rq-stat-value">{upCount}</div>
        </div>
        <div className="cache-stat-card rq-stat">
          <div className="cache-stat-label">Thumbs down</div>
          <div className="cache-stat-value rq-stat-value">{downCount}</div>
        </div>
        <div className="cache-stat-card rq-stat">
          <div className="cache-stat-label">Satisfaction</div>
          <div className="cache-stat-value rq-stat-value">{satisfaction}</div>
        </div>
        <div className="cache-stat-card rq-stat">
          <div className="cache-stat-label">Corrections</div>
          <div className="cache-stat-value rq-stat-value">{corrections.length}</div>
        </div>
      </div>

      <div className="rq-toolbar">
        <div className="tab-bar">
          <button
            className={`tab-bar-item ${tab === "attention" ? "active" : ""}`}
            onClick={() => setTab("attention")}
          >
            Needs attention{downCount > 0 ? ` (${downCount})` : ""}
          </button>
          <button
            className={`tab-bar-item ${tab === "positive" ? "active" : ""}`}
            onClick={() => setTab("positive")}
          >
            Positive{upCount > 0 ? ` (${upCount})` : ""}
          </button>
          <button
            className={`tab-bar-item ${tab === "corrections" ? "active" : ""}`}
            onClick={() => setTab("corrections")}
          >
            Corrections{corrections.length > 0 ? ` (${corrections.length})` : ""}
          </button>
        </div>
        <button className="btn btn-secondary" onClick={load} disabled={loading}>
          {loading ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      {tab === "attention" && (
        <FeedbackList
          rows={down}
          emptyTitle="No thumbs-down yet"
          emptyHint="When teammates flag an answer as unhelpful, it shows up here for triage."
        />
      )}
      {tab === "positive" && (
        <FeedbackList
          rows={up}
          emptyTitle="No thumbs-up yet"
          emptyHint="Answers people found helpful will be listed here."
        />
      )}
      {tab === "corrections" && <CorrectionsList rows={corrections} />}
    </div>
  );
}

function FeedbackList({
  rows,
  emptyTitle,
  emptyHint,
}: {
  rows: FeedbackListItem[];
  emptyTitle: string;
  emptyHint: string;
}) {
  if (rows.length === 0) {
    return (
      <div className="card-section rq-empty">
        <div className="rq-empty-title">{emptyTitle}</div>
        <div className="rq-empty-hint">{emptyHint}</div>
      </div>
    );
  }

  return (
    <div className="rq-list">
      {rows.map((r) => {
        const positive = r.direction > 0;
        return (
          <div key={r.id} className="card-section rq-card">
            <div className="rq-card-head">
              <span className={`badge ${positive ? "badge-grounded" : "badge-stale"}`}>
                {positive ? "Thumbs up" : "Thumbs down"}
              </span>
              <span className="rq-meta">
                {r.user_id || "anonymous"} · {fmtTimestamp(r.created_at)}
              </span>
            </div>
            {r.query_snippet && (
              <div className="rq-field">
                <div className="rq-field-label">Query</div>
                <div className="rq-field-value">{r.query_snippet}</div>
              </div>
            )}
            {r.answer_snippet && (
              <div className="rq-field">
                <div className="rq-field-label">Answer</div>
                <div className="rq-field-value rq-muted">{r.answer_snippet}</div>
              </div>
            )}
            {r.note && (
              <div className="rq-field">
                <div className="rq-field-label">Note</div>
                <div className="rq-field-value">{r.note}</div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function CorrectionsList({ rows }: { rows: CorrectionListItem[] }) {
  if (rows.length === 0) {
    return (
      <div className="card-section rq-empty">
        <div className="rq-empty-title">No corrections yet</div>
        <div className="rq-empty-hint">
          When a user submits the correct answer to a wrong response, it is
          stored as a high-weight chunk and shown here.
        </div>
      </div>
    );
  }

  return (
    <div className="rq-list">
      {rows.map((c) => (
        <div key={c.chunk_id} className="card-section rq-card">
          <div className="rq-card-head">
            <span className="chip chip-runbook">correction</span>
            <span className="rq-meta">
              {c.user_id || "anonymous"} · {fmtTimestamp(c.created_at)}
            </span>
          </div>
          {c.original_question && (
            <div className="rq-field">
              <div className="rq-field-label">Question</div>
              <div className="rq-field-value">{c.original_question}</div>
            </div>
          )}
          {c.correct_answer && (
            <div className="rq-field">
              <div className="rq-field-label">Correct answer</div>
              <div className="rq-field-value">{c.correct_answer}</div>
            </div>
          )}
          {c.wrong_answer && (
            <div className="rq-field">
              <div className="rq-field-label">Was answered</div>
              <div className="rq-field-value rq-muted">{c.wrong_answer}</div>
            </div>
          )}
          {c.evidence_url && (
            <div className="rq-field">
              <div className="rq-field-label">Evidence</div>
              <div className="rq-field-value">
                {safeHttpUrl(c.evidence_url) ? (
                  <a href={safeHttpUrl(c.evidence_url)!} target="_blank" rel="noreferrer">
                    {c.evidence_url}
                  </a>
                ) : (
                  <span className="rq-muted">{c.evidence_url}</span>
                )}
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
