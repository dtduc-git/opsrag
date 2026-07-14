import { useMemo, useState } from "react";
import { channelPlatformLabel, slackPermalink, normalizeTitle, type Session } from "../api";

interface Props {
  sessions: Session[];
  activeThread: string | null;
  onSelect: (threadId: string) => void;
  onNew: () => void;
  onDelete: (threadId: string) => void;
  // Read-only browse (public Channels view): hide the delete affordance and
  // the "Start a conversation" empty-state CTA. Also surfaces a per-row
  // platform badge when a conversation carries a `platform`.
  readOnly?: boolean;
  // Slack workspace base URL (from /ui-config). When set and a row's
  // thread_id is a `slack-thread:` id, a small "open in Slack" icon is
  // shown next to the title (hover-revealed via CSS).
  workspaceUrl?: string | null;
}

// Relative time from an ISO 8601 string. Null/unparseable -> "".
function ago(iso: string | null | undefined): string {
  if (!iso) return "";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "";
  const secs = Math.floor((Date.now() - t) / 1000);
  if (secs < 60) return "just now";
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days === 1) return "Yesterday";
  if (days < 7) return `${days}d ago`;
  return new Date(t).toLocaleDateString();
}

// Sort key: most-recent updated_at first; missing/unparseable sorts last.
function updatedKey(s: Session): number {
  if (!s.updated_at) return -Infinity;
  const t = Date.parse(s.updated_at);
  return Number.isNaN(t) ? -Infinity : t;
}

export default function ConversationListPane({
  sessions,
  activeThread,
  onSelect,
  onNew,
  onDelete,
  readOnly = false,
  workspaceUrl,
}: Props) {
  const [search, setSearch] = useState("");

  const sorted = useMemo(
    () => [...sessions].sort((a, b) => updatedKey(b) - updatedKey(a)),
    [sessions],
  );

  const q = search.trim().toLowerCase();
  const filtered = useMemo(() => {
    if (!q) return sorted;
    return sorted.filter((s) => {
      const hay = `${s.title ?? ""} ${s.preview ?? ""}`.toLowerCase();
      return hay.includes(q);
    });
  }, [sorted, q]);

  function handleDelete(e: React.MouseEvent, threadId: string) {
    e.stopPropagation();
    if (window.confirm("Delete this conversation? This cannot be undone.")) {
      onDelete(threadId);
    }
  }

  return (
    <div className="mdl">
      <div className="mdl-toolbar">
        <input
          className="mdl-search"
          type="text"
          placeholder="Search conversations…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      {sessions.length === 0 ? (
        <div className="mdl-empty">
          <div>No conversations yet.</div>
          {!readOnly && (
            <button className="mdl-empty-cta" type="button" onClick={onNew}>
              Start a conversation
            </button>
          )}
        </div>
      ) : filtered.length === 0 ? (
        <div className="mdl-empty">
          <div>No conversations match.</div>
        </div>
      ) : (
        <div className="mdl-list">
          {filtered.map((s) => {
            const title = normalizeTitle(s.title) || `Conversation ${s.thread_id.slice(0, 8)}`;
            const permalink = slackPermalink(s.thread_id, workspaceUrl);
            const when = ago(s.updated_at);
            const turns = s.turn_count || 0;
            const meta = `${when ? when + " · " : ""}${turns} turn${turns === 1 ? "" : "s"}`;
            return (
              <button
                key={s.thread_id}
                type="button"
                className={`mdl-row ${s.thread_id === activeThread ? "active" : ""}`}
                onClick={() => onSelect(s.thread_id)}
              >
                <span className="mdl-title-row">
                  <span className="mdl-title" title={s.title ?? undefined}>{title}</span>
                  {permalink && (
                    <span
                      className="mdl-slack-link"
                      role="link"
                      tabIndex={-1}
                      aria-label="Open Slack thread"
                      title="Open Slack thread"
                      onClick={(e) => { e.stopPropagation(); window.open(permalink, "_blank", "noopener,noreferrer"); }}
                    >
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                        <path d="M10 14 21 3M15 3h6v6M21 14v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5" />
                      </svg>
                    </span>
                  )}
                </span>
                {s.preview ? <span className="mdl-preview">{s.preview}</span> : null}
                <span className="mdl-meta">
                  {s.platform ? (
                    <span className="badge badge-type" style={{ marginRight: 6 }}>
                      {channelPlatformLabel(s.platform)}
                    </span>
                  ) : null}
                  {meta}
                </span>
                {!readOnly && (
                  <span
                    className="mdl-del"
                    role="button"
                    tabIndex={-1}
                    aria-label="Delete conversation"
                    title="Delete conversation"
                    onClick={(e) => handleDelete(e, s.thread_id)}
                  >
                    ✕
                  </span>
                )}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
