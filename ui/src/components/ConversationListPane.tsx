import { useMemo, useState } from "react";
import type { Session } from "../api";

interface Props {
  sessions: Session[];
  activeThread: string | null;
  onSelect: (threadId: string) => void;
  onNew: () => void;
  onDelete: (threadId: string) => void;
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
          <button className="mdl-empty-cta" type="button" onClick={onNew}>
            Start a conversation
          </button>
        </div>
      ) : filtered.length === 0 ? (
        <div className="mdl-empty">
          <div>No conversations match.</div>
        </div>
      ) : (
        <div className="mdl-list">
          {filtered.map((s) => {
            const title = s.title || `Conversation ${s.thread_id.slice(0, 8)}`;
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
                <span className="mdl-title">{title}</span>
                {s.preview ? <span className="mdl-preview">{s.preview}</span> : null}
                <span className="mdl-meta">{meta}</span>
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
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
