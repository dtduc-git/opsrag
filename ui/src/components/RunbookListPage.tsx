import { useEffect, useState } from "react";
import {
  listRunbooks,
  deleteRunbook,
  RUNBOOK_FAILURE_CLASSES,
  failureClassLabel,
  type Runbook,
  type RunbookFailureClass,
} from "../api";

interface Props {
  onEdit: (id: string | null) => void;
}

/** Runbook list page — shows hand-authored runbooks with filter + actions.
 *
 * "+ New runbook" → onEdit(null) opens the editor on a blank form.
 * Row click → onEdit(id) opens the editor pre-filled. */
export function RunbookListPage({ onEdit }: Props) {
  const [items, setItems] = useState<Runbook[]>([]);
  const [serviceFilter, setServiceFilter] = useState<string>("");
  const [kindFilter, setKindFilter] = useState<RunbookFailureClass | "">("");
  const [showDisabled, setShowDisabled] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await listRunbooks({
        service: serviceFilter.trim() || undefined,
        issue_kind: kindFilter || undefined,
        enabled_only: !showDisabled,
        limit: 200,
      });
      setItems(res.runbooks);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [serviceFilter, kindFilter, showDisabled]);

  const handleDelete = async (rb: Runbook) => {
    if (!confirm(`Soft-delete runbook "${rb.title}"? (Can be restored from version history.)`)) return;
    try {
      await deleteRunbook(rb.id);
      await refresh();
    } catch (e) {
      alert(`Delete failed: ${e instanceof Error ? e.message : e}`);
    }
  };

  return (
    <div className="runbook-list">
      <div className="runbook-list__filters">
        <input
          type="text"
          placeholder="Filter by service (e.g. my-service)"
          value={serviceFilter}
          onChange={(e) => setServiceFilter(e.target.value)}
        />
        <select value={kindFilter} onChange={(e) => setKindFilter(e.target.value as RunbookFailureClass | "")}>
          <option value="">All failure classes</option>
          {RUNBOOK_FAILURE_CLASSES.map((c) => (
            <option key={c} value={c}>{failureClassLabel(c)}</option>
          ))}
        </select>
        <label className="runbook-list__checkbox">
          <input
            type="checkbox"
            checked={showDisabled}
            onChange={(e) => setShowDisabled(e.target.checked)}
          />
          Show soft-deleted
        </label>
        <button className="btn-ghost" onClick={refresh} disabled={loading}>
          {loading ? "Loading…" : "↻ Refresh"}
        </button>
      </div>

      {error && <div className="runbook-list__error">{error}</div>}

      {!loading && items.length === 0 && !error && (
        <div className="runbook-list__empty">
          <p>No runbooks yet.</p>
          <p>Click <b>+ New runbook</b> to author one — they take priority over RAG-indexed runbooks during Investigate-mode root-cause analysis.</p>
        </div>
      )}

      <table className="runbook-list__table">
        <thead>
          <tr>
            <th>Title</th>
            <th>Service</th>
            <th>Issue kind</th>
            <th>Sev floor</th>
            <th>Priority</th>
            <th>Source</th>
            <th>Used / 👍 / 👎</th>
            <th>Updated</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {items.map((rb) => (
            <tr key={rb.id} className={!rb.enabled ? "row--disabled" : ""}>
              <td>
                <a className="link" onClick={() => onEdit(rb.id)}>{rb.title}</a>
                {rb.tags.length > 0 && (
                  <div className="runbook-tags">
                    {rb.tags.map((t) => <span key={t} className="runbook-tag">{t}</span>)}
                  </div>
                )}
              </td>
              <td>{rb.service || <span className="muted">—</span>}</td>
              <td>{rb.issue_kind ? failureClassLabel(rb.issue_kind) : <span className="muted">—</span>}</td>
              <td>{rb.severity_min || <span className="muted">—</span>}</td>
              <td>{rb.priority}</td>
              <td>
                <span className={`source-badge source-badge--${rb.source}`}>{rb.source}</span>
              </td>
              <td>
                <span className="muted">{rb.used_count}</span>
                {" / "}
                <span style={{ color: "var(--accent)" }}>{rb.thumbs_up_count}</span>
                {" / "}
                <span style={{ color: "var(--danger, #c33)" }}>{rb.thumbs_down_count}</span>
              </td>
              <td><span className="muted">{new Date(rb.updated_at).toLocaleString()}</span></td>
              <td>
                <div className="runbook-list__row-actions">
                  <button className="rb-btn rb-btn--edit" onClick={() => onEdit(rb.id)}>Edit</button>
                  <button className="rb-btn rb-btn--delete" onClick={() => handleDelete(rb)}>Delete</button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
