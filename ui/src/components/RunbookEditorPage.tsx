import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  getRunbook,
  createRunbook,
  updateRunbook,
  fetchRunbookVersions,
  RUNBOOK_FAILURE_CLASSES,
  failureClassLabel,
  RUNBOOK_SEVERITIES,
  type Runbook,
  type RunbookVersion,
  type RunbookFailureClass,
  type RunbookSeverity,
} from "../api";

interface Props {
  /** When null → blank editor (create new). When set → load + edit existing. */
  runbookId: string | null;
  /** Optional pre-filled markdown body (used by "Save as runbook" CTA from
   *  a completed investigation → Pro generator). */
  initialDraft?: {
    title?: string;
    body?: string;
    service?: string;
    issue_kind?: string;
    source_investigation_id?: string;
  };
  onSaved: (rb: Runbook) => void;
  onCancel: () => void;
}

const TEMPLATE_BODY = `# What's Happening?

## Description
- What is being observed (symptoms, errors, alerts, logs)
- When it started and how it was detected

> **Example:** API latency increased from 200 ms to 2 s at 14:05 UTC (Datadog alert \`api.p95_latency\`).

## Impact

### Who/What is Affected
- Users, systems, or regions impacted
- % of traffic or # of users affected
- Business effect (e.g. checkout unavailable)

> **Example:** ~40% of EU users can't log in (\`/auth\` endpoints return 500s).

### Severity
- SEV1 — Full outage
- SEV2 — Major degradation
- SEV3 — Partial failure
- SEV4 — Minor / alert only

# Why?

## Possible Causes
- Recent deploy or config change
- Dependency outage
- Resource exhaustion (CPU, memory, rate limits)

# How to Troubleshoot / Solve
1. **Confirm the issue** — check dashboards and logs to verify scope and symptoms.
2. **Check recent changes** — identify any deploys or config updates; roll back if needed.
3. **Validate dependencies** — ensure databases, APIs, and queues are healthy.
4. **Mitigate** — restart services, clear caches, or scale resources as applicable.
5. **Verify recovery** — monitor metrics, test endpoints, and confirm normal operation.

# Related Links
- Dashboards:
- Alerts:
- Logs:
- Postmortem:
- Incident:

# Notifications
- **Slack:** \`#oncall\`, \`#infrastructure-alerts\`
- **PagerDuty:**
- **Incident lead:** @username
- **Stakeholder updates:** \`#status-updates\`
`;

export function RunbookEditorPage({ runbookId, initialDraft, onSaved, onCancel }: Props) {
  const isEditing = Boolean(runbookId);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Form state
  const [title, setTitle] = useState("");
  const [service, setService] = useState("");
  const [issueKind, setIssueKind] = useState<RunbookFailureClass | "">("");
  const [severityMin, setSeverityMin] = useState<RunbookSeverity | "">("");
  const [priority, setPriority] = useState<number>(100);
  const [tags, setTags] = useState<string>("");
  const [body, setBody] = useState<string>(TEMPLATE_BODY);
  const [changeNote, setChangeNote] = useState<string>("");

  // Tabs
  const [activeTab, setActiveTab] = useState<"edit" | "preview" | "versions">("edit");
  const [versions, setVersions] = useState<RunbookVersion[]>([]);

  useEffect(() => {
    if (!runbookId) {
      // Create mode — apply initialDraft (from "Save as runbook" CTA in
      // either Chat or Investigate page). If no prop was passed, check
      // sessionStorage for a stashed draft (Investigate page navigates
      // via window.location.hash so React props don't carry across).
      let draft = initialDraft;
      if (!draft) {
        try {
          const stashed = sessionStorage.getItem("opsrag-runbook-draft");
          if (stashed) {
            draft = JSON.parse(stashed);
            // Consume — don't repopulate on subsequent visits.
            sessionStorage.removeItem("opsrag-runbook-draft");
          }
        } catch { /* malformed stash — fall back to template */ }
      }
      if (draft) {
        if (draft.title) setTitle(draft.title);
        if (draft.body) setBody(draft.body);
        if (draft.service) setService(draft.service);
        if (draft.issue_kind && (RUNBOOK_FAILURE_CLASSES as readonly string[]).includes(draft.issue_kind))
          setIssueKind(draft.issue_kind as RunbookFailureClass);
      }
      return;
    }
    // Edit mode — load existing runbook.
    setLoading(true);
    getRunbook(runbookId)
      .then((rb: Runbook) => {
        setTitle(rb.title);
        setService(rb.service || "");
        setIssueKind(rb.issue_kind || "");
        setSeverityMin(rb.severity_min || "");
        setPriority(rb.priority);
        setTags((rb.tags || []).join(", "));
        setBody(rb.body_markdown);
        setError(null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runbookId]);

  const loadVersions = async () => {
    if (!runbookId) return;
    try {
      const vs = await fetchRunbookVersions(runbookId);
      setVersions(vs);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const handleSave = async () => {
    if (!title.trim()) {
      setError("Title is required.");
      return;
    }
    if (!body.trim()) {
      setError("Body is required.");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const payload = {
        title: title.trim(),
        body_markdown: body,
        service: service.trim() || null,
        issue_kind: (issueKind || null) as RunbookFailureClass | null,
        severity_min: (severityMin || null) as RunbookSeverity | null,
        priority,
        tags: tags
          .split(",")
          .map((t) => t.trim())
          .filter(Boolean),
      };
      const saved = isEditing
        ? await updateRunbook(runbookId!, { ...payload, change_note: changeNote || undefined })
        : await createRunbook(payload);
      onSaved(saved);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  if (loading) return <div className="runbook-editor"><p>Loading…</p></div>;

  return (
    <div className="runbook-editor">
      <div className="runbook-editor__header">
        <h1>{isEditing ? "Edit runbook" : "New runbook"}</h1>
        <div className="runbook-editor__header-actions">
          <button className="btn-ghost" onClick={onCancel}>Cancel</button>
          <button className="btn-primary" onClick={handleSave} disabled={saving}>
            {saving ? "Saving…" : "Save"}
          </button>
        </div>
      </div>

      {error && <div className="runbook-editor__error">{error}</div>}
      {initialDraft?.source_investigation_id && (
        <div className="runbook-editor__provenance">
          Drafted by Pro from investigation <code>{initialDraft.source_investigation_id}</code> — review and edit before saving.
        </div>
      )}

      <div className="runbook-editor__form">
        <div className="form-row">
          <label>Title<span className="required">*</span></label>
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="e.g. my-service SSO 503 outage"
            maxLength={200}
          />
        </div>
        <div className="form-row form-row--inline">
          <div>
            <label>Service</label>
            <input
              type="text"
              value={service}
              onChange={(e) => setService(e.target.value)}
              placeholder="my-service"
              maxLength={80}
            />
          </div>
          <div>
            <label>Issue kind</label>
            <select value={issueKind} onChange={(e) => setIssueKind(e.target.value as RunbookFailureClass | "")}>
              <option value="">—</option>
              {RUNBOOK_FAILURE_CLASSES.map((c) => (
                <option key={c} value={c}>{failureClassLabel(c)}</option>
              ))}
            </select>
          </div>
          <div>
            <label>Severity floor</label>
            <select value={severityMin} onChange={(e) => setSeverityMin(e.target.value as RunbookSeverity | "")}>
              <option value="">—</option>
              {RUNBOOK_SEVERITIES.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>
          <div>
            <label>Priority</label>
            <input
              type="number"
              value={priority}
              min={0}
              max={10000}
              onChange={(e) => setPriority(parseInt(e.target.value, 10) || 100)}
              title="Higher = retrieved/weighted higher. Default 100. Hand-authored ALWAYS rank above RAG."
            />
          </div>
        </div>
        <div className="form-row">
          <label>Tags (comma-separated)</label>
          <input
            type="text"
            value={tags}
            onChange={(e) => setTags(e.target.value)}
            placeholder="sso, login, payments, cloudsql"
          />
        </div>
        {isEditing && (
          <div className="form-row">
            <label>Change note (optional)</label>
            <input
              type="text"
              value={changeNote}
              onChange={(e) => setChangeNote(e.target.value)}
              placeholder="why this edit (shown in version history)"
              maxLength={500}
            />
          </div>
        )}

        <div className="runbook-editor__tabs">
          <button className={`tab ${activeTab === "edit" ? "tab--active" : ""}`} onClick={() => setActiveTab("edit")}>
            Edit
          </button>
          <button className={`tab ${activeTab === "preview" ? "tab--active" : ""}`} onClick={() => setActiveTab("preview")}>
            Preview
          </button>
          {isEditing && (
            <button
              className={`tab ${activeTab === "versions" ? "tab--active" : ""}`}
              onClick={() => { setActiveTab("versions"); loadVersions(); }}
            >
              Version history
            </button>
          )}
        </div>

        {activeTab === "edit" && (
          <textarea
            className="runbook-editor__textarea"
            value={body}
            onChange={(e) => setBody(e.target.value)}
            spellCheck={false}
          />
        )}
        {activeTab === "preview" && (
          <div className="runbook-editor__preview markdown-body">
            {/* remark-gfm: tables / task-lists / strikethrough — same as
                ChatMessage; without it runbook verdict tables render as raw
                pipe text. */}
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{body || "_(empty)_"}</ReactMarkdown>
          </div>
        )}
        {activeTab === "versions" && (
          <div className="runbook-editor__versions">
            {versions.length === 0 && <p>No versions yet.</p>}
            {versions.map((v) => (
              <div key={v.id} className="runbook-version">
                <div className="runbook-version__header">
                  <b>v{v.version_num}</b>
                  <span className="muted">
                    {new Date(v.edited_at).toLocaleString()} by {v.edited_by || "unknown"}
                  </span>
                  {v.change_note && <span className="runbook-version__note">"{v.change_note}"</span>}
                </div>
                <div className="runbook-version__meta">
                  <span>service: {v.service || "—"}</span>
                  <span>issue_kind: {v.issue_kind || "—"}</span>
                  <span>sev: {v.severity_min || "—"}</span>
                  <span>priority: {v.priority ?? "—"}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
