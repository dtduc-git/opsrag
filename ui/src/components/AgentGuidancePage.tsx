import { useEffect, useState } from "react";
import { fetchAgentGuidance, saveAgentGuidance, type AgentGuidance } from "../api";

/**
 * Agent Guidance — deployment-wide custom instructions (think CLAUDE.md):
 * free-text rules the agent ALWAYS honors, injected into both RAG answers and
 * chat. Edited live here; takes effect on the next query (no redeploy).
 */
export default function AgentGuidancePage() {
  const [text, setText] = useState("");
  const [orig, setOrig] = useState("");
  const [meta, setMeta] = useState<AgentGuidance | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState<{ kind: "ok" | "err"; msg: string } | null>(null);

  useEffect(() => {
    let alive = true;
    fetchAgentGuidance()
      .then((g) => { if (!alive) return; setText(g.custom_instructions || ""); setOrig(g.custom_instructions || ""); setMeta(g); })
      .catch((e) => { if (alive) setStatus({ kind: "err", msg: String(e.message || e) }); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, []);

  const dirty = text !== orig;

  async function onSave() {
    setSaving(true);
    setStatus(null);
    try {
      const g = await saveAgentGuidance(text);
      setOrig(g.custom_instructions || "");
      setMeta(g);
      setStatus({ kind: "ok", msg: "Saved — takes effect on the next query (no restart)." });
    } catch (e: any) {
      setStatus({ kind: "err", msg: String(e.message || e) });
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="page">
      <p style={{ color: "var(--fg-dim)", fontSize: 13, marginBottom: 16, lineHeight: 1.5 }}>
        Deployment-wide instructions the agent <strong>always</strong> follows — injected into
        both grounded answers and chat (like a <code>CLAUDE.md</code> for OpsRAG). Use it for
        org conventions, escalation / on-call policy, edge-case rules, or tone. Saved here it
        applies on the <strong>next query</strong> — no redeploy. Per-question facts still come
        from your indexed sources; this is the always-on layer on top.
      </p>

      <div className="panel" style={{ border: "1px solid var(--line)", borderRadius: "var(--r-md, 10px)" }}>
        <div className="panel-head" style={{ marginBottom: 10 }}>
          <div>
            <div className="panel-title">Custom instructions</div>
            <div className="panel-sub">
              {loading ? "loading…" : meta?.source === "db"
                ? `live · ${meta?.updated_by ? "updated by " + meta.updated_by : "saved"}${meta?.updated_at ? " · " + new Date(meta.updated_at).toLocaleString() : ""}`
                : meta?.source === "config" ? "from config seed (deployment.custom_instructions) — save to make it live-editable"
                : "not set yet"}
            </div>
          </div>
        </div>

        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={loading}
          placeholder={"e.g.\nWhen the question is about a production incident, remind the user to page the on-call via /oncall in Slack before deep-diving.\nPrefer concise, runbook-style answers with concrete commands.\nNever recommend a hard delete without a backup step."}
          spellCheck={false}
          style={{
            width: "100%", minHeight: 320, resize: "vertical",
            background: "var(--bg-2)", color: "var(--fg)",
            border: "1px solid var(--line)", borderRadius: 8,
            padding: "12px 14px", fontFamily: "var(--mono)", fontSize: 13, lineHeight: 1.55,
          }}
        />

        <div style={{ display: "flex", alignItems: "center", gap: 14, marginTop: 12 }}>
          <button className="btn-primary" onClick={onSave} disabled={!dirty || saving || loading}>
            {saving ? "Saving…" : "Save guidance"}
          </button>
          {dirty && !saving && (
            <button className="btn secondary" onClick={() => setText(orig)}>Revert</button>
          )}
          <span style={{ fontSize: 12, color: status?.kind === "err" ? "var(--red)" : "var(--fg-mute)" }}>
            {status?.msg || (dirty ? "unsaved changes" : "")}
          </span>
          <span style={{ marginLeft: "auto", fontSize: 11, color: "var(--fg-mute)", fontFamily: "var(--mono)" }}>
            {text.length.toLocaleString()} chars
          </span>
        </div>
      </div>
    </div>
  );
}
