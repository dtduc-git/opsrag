import { useCallback, useEffect, useState } from "react";

import {
  fetchMcpAudit,
  fetchMcpAuditSummary,
  type McpAuditRow,
  type McpAuditSummary,
  type MeResponse,
} from "../api";
import { IconBolt, IconClock, IconDatabase, IconHash, IconKey, IconMessage } from "./icons";

const PAGE_SIZE = 100;

const WINDOWS: { label: string; minutes: number | undefined }[] = [
  { label: "Last hour", minutes: 60 },
  { label: "Last 24h", minutes: 60 * 24 },
  { label: "Last 7 days", minutes: 60 * 24 * 7 },
  { label: "All time", minutes: undefined },
];

function fmt(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function fmtTs(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return isNaN(d.getTime()) ? iso : d.toLocaleString();
}

function shortId(s: string | null): string {
  if (!s) return "—";
  return s.length > 10 ? `${s.slice(0, 8)}…` : s;
}

function StatusPill({ status }: { status: string }) {
  const color =
    status === "ok" ? "#16a34a" : status === "denied" ? "#d97706" : "#dc2626";
  return (
    <span
      style={{
        color,
        fontWeight: 600,
        fontSize: "0.78rem",
        textTransform: "uppercase",
        letterSpacing: "0.03em",
      }}
    >
      {status}
    </span>
  );
}

export default function MCPAuditPage({ me }: { me: MeResponse }) {
  const [summary, setSummary] = useState<McpAuditSummary | null>(null);
  const [rows, setRows] = useState<McpAuditRow[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);

  // filters
  const [tool, setTool] = useState("");
  const [user, setUser] = useState("");
  const [status, setStatus] = useState("");
  const [windowMin, setWindowMin] = useState<number | undefined>(60 * 24);

  const load = useCallback(
    async (reset: boolean) => {
      setLoading(true);
      const nextOffset = reset ? 0 : offset;
      try {
        const list = await fetchMcpAudit({
          tool: tool || undefined,
          user: user || undefined,
          status: status || undefined,
          sinceMinutes: windowMin,
          limit: PAGE_SIZE,
          offset: nextOffset,
        });
        setRows((prev) => (reset ? list.rows : [...prev, ...list.rows]));
        setTotal(list.total);
        setOffset(nextOffset + list.rows.length);
        if (reset) {
          fetchMcpAuditSummary(windowMin).then(setSummary).catch(() => setSummary(null));
        }
      } finally {
        setLoading(false);
      }
    },
    [tool, user, status, windowMin, offset],
  );

  // Reload from the top whenever a filter changes.
  useEffect(() => {
    load(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tool, user, status, windowMin]);

  if (!me?.is_admin) {
    return (
      <div className="page">
        <div className="page-subtitle">Admin only — you do not have access to the MCP audit log.</div>
      </div>
    );
  }

  return (
    <div className="page">
      <div className="page-title">Tool-call activity</div>
      <div className="page-subtitle">
        Filter by tool, user, status, or time window. Arguments are shown only as a
        sha256 <code>args_hash</code>.
      </div>

      {summary && (
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-icon"><IconMessage /></div>
            <div className="stat-label">Total calls</div>
            <div className="stat-value">{fmt(summary.total_calls)}</div>
          </div>
          <div className="stat-card">
            <div className="stat-icon"><IconBolt /></div>
            <div className="stat-label">Errors</div>
            <div className="stat-value">{fmt(summary.error_count)}</div>
            <div className="stat-sub">{fmt(summary.denied_count)} denied (rate-limit / unknown)</div>
          </div>
          <div className="stat-card">
            <div className="stat-icon"><IconKey /></div>
            <div className="stat-label">Distinct users</div>
            <div className="stat-value">{fmt(summary.distinct_users)}</div>
          </div>
          <div className="stat-card">
            <div className="stat-icon"><IconDatabase /></div>
            <div className="stat-label">Distinct tools</div>
            <div className="stat-value">{fmt(summary.distinct_tools)}</div>
            <div className="stat-sub">
              top: {summary.top_tools.slice(0, 3).map((t) => t.tool_name).join(", ") || "—"}
            </div>
          </div>
        </div>
      )}

      <div className="card-section">
      <div className="card-section-title" style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <span>Tool calls</span>
        <span style={{ color: "var(--fg-dim)", fontWeight: 500 }}>{total} total</span>
      </div>
      <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap", alignItems: "flex-end", margin: "0 0 1rem" }}>
        <div className="field" style={{ minWidth: 240, flex: "1 1 240px" }}>
          <label>Tool</label>
          <input
            placeholder="e.g. datadog_search_spans"
            value={tool}
            onChange={(e) => setTool(e.target.value)}
          />
        </div>
        <div className="field" style={{ minWidth: 180, flex: "1 1 180px" }}>
          <label>User</label>
          <input placeholder="user oid" value={user} onChange={(e) => setUser(e.target.value)} />
        </div>
        <div className="field" style={{ minWidth: 130 }}>
          <label>Status</label>
          <select value={status} onChange={(e) => setStatus(e.target.value)}>
            <option value="">all</option>
            <option value="ok">ok</option>
            <option value="denied">denied</option>
            <option value="error">error</option>
          </select>
        </div>
        <div className="field" style={{ minWidth: 130 }}>
          <label>Window</label>
          <select
            value={windowMin === undefined ? "" : String(windowMin)}
            onChange={(e) => setWindowMin(e.target.value === "" ? undefined : Number(e.target.value))}
          >
            {WINDOWS.map((w) => (
              <option key={w.label} value={w.minutes === undefined ? "" : String(w.minutes)}>
                {w.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <table className="tbl">
        <thead>
          <tr>
            <th>Time</th>
            <th>User</th>
            <th>Token</th>
            <th>Tool</th>
            <th>Status</th>
            <th style={{ textAlign: "right" }}>Latency</th>
            <th>Args hash</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              <td style={{ whiteSpace: "nowrap" }}>
                <IconClock /> {fmtTs(r.occurred_at)}
              </td>
              <td title={r.user_oid ?? ""}>{shortId(r.user_oid)}</td>
              <td title={r.token_id ?? ""}>{shortId(r.token_id)}</td>
              <td><code>{r.tool_name}</code></td>
              <td><StatusPill status={r.status} /></td>
              <td style={{ textAlign: "right" }}>{r.latency_ms != null ? `${r.latency_ms} ms` : "—"}</td>
              <td title={r.error ?? r.args_hash ?? ""}>
                <code className="dim"><IconHash /> {r.args_hash ? r.args_hash.slice(0, 10) : "—"}</code>
              </td>
            </tr>
          ))}
          {rows.length === 0 && !loading && (
            <tr>
              <td colSpan={7} className="dim" style={{ textAlign: "center", padding: "2rem" }}>
                No MCP tool calls in this window.
              </td>
            </tr>
          )}
        </tbody>
      </table>

      <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginTop: "0.75rem" }}>
        <span className="dim">{rows.length} of {total} shown</span>
        {rows.length < total && (
          <button className="btn-secondary" disabled={loading} onClick={() => load(false)}>
            {loading ? "Loading…" : "Load more"}
          </button>
        )}
      </div>
      </div>
    </div>
  );
}
