import { useState, useEffect, useMemo } from "react";
import {
  fetchUsage, fetchUsageByUser, fetchUsageMine,
  microsToUsd,
  type UsageSummary, type UsageByUser, type UsageMine, type MeResponse,
} from "../api";
import { IconMessage, IconHash, IconClock, IconCoin, IconStack, IconDatabase } from "./icons";

function fmt(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return n.toLocaleString();
}

function fmtCost(n: number): string {
  // Exactly zero (no usage / no pricing) reads as an empty cell, not "$0.0000".
  if (!n || n <= 0) return "—";
  // Tiny-but-real costs shouldn't look like zero.
  if (n < 0.01) return "<$0.01";
  // Anything a cent or more: 2 decimals (e.g. $5.05, $18.44).
  return "$" + n.toFixed(2);
}

function fmtTime(s: number): string {
  if (s < 60) return Math.round(s) + "s";
  if (s < 3600) return Math.round(s / 60) + "m";
  return (s / 3600).toFixed(1) + "h";
}

function fmtTimestamp(iso: string | null): string {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return iso;
    return d.toLocaleString();
  } catch {
    return iso;
  }
}

type Tab = "overall" | "by-user" | "mine";

interface Props {
  me?: MeResponse | null;
}

export default function UsagePage({ me = null }: Props) {
  const [data, setData] = useState<UsageSummary | null>(null);
  const [byUser, setByUser] = useState<UsageByUser[]>([]);
  const [mine, setMine] = useState<UsageMine | null>(null);
  // Non-admins only get the "Mine" tab (personal tracking); Overall + By-user
  // are org-wide (admin-only). Default the landing tab accordingly.
  const [tab, setTab] = useState<Tab>(me?.is_admin ? "overall" : "mine");

  // Sortable column state for the by-user table.
  type SortKey = "email" | "query_count" | "prompt_tokens" | "completion_tokens" | "cost_usd_micros" | "last_active_at";
  const [sortKey, setSortKey] = useState<SortKey>("cost_usd_micros");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  useEffect(() => {
    // Overall is the org-wide aggregate -> admin-only (the endpoint now 403s
    // non-admins). Don't poll it for regular users.
    if (!me?.is_admin) return;
    const loadOverall = () => fetchUsage().then(setData).catch(() => {});
    loadOverall();
    const iv = setInterval(loadOverall, 5000);
    return () => clearInterval(iv);
  }, [me?.is_admin]);

  // Per-user data only loads when the user actually opens those tabs —
  // keeps initial render cheap and avoids unnecessary 403s for non-admins.
  useEffect(() => {
    if (tab === "by-user" && me?.is_admin) {
      fetchUsageByUser().then(setByUser).catch(() => setByUser([]));
    }
  }, [tab, me?.is_admin]);

  useEffect(() => {
    if (tab === "mine" && me && !me.is_anonymous) {
      fetchUsageMine().then(setMine).catch(() => setMine(null));
    }
  }, [tab, me]);

  const sortedByUser = useMemo(() => {
    const arr = [...byUser];
    arr.sort((a, b) => {
      const av = a[sortKey] as unknown;
      const bv = b[sortKey] as unknown;
      // null/empty sorts last regardless of direction.
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      let cmp = 0;
      if (typeof av === "number" && typeof bv === "number") cmp = av - bv;
      else cmp = String(av).localeCompare(String(bv));
      return sortDir === "asc" ? cmp : -cmp;
    });
    return arr;
  }, [byUser, sortKey, sortDir]);

  const onSort = (k: SortKey) => {
    if (k === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(k);
      // Default to descending for numeric, ascending for strings.
      setSortDir(k === "email" ? "asc" : "desc");
    }
  };

  const showOverallTab = Boolean(me?.is_admin);
  const showByUserTab = Boolean(me?.is_admin);
  const showMineTab = Boolean(me && !me.is_anonymous);

  // If the active tab gets hidden underneath the user (e.g. me loads late, or
  // a non-admin lands on overall), drop to the best available tab.
  useEffect(() => {
    const fallback: Tab = showOverallTab ? "overall" : showMineTab ? "mine" : "overall";
    if (tab === "overall" && !showOverallTab) setTab(fallback);
    if (tab === "by-user" && !showByUserTab) setTab(fallback);
    if (tab === "mine" && !showMineTab) setTab(fallback);
  }, [tab, showOverallTab, showByUserTab, showMineTab]);

  return (
    <div className="page">
      <div className="page-title">Usage &amp; Cost</div>
      <div className="page-subtitle">Live telemetry — auto-refreshed every 5s. Costs split by purpose so indexing vs query spend is visible at a glance.</div>

      <div className="tab-bar">
        {showOverallTab && (
          <button
            className={`tab-bar-item ${tab === "overall" ? "active" : ""}`}
            onClick={() => setTab("overall")}
          >Overall</button>
        )}
        {showByUserTab && (
          <button
            className={`tab-bar-item ${tab === "by-user" ? "active" : ""}`}
            onClick={() => setTab("by-user")}
          >By user</button>
        )}
        {showMineTab && (
          <button
            className={`tab-bar-item ${tab === "mine" ? "active" : ""}`}
            onClick={() => setTab("mine")}
          >Mine</button>
        )}
      </div>

      {tab === "overall" && <OverallView data={data} />}
      {tab === "by-user" && (
        <ByUserView
          rows={sortedByUser}
          sortKey={sortKey}
          sortDir={sortDir}
          onSort={onSort}
        />
      )}
      {tab === "mine" && <MineView mine={mine} byUser={byUser} me={me} />}
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────────
// Overall view (preserves existing system-wide content).
// ──────────────────────────────────────────────────────────────────────

function OverallView({ data }: { data: UsageSummary | null }) {
  if (!data) return <p>Loading…</p>;

  const models = Object.entries(data.models);
  const maxTokens = Math.max(1, ...models.map(([, m]) => m.input_tokens + m.output_tokens));

  const purposes = Object.entries(data.by_purpose ?? {}).sort(
    (a, b) => b[1].estimated_cost_usd - a[1].estimated_cost_usd,
  );
  const indexingCost = data.indexing_cost_usd ?? 0;
  const queryCost = data.query_cost_usd ?? 0;
  const indexingCalls = purposes
    .filter(([, p]) => p.category === "indexing")
    .reduce((s, [, p]) => s + p.call_count, 0);
  const queryCalls = purposes
    .filter(([, p]) => p.category === "query")
    .reduce((s, [, p]) => s + p.call_count, 0);
  const purposeLabels: Record<string, string> = {
    "generation": "Answer generation",
    "query-rewrite": "Query rewrite (coreference)",
    "embed-query": "Embed query",
    "embed-index": "Embed for indexing",
    "rerank": "Reranker (Vertex)",
    "contextual-chunk": "Contextual chunking",
    "grade": "Grade documents",
    "route": "Route query",
    "hallucination-check": "Hallucination check",
    "entity-extract": "Entity extraction",
    "unknown": "Untagged calls",
  };

  return (
    <>
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon"><IconMessage /></div>
          <div className="stat-label">Total calls</div>
          <div className="stat-value">{fmt(data.total_calls)}</div>
          <div className="stat-sub">{fmtTime(data.uptime_seconds)} uptime · {data.active_sessions} active session{data.active_sessions === 1 ? "" : "s"}</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon"><IconCoin /></div>
          <div className="stat-label">Total cost</div>
          <div className="stat-value cost">{fmtCost(data.total_estimated_cost_usd)}</div>
          <div className="stat-sub">{data.total_calls > 0 ? fmtCost(data.total_estimated_cost_usd / data.total_calls) : "$0.00"} / call</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon"><IconStack /></div>
          <div className="stat-label">Indexing spend</div>
          <div className="stat-value">{fmtCost(indexingCost)}</div>
          <div className="stat-sub">{fmt(indexingCalls)} call{indexingCalls === 1 ? "" : "s"} · embed + contextual chunking</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon"><IconDatabase /></div>
          <div className="stat-label">Query spend</div>
          <div className="stat-value">{fmtCost(queryCost)}</div>
          <div className="stat-sub">{fmt(queryCalls)} call{queryCalls === 1 ? "" : "s"} · generation + rerank + grade + …</div>
        </div>
      </div>

      <div className="stats-grid" style={{ marginTop: 0 }}>
        <div className="stat-card">
          <div className="stat-icon"><IconHash /></div>
          <div className="stat-label">Input tokens</div>
          <div className="stat-value">{fmt(data.total_input_tokens)}</div>
          <div className="stat-sub">avg {data.total_calls > 0 ? Math.round(data.total_input_tokens / data.total_calls) : 0} / call</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon"><IconHash /></div>
          <div className="stat-label">Output tokens</div>
          <div className="stat-value">{fmt(data.total_output_tokens)}</div>
          <div className="stat-sub">avg {data.total_calls > 0 ? Math.round(data.total_output_tokens / data.total_calls) : 0} / call</div>
        </div>
      </div>

      {purposes.length > 0 && (
        <div className="card-section">
          <div className="card-section-title">
            <span>Cost by purpose</span>
            <span style={{ color: "var(--text-3)", fontWeight: 500, textTransform: "none", letterSpacing: 0 }}>{purposes.length} categor{purposes.length === 1 ? "y" : "ies"}</span>
          </div>
          <table className="tbl">
            <thead>
              <tr>
                <th>Purpose</th>
                <th>Category</th>
                <th>Calls</th>
                <th>Input</th>
                <th>Output</th>
                <th>Avg latency</th>
                <th style={{ textAlign: "right" }}>Cost</th>
              </tr>
            </thead>
            <tbody>
              {purposes.map(([name, p]) => (
                <tr key={name}>
                  <td>{purposeLabels[name] ?? name}</td>
                  <td>
                    <span style={{
                      display: "inline-block",
                      padding: "2px 8px",
                      borderRadius: 4,
                      fontSize: 11,
                      fontWeight: 600,
                      background: p.category === "indexing" ? "rgba(125,211,160,0.12)" : "rgba(109,180,232,0.12)",
                      color: p.category === "indexing" ? "var(--accent)" : "var(--info, #6db4e8)",
                    }}>{p.category}</span>
                  </td>
                  <td>{fmt(p.call_count)}</td>
                  <td>{fmt(p.input_tokens)}</td>
                  <td>{fmt(p.output_tokens)}</td>
                  <td>{Math.round(p.avg_latency_ms)} ms</td>
                  <td className="mono" style={{ textAlign: "right" }}>{fmtCost(p.estimated_cost_usd)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {models.length > 0 && (
        <>
          <div className="card-section">
            <div className="card-section-title">
              <span>Token usage by model</span>
              <span style={{ color: "var(--text-3)", fontWeight: 500, textTransform: "none", letterSpacing: 0 }}>{models.length} model{models.length === 1 ? "" : "s"}</span>
            </div>
            <div className="bar-chart">
              {models.map(([name, m]) => {
                const total = m.input_tokens + m.output_tokens;
                const pct = (total / maxTokens) * 100;
                const inputPct = total > 0 ? (m.input_tokens / total) * pct : 0;
                const outputPct = pct - inputPct;
                const shortName = name.split(".").pop() ?? name;
                return (
                  <div key={name}>
                    <div className="bar-row">
                      <div className="bar-label" title={name}>{shortName}</div>
                      <div className="bar-track">
                        <div className="bar-fill input" style={{ width: `${inputPct}%` }}>
                          {inputPct > 15 ? fmt(m.input_tokens) : ""}
                        </div>
                      </div>
                    </div>
                    <div className="bar-row" style={{ marginTop: -2 }}>
                      <div className="bar-label" style={{ opacity: 0 }}>.</div>
                      <div className="bar-track">
                        <div className="bar-fill output" style={{ width: `${outputPct}%` }}>
                          {outputPct > 15 ? fmt(m.output_tokens) : ""}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
              <div className="bar-legend">
                <span><span className="swatch" style={{ background: "linear-gradient(90deg, var(--primary), var(--primary-2))" }} /> Input</span>
                <span><span className="swatch" style={{ background: "linear-gradient(90deg, var(--accent), var(--accent-2))" }} /> Output</span>
              </div>
            </div>
          </div>

          <div className="card-section">
            <div className="card-section-title">
              <span>Per-model breakdown</span>
              <span style={{ color: "var(--text-3)", fontWeight: 500, textTransform: "none", letterSpacing: 0 }}><IconClock /></span>
            </div>
            <table className="tbl">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Calls</th>
                  <th>Input</th>
                  <th>Output</th>
                  <th>Avg latency</th>
                  <th style={{ textAlign: "right" }}>Cost</th>
                </tr>
              </thead>
              <tbody>
                {models.map(([name, m]) => {
                  const purposeList = Object.entries(m.by_purpose ?? {});
                  return (
                    <>
                      <tr key={name}>
                        <td className="mono">{name}</td>
                        <td>{m.call_count}</td>
                        <td>{fmt(m.input_tokens)}</td>
                        <td>{fmt(m.output_tokens)}</td>
                        <td>{Math.round(m.avg_latency_ms)} ms</td>
                        <td className="mono" style={{ textAlign: "right" }}>{fmtCost(m.estimated_cost_usd)}</td>
                      </tr>
                      {purposeList.map(([purpose, pu]) => (
                        <tr key={`${name}-${purpose}`} style={{ background: "rgba(0,0,0,0.015)" }}>
                          <td className="mono" style={{ paddingLeft: 24, color: "var(--text-3)", fontSize: 11 }}>↳ {purposeLabels[purpose] ?? purpose}</td>
                          <td style={{ color: "var(--text-3)", fontSize: 11 }}>{fmt(pu.call_count)}</td>
                          <td style={{ color: "var(--text-3)", fontSize: 11 }}>{fmt(pu.input_tokens)}</td>
                          <td style={{ color: "var(--text-3)", fontSize: 11 }}>{fmt(pu.output_tokens)}</td>
                          <td style={{ color: "var(--text-3)", fontSize: 11 }}>{Math.round(pu.avg_latency_ms)} ms</td>
                          <td style={{ textAlign: "right" }}>—</td>
                        </tr>
                      ))}
                    </>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </>
  );
}

// ──────────────────────────────────────────────────────────────────────
// By-user view (admin-only).
// ──────────────────────────────────────────────────────────────────────

type SortKey = "email" | "query_count" | "prompt_tokens" | "completion_tokens" | "cost_usd_micros" | "last_active_at";

function ByUserView({
  rows, sortKey, sortDir, onSort,
}: {
  rows: UsageByUser[];
  sortKey: SortKey;
  sortDir: "asc" | "desc";
  onSort: (k: SortKey) => void;
}) {
  const arrow = (k: SortKey) => {
    if (k !== sortKey) return "";
    return sortDir === "asc" ? " ↑" : " ↓";
  };

  if (rows.length === 0) {
    return (
      <div className="card-section" style={{ padding: 22, color: "var(--text-3)", fontSize: 13 }}>
        No per-user usage data yet. Once authenticated users start querying, their totals will appear here.
      </div>
    );
  }

  return (
    <div className="card-section">
      <div className="card-section-title">
        <span>Per-user usage</span>
        <span style={{ color: "var(--text-3)", fontWeight: 500, textTransform: "none", letterSpacing: 0 }}>
          {rows.length} user{rows.length === 1 ? "" : "s"}
        </span>
      </div>
      <table className="tbl">
        <thead>
          <tr>
            <th style={{ cursor: "pointer" }} onClick={() => onSort("email")}>User{arrow("email")}</th>
            <th style={{ cursor: "pointer" }} onClick={() => onSort("query_count")}>Queries{arrow("query_count")}</th>
            <th style={{ cursor: "pointer" }} onClick={() => onSort("prompt_tokens")}>Input tokens{arrow("prompt_tokens")}</th>
            <th style={{ cursor: "pointer" }} onClick={() => onSort("completion_tokens")}>Output tokens{arrow("completion_tokens")}</th>
            <th style={{ cursor: "pointer", textAlign: "right" }} onClick={() => onSort("cost_usd_micros")}>Cost{arrow("cost_usd_micros")}</th>
            <th style={{ cursor: "pointer" }} onClick={() => onSort("last_active_at")}>Last active{arrow("last_active_at")}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.user_oid}>
              <td>
                <div style={{ fontWeight: 600 }}>{r.display_name || r.email}</div>
                {r.display_name && (
                  <div style={{ fontSize: 11, color: "var(--text-3)" }}>{r.email}</div>
                )}
              </td>
              <td>{fmt(r.query_count)}</td>
              <td>{fmt(r.prompt_tokens)}</td>
              <td>{fmt(r.completion_tokens)}</td>
              <td className="mono" style={{ textAlign: "right" }}>{microsToUsd(r.cost_usd_micros)}</td>
              <td style={{ fontSize: 12, color: "var(--text-2)" }}>{fmtTimestamp(r.last_active_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────────
// Mine view (anyone signed in).
// ──────────────────────────────────────────────────────────────────────

function MineView({
  mine, byUser, me,
}: {
  mine: UsageMine | null;
  byUser: UsageByUser[];
  me: MeResponse | null;
}) {
  if (!me || me.is_anonymous) {
    return (
      <div className="card-section" style={{ padding: 22, color: "var(--text-3)", fontSize: 13 }}>
        Sign in to see your usage.
      </div>
    );
  }
  if (!mine) {
    return (
      <div className="card-section" style={{ padding: 22, color: "var(--text-3)", fontSize: 13 }}>
        No usage recorded for your account yet — ask a question to start tracking.
      </div>
    );
  }

  // If the admin-only by_user data happens to be loaded (e.g. admin user
  // viewing their own row), surface their share-of-total. Otherwise the
  // share bar quietly hides; spec says "if available".
  const totalCostMicros = byUser.reduce((s, r) => s + r.cost_usd_micros, 0);
  const sharePct = totalCostMicros > 0 ? (mine.cost_usd_micros / totalCostMicros) * 100 : null;

  return (
    <>
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon"><IconMessage /></div>
          <div className="stat-label">Your queries</div>
          <div className="stat-value">{fmt(mine.query_count)}</div>
          <div className="stat-sub">last active {fmtTimestamp(mine.last_active_at)}</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon"><IconCoin /></div>
          <div className="stat-label">Your cost</div>
          <div className="stat-value cost">{microsToUsd(mine.cost_usd_micros)}</div>
          <div className="stat-sub">{mine.query_count > 0 ? microsToUsd(Math.round(mine.cost_usd_micros / mine.query_count)) : "$0.00"} / query</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon"><IconHash /></div>
          <div className="stat-label">Input tokens</div>
          <div className="stat-value">{fmt(mine.prompt_tokens)}</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon"><IconHash /></div>
          <div className="stat-label">Output tokens</div>
          <div className="stat-value">{fmt(mine.completion_tokens)}</div>
        </div>
      </div>

      {sharePct !== null && (
        <div className="card-section">
          <div className="card-section-title">
            <span>Your share of cost</span>
            <span style={{ color: "var(--text-3)", fontWeight: 500, textTransform: "none", letterSpacing: 0 }}>
              {sharePct.toFixed(1)}% of {microsToUsd(totalCostMicros)} total
            </span>
          </div>
          <div className="bar-chart">
            <div className="bar-row">
              <div className="bar-label">You</div>
              <div className="bar-track">
                <div className="bar-fill input" style={{ width: `${Math.max(2, sharePct)}%` }}>
                  {sharePct > 8 ? microsToUsd(mine.cost_usd_micros) : ""}
                </div>
              </div>
            </div>
            <div className="bar-legend">
              <span>Out of {byUser.length} tracked user{byUser.length === 1 ? "" : "s"}</span>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
