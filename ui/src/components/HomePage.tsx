import { useEffect, useState } from "react";
import {
  fetchUsage,
  fetchUsageWeekly,
  fetchIndexing,
  fetchInvestigationHistory,
  type Session,
  type MeResponse,
  type UsageSummary,
  type UsageWeek,
  type IndexingSummary,
  type IndexingRepo,
  type InvestigationHistoryItem,
} from "../api";
import {
  IconChat,
  IconChart,
  IconDatabase,
  IconBolt,
  IconPlus,
  IconFile,
  IconCheck,
} from "./icons";

// ── Local formatting helpers (mirrors UsagePage's fmt/fmtCost style) ─────
function fmt(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return n.toLocaleString();
}

function fmtCost(n: number): string {
  if (n <= 0) return "$0";
  if (n < 0.01) return "$" + n.toFixed(4);
  if (n < 1) return "$" + n.toFixed(3);
  return "$" + n.toFixed(2);
}

function truncate(s: string, max: number): string {
  if (!s) return "";
  if (s.length <= max) return s;
  return s.slice(0, max).trimEnd() + "…";
}

function threadLabel(threadId: string): string {
  return threadId.split("_").slice(1).join("_") || threadId;
}

// Relative time from an age in seconds (investigation history carries
// `age_seconds`). Defensive — never throws on odd input.
function ago(seconds: number | null | undefined): string {
  if (seconds == null || !isFinite(seconds) || seconds < 0) return "";
  if (seconds < 60) return "just now";
  const m = Math.floor(seconds / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  if (d === 1) return "Yesterday";
  if (d < 7) return `${d}d ago`;
  const w = Math.floor(d / 7);
  return `${w}w ago`;
}

// Map a raw indexing status onto a labelled health state + dot colour class.
function repoHealth(status: string): { label: string; cls: string } {
  const v = (status || "").toLowerCase();
  if (v === "done" || v === "ready" || v === "healthy") return { label: "Healthy", cls: "ok" };
  if (v.includes("run") || v.includes("sync") || v.includes("progress") || v.includes("index"))
    return { label: "Syncing", cls: "sync" };
  if (v.includes("error") || v.includes("fail")) return { label: "Error", cls: "err" };
  if (v.includes("stale")) return { label: "Stale", cls: "stale" };
  return { label: status || "Unknown", cls: "neutral" };
}

// Map a free-form investigation outcome onto a status pill + progress %.
function invStatus(outcome: string): { label: string; cls: string; pct: number } {
  const o = (outcome || "").toLowerCase();
  if (o.includes("resolved") || o.includes("root") || o.includes("validated") || o.includes("found") || o.includes("complete"))
    return { label: "Resolved", cls: "ok", pct: 100 };
  if (o.includes("progress") || o.includes("running") || o.includes("active") || o.includes("pending"))
    return { label: "In progress", cls: "warn", pct: 60 };
  if (o.includes("inconclusive") || o.includes("unknown") || o.includes("partial"))
    return { label: "Inconclusive", cls: "stale", pct: 45 };
  return { label: outcome || "—", cls: "neutral", pct: 50 };
}

// Inline glyphs the shared icon set doesn't provide.
const ArrowRight = () => (
  <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><path d="M3 8h10M9 4l4 4-4 4" /></svg>
);
const LockGlyph = () => (
  <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><rect x="3" y="7" width="10" height="7" rx="1.5" /><path d="M5 7V5a3 3 0 0 1 6 0v2" /></svg>
);
const SearchGlyph = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"><circle cx="7" cy="7" r="4.5" /><path d="M11 11l3 3" /></svg>
);

// Rotating accent dots for the conversation list (purely decorative rhythm).
const DOT_TONES = ["ok", "sync", "stale", "neutral"];

interface Props {
  me?: MeResponse | null;
  sessions: Session[];
  onNavigate: (page: string) => void;
  onNewChat: () => void;
  onOpenChat: (threadId: string) => void;
  brandName?: string;
}

export default function HomePage({
  me,
  sessions,
  onNavigate,
  onNewChat,
  onOpenChat,
  brandName = "OpsRAG",
}: Props) {
  const [usage, setUsage] = useState<UsageSummary | null>(null);
  const [usageWeeks, setUsageWeeks] = useState<UsageWeek[] | null>(null);
  const [indexing, setIndexing] = useState<IndexingSummary | null>(null);
  const [investigations, setInvestigations] = useState<InvestigationHistoryItem[] | null>(null);

  // Fetch dashboard data on mount. Every card tolerates failure on its own —
  // none of these should ever throw or blank the page.
  useEffect(() => {
    let cancelled = false;
    fetchUsage().then((d) => { if (!cancelled) setUsage(d); }).catch(() => { if (!cancelled) setUsage(null); });
    fetchUsageWeekly().then((d) => { if (!cancelled) setUsageWeeks(d); }).catch(() => { if (!cancelled) setUsageWeeks([]); });
    fetchIndexing().then((d) => { if (!cancelled) setIndexing(d); }).catch(() => { if (!cancelled) setIndexing(null); });
    fetchInvestigationHistory(6).then((d) => { if (!cancelled) setInvestigations(d); }).catch(() => { if (!cancelled) setInvestigations([]); });
    return () => { cancelled = true; };
  }, []);

  // ── Greeting header (time-of-day + first name) ─────────────────────────
  const hour = new Date().getHours();
  const tod = hour < 12 ? "morning" : hour < 18 ? "afternoon" : "evening";
  const firstName =
    me?.name?.trim().split(/\s+/)[0] ||
    (me?.email ? me.email.split("@")[0] : "");
  const greeting = firstName ? `Good ${tod}, ${firstName}` : `Good ${tod}`;

  const srcCount = indexing?.total_repos ?? 0;
  const chunkTotal = indexing?.total_chunks ?? 0;
  const healthLine = indexing
    ? `Your knowledge base is healthy. ${srcCount} source${srcCount === 1 ? "" : "s"} indexed${chunkTotal ? `, ${fmt(chunkTotal)} chunks total` : ""}.`
    : `Welcome to ${brandName} — your command center for agentic GraphRAG.`;

  // Role label for the scope-preview pill. Driven by the REAL roles/scopes
  // from /me now (was a hardcoded is_admin guess). Prefer an explicit
  // "admin" role/scope; fall back to is_admin, else "Member".
  const isAdmin =
    (me?.roles?.includes("admin") ?? false) ||
    (me?.scopes?.includes("admin") ?? false) ||
    (me?.is_admin ?? false);
  const roleLabel = isAdmin ? "Admin" : "Member";

  // ── Usage card ─────────────────────────────────────────────────────────
  const now = new Date();
  const usageRange = `${now.toLocaleString("default", { month: "short" })} 1 – ${now.getDate()}`;
  const totalTokens = usage ? usage.total_input_tokens + usage.total_output_tokens : 0;

  // Weekly bar chart, driven by GET /usage/weekly (6 buckets, oldest-first,
  // current week last). Each bar's height is scaled to the busiest week so
  // magnitudes are comparable; the most-recent week is emphasized. If the
  // backend has no per-week data (no DB / no activity), we fall back to a
  // labelled-but-empty axis with a clear caption.
  const CHART_WEEKS = 6;
  const fallbackWeeks: UsageWeek[] = [];
  for (let i = CHART_WEEKS - 1; i >= 0; i--) {
    const d = new Date(now);
    d.setDate(now.getDate() - i * 7);
    fallbackWeeks.push({
      week_start: d.toISOString().slice(0, 10),
      tokens: 0, input_tokens: 0, output_tokens: 0, call_count: 0, cost_usd: 0,
    });
  }
  const weeks: UsageWeek[] = usageWeeks && usageWeeks.length ? usageWeeks : fallbackWeeks;
  // Scale by tokens (the headline metric on this card).
  const maxWeekTokens = weeks.reduce((m, w) => Math.max(m, w.tokens), 0);
  const hasWeeklyData = maxWeekTokens > 0;
  const weekBars = weeks.map((w, i) => {
    // Floor non-zero weeks at 6% so a tiny-but-real week is still visible.
    const ratio = maxWeekTokens > 0 ? w.tokens / maxWeekTokens : 0;
    const pct = w.tokens > 0 ? Math.max(6, Math.round(ratio * 100)) : 0;
    const d = new Date(w.week_start + "T00:00:00");
    const label = `${d.getMonth() + 1}/${d.getDate()}`;
    return {
      key: w.week_start || `wk-${i}`,
      label,
      pct,
      current: i === weeks.length - 1,
      tokens: w.tokens,
      cost: w.cost_usd,
    };
  });
  const chartCaption = hasWeeklyData
    ? "Weekly tokens — current week highlighted."
    : "Weekly breakdown builds as you query.";

  const recentSessions = sessions.slice(0, 4);
  const repos: IndexingRepo[] = indexing?.repos ?? [];
  const topRepos = repos.slice(0, 6);

  return (
    <div className="home">
      {/* ── Greeting header ── */}
      <header className="home-header">
        <div className="home-greet">
          <h1>{greeting}</h1>
          <p>{healthLine}</p>
        </div>
        {/* Real role indicator — reflects the viewer's applied scopes/role
            from /me (nav + pages adapt to these). */}
        <div className="home-scope-pill" title="Your navigation and pages reflect your role's scopes.">
          <LockGlyph />
          <span>You're seeing the <strong>{roleLabel}</strong> layout.</span>
        </div>
      </header>

      <div className="home-grid">
        {/* ── Continue a conversation ── */}
        <section className="home-card">
          <div className="home-card-head">
            <div className="home-card-title">
              <span className="home-card-icon"><IconChat /></span>
              <div>
                <h3>Continue a conversation</h3>
                <p>Pick up where you left off</p>
              </div>
            </div>
            <button className="home-link" onClick={() => onNavigate("chat")}>View all <ArrowRight /></button>
          </div>

          {recentSessions.length === 0 ? (
            <div className="home-card-empty">
              <p>No conversations yet.</p>
              <button className="btn btn-primary" onClick={onNewChat}><IconPlus /> New chat</button>
            </div>
          ) : (
            <ul className="home-conv-list">
              {recentSessions.map((s, i) => {
                const title = s.title?.trim() || `Conversation ${threadLabel(s.thread_id)}`;
                const turns = s.turn_count ?? s.checkpoint_count;
                return (
                  <li key={s.thread_id}>
                    <button className="home-conv-row" onClick={() => onOpenChat(s.thread_id)} title={title}>
                      <span className={`home-dot ${DOT_TONES[i % DOT_TONES.length]}`} />
                      <span className="home-conv-body">
                        <span className="home-conv-title">{title}</span>
                        <span className="home-conv-meta">
                          {turns} turn{turns === 1 ? "" : "s"}
                        </span>
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </section>

        {/* ── Usage this month ── */}
        <section className="home-card">
          <div className="home-card-head">
            <div className="home-card-title">
              <span className="home-card-icon"><IconChart /></span>
              <div>
                <h3>Usage this month</h3>
                <p>{usageRange}</p>
              </div>
            </div>
            <button className="home-link" onClick={() => onNavigate("usage")}>Details <ArrowRight /></button>
          </div>

          <div className="home-usage-tiles">
            <div className="home-usage-tile">
              <span className="home-usage-label">Tokens</span>
              <span className="home-usage-value">{usage ? fmt(totalTokens) : "—"}</span>
              <span className="home-usage-sub">{usage ? `${fmt(usage.total_calls)} calls` : "loading…"}</span>
            </div>
            <div className="home-usage-tile">
              <span className="home-usage-label">Est. cost</span>
              <span className="home-usage-value">{usage ? fmtCost(usage.total_estimated_cost_usd) : "—"}</span>
              <span className="home-usage-sub">this month</span>
            </div>
          </div>

          <div className="home-chart">
            <div className="home-chart-bars">
              {weekBars.map((b) => (
                <div className="home-chart-col" key={b.key}>
                  <div
                    className={`home-chart-bar${b.current ? " is-current" : ""}`}
                    style={{ height: `${b.pct}%` }}
                    title={`Week of ${b.label}: ${fmt(b.tokens)} tokens · ${fmtCost(b.cost)}`}
                  />
                  <span className={`home-chart-label${b.current ? " is-current" : ""}`}>{b.label}</span>
                </div>
              ))}
            </div>
            {hasWeeklyData ? (
              <div className="home-chart-caption">{chartCaption}</div>
            ) : (
              <div className="home-chart-note">{chartCaption}</div>
            )}
          </div>
        </section>

        {/* ── Indexing health ── */}
        <section className="home-card">
          <div className="home-card-head">
            <div className="home-card-title">
              <span className="home-card-icon"><IconDatabase /></span>
              <div>
                <h3>Indexing health</h3>
                <p>{indexing ? `${srcCount} source${srcCount === 1 ? "" : "s"} · ${fmt(chunkTotal)} chunks total` : "loading…"}</p>
              </div>
            </div>
            <button className="home-link" onClick={() => onNavigate("indexing")}>Manage <ArrowRight /></button>
          </div>

          {!indexing ? (
            <div className="home-card-empty"><p>Loading indexing status…</p></div>
          ) : topRepos.length === 0 ? (
            <div className="home-card-empty"><p>No sources indexed yet.</p>
              <button className="btn btn-secondary" onClick={() => onNavigate("sources")}>Add a source</button>
            </div>
          ) : (
            <table className="home-table">
              <thead>
                <tr><th>Source</th><th>Status</th><th className="num">Chunks</th></tr>
              </thead>
              <tbody>
                {topRepos.map((r) => {
                  const h = repoHealth(r.status);
                  return (
                    <tr key={`${r.repo}@${r.branch}`}>
                      <td className="home-td-source" title={r.repo}>
                        <span className="home-src-icon"><IconDatabase /></span>
                        {r.display_name || r.repo}
                      </td>
                      <td>
                        <span className="home-status"><span className={`home-dot ${h.cls}`} />{h.label}</span>
                      </td>
                      <td className="num home-td-chunks">{r.total_chunks.toLocaleString()}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </section>

        {/* ── Quick actions ── */}
        <section className="home-card">
          <div className="home-card-head">
            <div className="home-card-title">
              <span className="home-card-icon"><IconBolt /></span>
              <div>
                <h3>Quick actions</h3>
                <p>Jump straight in</p>
              </div>
            </div>
          </div>
          <div className="home-qa-grid">
            <button className="home-qa" onClick={onNewChat}>
              <span className="home-qa-icon"><IconChat /></span>
              <span className="home-qa-body"><span className="home-qa-title">New chat</span><span className="home-qa-sub">Ask the assistant</span></span>
            </button>
            <button className="home-qa" onClick={() => onNavigate("investigate")}>
              <span className="home-qa-icon"><SearchGlyph /></span>
              <span className="home-qa-body"><span className="home-qa-title">New investigation</span><span className="home-qa-sub">Multi-step trace</span></span>
            </button>
            <button className="home-qa" onClick={() => onNavigate("sources")}>
              <span className="home-qa-icon"><IconPlus /></span>
              <span className="home-qa-body"><span className="home-qa-title">Add source</span><span className="home-qa-sub">Connect &amp; index</span></span>
            </button>
            <button className="home-qa" onClick={() => onNavigate("runbooks")}>
              <span className="home-qa-icon"><IconFile /></span>
              <span className="home-qa-body"><span className="home-qa-title">Browse runbooks</span><span className="home-qa-sub">Hand-authored playbooks</span></span>
            </button>
          </div>
        </section>

        {/* ── Recent investigations (full width) ── */}
        <section className="home-card home-card-full">
          <div className="home-card-head">
            <div className="home-card-title">
              <span className="home-card-icon"><IconBolt /></span>
              <div>
                <h3>Recent investigations</h3>
                <p>Multi-step agentic traces across your knowledge</p>
              </div>
            </div>
            <button className="home-link" onClick={() => onNavigate("investigate")}>View all <ArrowRight /></button>
          </div>

          {investigations === null ? (
            <div className="home-card-empty"><p>Loading investigations…</p></div>
          ) : investigations.length === 0 ? (
            <div className="home-card-empty">
              <p>No investigations yet. Kick one off from an alert to see it here.</p>
              <button className="btn btn-secondary" onClick={() => onNavigate("investigate")}>Start an investigation</button>
            </div>
          ) : (
            <ul className="home-inv-list">
              {investigations.map((inv) => {
                const st = invStatus(inv.outcome);
                return (
                  <li key={inv.investigation_id}>
                    <button className="home-inv-row" onClick={() => onNavigate("investigate")} title={inv.alert_text}>
                      <span className={`home-inv-icon ${st.cls}`}>
                        {st.cls === "ok" ? <IconCheck /> : <IconBolt />}
                      </span>
                      <span className="home-inv-body">
                        <span className="home-inv-title">{truncate(inv.alert_text, 70)}</span>
                        {inv.final_root_cause && (
                          <span className="home-inv-meta">Root cause: {truncate(inv.final_root_cause, 80)}</span>
                        )}
                      </span>
                      <span className="home-inv-right">
                        <span className={`badge home-inv-badge ${st.cls}`}>{st.label}</span>
                        <span className="home-inv-bar"><span className="home-inv-bar-fill" style={{ width: `${st.pct}%` }} /></span>
                        <span className="home-inv-time">{ago(inv.age_seconds)}</span>
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </section>
      </div>
    </div>
  );
}
