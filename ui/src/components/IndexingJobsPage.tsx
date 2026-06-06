import { useEffect, useState } from "react";
import { fetchIndexingJobs, type IndexingJob, type IndexingJobsSummary } from "../api";

// Relative time from an epoch-seconds timestamp.
function ago(epochSeconds: number): string {
  if (!epochSeconds) return "—";
  const secs = Date.now() / 1000 - epochSeconds;
  if (secs < 0) return "just now";
  if (secs < 60) return "just now";
  const m = Math.floor(secs / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  if (d === 1) return "Yesterday";
  if (d < 7) return `${d}d ago`;
  return new Date(epochSeconds * 1000).toLocaleDateString();
}

function fmtDuration(seconds: number): string {
  if (!seconds || seconds < 1) return "<1s";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  if (m < 60) return s ? `${m}m ${s}s` : `${m}m`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

const STATUS_META: Record<IndexingJob["status"], { label: string; cls: string }> = {
  running: { label: "Running", cls: "sync" },
  success: { label: "Successful", cls: "ok" },
  failed: { label: "Failed", cls: "err" },
};

type Filter = "all" | "success" | "failed" | "running";

export default function IndexingJobsPage() {
  const [data, setData] = useState<IndexingJobsSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>("all");

  const load = () => {
    setLoading(true);
    fetchIndexingJobs()
      .then((d) => { setData(d); setErr(null); })
      .catch((e) => setErr(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  };

  // Poll while anything is running so a live job advances to its terminal
  // state without a manual refresh.
  useEffect(() => {
    load();
  }, []);
  useEffect(() => {
    if (!data?.running) return;
    const iv = setInterval(load, 4000);
    return () => clearInterval(iv);
  }, [data?.running]);

  const jobs = data?.jobs ?? [];
  const shown = filter === "all" ? jobs : jobs.filter((j) => j.status === filter);

  return (
    <div className="page">
      <div className="ij-toolbar">
        <div className="ij-stats">
          <span className="ij-stat"><strong>{data?.total ?? 0}</strong> runs</span>
          <span className="ij-stat ok"><strong>{jobs.filter((j) => j.status === "success").length}</strong> successful</span>
          <span className="ij-stat err"><strong>{data?.failed ?? 0}</strong> failed</span>
          {(data?.running ?? 0) > 0 && <span className="ij-stat sync"><strong>{data?.running}</strong> running</span>}
        </div>
        <div className="ij-filters">
          {(["all", "success", "failed", "running"] as Filter[]).map((f) => (
            <button
              key={f}
              className={`ij-filter-btn ${filter === f ? "active" : ""}`}
              onClick={() => setFilter(f)}
            >
              {f === "all" ? "All" : STATUS_META[f].label}
            </button>
          ))}
          <button className="btn btn-secondary" onClick={load} disabled={loading}>
            {loading ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </div>

      {err ? (
        <div className="card-section ij-error">Could not load jobs: {err}</div>
      ) : shown.length === 0 ? (
        <div className="empty-state">
          <p>{loading ? "Loading jobs…" : filter === "all" ? "No indexing jobs yet. Trigger a reindex from Sources to see runs here." : `No ${filter} jobs.`}</p>
        </div>
      ) : (
        <div className="ij-list">
          {shown.map((j) => {
            const st = STATUS_META[j.status];
            return (
              <div key={j.id} className={`ij-row ${j.status}`}>
                <div className="ij-row-head">
                  <span className={`home-dot ${st.cls}`} />
                  <span className="ij-source" title={j.repo}>
                    {j.display_name || j.repo}
                    {j.branch && j.branch !== j.source_type && <span className="ij-branch">@{j.branch}</span>}
                  </span>
                  <span className={`badge ij-badge ${st.cls}`}>{st.label}</span>
                  {j.kind === "restored" && (
                    <span className="badge ij-badge neutral" title="Reconstructed from the vector store at startup">restored</span>
                  )}
                  <span className="ij-meta">
                    <span title="source type">{j.source_type}</span>
                    <span>·</span>
                    <span>{j.chunks_indexed.toLocaleString()} chunks</span>
                    <span>·</span>
                    <span title="duration">{j.status === "running" ? "running…" : fmtDuration(j.duration_seconds)}</span>
                  </span>
                  <span className="ij-time">{ago(j.started_at)}</span>
                </div>
                {j.status === "failed" && j.error && (
                  <div className="ij-error-detail">
                    <span className="ij-error-label">Error</span>
                    <code>{j.error}</code>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
