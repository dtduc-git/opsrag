import { useState, useEffect } from "react";
import { fetchIndexing, indexRepo, indexSource, type IndexingSummary, type IndexingRepo } from "../api";
import { IconStack, IconFile, IconDatabase, IconSync, IconPlus, IconClose } from "./icons";

// repo path: at least one slash, kebab-friendly characters, no leading/trailing slash.
const REPO_RE = /^[a-zA-Z0-9._-]+(?:\/[a-zA-Z0-9._-]+)+$/;
const BRANCH_RE = /^[a-zA-Z0-9._/-]+$/;

function AddRepoForm({ onSubmitted }: { onSubmitted: () => void }) {
  const [repo, setRepo] = useState("");
  const [branch, setBranch] = useState("master");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const repoOk = REPO_RE.test(repo.trim());
  const branchOk = BRANCH_RE.test(branch.trim());

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (!repoOk || !branchOk) {
      setError("Repo path needs at least one slash; branch can't be empty.");
      return;
    }
    setBusy(true);
    try {
      await indexRepo(repo.trim(), branch.trim());
      setRepo("");
      onSubmitted();
      // Scroll the new card into view once the next poll cycle runs.
      setTimeout(() => {
        const el = document.getElementById(`repo-card-${repo.trim()}`);
        if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
      }, 600);
    } catch (exc: unknown) {
      const msg = exc instanceof Error ? exc.message : String(exc);
      setError(msg);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="add-repo-form" onSubmit={submit}>
      <div className="add-repo-fields">
        <div className="add-repo-field">
          <label>Repository</label>
          <input
            type="text"
            placeholder="devops/some-repo or saas/my-service"
            value={repo}
            onChange={(e) => setRepo(e.target.value)}
            disabled={busy}
            autoComplete="off"
            spellCheck={false}
          />
        </div>
        <div className="add-repo-field branch">
          <label>Branch</label>
          <input
            type="text"
            placeholder="master"
            value={branch}
            onChange={(e) => setBranch(e.target.value)}
            disabled={busy}
            autoComplete="off"
            spellCheck={false}
          />
        </div>
        <button type="submit" className="btn-primary" disabled={busy || !repoOk || !branchOk}>
          {busy ? "Queuing…" : "Index now"}
        </button>
      </div>
      {error && <div className="add-repo-error">{error}</div>}
    </form>
  );
}

// The backend stores one progress row per (repo, branch). When the same repo
// was indexed under multiple branch labels (e.g. `main` and `master`) the API
// returns it twice — the content is the same, so the Sources page must show a
// single card. Collapse duplicates to the canonical branch: most chunks, then
// most total_files, then most indexed_files. Preserves first-seen order.
function dedupReposByRepo(repos: IndexingRepo[]): IndexingRepo[] {
  const best = new Map<string, IndexingRepo>();
  const order: string[] = [];
  for (const r of repos) {
    const existing = best.get(r.repo);
    if (!existing) {
      best.set(r.repo, r);
      order.push(r.repo);
      continue;
    }
    const better =
      r.total_chunks !== existing.total_chunks
        ? r.total_chunks > existing.total_chunks
        : r.total_files !== existing.total_files
        ? r.total_files > existing.total_files
        : r.indexed_files > existing.indexed_files;
    if (better) best.set(r.repo, r);
  }
  return order.map((k) => best.get(k)!);
}

function fmt(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return n.toLocaleString();
}

function fmtTime(s: number): string {
  if (s <= 0) return "—";
  if (s < 60) return Math.round(s) + "s";
  if (s < 3600) return Math.round(s / 60) + "m " + Math.round(s % 60) + "s";
  return (s / 3600).toFixed(1) + "h";
}

function RepoCard({ repo, onSynced }: { repo: IndexingRepo; onSynced: () => void }) {
  const isDone = repo.status === "done";
  const isFailed = repo.status === "failed";
  const isActive = repo.status === "indexing" || repo.status === "listing";
  const fillClass = isDone ? "done" : isActive ? "" : "idle";
  const width = isFailed ? 100 : Math.max(repo.percent, 0);

  const [syncing, setSyncing] = useState(false);
  const [syncError, setSyncError] = useState<string | null>(null);

  // Disable while a sync request is in flight OR while the repo is already
  // mid-run (no value in queueing on top of itself).
  const syncDisabled = syncing || isActive;

  async function handleSync() {
    setSyncError(null);
    setSyncing(true);
    try {
      if (repo.source_type && repo.source_type !== "git") {
        // Non-git: repo string is `<source_type>:<scope>` (e.g. `confluence:SRE`).
        const scope = repo.repo.startsWith(`${repo.source_type}:`)
          ? repo.repo.slice(repo.source_type.length + 1)
          : repo.repo;
        await indexSource(repo.source_type, scope);
      } else {
        await indexRepo(repo.repo, repo.branch);
      }
      onSynced();
    } catch (exc: unknown) {
      setSyncError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      // Brief disable even on success so the user sees the click registered;
      // the next poll cycle (3s) will flip the status to listing/indexing.
      setTimeout(() => setSyncing(false), 600);
    }
  }

  return (
    <div className="repo-card" id={`repo-card-${repo.repo}`}>
      <div className="repo-header">
        <span
          className="repo-name"
          title={repo.display_name ? repo.repo : undefined}
        >
          {repo.display_name || repo.repo}
        </span>
        <div className="repo-header-actions">
          <span className={`repo-status status-${repo.status}`}>{repo.status}</span>
          <button
            type="button"
            className="repo-sync"
            onClick={handleSync}
            disabled={syncDisabled}
            title={isActive ? "Already running" : `Reindex ${repo.display_name || repo.repo}`}
          >
            <IconSync />
            <span>{syncing ? "Indexing…" : "Index now"}</span>
          </button>
        </div>
      </div>
      {syncError && <div className="repo-error">{syncError.slice(0, 240)}</div>}
      <div className="progress-bar">
        <div
          className={`progress-fill ${fillClass}`}
          style={{ width: `${width}%` }}
        />
      </div>
      <div className="repo-stats">
        <span>Branch <strong>{repo.branch}</strong></span>
        <span title="Files that produced ≥1 indexed chunk">
          Indexed <strong>{fmt(repo.indexed_files)}</strong>/{fmt(repo.total_files)}
        </span>
        {repo.skipped_files > 0 && (
          <span title="Files no parser claimed (binary, oversized, or unsupported)" style={{ color: "var(--text-3)" }}>
            Skipped <strong>{fmt(repo.skipped_files)}</strong>
          </span>
        )}
        <span>Chunks <strong>{fmt(repo.total_chunks)}</strong></span>
        <span>Entities <strong>{fmt(repo.entities_found)}</strong></span>
        <span>Time <strong>{fmtTime(repo.elapsed_seconds)}</strong></span>
        {repo.percent > 0 && isActive && <span className="repo-percent">{repo.percent.toFixed(1)}%</span>}
      </div>
      {isFailed && repo.error && <div className="repo-error">{repo.error.slice(0, 240)}</div>}
    </div>
  );
}

export default function IndexingPage() {
  const [data, setData] = useState<IndexingSummary | null>(null);
  const [addOpen, setAddOpen] = useState(false);

  useEffect(() => {
    const load = () => fetchIndexing().then(setData).catch(() => {});
    load();
    const iv = setInterval(load, 3000);
    return () => clearInterval(iv);
  }, []);

  if (!data) return <div className="page"><p>Loading…</p></div>;

  // Collapse repos indexed under multiple branches to a single canonical entry,
  // then derive every displayed count from the deduped list so the top-line
  // stats, per-group counts, and rendered cards all agree.
  const repos = dedupReposByRepo(data.repos);
  const totalRepos = repos.length;
  const totalFiles = repos.reduce((s, r) => s + r.total_files, 0);
  const totalIndexed = repos.reduce((s, r) => s + r.indexed_files, 0);
  const totalChunks = repos.reduce((s, r) => s + r.total_chunks, 0);
  const active = repos.filter((r) => r.status === "indexing" || r.status === "listing");

  const reload = () => fetchIndexing().then(setData).catch(() => {});

  return (
    <div className="page">
      <div className="page-title-row">
        <div>
          <div className="page-title">Indexing</div>
          <div className="page-subtitle">Repository ingestion progress — auto-refreshed every 3s.</div>
        </div>
        <button
          type="button"
          className={addOpen ? "btn-secondary" : "btn-primary"}
          onClick={() => setAddOpen((v) => !v)}
        >
          {addOpen ? <IconClose /> : <IconPlus />}
          <span>{addOpen ? "Close" : "Add repo"}</span>
        </button>
      </div>

      {addOpen && (
        <AddRepoForm
          onSubmitted={() => {
            reload();
            setAddOpen(false);
          }}
        />
      )}

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon"><IconStack /></div>
          <div className="stat-label">Repositories</div>
          <div className="stat-value">{totalRepos}</div>
          <div className="stat-sub">{active.length} active · {repos.filter(r => r.status === "done").length} done</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon"><IconFile /></div>
          <div className="stat-label">Files</div>
          <div className="stat-value">{fmt(totalFiles)}</div>
          <div className="stat-sub">{fmt(totalIndexed)} processed</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon"><IconDatabase /></div>
          <div className="stat-label">Chunks</div>
          <div className="stat-value">{fmt(totalChunks)}</div>
          <div className="stat-sub">In vector store</div>
        </div>
      </div>

      {repos.length === 0 ? (
        <div className="card-section" style={{ padding: 32, textAlign: "center" }}>
          <p style={{ color: "var(--text-2)", fontSize: 13 }}>
            No repos configured. Add entries under <code>scm.repos</code> in <code>config-local.yaml</code> to start indexing.
          </p>
        </div>
      ) : (
        <SourceGroups repos={repos} onSynced={reload} />
      )}
    </div>
  );
}

// Group repos by source_type ("git" | "confluence" | ...) with collapsible
// sections so the dashboard scales as more sources land in Phase 2-3.
function SourceGroups({ repos, onSynced }: { repos: IndexingRepo[]; onSynced: () => void }) {
  const groups = new Map<string, IndexingRepo[]>();
  for (const r of repos) {
    const key = r.source_type ?? "git";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(r);
  }
  // Stable order: git first, then alphabetical for the rest.
  const ordered = Array.from(groups.entries()).sort(([a], [b]) => {
    if (a === "git") return -1;
    if (b === "git") return 1;
    return a.localeCompare(b);
  });
  return (
    <>
      {ordered.map(([sourceType, items]) => (
        <SourceGroup
          key={sourceType}
          sourceType={sourceType}
          repos={items}
          onSynced={onSynced}
        />
      ))}
    </>
  );
}

const SOURCE_LABELS: Record<string, string> = {
  git: "Repositories (Git)",
  confluence: "Confluence",
  rootly: "Rootly · Incidents",
  slack: "Slack archive",
};

const SOURCE_HINTS: Record<string, string> = {
  git: "GitLab + GitHub repos cloned via SSH; ingested by file-glob patterns.",
  confluence: "Atlassian Cloud wiki pages, ADF→Markdown rendered. One scope = one space.",
  rootly: "Rootly incidents (alerts not indexed — RAG scope decision).",
  slack: "Allowlisted channels; threads summarized at ingest time.",
};

function SourceGroup({ sourceType, repos, onSynced }: { sourceType: string; repos: IndexingRepo[]; onSynced: () => void }) {
  const [open, setOpen] = useState(true);
  const total = repos.length;
  const active = repos.filter((r) => r.status === "indexing" || r.status === "listing").length;
  const done = repos.filter((r) => r.status === "done").length;
  const failed = repos.filter((r) => r.status === "failed").length;
  const totalChunks = repos.reduce((s, r) => s + r.total_chunks, 0);
  const totalIndexed = repos.reduce((s, r) => s + r.indexed_files, 0);
  const totalFiles = repos.reduce((s, r) => s + r.total_files, 0);
  const label = SOURCE_LABELS[sourceType] ?? sourceType;
  const hint = SOURCE_HINTS[sourceType] ?? "";
  return (
    <div className="source-group">
      <button
        type="button"
        className="source-group-header"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <div className="source-group-title">
          <span className="source-group-chevron" style={{ transform: open ? "rotate(90deg)" : "rotate(0deg)" }}>▶</span>
          <span className="source-group-name">{label}</span>
          <span className="source-group-count">{total}</span>
        </div>
        <div className="source-group-stats">
          <span title="Active right now">{active > 0 ? `${active} active` : `${done} done`}</span>
          {failed > 0 && <span style={{ color: "var(--danger, #e8806b)" }}>{failed} failed</span>}
          <span>{fmt(totalIndexed)}/{fmt(totalFiles)} files</span>
          <span>{fmt(totalChunks)} chunks</span>
        </div>
      </button>
      {open && hint && (
        <div className="source-group-hint">{hint}</div>
      )}
      {open && (
        <div className="source-group-body">
          {repos.map((r) => (
            <RepoCard key={`${r.repo}@${r.branch}`} repo={r} onSynced={onSynced} />
          ))}
        </div>
      )}
    </div>
  );
}
