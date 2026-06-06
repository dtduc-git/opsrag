import { useState, useEffect, useCallback } from "react";
import { fetchCacheSummary, purgeCache, type CacheSummary, type CachePurgeRequest } from "../api";
import { IconClose, IconSync } from "./icons";

type Target = "qa" | "investigation" | "tool" | "all";
type Strategy = CachePurgeRequest["strategy"];

// Allowed strategies per target — keeps the form coherent.
const STRATEGY_OPTIONS: Record<Target, { value: Strategy; label: string; needs?: string[] }[]> = {
  qa: [
    { value: "all",                label: "All Q&A cache (nuke)" },
    { value: "older_than",         label: "Older than N hours",          needs: ["older_than_hours"] },
    { value: "repo",               label: "By source repo",              needs: ["repo"] },
    { value: "quality_low",        label: "Low-quality entries (👎)" },
    { value: "question_contains",  label: "Question contains text",      needs: ["question_substring"] },
  ],
  investigation: [
    { value: "all",                label: "All investigation cache (nuke)" },
    { value: "older_than",         label: "Older than N hours",          needs: ["older_than_hours"] },
    { value: "thumbs_down",        label: "Investigations with 👎" },
    { value: "question_contains",  label: "Question contains text",      needs: ["question_substring"] },
  ],
  tool: [
    { value: "all",                label: "All tool-output cache" },
    { value: "tool_name",          label: "By tool name",                needs: ["tool_name"] },
  ],
  all: [
    { value: "all",                label: "💥 NUKE ALL 3 caches" },
  ],
};

function StatCard({ title, items }: { title: string; items: { label: string; value: string | number }[] }) {
  return (
    <div className="cache-stat-card">
      <div className="cache-stat-title">{title}</div>
      <div className="cache-stat-grid">
        {items.map((it) => (
          <div key={it.label} className="cache-stat-row">
            <span className="cache-stat-label">{it.label}</span>
            <span className="cache-stat-value">{it.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function CachePage() {
  const [summary, setSummary] = useState<CacheSummary | null>(null);
  const [target, setTarget] = useState<Target>("qa");
  const [strategy, setStrategy] = useState<Strategy>("older_than");
  const [olderHours, setOlderHours] = useState<number>(168);
  const [repo, setRepo] = useState("");
  const [questionSubstring, setQuestionSubstring] = useState("");
  const [toolName, setToolName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setSummary(await fetchCacheSummary());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 10_000);
    return () => clearInterval(id);
  }, [refresh]);

  // Reset strategy to first allowed when target changes.
  useEffect(() => {
    const opts = STRATEGY_OPTIONS[target];
    if (!opts.find((o) => o.value === strategy)) {
      setStrategy(opts[0].value);
    }
  }, [target, strategy]);

  const currentStrategy = STRATEGY_OPTIONS[target].find((o) => o.value === strategy);
  const needs = currentStrategy?.needs ?? [];
  const isNuke = (target === "all" && strategy === "all") || strategy === "all";

  async function submit() {
    setError(null);
    setLastResult(null);
    if (isNuke) {
      const ok = window.confirm(
        target === "all"
          ? "Confirm: NUKE all Q&A + Investigation + Tool caches. This cannot be undone."
          : `Confirm: drop all of the ${target} cache. This cannot be undone.`,
      );
      if (!ok) return;
    }
    setBusy(true);
    try {
      const body: CachePurgeRequest = { target, strategy };
      if (needs.includes("older_than_hours")) body.older_than_hours = olderHours;
      if (needs.includes("repo")) body.repo = repo.trim();
      if (needs.includes("question_substring")) body.question_substring = questionSubstring.trim();
      if (needs.includes("tool_name")) body.tool_name = toolName.trim();
      const res = await purgeCache(body);
      const counts = [
        res.purged_qa ? `Q&A: ${res.purged_qa}` : null,
        res.purged_investigation ? `Investigation: ${res.purged_investigation}` : null,
        res.purged_tool ? `Tool: ${res.purged_tool}` : null,
      ].filter(Boolean).join(" · ");
      setLastResult(counts || (res.detail ?? "Purged (Qdrant doesn't return count)"));
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const fmtSecs = (s?: number) => (s ? `${Math.round(s / 86400)}d` : "—");
  const tool = summary?.tool;
  const totalToolHits = (tool?.hits ?? 0) + (tool?.negative_hits ?? 0);
  const hitRate = tool && totalToolHits + tool.misses > 0
    ? `${((totalToolHits / (totalToolHits + tool.misses)) * 100).toFixed(1)}%`
    : "—";

  return (
    <div className="cache-page">
      <div className="main-header">
        <h2>Cache Control</h2>
        <span className="sub">Q&A · Investigation · Tool-output</span>
      </div>

      <div className="cache-stats">
        <StatCard
          title="Q&A semantic cache"
          items={[
            { label: "Entries",    value: summary?.qa.points_count ?? 0 },
            { label: "Threshold",  value: summary?.qa.threshold ?? "—" },
            { label: "TTL default", value: fmtSecs(summary?.qa.default_ttl_seconds) },
          ]}
        />
        <StatCard
          title="Investigation cache"
          items={[
            { label: "Entries",   value: summary?.investigation.total ?? 0 },
            { label: "Available", value: summary?.investigation.available ? "yes" : "no" },
          ]}
        />
        <StatCard
          title="Tool-output micro-cache"
          items={[
            { label: "Size",        value: tool?.size ?? 0 },
            { label: "Hits",        value: tool?.hits ?? 0 },
            { label: "Misses",      value: tool?.misses ?? 0 },
            { label: "Neg. hits",   value: tool?.negative_hits ?? 0 },
            { label: "Hit rate",    value: hitRate },
            { label: "Evictions",   value: tool?.evictions ?? 0 },
          ]}
        />
      </div>

      {tool && Object.keys(tool.by_tool).length > 0 && (
        <div className="cache-by-tool">
          <h3>Per-tool stats</h3>
          <table>
            <thead>
              <tr><th>Tool</th><th>Hits</th><th>Misses</th><th>Neg. hits</th><th>Hit rate</th></tr>
            </thead>
            <tbody>
              {Object.entries(tool.by_tool).map(([name, s]) => {
                const total = s.hits + s.negative_hits + s.misses;
                const hr = total > 0 ? `${(((s.hits + s.negative_hits) / total) * 100).toFixed(0)}%` : "—";
                return (
                  <tr key={name}>
                    <td className="mono">{name}</td>
                    <td>{s.hits}</td>
                    <td>{s.misses}</td>
                    <td>{s.negative_hits}</td>
                    <td>{hr}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      <div className="cache-purge-card">
        <h3>Purge cache</h3>
        <p className="cache-purge-hint">
          Cloudflare-style multi-strategy purge. Pick target + strategy, fill in params, hit Purge.
        </p>

        <div className="cache-purge-grid">
          <div className="cache-purge-field">
            <label>Target cache</label>
            <select value={target} onChange={(e) => setTarget(e.target.value as Target)}>
              <option value="qa">Q&A semantic</option>
              <option value="investigation">Investigation</option>
              <option value="tool">Tool-output</option>
              <option value="all">All caches</option>
            </select>
          </div>

          <div className="cache-purge-field">
            <label>Strategy</label>
            <select value={strategy} onChange={(e) => setStrategy(e.target.value as Strategy)}>
              {STRATEGY_OPTIONS[target].map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>

          {needs.includes("older_than_hours") && (
            <div className="cache-purge-field">
              <label>Older than (hours)</label>
              <input type="number" min={1} value={olderHours} onChange={(e) => setOlderHours(Number(e.target.value))} />
            </div>
          )}
          {needs.includes("repo") && (
            <div className="cache-purge-field">
              <label>Repo (e.g. confluence:SRE)</label>
              <input type="text" value={repo} onChange={(e) => setRepo(e.target.value)} placeholder="confluence:SRE or devops/gitops-repo" />
            </div>
          )}
          {needs.includes("question_substring") && (
            <div className="cache-purge-field">
              <label>Question contains</label>
              <input type="text" value={questionSubstring} onChange={(e) => setQuestionSubstring(e.target.value)} placeholder="kafka, sre goals 2025, ..." />
            </div>
          )}
          {needs.includes("tool_name") && (
            <div className="cache-purge-field">
              <label>Tool name</label>
              <input type="text" value={toolName} onChange={(e) => setToolName(e.target.value)} placeholder="prometheus_query, k8s_list_pods, ..." />
            </div>
          )}
          {/* Purge button sits in the same auto-fit grid as the fields
              so it lines up with Target / Strategy / Older-than on a
              single row when there's space. */}
          <div className="cache-purge-actions">
            <button
              className={`btn ${isNuke ? "btn-danger" : "btn-primary"}`}
              disabled={busy}
              onClick={submit}
            >
              {busy ? <IconSync /> : <IconClose />}
              {busy ? "Purging…" : isNuke ? "Nuke cache" : "Purge"}
            </button>
          </div>
        </div>

        {(lastResult || error) && (
          <div className="cache-purge-feedback">
            {lastResult && <span className="cache-purge-result">✓ {lastResult}</span>}
            {error && <span className="cache-purge-error"><IconClose /> {error}</span>}
          </div>
        )}
      </div>
    </div>
  );
}
