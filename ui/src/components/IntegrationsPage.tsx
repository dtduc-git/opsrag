import { useState, useEffect, useCallback, useMemo } from "react";
import { fetchIntegrations, type Integration, type IntegrationsSummary } from "../api";

// Sort: enabled first, then alphabetically by display_name.
function sortIntegrations(items: Integration[]): Integration[] {
  return [...items].sort((a, b) => {
    if (a.enabled !== b.enabled) return a.enabled ? -1 : 1;
    return a.display_name.localeCompare(b.display_name);
  });
}

function IntegrationCard({ integration }: { integration: Integration }) {
  const [expanded, setExpanded] = useState(false);
  const { display_name, name, enabled, tool_count, tool_names, has_health_probe, required_env } = integration;
  const hasTools = tool_names.length > 0;

  return (
    <div className={`cache-card integration-card${enabled ? "" : " disabled"}`}>
      <div className="cache-card-head">
        <div className="title">
          <h4>{display_name}</h4>
        </div>
        <span className={`badge ${enabled ? "badge-grounded" : "badge-neutral"}`}>
          {enabled ? "enabled" : "disabled"}
        </span>
      </div>

      <div className="integration-name mono">{name}</div>

      <div className="integration-meta">
        <span className="chip">{tool_count} tool{tool_count === 1 ? "" : "s"}</span>
        {has_health_probe ? (
          <span className="chip chip-match">probeable</span>
        ) : (
          <span className="chip">no probe</span>
        )}
      </div>

      <div className="integration-env">
        {required_env.length > 0 ? (
          <>
            <span className="detected-label">needs env:</span>
            <div className="detected-chips">
              {required_env.map((env) => (
                <span key={env} className="chip">{env}</span>
              ))}
            </div>
          </>
        ) : (
          <span className="integration-env-none">no env required</span>
        )}
      </div>

      {hasTools && (
        <div className="integration-tools">
          <button
            type="button"
            className="btn btn-secondary integration-toggle"
            onClick={() => setExpanded((v) => !v)}
          >
            {expanded ? "Hide tools" : `Show ${tool_count} tool${tool_count === 1 ? "" : "s"}`}
          </button>
          {expanded && (
            <div className="detected-chips integration-tool-chips">
              {tool_names.map((tool) => (
                <span key={tool} className="chip mono">{tool}</span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function IntegrationsPage() {
  const [summary, setSummary] = useState<IntegrationsSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [enabledOnly, setEnabledOnly] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await fetchIntegrations();
      setSummary(data);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const sorted = useMemo(
    () => (summary ? sortIntegrations(summary.integrations) : []),
    [summary],
  );

  const visible = useMemo(
    () => (enabledOnly ? sorted.filter((i) => i.enabled) : sorted),
    [sorted, enabledOnly],
  );

  // Total tool count across enabled integrations only.
  const enabledToolCount = useMemo(
    () =>
      summary
        ? summary.integrations.reduce((acc, i) => acc + (i.enabled ? i.tool_count : 0), 0)
        : 0,
    [summary],
  );

  return (
    <div className="page">
      <div className="page-title-row" style={{ justifyContent: "flex-end" }}>
        <button type="button" className="btn btn-secondary" disabled={loading} onClick={refresh}>
          {loading ? "Refreshing…" : "Refresh"}
        </button>
      </div>
      <div className="integration-stats">
        <div className="cache-stat-card">
          <div className="cache-stat-row">
            <span className="cache-stat-label">Enabled</span>
            <span className="cache-stat-value">
              {summary ? `${summary.enabled_count} / ${summary.total}` : "—"} enabled
            </span>
          </div>
        </div>
        <div className="cache-stat-card">
          <div className="cache-stat-row">
            <span className="cache-stat-label">Tools (enabled)</span>
            <span className="cache-stat-value">{summary ? enabledToolCount : "—"}</span>
          </div>
        </div>
      </div>

      <div className="integration-controls">
        <div className="integration-filter">
          <button
            type="button"
            className={`btn btn-secondary integration-filter-btn${enabledOnly ? "" : " active"}`}
            onClick={() => setEnabledOnly(false)}
          >
            All
          </button>
          <button
            type="button"
            className={`btn btn-secondary integration-filter-btn${enabledOnly ? " active" : ""}`}
            onClick={() => setEnabledOnly(true)}
          >
            Enabled only
          </button>
        </div>
        <span className="integration-config-note">
          Integrations are enabled via config &amp; deploy — this page is read-only.
        </span>
      </div>

      {error && (
        <div className="card-section integration-error">
          Could not load integrations: {error}
        </div>
      )}

      {!error && summary && visible.length === 0 && (
        <div className="empty-state">
          <h3>No integrations</h3>
          <p>
            {enabledOnly
              ? "No integrations are currently enabled. Enable one via config and redeploy."
              : "The integration registry is empty."}
          </p>
        </div>
      )}

      {visible.length > 0 && (
        <div className="cache-grid integration-grid">
          {visible.map((integration) => (
            <IntegrationCard key={integration.name} integration={integration} />
          ))}
        </div>
      )}
    </div>
  );
}
