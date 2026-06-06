import { useState, useEffect, useRef, useCallback } from "react";
import * as echarts from "echarts";
import { fetchGraphView, type GraphView } from "../api";
import { IconStack, IconSync } from "./icons";

// Knowledge Graph page.
//
// Renders a view-scoped subgraph with ECharts (force-directed graph series).
// Three views — Business / Public facing / Private facing — each fetch a
// filtered subgraph from GET /graph/view?view=... and re-render the canvas.
//
// The graph store is provider-selected (config.knowledge_graph.provider).
// When the backend reports provider=="disabled" (the NullGraphStore default)
// we render an honest "graph is off" state. When the provider is real but no
// nodes come back we render an "empty for this view" state.

type ViewId = "business" | "public" | "private";

const VIEWS: { id: ViewId; label: string; labels: string[] }[] = [
  { id: "business", label: "Business", labels: ["Service", "Team", "Database", "Dependency"] },
  { id: "public", label: "Public facing", labels: ["Gateway", "Route", "Host", "Middleware", "Service"] },
  { id: "private", label: "Private facing", labels: ["Service", "Namespace", "Cluster", "Config", "Infra", "Repository", "Database"] },
];

// Node type → colour. Reused from the design-scratch mockup palette so the
// graph stays visually consistent with the rest of the dashboard. Unknown
// types fall back to a neutral grey.
const TYPE_COLORS: Record<string, string> = {
  Gateway: "#16a34a",
  Route: "#2563eb",
  Service: "#5b4cdb",
  Namespace: "#64748b",
  Cluster: "#9333ea",
  Host: "#db2777",
  Middleware: "#dc2626",
  Team: "#0891b2",
  Database: "#d97706",
  Dependency: "#0d9488",
  Config: "#7c6df0",
  Infra: "#0284c7",
  Repository: "#7c6df0",
  Runbook: "#ca8a04",
  Alert: "#e11d48",
  Person: "#475569",
};
const FALLBACK_COLOR = "#94a3b8";
const colorFor = (type: string): string => TYPE_COLORS[type] ?? FALLBACK_COLOR;

export default function KnowledgeGraphPage() {
  const [view, setView] = useState<ViewId>("business");
  const [data, setData] = useState<GraphView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // The ECharts instance is created once and reused across view switches /
  // refreshes; we only swap its `option`.
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);

  const refresh = useCallback(async (v: ViewId) => {
    setLoading(true);
    try {
      const g = await fetchGraphView(v);
      setData(g);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch on mount + whenever the active view changes.
  useEffect(() => {
    void refresh(view);
  }, [view, refresh]);

  // Init the chart once, dispose on unmount, and keep it sized to its box.
  useEffect(() => {
    if (!containerRef.current) return;
    const chart = echarts.init(containerRef.current);
    chartRef.current = chart;
    const onResize = () => chart.resize();
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  const provider = data?.provider ?? "—";
  const disabled = data?.provider === "disabled";
  // The always-on lightweight entity-graph (opsrag_entity_edges). Its
  // labels/relations don't map to the three Neo4j-schema views, so we render
  // it as a single unified graph instead of the view tabs.
  const isEntityGraph = data?.provider === "entity-graph";
  const isEmpty = !!data && !disabled && data.nodes.length === 0;
  const hasGraph = !!data && !disabled && data.nodes.length > 0;

  // Build + apply the ECharts option whenever we have a renderable graph.
  // Categories are the DISTINCT node types actually present in the subgraph,
  // each tinted via the palette so the legend doubles as a type key.
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    if (!hasGraph || !data) {
      chart.clear();
      return;
    }
    const cats = Array.from(new Set(data.nodes.map((n) => n.type)));
    chart.setOption(
      {
        tooltip: {},
        legend: [{ data: cats, top: 6, textStyle: { fontSize: 11 } }],
        series: [
          {
            type: "graph",
            layout: "force",
            roam: true,
            draggable: true,
            categories: cats.map((c) => ({ name: c, itemStyle: { color: colorFor(c) } })),
            label: { show: true, position: "bottom", fontSize: 10, color: "#3f3f46" },
            edgeLabel: {
              show: true,
              fontSize: 8,
              color: "#a0a0a8",
              formatter: (p: { data?: { type?: string } }) => p.data?.type ?? "",
            },
            force: { repulsion: 240, edgeLength: 90 },
            edgeSymbol: ["none", "arrow"],
            edgeSymbolSize: 6,
            lineStyle: { color: "#d0d0d0", curveness: 0.1 },
            data: data.nodes.map((n) => ({
              id: n.id,
              name: n.name,
              category: n.type,
              symbolSize: n.type === "Service" || n.type === "Gateway" ? 24 : 16,
            })),
            links: data.edges.map((e) => ({ source: e.source, target: e.target, type: e.type })),
          },
        ],
      },
      true, // notMerge — fully replace prior view's option
    );
    chart.resize();
  }, [data, hasGraph]);

  // Legend: for the Neo4j views, the view's fixed label set; for the active
  // entity-graph, the distinct node types actually present in the data.
  const activeLabels = isEntityGraph
    ? Array.from(new Set((data?.nodes ?? []).map((n) => n.type)))
    : VIEWS.find((v) => v.id === view)?.labels ?? [];

  return (
    <div className="page">
      <div className="kg-toolbar">
        {isEntityGraph ? (
          // The lightweight entity-graph is a single unified graph; the
          // Neo4j-schema view tabs don't apply, so show a label instead.
          <div className="tab-bar">
            <span className="tab-bar-item active" style={{ cursor: "default" }}>
              Active entity-graph
            </span>
          </div>
        ) : (
          <div className="tab-bar">
            {VIEWS.map((v) => (
              <button
                key={v.id}
                className={`tab-bar-item ${view === v.id ? "active" : ""}`}
                onClick={() => setView(v.id)}
              >
                {v.label}
              </button>
            ))}
          </div>
        )}
        <div className="kg-toolbar-right">
          <span className="cache-stat-label">Provider</span>
          <span className="cache-stat-value">{provider}</span>
          <span className={`badge ${hasGraph ? "badge-positive" : "badge-neutral"}`}>
            {disabled ? "off" : isEntityGraph ? "entity-graph" : hasGraph ? "neo4j" : "empty"}
          </span>
          <button className="btn btn-secondary" onClick={() => void refresh(view)} disabled={loading}>
            <IconSync /> Refresh
          </button>
        </div>
      </div>

      {/* Type legend for the active view (also serves as empty-state copy). */}
      <div className="kg-legend">
        {activeLabels.map((t) => (
          <span key={t} className="kg-legend-item">
            <span className="kg-legend-sw" style={{ background: colorFor(t) }} />
            {t}
          </span>
        ))}
      </div>

      {/* The ECharts canvas is always mounted (the init effect needs the ref);
          overlays sit on top for loading / disabled / empty / error states. */}
      <div className="kg-canvas-wrap">
        <div ref={containerRef} className="kg-canvas" />

        {loading && (
          <div className="kg-overlay">
            <p>Loading graph…</p>
          </div>
        )}

        {!loading && error && (
          <div className="kg-overlay">
            <div className="empty-state">
              <div className="icon"><IconStack /></div>
              <h3>Could not load this view</h3>
              <p>{error}. The retrieval engine is unaffected — it falls back to vector search.</p>
            </div>
          </div>
        )}

        {!loading && !error && disabled && (
          <div className="kg-overlay">
            <div className="empty-state">
              <div className="icon"><IconStack /></div>
              <h3>Knowledge graph is off</h3>
              <p>
                Knowledge graph is off (<code>knowledge_graph.provider=none</code>).
                Set a provider (e.g. <code>"neo4j"</code>) and reindex to populate it.
              </p>
            </div>
          </div>
        )}

        {!loading && !error && isEmpty && isEntityGraph && (
          <div className="kg-overlay">
            <div className="empty-state">
              <div className="icon"><IconStack /></div>
              <h3>Entity-graph is empty</h3>
              <p>
                No entity edges yet. Re-index a repo, or run{" "}
                <code>POST /api/admin/light-graph/backfill</code> to derive edges
                from already-indexed chunks, then refresh.
              </p>
            </div>
          </div>
        )}

        {!loading && !error && isEmpty && !isEntityGraph && (
          <div className="kg-overlay">
            <div className="empty-state">
              <div className="icon"><IconStack /></div>
              <h3>Graph is empty for this view</h3>
              <p>
                Graph is empty for this view — index a repo with{" "}
                <code>knowledge_graph.provider=neo4j</code>, then refresh.
              </p>
            </div>
          </div>
        )}

        {hasGraph && data?.truncated && (
          <div className="kg-truncated">Showing a capped subset of this view.</div>
        )}
      </div>
    </div>
  );
}
