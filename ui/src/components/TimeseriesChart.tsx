import { useMemo } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts";

// — inline Prometheus chart. The backend emits a
// `render_component` SSE event per chartable tool result; App.tsx
// attaches the props to the current assistant message, ChatMessage
// renders this component beneath the markdown. Vanilla CSS (no
// Tailwind) — colors come from the `--primary` / `--surface` /
// `--border` CSS vars so the chart matches both themes.

export interface TimeseriesPoint {
  ts: number;       // Unix seconds
  value: number;
}

export interface TimeseriesSeries {
  label: string;
  labels?: Record<string, string>;
  points: TimeseriesPoint[];
}

export interface TimeseriesChartProps {
  metric_label: string;
  unit?: string | null;
  threshold?: number | null;
  query?: string | null;
  series: TimeseriesSeries[];
  source?: string | null;
}

// Recharts palette — tuned to read well on both dark and light bgs.
// Primary first so single-series queries pick up the brand colour.
const SERIES_COLORS = [
  "var(--primary)",
  "#22c55e",   // emerald
  "#f59e0b",   // amber
  "#ec4899",   // pink
  "#06b6d4",   // cyan
  "#a855f7",   // violet
  "#ef4444",   // red
  "#84cc16",   // lime
];

function formatValue(v: number, unit?: string | null): string {
  if (!Number.isFinite(v)) return "—";
  if (unit === "bytes") {
    const units = ["B", "KB", "MB", "GB", "TB"];
    let i = 0;
    let n = Math.abs(v);
    while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
    return `${(v < 0 ? -n : n).toFixed(n >= 100 ? 0 : n >= 10 ? 1 : 2)} ${units[i]}`;
  }
  if (unit === "seconds") {
    if (Math.abs(v) >= 60) return `${(v / 60).toFixed(1)}m`;
    if (Math.abs(v) >= 1) return `${v.toFixed(2)}s`;
    return `${(v * 1000).toFixed(0)}ms`;
  }
  if (unit === "percent") return `${v.toFixed(1)}%`;
  // Generic — adapt precision to magnitude so a CPU rate like 0.0023
  // doesn't render as "0".
  const abs = Math.abs(v);
  if (abs === 0) return "0";
  if (abs >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (abs >= 1) return v.toFixed(2);
  if (abs >= 0.01) return v.toFixed(4);
  return v.toExponential(2);
}

function formatTime(ts: number, spanSeconds: number): string {
  const d = new Date(ts * 1000);
  // Drop the date when the window is shorter than 24h — keeps axis labels readable.
  if (spanSeconds <= 86400) {
    return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
  }
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

interface TooltipPayloadEntry {
  dataKey?: string | number;
  name?: string;
  value?: number;
  color?: string;
}

function ChartTooltip({ active, payload, label, unit, spanSeconds }: {
  active?: boolean;
  payload?: TooltipPayloadEntry[];
  label?: number;
  unit?: string | null;
  spanSeconds: number;
}) {
  if (!active || !payload || payload.length === 0 || label == null) return null;
  return (
    <div className="ts-chart-tooltip">
      <div className="ts-chart-tooltip-time">{formatTime(label, spanSeconds)}</div>
      <div className="ts-chart-tooltip-rows">
        {payload.map((p) => (
          <div className="ts-chart-tooltip-row" key={String(p.dataKey)}>
            <span className="ts-chart-tooltip-dot" style={{ background: p.color || "var(--primary)" }} />
            <span className="ts-chart-tooltip-label">{p.name}</span>
            <span className="ts-chart-tooltip-value">{formatValue(Number(p.value), unit)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * Strip the longest shared prefix from a list of series keys so the
 * legend shows the discriminating tail (e.g. pod hash) instead of
 * 30 chars of identical `<service>-appservice-<consumer>-`
 * prefix that just wastes pixels.
 *
 * Returns the original key when shortening would leave fewer than 4
 * chars (paranoid edge case for very-similar 2-series queries).
 */
function stripCommonPrefix(keys: string[]): { shortened: string[]; prefix: string } {
  if (keys.length < 2) return { shortened: keys, prefix: "" };
  const sorted = [...keys].sort();
  const first = sorted[0];
  const last = sorted[sorted.length - 1];
  let i = 0;
  while (i < first.length && i < last.length && first[i] === last[i]) i++;
  // Don't cut mid-word: rewind to the previous `-` / `_` / `.` boundary.
  while (i > 0 && !["-", "_", "."].includes(first[i - 1])) i--;
  if (i < 4) return { shortened: keys, prefix: "" };
  const tooShortAfter = keys.some((k) => k.length - i < 4);
  if (tooShortAfter) return { shortened: keys, prefix: "" };
  return {
    shortened: keys.map((k) => k.slice(i)),
    prefix: first.slice(0, i),
  };
}


export default function TimeseriesChart(props: TimeseriesChartProps) {
  const { metric_label, unit, threshold, query, series, source } = props;

  // Build a wide-format dataset keyed by ts so Recharts can render N
  // series sharing the same x-axis. We can't assume every series
  // reported a value at every ts (Prometheus often has gaps), so we
  // merge by timestamp and leave missing values as undefined — the
  // Line component's `connectNulls={false}` then draws a gap rather
  // than a phantom line.
  const { rows, seriesKeys, stats, spanSeconds } = useMemo(() => {
    const tsToRow = new Map<number, Record<string, number | undefined>>();
    const keys: string[] = [];
    let total = 0;
    let count = 0;
    let peak = -Infinity;
    let latestVal: number | null = null;
    let latestTs = -Infinity;
    let firstTs = Infinity;
    let lastTs = -Infinity;
    series.forEach((s, i) => {
      // Disambiguate duplicate labels (rare but possible — two pods
      // can share a name across namespaces in aggregate queries).
      let key = s.label || `series ${i + 1}`;
      if (keys.includes(key)) key = `${key} (${i + 1})`;
      keys.push(key);
      for (const pt of s.points) {
        if (!Number.isFinite(pt.ts) || !Number.isFinite(pt.value)) continue;
        const row = tsToRow.get(pt.ts) || { ts: pt.ts } as Record<string, number | undefined>;
        row[key] = pt.value;
        tsToRow.set(pt.ts, row);
        total += pt.value;
        count++;
        if (pt.value > peak) peak = pt.value;
        if (pt.ts > latestTs) { latestTs = pt.ts; latestVal = pt.value; }
        if (pt.ts < firstTs) firstTs = pt.ts;
        if (pt.ts > lastTs) lastTs = pt.ts;
      }
    });
    const sortedRows = Array.from(tsToRow.values()).sort((a, b) => (a.ts as number) - (b.ts as number));
    const span = Number.isFinite(firstTs) && Number.isFinite(lastTs) ? lastTs - firstTs : 0;
    return {
      rows: sortedRows,
      seriesKeys: keys,
      stats: {
        latest: latestVal,
        avg: count > 0 ? total / count : null,
        peak: Number.isFinite(peak) ? peak : null,
        points: count,
      },
      spanSeconds: span,
    };
  }, [series]);

  const seriesCount = seriesKeys.length;
  const showLegend = seriesCount > 1 && seriesCount <= 8;
  const { shortened: legendLabels, prefix: commonPrefix } = useMemo(
    () => stripCommonPrefix(seriesKeys),
    [seriesKeys]
  );

  // Empty state — render a tiny placeholder instead of nothing so the
  // user sees the chart slot but understands why it's blank (e.g. the
  // query returned an empty matrix but the agent still produced an
  // answer about why).
  if (rows.length === 0) {
    return (
      <div className="ts-chart-card">
        <div className="ts-chart-head">
          <div className="ts-chart-title">{metric_label}</div>
          {source && <span className="ts-chart-source">{source}</span>}
        </div>
        <div className="ts-chart-empty">No data points in the returned timeseries.</div>
      </div>
    );
  }

  return (
    <div className="ts-chart-card">
      <div className="ts-chart-head">
        <div className="ts-chart-title-row">
          <span className="ts-chart-title">{metric_label}</span>
          {unit && <span className="ts-chart-unit">({unit})</span>}
        </div>
        {source && <span className="ts-chart-source">{source}</span>}
      </div>
      {query && (
        <div className="ts-chart-query" title={query}>
          <code>{query}</code>
        </div>
      )}
      <div className="ts-chart-canvas">
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={rows} margin={{ top: 8, right: 16, left: 4, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
            <XAxis
              dataKey="ts"
              type="number"
              scale="time"
              domain={["dataMin", "dataMax"]}
              tickFormatter={(v: number) => formatTime(v, spanSeconds)}
              stroke="var(--text-3)"
              tick={{ fill: "var(--text-2)", fontSize: 11 }}
              minTickGap={48}
            />
            <YAxis
              tickFormatter={(v: number) => formatValue(v, unit)}
              stroke="var(--text-3)"
              tick={{ fill: "var(--text-2)", fontSize: 11 }}
              width={64}
              domain={["auto", "auto"]}
            />
            <Tooltip
              content={<ChartTooltip unit={unit ?? null} spanSeconds={spanSeconds} />}
              cursor={{ stroke: "var(--primary)", strokeOpacity: 0.3, strokeWidth: 1 }}
            />
            {threshold != null && Number.isFinite(threshold) && (
              <ReferenceLine
                y={threshold}
                stroke="var(--primary)"
                strokeDasharray="4 4"
                strokeOpacity={0.65}
                label={{
                  value: `threshold ${formatValue(threshold, unit)}`,
                  fill: "var(--text-2)",
                  fontSize: 10,
                  position: "insideTopRight",
                }}
              />
            )}
            {seriesKeys.map((key, i) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                name={key}
                stroke={SERIES_COLORS[i % SERIES_COLORS.length]}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, strokeWidth: 0 }}
                isAnimationActive={false}
                connectNulls={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      {showLegend && (
        <div className="ts-chart-legend">
          {commonPrefix && (
            <span className="ts-chart-legend-prefix" title={`Common prefix on every series: ${commonPrefix}`}>
              <code>{commonPrefix}*</code>
            </span>
          )}
          {seriesKeys.map((key, i) => (
            <span
              key={key}
              className="ts-chart-legend-item"
              title={key}
            >
              <span
                className="ts-chart-legend-swatch"
                style={{ background: SERIES_COLORS[i % SERIES_COLORS.length] }}
              />
              <span className="ts-chart-legend-label">{legendLabels[i]}</span>
            </span>
          ))}
        </div>
      )}
      <div className="ts-chart-stats">
        <Stat label="Latest" value={stats.latest != null ? formatValue(stats.latest, unit) : "—"} />
        <Stat label="Avg" value={stats.avg != null ? formatValue(stats.avg, unit) : "—"} />
        <Stat label="Peak" value={stats.peak != null ? formatValue(stats.peak, unit) : "—"} />
        <Stat label="Points" value={String(stats.points)} />
        <Stat label="Series" value={String(seriesCount)} />
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="ts-chart-stat">
      <div className="ts-chart-stat-label">{label}</div>
      <div className="ts-chart-stat-value">{value}</div>
    </div>
  );
}
