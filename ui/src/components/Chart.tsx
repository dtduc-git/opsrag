import { useMemo } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

// Generic, tool-agnostic chart. The reasoner calls the `render_chart` engine
// tool with a spec; the backend emits it as a `render_component: Chart` event
// (live) and rebuilds it identically on replay. Any tool (billing, k8s, ...)
// can visualize data through this — no per-source frontend code. Reuses the
// `ts-chart-*` CSS (see TimeseriesChart) so it matches both themes.

export interface ChartPoint {
  x: string;   // time or category label
  y: number;
}

export interface ChartSeries {
  label: string;
  points: ChartPoint[];
}

export interface ChartProps {
  type: "line" | "bar" | "pie";
  title: string;
  unit?: string | null;
  x_label?: string | null;
  y_label?: string | null;
  series: ChartSeries[];
}

// Shared with TimeseriesChart's palette — primary first so single-series
// charts pick up the brand colour.
const SERIES_COLORS = [
  "var(--primary)",
  "#22c55e",
  "#f59e0b",
  "#ec4899",
  "#06b6d4",
  "#a855f7",
  "#ef4444",
  "#84cc16",
];

function formatValue(v: number, unit?: string | null): string {
  if (!Number.isFinite(v)) return "—";
  const abs = Math.abs(v);
  let body: string;
  if (abs === 0) body = "0";
  else if (abs >= 1000) body = v.toLocaleString(undefined, { maximumFractionDigits: abs >= 100000 ? 0 : 2 });
  else if (abs >= 1) body = v.toFixed(2);
  else if (abs >= 0.01) body = v.toFixed(4);
  else body = v.toExponential(2);
  if (!unit) return body;
  // Currency-style units read better as a prefix.
  if (unit === "USD" || unit === "$") return `$${body}`;
  if (unit === "percent" || unit === "%") return `${body}%`;
  return `${body} ${unit}`;
}

interface TooltipPayloadEntry {
  dataKey?: string | number;
  name?: string;
  value?: number;
  color?: string;
  payload?: Record<string, unknown>;
}

function ChartTooltip({ active, payload, label, unit }: {
  active?: boolean;
  payload?: TooltipPayloadEntry[];
  label?: string | number;
  unit?: string | null;
}) {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div className="ts-chart-tooltip">
      {label != null && <div className="ts-chart-tooltip-time">{String(label)}</div>}
      <div className="ts-chart-tooltip-rows">
        {payload.map((p) => (
          <div className="ts-chart-tooltip-row" key={String(p.dataKey ?? p.name)}>
            <span className="ts-chart-tooltip-dot" style={{ background: p.color || "var(--primary)" }} />
            <span className="ts-chart-tooltip-label">{p.name}</span>
            <span className="ts-chart-tooltip-value">{formatValue(Number(p.value), unit)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function Chart(props: ChartProps) {
  const { type, title, unit, series } = props;

  // Wide-format rows keyed by x-label so N series share one x-axis (line/bar).
  const { rows, seriesKeys } = useMemo(() => {
    const xToRow = new Map<string, Record<string, string | number | undefined>>();
    const order: string[] = [];
    const keys: string[] = [];
    (series || []).forEach((s, i) => {
      let key = s.label || `series ${i + 1}`;
      if (keys.includes(key)) key = `${key} (${i + 1})`;
      keys.push(key);
      for (const pt of s.points || []) {
        if (!Number.isFinite(pt.y)) continue;
        const x = String(pt.x ?? "");
        let row = xToRow.get(x);
        if (!row) {
          row = { x };
          xToRow.set(x, row);
          order.push(x);
        }
        row[key] = pt.y;
      }
    });
    return { rows: order.map((x) => xToRow.get(x)!), seriesKeys: keys };
  }, [series]);

  // Pie uses the first series' points as slices (share-of-total).
  const pieData = useMemo(() => {
    const first = (series || [])[0];
    if (!first) return [] as { name: string; value: number }[];
    return (first.points || [])
      .filter((p) => Number.isFinite(p.y))
      .map((p) => ({ name: String(p.x ?? ""), value: p.y }));
  }, [series]);

  const hasData = type === "pie" ? pieData.length > 0 : rows.length > 0;

  return (
    <div className="ts-chart-card">
      <div className="ts-chart-head">
        <div className="ts-chart-title-row">
          <span className="ts-chart-title">{title}</span>
          {unit && <span className="ts-chart-unit">({unit})</span>}
        </div>
        <span className="ts-chart-source">render_chart</span>
      </div>
      {!hasData ? (
        <div className="ts-chart-empty">No data points to plot.</div>
      ) : (
        <div className="ts-chart-canvas">
          <ResponsiveContainer width="100%" height={260}>
            {type === "line" ? (
              <LineChart data={rows} margin={{ top: 8, right: 16, left: 4, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
                <XAxis dataKey="x" stroke="var(--text-3)" tick={{ fill: "var(--text-2)", fontSize: 11 }} minTickGap={24} />
                <YAxis tickFormatter={(v: number) => formatValue(v, unit)} stroke="var(--text-3)" tick={{ fill: "var(--text-2)", fontSize: 11 }} width={64} domain={["auto", "auto"]} />
                <Tooltip content={<ChartTooltip unit={unit ?? null} />} cursor={{ stroke: "var(--primary)", strokeOpacity: 0.3 }} />
                {seriesKeys.length > 1 && <Legend wrapperStyle={{ fontSize: 11 }} />}
                {seriesKeys.map((key, i) => (
                  <Line key={key} type="monotone" dataKey={key} name={key} stroke={SERIES_COLORS[i % SERIES_COLORS.length]} strokeWidth={2} dot={{ r: 2 }} isAnimationActive={false} connectNulls />
                ))}
              </LineChart>
            ) : type === "bar" ? (
              <BarChart data={rows} margin={{ top: 8, right: 16, left: 4, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
                <XAxis dataKey="x" stroke="var(--text-3)" tick={{ fill: "var(--text-2)", fontSize: 11 }} minTickGap={8} />
                <YAxis tickFormatter={(v: number) => formatValue(v, unit)} stroke="var(--text-3)" tick={{ fill: "var(--text-2)", fontSize: 11 }} width={64} domain={["auto", "auto"]} />
                <Tooltip content={<ChartTooltip unit={unit ?? null} />} cursor={{ fill: "var(--primary)", fillOpacity: 0.08 }} />
                {seriesKeys.length > 1 && <Legend wrapperStyle={{ fontSize: 11 }} />}
                {seriesKeys.map((key, i) => (
                  <Bar key={key} dataKey={key} name={key} fill={SERIES_COLORS[i % SERIES_COLORS.length]} isAnimationActive={false} radius={[2, 2, 0, 0]} />
                ))}
              </BarChart>
            ) : (
              <PieChart>
                <Tooltip content={<ChartTooltip unit={unit ?? null} />} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} isAnimationActive={false} label={(e: { name?: string }) => e.name ?? ""}>
                  {pieData.map((_, i) => (
                    <Cell key={i} fill={SERIES_COLORS[i % SERIES_COLORS.length]} />
                  ))}
                </Pie>
              </PieChart>
            )}
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
