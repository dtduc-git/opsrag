// — structured-JSON architecture/component diagrams.
//
// Replaces the mermaid renderer. Backend emits a fenced
// ```diagram-json``` code block with `{nodes, edges}`; ChatMessage
// passes the parsed JSON here. We run dagre for auto-layout and
// React Flow for the actual rendering — same stack the
// InvestigationPage already uses, so styles, controls, and zoom
// behavior are consistent across the app.
//
// Why structured JSON instead of mermaid:
//   1. Zero parse errors (vs. mermaid's "syntax error in text" on
//      partial / unicode-arrow streams we kept hitting).
//   2. Auto-layout via dagre — no LLM coordinate math.
//   3. Stylable per-`kind` (service/storage/queue/gateway/external).
//   4. Reuses the @xyflow/react + @dagrejs/dagre we already ship.

import { useMemo } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  BackgroundVariant,
  Controls,
  MarkerType,
  Position,
  type Node as RFNode,
  type Edge as RFEdge,
} from "@xyflow/react";
import dagre from "@dagrejs/dagre";
import "@xyflow/react/dist/style.css";

export type DiagramNodeKind =
  | "actor"      // human / external client
  | "service"    // internal microservice / application
  | "storage"    // database / GCS bucket / volume
  | "queue"      // Kafka topic / Pub/Sub / SQS / Redis queue
  | "gateway"    // Kong / Istio / load balancer
  | "external";  // third-party SaaS (Ada, SendGrid, etc.)

export interface DiagramNode {
  id: string;
  label: string;
  kind?: DiagramNodeKind;
  /** Optional repo / source-path hint, rendered as subtitle line. */
  repo?: string;
}

export interface DiagramEdge {
  from: string;
  to: string;
  /** Optional edge label (e.g. "HTTP POST /api/v1/widget"). */
  label?: string;
  /** When true the edge is dashed to convey async / eventual delivery. */
  async?: boolean;
}

export interface DiagramData {
  nodes: DiagramNode[];
  edges: DiagramEdge[];
  /** Layout direction. Default `LR` (left-to-right flow diagrams). */
  direction?: "TB" | "LR";
  /** Optional title rendered in the card header. */
  title?: string;
}

// Brand-aligned palettes per node kind. Picked from the app's accent
// palette in index.css; intentionally low-contrast so
// labels stay legible in both light and dark UI themes.
const NODE_KIND_STYLE: Record<DiagramNodeKind, { bg: string; border: string; fg: string }> = {
  actor:    { bg: "#fef3c7", border: "#f59e0b", fg: "#78350f" },
  service:  { bg: "#dbeafe", border: "#3b82f6", fg: "#1e3a8a" },
  storage:  { bg: "#e5e7eb", border: "#6b7280", fg: "#1f2937" },
  queue:    { bg: "#fed7aa", border: "#f97316", fg: "#7c2d12" },
  gateway:  { bg: "#ede9fe", border: "#8b5cf6", fg: "#4c1d95" },
  external: { bg: "#fce7f3", border: "#ec4899", fg: "#831843" },
};

const NODE_W = 200;
const NODE_H = 64;

function buildLayout(nodes: DiagramNode[], edges: DiagramEdge[], dir: "TB" | "LR"): dagre.graphlib.Graph {
  const g = new dagre.graphlib.Graph();
  g.setGraph({
    rankdir: dir,
    // Wider separation so edge labels (up to ~5 short words) don't
    // overlap with adjacent nodes or each other. Bumped after operator
    // feedback that the first-cut layout was too cramped to read.
    ranksep: dir === "LR" ? 160 : 110,
    nodesep: dir === "LR" ? 60 : 96,
    edgesep: 28,
    marginx: 32,
    marginy: 24,
  });
  g.setDefaultEdgeLabel(() => ({}));
  for (const n of nodes) {
    // Tall nodes when a repo subtitle is present so the text fits.
    const h = n.repo ? NODE_H + 16 : NODE_H;
    g.setNode(n.id, { width: NODE_W, height: h });
  }
  for (const e of edges) {
    if (g.hasNode(e.from) && g.hasNode(e.to)) {
      g.setEdge(e.from, e.to);
    }
  }
  dagre.layout(g);
  return g;
}

export default function DiagramComponent({ data }: { data: DiagramData }) {
  const dir: "TB" | "LR" = data.direction ?? "LR";

  const { rfNodes, rfEdges, height } = useMemo(() => {
    const g = buildLayout(data.nodes, data.edges, dir);

    const rfNodes: RFNode[] = data.nodes.map((n) => {
      const pos = g.node(n.id);
      const style = NODE_KIND_STYLE[n.kind ?? "service"];
      const nodeH = n.repo ? NODE_H + 16 : NODE_H;
      return {
        id: n.id,
        position: {
          x: (pos?.x ?? 0) - NODE_W / 2,
          y: (pos?.y ?? 0) - nodeH / 2,
        },
        data: {
          label: (
            <div className="diagram-node-body">
              <div className="diagram-node-label">{n.label}</div>
              {n.repo && <div className="diagram-node-sub">{n.repo}</div>}
            </div>
          ),
        },
        type: "default",
        style: {
          background: style.bg,
          border: `1px solid ${style.border}`,
          color: style.fg,
          borderRadius: 8,
          fontSize: 12,
          fontWeight: 500,
          padding: "8px 10px",
          width: NODE_W,
          minHeight: nodeH,
          textAlign: "center" as const,
        },
        sourcePosition: dir === "LR" ? Position.Right : Position.Bottom,
        targetPosition: dir === "LR" ? Position.Left  : Position.Top,
      };
    });

    const rfEdges: RFEdge[] = data.edges.map((e, i) => ({
      id: `e-${i}-${e.from}-${e.to}`,
      source: e.from,
      target: e.to,
      label: e.label,
      labelStyle: { fontSize: 10, fontWeight: 500 },
      labelBgStyle: { fill: "rgba(255,255,255,0.9)" },
      labelBgPadding: [4, 2],
      labelBgBorderRadius: 3,
      style: {
        stroke: "#94a3b8",
        strokeWidth: 1.5,
        strokeDasharray: e.async ? "6 4" : undefined,
      },
      markerEnd: { type: MarkerType.ArrowClosed, color: "#94a3b8", width: 16, height: 16 },
    }));

    // Approximate height from dagre's bounding box so the chat
    // doesn't reserve excessive space for a 3-node diagram.
    const graphHeight = g.graph().height ?? 200;
    return {
      rfNodes,
      rfEdges,
      height: Math.min(720, Math.max(220, graphHeight + 60)),
    };
  }, [data, dir]);

  return (
    <div className="diagram-card">
      {data.title && <div className="diagram-title">{data.title}</div>}
      <ReactFlowProvider>
        <div className="diagram-canvas" style={{ height }}>
          <ReactFlow
            nodes={rfNodes}
            edges={rfEdges}
            fitView
            fitViewOptions={{ padding: 0.15 }}
            nodesDraggable={false}
            nodesConnectable={false}
            elementsSelectable
            zoomOnScroll={false}
            zoomOnPinch
            panOnScroll
            panOnDrag
            minZoom={0.4}
            maxZoom={2}
            proOptions={{ hideAttribution: true }}
          >
            <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#cbd5e1" />
            <Controls showInteractive={false} position="bottom-right" />
          </ReactFlow>
        </div>
      </ReactFlowProvider>
      <DiagramLegend nodes={data.nodes} />
    </div>
  );
}

function DiagramLegend({ nodes }: { nodes: DiagramNode[] }) {
  // Only show legend chips for kinds that actually appear in this diagram.
  const kinds = new Set(nodes.map((n) => n.kind ?? "service"));
  if (kinds.size === 0) return null;
  return (
    <div className="diagram-legend">
      {Array.from(kinds).map((k) => {
        const s = NODE_KIND_STYLE[k];
        return (
          <span key={k} className="diagram-legend-chip" style={{ background: s.bg, color: s.fg, border: `1px solid ${s.border}` }}>
            {k}
          </span>
        );
      })}
    </div>
  );
}
