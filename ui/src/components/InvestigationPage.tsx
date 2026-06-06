import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  useInvestigationEvents,
  type InvestigationEvent,
} from "../hooks/useInvestigationEvents";

/** Option B refactor — canvas-style investigate page ported from
 *  gn-agentic-platform's `investigation-flow.tsx`, themed for OpsRAG
 *  (light mode, accent purple, no dark hexes). Visual structure:
 *
 *    AlertSourceCard    (red eyebrow + ALERT pill)
 *       │
 *    StepTile "Initial Investigation"  ← clickable
 *       │  (expands to 3 probe cards → 3 arrows → INITIAL FINDINGS box)
 *       │
 *    StepTile "Hypothesis Investigation"  ← clickable
 *       │  (expands to a row of hypothesis cards with status-coloured
 *       │   ring shadows + circular status badge on the left side)
 *       │
 *    ConclusionCard  (emerald eyebrow + body + Save-as-runbook)
 *
 *  Click a hypothesis card → side panel slides in from the right with
 *  full evidence + tools.
 */

// ───────────────────────────────────────────────────────────────
// Types (carry-over from prior event-driven implementation)
// ───────────────────────────────────────────────────────────────

type HypoStatus = "open" | "confirmed" | "ruled_out" | "untested";

interface InsightCardPayload {
  what_we_know?: string;
  what_weve_seen?: string;
  what_runbook_says?: string;
  open_questions?: string;
  elapsed_ms?: number;
  error?: string | null;
}

interface Hypothesis {
  id: string;
  text: string;
  discriminating_tools?: string[];
  status: HypoStatus;
  evidence?: string;
  confidence?: number;
}

interface ToolCall {
  name: string;
  args?: Record<string, unknown>;
  summary?: string;
  error?: string;
  latency_ms?: number;
}

interface LaneSnapshot {
  hits?: Array<Record<string, unknown>>;
  elapsed_ms?: number;
  summary?: string;
  skipped?: string;
  error?: string;
}

interface InvestigationState {
  investigationId: string | null;
  alertText: string;
  incidentTarget: string | null;
  status: "idle" | "running" | "complete" | "failed";
  insight: InsightCardPayload | null;
  insightElapsedMs: number | null;
  hypotheses: Hypothesis[];
  toolCalls: ToolCall[];
  conclusion: string | null;
  errorDetail: string | null;
  steps: Array<{ type: string; label: string; ts: string }>;
  latestSeq: number;
  laneA: LaneSnapshot | null;
  laneB: LaneSnapshot | null;
  laneC: LaneSnapshot | null;
}

const EMPTY_STATE: InvestigationState = {
  investigationId: null,
  alertText: "",
  incidentTarget: null,
  status: "idle",
  insight: null,
  insightElapsedMs: null,
  hypotheses: [],
  toolCalls: [],
  conclusion: null,
  errorDetail: null,
  steps: [],
  latestSeq: 0,
  laneA: null,
  laneB: null,
  laneC: null,
};

// ───────────────────────────────────────────────────────────────
// Reducer
// ───────────────────────────────────────────────────────────────

type Action =
  | { kind: "reset" }
  | { kind: "set_id"; investigationId: string; alertText: string }
  | { kind: "snapshot_loaded"; snapshot: { investigation: Record<string, unknown>; events: InvestigationEvent[] } }
  | { kind: "event"; event: InvestigationEvent };

function reducer(state: InvestigationState, action: Action): InvestigationState {
  switch (action.kind) {
    case "reset":
      return { ...EMPTY_STATE };
    case "set_id":
      return {
        ...EMPTY_STATE,
        investigationId: action.investigationId,
        alertText: action.alertText,
        status: "running",
      };
    case "snapshot_loaded": {
      let s: InvestigationState = {
        ...EMPTY_STATE,
        investigationId: (action.snapshot.investigation.id as string) ?? null,
        alertText: (action.snapshot.investigation.alert_text as string) ?? "",
        incidentTarget: (action.snapshot.investigation.incident_target as string) ?? null,
        status:
          (action.snapshot.investigation.status as string) === "completed"
            ? "complete"
            : (action.snapshot.investigation.status as string) === "failed"
              ? "failed"
              : "running",
        conclusion: (action.snapshot.investigation.root_cause as string) ?? null,
      };
      for (const ev of action.snapshot.events ?? []) {
        s = applyEvent(s, ev);
      }
      return s;
    }
    case "event":
      return applyEvent(state, action.event);
    default:
      return state;
  }
}

function applyEvent(s: InvestigationState, ev: InvestigationEvent): InvestigationState {
  const p = ev.payload || {};
  const next: InvestigationState = {
    ...s,
    latestSeq: Math.max(s.latestSeq, ev.sequence),
    steps: [...s.steps, { type: ev.type, label: humanizeEventType(ev.type), ts: ev.ts }],
  };
  switch (ev.type) {
    case "investigation_started":
      return {
        ...next,
        alertText: (p.alert_text as string) ?? next.alertText,
        incidentTarget: (p.incident_target as string) ?? next.incidentTarget,
        status: "running",
      };
    case "lane_a_completed":
      return { ...next, laneA: p as LaneSnapshot };
    case "lane_b_completed":
      return { ...next, laneB: p as LaneSnapshot };
    case "lane_c_completed":
      return { ...next, laneC: p as LaneSnapshot };
    case "insight_ready":
      return {
        ...next,
        insight: (p.insight_card as InsightCardPayload) ?? null,
        insightElapsedMs:
          ((p.insight_card as InsightCardPayload | undefined)?.elapsed_ms) ?? null,
      };
    case "hypotheses_generated": {
      const raw = (p.hypotheses as Array<Record<string, unknown>>) ?? [];
      return {
        ...next,
        incidentTarget: (p.incident_target as string) ?? next.incidentTarget,
        hypotheses: raw.map((h, i) => ({
          id: (h.id as string) ?? `h${i + 1}`,
          text: (h.text as string) ?? "",
          discriminating_tools: (h.discriminating_tools as string[]) ?? [],
          status: ((h.status as HypoStatus) ?? "open"),
        })),
      };
    }
    case "hypothesis_evaluated": {
      const hid = (p.hypothesis_id as string) ?? "";
      const status = (p.status as HypoStatus) ?? "open";
      const evidence = (p.evidence as string) ?? "";
      const confidence = (p.confidence as number) ?? 0;
      return {
        ...next,
        hypotheses: next.hypotheses.map((h) =>
          h.id === hid ? { ...h, status, evidence, confidence } : h,
        ),
      };
    }
    case "tool_called":
      return {
        ...next,
        toolCalls: [
          ...next.toolCalls,
          { name: (p.name as string) ?? "?", args: (p.args as Record<string, unknown>) ?? {} },
        ],
      };
    case "tool_result": {
      const tname = (p.name as string) ?? "?";
      const summary = (p.summary as string) ?? "";
      const errorMsg = (p.error as string) ?? undefined;
      const latency = (p.latency_ms as number) ?? undefined;
      const tools = [...next.toolCalls];
      for (let i = tools.length - 1; i >= 0; i--) {
        if (tools[i].name === tname && !tools[i].summary && !tools[i].error) {
          tools[i] = { ...tools[i], summary, error: errorMsg, latency_ms: latency };
          return { ...next, toolCalls: tools };
        }
      }
      return {
        ...next,
        toolCalls: [...next.toolCalls, { name: tname, summary, error: errorMsg, latency_ms: latency }],
      };
    }
    case "conclusion_ready":
      return { ...next, conclusion: (p.answer as string) ?? null };
    case "investigation_completed":
      return {
        ...next,
        status: "complete",
        conclusion: next.conclusion || ((p.root_cause as string) ?? null),
      };
    case "investigation_failed":
      return {
        ...next,
        status: "failed",
        errorDetail: (p.error as string) ?? "investigation failed",
      };
    default:
      return next;
  }
}

function humanizeEventType(t: string): string {
  const m: Record<string, string> = {
    investigation_started: "Investigation started",
    initial_investigation_started: "Initial investigation",
    lane_a_completed: "Runbook lane",
    lane_b_completed: "Historical lane",
    lane_c_completed: "Live probe lane",
    insight_ready: "Insight synthesized",
    hypotheses_generated: "Hypotheses enumerated",
    reasoner_step: "Reasoner step",
    tool_called: "Tool called",
    tool_result: "Tool result",
    hypothesis_evaluated: "Hypothesis evaluated",
    conclusion_ready: "Conclusion written",
    investigation_completed: "Investigation completed",
    investigation_failed: "Investigation failed",
  };
  return m[t] ?? t;
}

// ───────────────────────────────────────────────────────────────
// Inline icons (lucide-react is not installed; keep deps tight)
// ───────────────────────────────────────────────────────────────

const I_check = (cls = "size-3") => (
  <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <polyline points="20 6 9 17 4 12" />
  </svg>
);
const I_x = (cls = "size-3") => (
  <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
  </svg>
);
const I_help = (cls = "size-3") => (
  <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <circle cx="12" cy="12" r="10" />
    <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
    <line x1="12" y1="17" x2="12.01" y2="17" />
  </svg>
);
const I_max = (cls = "size-3") => (
  <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <polyline points="15 3 21 3 21 9" /><polyline points="9 21 3 21 3 15" />
    <line x1="21" y1="3" x2="14" y2="10" /><line x1="3" y1="21" x2="10" y2="14" />
  </svg>
);
const I_book = (cls = "size-3") => (
  <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" /><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
  </svg>
);
const I_brain = (cls = "size-3") => (
  <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d="M9.5 2a4.5 4.5 0 0 0-4.5 4.5c0 1.06.37 2.04 1 2.81A4.5 4.5 0 0 0 5 18.5a4.5 4.5 0 0 0 4.5 4.5h0V2z" />
    <path d="M14.5 2a4.5 4.5 0 0 1 4.5 4.5c0 1.06-.37 2.04-1 2.81A4.5 4.5 0 0 1 19 18.5a4.5 4.5 0 0 1-4.5 4.5h0V2z" />
  </svg>
);
const I_search = (cls = "size-3") => (
  <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
  </svg>
);
const I_alert = (cls = "size-3") => (
  <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
    <line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" />
  </svg>
);

// ───────────────────────────────────────────────────────────────
// URL hash helpers
// ───────────────────────────────────────────────────────────────

function setHashId(invId: string | null) {
  if (invId) window.location.hash = `investigate?id=${invId}`;
  else window.location.hash = "investigate";
}

// ───────────────────────────────────────────────────────────────
// Main component
// ───────────────────────────────────────────────────────────────

export interface InvestigationPageProps {
  loadInvestigationId?: string | null;
  onInvestigationCompleted?: (investigationId: string) => void;
  onNewInvestigation?: () => void;
}

export default function InvestigationPage({
  loadInvestigationId = null,
  onInvestigationCompleted,
  onNewInvestigation,
}: InvestigationPageProps) {
  const [state, dispatch] = useReducer(reducer, EMPTY_STATE);
  const alertInputRef = useRef<HTMLTextAreaElement>(null);
  const [openHypoId, setOpenHypoId] = useState<string | null>(null);
  // The two big spine tiles are clickable to expand/collapse. They
  // auto-open on first data arrival so the operator doesn't have to
  // click around — but stay user-controllable thereafter.
  const [initialExpanded, setInitialExpanded] = useState(false);
  const [hypoExpanded, setHypoExpanded] = useState(false);

  const targetId = loadInvestigationId || state.investigationId;

  // Snapshot fetch on mount / id change.
  useEffect(() => {
    if (!targetId) return;
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch(`/api/investigations/${encodeURIComponent(targetId)}`);
        if (!r.ok) throw new Error(`snapshot ${r.status}`);
        const data = await r.json();
        if (cancelled) return;
        dispatch({ kind: "snapshot_loaded", snapshot: data });
      } catch (exc) {
        if (cancelled) return;
        dispatch({ kind: "event", event: {
          sequence: 0, type: "investigation_failed",
          payload: { error: exc instanceof Error ? exc.message : String(exc) },
          tags: [], ts: new Date().toISOString(),
        }});
      }
    })();
    return () => { cancelled = true; };
  }, [targetId]);

  // Event subscription.
  const handleEvent = useCallback((ev: InvestigationEvent) => {
    dispatch({ kind: "event", event: ev });
  }, []);
  const { connected, latestSeq } = useInvestigationEvents({
    investigationId: targetId,
    initialSinceSeq: state.latestSeq,
    onEvent: handleEvent,
  });

  // Auto-expand tiles when their data arrives.
  useEffect(() => {
    if (state.insight && !initialExpanded) setInitialExpanded(true);
  }, [state.insight, initialExpanded]);
  useEffect(() => {
    if (state.hypotheses.length > 0 && !hypoExpanded) setHypoExpanded(true);
  }, [state.hypotheses.length, hypoExpanded]);

  // Notify parent when complete.
  useEffect(() => {
    if (state.status === "complete" && state.investigationId) {
      onInvestigationCompleted?.(state.investigationId);
    }
  }, [state.status, state.investigationId, onInvestigationCompleted]);

  // Submit handler.
  const handleRun = useCallback(async () => {
    const text = (alertInputRef.current?.value || "").trim();
    if (!text) return;
    dispatch({ kind: "reset" });
    setOpenHypoId(null);
    setInitialExpanded(false);
    setHypoExpanded(false);
    onNewInvestigation?.();
    try {
      const r = await fetch("/api/investigations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ alert_text: text }),
      });
      if (!r.ok) throw new Error(`create ${r.status}: ${await r.text()}`);
      const { investigation_id } = await r.json();
      dispatch({ kind: "set_id", investigationId: investigation_id, alertText: text });
      setHashId(investigation_id);
    } catch (exc) {
      dispatch({ kind: "event", event: {
        sequence: 0, type: "investigation_failed",
        payload: { error: exc instanceof Error ? exc.message : String(exc) },
        tags: [], ts: new Date().toISOString(),
      }});
    }
  }, [onNewInvestigation]);

  const openHypo = state.hypotheses.find((h) => h.id === openHypoId) ?? null;
  const conclusionVisible =
    !!state.conclusion ||
    state.status === "failed" ||
    (state.status === "running" && state.hypotheses.length > 0);

  // ─── Render ─────────────────────────────────────────────────
  return (
    <div className="inv2-page">
      {/* Live indicator only — the topbar already has the +New button. */}
      <header className="inv2-header">
        <div className="inv2-header-left">
          <span className="inv2-live-dot" aria-hidden />
          <span className="inv2-eyebrow">
            {state.status === "running" ? "Live · streaming" :
             state.status === "complete" ? "Complete" :
             state.status === "failed" ? "Failed" : "Ready"}
            {connected && state.status === "running" && (
              <span className="inv2-seq"> · seq {latestSeq}</span>
            )}
          </span>
        </div>
      </header>

      {/* Empty state: paste-an-alert form */}
      {!targetId && (
        <section className="inv2-empty">
          <h1 className="inv2-empty-title">
            Paste an <em>alert</em>. Get the <em>root cause</em>.
          </h1>
          <p className="inv2-empty-blurb">
            3-lane retrieval → Pro Insight → competing hypotheses → live tool
            checks → per-hypothesis structured verdicts. Cards flip status
            live as evidence arrives.
          </p>
          <div className="inv2-empty-card">
            <textarea
              ref={alertInputRef}
              className="inv2-alert-input"
              placeholder="[P2][prod/k8s/<service>] High 503 rate (> 19%) from the gateway to <host>"
              rows={5}
            />
            <div className="inv2-empty-foot">
              <span className="inv2-model-chip">Pro · reasoner / insight</span>
              <span className="inv2-model-chip inv2-model-chip--flash">Flash · lane probes / evaluator</span>
              <button className="inv2-investigate-btn" onClick={handleRun}>
                Investigate
              </button>
            </div>
          </div>
        </section>
      )}

      {/* Canvas — vertical spine */}
      {targetId && (
        <div className="inv2-canvas">
          {/* SOURCE */}
          <AlertSourceCard alert={state.alertText} target={state.incidentTarget} />

          <Connector />

          {/* INITIAL INVESTIGATION */}
          <StepTile
            label="Initial Investigation"
            countLabel={`${stepsForLanes(state)} probes`}
            expanded={initialExpanded}
            done={!!state.insight}
            onClick={() => setInitialExpanded((v) => !v)}
          />
          {initialExpanded && (
            <InitialInvestigationExpansion
              state={state}
              onClose={() => setInitialExpanded(false)}
            />
          )}

          <Connector />

          {/* HYPOTHESIS INVESTIGATION */}
          <StepTile
            label="Hypothesis Investigation"
            countLabel={`${state.hypotheses.length} hypotheses`}
            expanded={hypoExpanded}
            done={state.hypotheses.length > 0}
            onClick={() => setHypoExpanded((v) => !v)}
          />
          {hypoExpanded && (
            <HypothesisInvestigationExpansion
              hypotheses={state.hypotheses}
              status={state.status}
              onClose={() => setHypoExpanded(false)}
              onOpenHypothesis={(id) => setOpenHypoId(id)}
            />
          )}

          {/* CONCLUSION */}
          {conclusionVisible && (
            <>
              <Connector active={state.status === "complete"} />
              <ConclusionCard
                conclusion={state.conclusion}
                status={state.status}
                errorDetail={state.errorDetail}
              />
            </>
          )}

          {/* Pipeline detail — collapsed at bottom */}
          {state.steps.length > 0 && (
            <details className="inv2-pipeline-details">
              <summary>
                Pipeline detail · {state.steps.length} events
                {state.status === "running" && " · running…"}
                {state.status === "complete" && " · complete"}
                {state.status === "failed" && " · failed"}
              </summary>
              <ol className="inv2-pipeline-list">
                {state.steps.map((s, i) => (
                  <li key={i} className="inv2-pipeline-step">
                    <span className="inv2-pipeline-icon">{I_check("size-3")}</span>
                    <span className="inv2-pipeline-label">{s.label}</span>
                    <span className="inv2-pipeline-type">{s.type}</span>
                  </li>
                ))}
                {state.toolCalls.map((tc, i) => (
                  <li key={`t${i}`} className={`inv2-pipeline-step ${tc.error ? "inv2-pipeline-step--error" : ""}`}>
                    <span className="inv2-pipeline-icon">
                      {tc.error ? I_x("size-3") : I_check("size-3")}
                    </span>
                    <span className="inv2-pipeline-label">
                      <code>{tc.name}</code>
                      {tc.summary ? ` · ${tc.summary.slice(0, 80)}` : ""}
                      {tc.error ? <span className="inv2-pipeline-err"> {tc.error.slice(0, 100)}</span> : null}
                    </span>
                    {typeof tc.latency_ms === "number" && (
                      <span className="inv2-pipeline-elapsed">{tc.latency_ms}ms</span>
                    )}
                  </li>
                ))}
              </ol>
            </details>
          )}
        </div>
      )}

      {/* Side panel — slides in from right */}
      <HypothesisSidePanel
        hypothesis={openHypo}
        toolCalls={state.toolCalls.filter((tc) =>
          openHypo?.discriminating_tools?.includes(tc.name),
        )}
        onClose={() => setOpenHypoId(null)}
      />
    </div>
  );
}

function stepsForLanes(s: InvestigationState): number {
  // Count of lane probes that have a result yet (out of 3).
  return [s.laneA, s.laneB, s.laneC].filter(Boolean).length;
}

// ───────────────────────────────────────────────────────────────
// Sub-components — themed in OpsRAG light mode
// ───────────────────────────────────────────────────────────────

function AlertSourceCard({ alert, target }: { alert: string; target: string | null }) {
  return (
    <article className="inv2-source">
      <header className="inv2-source-head">
        <span className="inv2-source-icon" aria-hidden>{I_alert("size-3")}</span>
        <span className="inv2-source-eyebrow">SOURCE · ALERT</span>
        {target && <span className="inv2-source-target">target · <b>{target}</b></span>}
      </header>
      <div className="inv2-source-body">
        <p>{alert || "(no alert text)"}</p>
      </div>
    </article>
  );
}

function Connector({ active = false }: { active?: boolean }) {
  return (
    <div className="inv2-connector" aria-hidden>
      <svg width="20" height="36" viewBox="0 0 20 36">
        <defs>
          <marker id={`a-${active}`} viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill={active ? "#0EBA8F" : "#B4BCD4"} />
          </marker>
        </defs>
        <line x1="10" x2="10" y1="0" y2="30" stroke={active ? "#0EBA8F" : "#B4BCD4"} strokeWidth="1.4" markerEnd={`url(#a-${active})`} />
      </svg>
    </div>
  );
}

function StepTile({
  label, countLabel, expanded, done, onClick,
}: {
  label: string; countLabel: string; expanded: boolean; done: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-expanded={expanded}
      className={`inv2-step-tile ${expanded ? "inv2-step-tile--expanded" : ""}`}
    >
      <span className={`inv2-step-badge ${done ? "inv2-step-badge--done" : ""}`}>
        {done ? I_check("size-3") : I_help("size-3")}
      </span>
      <span className="inv2-step-label">{label}</span>
      <span className="inv2-step-count">{countLabel}</span>
      <span className="inv2-step-action" aria-hidden>
        {expanded ? I_x("size-3") : I_max("size-3")}
      </span>
    </button>
  );
}

function InitialInvestigationExpansion({
  state, onClose,
}: {
  state: InvestigationState;
  onClose: () => void;
}) {
  // Build 3 probes from lane snapshots — defaults preserve the visual
  // even before lanes complete.
  const probes = [
    {
      kind: "runbook" as const,
      icon: I_book,
      label: "Runbook",
      title:
        state.laneA?.hits?.length
          ? `${state.laneA.hits.length} matching runbook${state.laneA.hits.length === 1 ? "" : "s"}`
          : "Checking the runbook catalog for SRE guidance",
      status: laneStatusText(state.laneA, "No runbook match"),
      positive: !!state.laneA?.hits?.length,
    },
    {
      kind: "memory" as const,
      icon: I_brain,
      label: "Memory",
      title:
        state.laneB?.hits?.length
          ? `${state.laneB.hits.length} similar past investigation${state.laneB.hits.length === 1 ? "" : "s"}`
          : "Recalling past investigations for related context",
      status: laneStatusText(state.laneB, "No memories found"),
      positive: !!state.laneB?.hits?.length,
    },
    {
      kind: "general" as const,
      icon: I_search,
      label: "General Search",
      title: "Scanning your services for related signals",
      status: state.laneC?.skipped
        ? state.laneC.skipped
        : (state.laneC ? "Probe complete" : "Probing…"),
      positive: !!state.laneC && !state.laneC.skipped,
    },
  ];

  const insight = state.insight;
  const findings: string[] = [];
  if (insight) {
    if (insight.what_we_know) findings.push(insight.what_we_know);
    if (insight.what_weve_seen) findings.push(insight.what_weve_seen);
    if (insight.what_runbook_says) findings.push(insight.what_runbook_says);
  }

  return (
    <div className="inv2-expansion">
      <span className="inv2-expansion-tab">Initial Investigation</span>
      <button
        type="button"
        onClick={onClose}
        aria-label="Collapse initial investigation"
        className="inv2-expansion-close"
      >
        {I_x("size-3")}
      </button>
      <div className="inv2-expansion-meta">
        {state.insightElapsedMs
          ? `Synthesized in ${(state.insightElapsedMs / 1000).toFixed(1)}s`
          : "Synthesizing…"}
      </div>

      <div className="inv2-probe-row">
        {probes.map((p) => {
          const Icon = p.icon;
          return (
            <article key={p.kind} className="inv2-probe">
              <header className="inv2-probe-head">
                <span className="inv2-probe-icon" aria-hidden>{Icon("size-3")}</span>
                <span className="inv2-probe-label">{p.label}</span>
              </header>
              <p className="inv2-probe-title">{p.title}</p>
              <span className={`inv2-probe-status ${p.positive ? "inv2-probe-status--positive" : ""}`}>
                {p.positive ? I_check("size-3") : null}
                {p.status}
              </span>
            </article>
          );
        })}
      </div>

      {/* 3 fanning arrows */}
      <div className="inv2-fan-arrows" aria-hidden>
        {[0, 1, 2].map((i) => (
          <svg key={i} width="100%" height="28" viewBox="0 0 100 28" preserveAspectRatio="none">
            <defs>
              <marker id={`ifan-${i}`} viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#B4BCD4" />
              </marker>
            </defs>
            <line x1="50" x2="50" y1="0" y2="22" stroke="#B4BCD4" strokeWidth="1.4" markerEnd={`url(#ifan-${i})`} />
          </svg>
        ))}
      </div>

      {/* INITIAL FINDINGS box */}
      <div className="inv2-findings">
        <header className="inv2-findings-head">
          <span>Initial Findings</span>
        </header>
        <ul className="inv2-findings-list">
          {findings.length === 0 ? (
            <li className="inv2-findings-placeholder">Waiting for Pro to synthesize…</li>
          ) : findings.map((f, i) => (
            <li key={i} className="inv2-findings-item">{f}</li>
          ))}
          {insight?.open_questions && (
            <li className="inv2-findings-item inv2-findings-item--question">
              <b>Open questions:</b> {insight.open_questions}
            </li>
          )}
        </ul>
      </div>
    </div>
  );
}

function laneStatusText(lane: LaneSnapshot | null, emptyLabel: string): string {
  if (!lane) return "Probing…";
  if (lane.error) return `Error: ${lane.error.slice(0, 60)}`;
  if (lane.skipped) return lane.skipped.replace(/_/g, " ");
  if (lane.hits?.length) {
    const ms = lane.elapsed_ms ? ` · ${lane.elapsed_ms}ms` : "";
    return `${lane.hits.length} hit${lane.hits.length === 1 ? "" : "s"}${ms}`;
  }
  return emptyLabel;
}

function HypothesisInvestigationExpansion({
  hypotheses, status, onClose, onOpenHypothesis,
}: {
  hypotheses: Hypothesis[];
  status: InvestigationState["status"];
  onClose: () => void;
  onOpenHypothesis: (id: string) => void;
}) {
  return (
    <div className="inv2-expansion">
      <span className="inv2-expansion-tab inv2-expansion-tab--hypo">Hypotheses Investigation</span>
      <button
        type="button"
        onClick={onClose}
        aria-label="Collapse hypothesis investigation"
        className="inv2-expansion-close"
      >
        {I_x("size-3")}
      </button>
      <div className="inv2-expansion-meta">
        {hypotheses.length === 0
          ? (status === "running" ? "Enumerating competing hypotheses…" : "No hypotheses generated.")
          : `${hypotheses.length} competing hypotheses · click any card for full evidence`}
      </div>

      {hypotheses.length > 0 ? (
        <div className="inv2-hypo-grid">
          {hypotheses.map((h) => (
            <HypothesisCard
              key={h.id}
              hypothesis={h}
              onOpen={() => onOpenHypothesis(h.id)}
            />
          ))}
        </div>
      ) : (
        <div className="inv2-section-placeholder">Pro is enumerating 3-5 root-cause candidates…</div>
      )}
    </div>
  );
}

function HypothesisCard({ hypothesis, onOpen }: { hypothesis: Hypothesis; onOpen: () => void }) {
  const isConfirmed = hypothesis.status === "confirmed";
  const isEliminated = hypothesis.status === "ruled_out";
  const isUntested = hypothesis.status === "untested";
  const badge =
    isConfirmed ? { cls: "inv2-hbadge--confirmed", icon: I_check("size-3") } :
    isEliminated ? { cls: "inv2-hbadge--eliminated", icon: I_x("size-3") } :
    { cls: "inv2-hbadge--inconclusive", icon: I_help("size-3") };

  const cardCls =
    isConfirmed ? "inv2-hcard--confirmed" :
    isEliminated ? "inv2-hcard--eliminated" :
    isUntested ? "inv2-hcard--untested" :
    "inv2-hcard--open";

  return (
    <button
      type="button"
      onClick={onOpen}
      aria-label={`Open hypothesis ${hypothesis.id.toUpperCase()}`}
      className={`inv2-hcard-btn ${cardCls}`}
    >
      <span className={`inv2-hbadge ${badge.cls}`} aria-hidden>
        {badge.icon}
      </span>
      <article className="inv2-hcard">
        <div className="inv2-hcard-head">
          <span className="inv2-hcard-id">{hypothesis.id.toUpperCase()}</span>
          <span className={`inv2-hcard-pill inv2-hcard-pill--${hypothesis.status}`}>
            {hypothesis.status === "confirmed" ? "VALIDATED" :
             hypothesis.status === "ruled_out" ? "INVALIDATED" :
             hypothesis.status === "untested" ? "UNTESTED" : "OPEN"}
          </span>
          {typeof hypothesis.confidence === "number" && hypothesis.confidence > 0 && (
            <span className="inv2-hcard-conf">{Math.round(hypothesis.confidence * 100)}%</span>
          )}
        </div>
        <p className="inv2-hcard-text">{hypothesis.text}</p>
        {hypothesis.evidence && (
          <p className="inv2-hcard-evidence">{hypothesis.evidence.slice(0, 140)}</p>
        )}
      </article>
    </button>
  );
}

function ConclusionCard({
  conclusion, status, errorDetail,
}: {
  conclusion: string | null;
  status: InvestigationState["status"];
  errorDetail: string | null;
}) {
  return (
    <article className="inv2-conclusion">
      <header className="inv2-conclusion-head">
        <span className="inv2-conclusion-eyebrow">INVESTIGATION CONCLUSION</span>
        {status === "running" && !conclusion && (
          <span className="inv2-section-status">testing hypotheses with live tools…</span>
        )}
        {status === "failed" && <span className="inv2-section-status">failed</span>}
        {status === "complete" && <span className="inv2-section-status">complete</span>}
      </header>
      <div className="inv2-conclusion-inner">
        {conclusion ? (
          <div className="markdown-body">
            <ReactMarkdown>{conclusion}</ReactMarkdown>
          </div>
        ) : status === "failed" ? (
          <div className="inv2-error-banner">⚠ {errorDetail || "investigation failed"}</div>
        ) : (
          <div className="inv2-section-placeholder">
            Pro reasoner is calling discriminating tools to test each hypothesis…
          </div>
        )}
      </div>
    </article>
  );
}

function HypothesisSidePanel({
  hypothesis, toolCalls, onClose,
}: {
  hypothesis: Hypothesis | null;
  toolCalls: ToolCall[];
  onClose: () => void;
}) {
  const open = hypothesis !== null;
  // Trap close on ESC.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open || !hypothesis) return null;

  const statusLabel =
    hypothesis.status === "confirmed" ? "VALIDATED" :
    hypothesis.status === "ruled_out" ? "INVALIDATED" :
    hypothesis.status === "untested" ? "UNTESTED" : "OPEN";

  return (
    <>
      <div className="inv2-panel-overlay" onClick={onClose} aria-hidden />
      <aside className="inv2-panel" aria-label={`Hypothesis ${hypothesis.id} details`}>
        <header className="inv2-panel-head">
          <span className="inv2-panel-eyebrow">Hypothesis · {hypothesis.id.toUpperCase()}</span>
          <button type="button" onClick={onClose} className="inv2-panel-close" aria-label="Close">
            {I_x("size-4")}
          </button>
        </header>
        <div className="inv2-panel-body">
          <div className="inv2-panel-title-row">
            <h2 className="inv2-panel-title">{hypothesis.text}</h2>
            <span className={`inv2-hcard-pill inv2-hcard-pill--${hypothesis.status}`}>
              {statusLabel}
            </span>
          </div>
          {typeof hypothesis.confidence === "number" && hypothesis.confidence > 0 && (
            <div className="inv2-panel-conf">
              Confidence · <b>{Math.round(hypothesis.confidence * 100)}%</b>
            </div>
          )}

          {hypothesis.evidence && (
            <section className="inv2-panel-section">
              <h3 className="inv2-panel-section-title">Evidence</h3>
              <p className="inv2-panel-section-body">{hypothesis.evidence}</p>
            </section>
          )}

          {hypothesis.discriminating_tools && hypothesis.discriminating_tools.length > 0 && (
            <section className="inv2-panel-section">
              <h3 className="inv2-panel-section-title">Discriminating tools</h3>
              <div className="inv2-panel-tools">
                {hypothesis.discriminating_tools.map((t) => (
                  <code key={t} className="inv2-panel-tool">{t}</code>
                ))}
              </div>
            </section>
          )}

          {toolCalls.length > 0 && (
            <section className="inv2-panel-section">
              <h3 className="inv2-panel-section-title">Tool calls this run</h3>
              <ol className="inv2-panel-tool-calls">
                {toolCalls.map((tc, i) => (
                  <li key={i} className={`inv2-panel-toolcall ${tc.error ? "inv2-panel-toolcall--error" : ""}`}>
                    <div className="inv2-panel-toolcall-name">
                      <code>{tc.name}</code>
                      {typeof tc.latency_ms === "number" && (
                        <span className="inv2-panel-toolcall-latency">{tc.latency_ms}ms</span>
                      )}
                    </div>
                    {tc.summary && <div className="inv2-panel-toolcall-body">{tc.summary}</div>}
                    {tc.error && <div className="inv2-panel-toolcall-err">⚠ {tc.error}</div>}
                  </li>
                ))}
              </ol>
            </section>
          )}
        </div>
      </aside>
    </>
  );
}
