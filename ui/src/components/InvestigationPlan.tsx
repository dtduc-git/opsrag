// rec #3 — externalized reasoner plan, rendered inline under the
// assistant message. The backend emits `render_component: InvestigationPlan`
// whenever the reasoner called `update_plan` during this turn; ChatMessage
// dispatches the props here.
//
// Item statuses map to colors via `--success` / `--warning` / `--danger`
// CSS vars so both themes work.

import { IconBolt } from "./icons";

export type PlanStatus = "open" | "testing" | "validated" | "invalidated" | "deferred";

export interface PlanItem {
  id: string;
  hypothesis: string;
  status: PlanStatus;
  next_tool?: string | null;
  evidence_so_far?: string;
  confidence?: number;
}

export interface InvestigationPlanProps {
  items: PlanItem[];
}

const STATUS_META: Record<PlanStatus, { label: string; icon: string; cls: string }> = {
  open:        { label: "Open",        icon: "○", cls: "plan-status-open" },
  testing:     { label: "Testing",     icon: "…", cls: "plan-status-testing" },
  validated:   { label: "Validated",   icon: "✓", cls: "plan-status-validated" },
  invalidated: { label: "Invalidated", icon: "✗", cls: "plan-status-invalidated" },
  deferred:    { label: "Deferred",    icon: "↺", cls: "plan-status-deferred" },
};

export default function InvestigationPlan({ items }: InvestigationPlanProps) {
  if (!items || items.length === 0) return null;
  const counts = items.reduce(
    (acc, it) => ({ ...acc, [it.status]: (acc[it.status] ?? 0) + 1 }),
    {} as Record<string, number>
  );

  return (
    <div className="plan-card">
      <div className="plan-head">
        <IconBolt />
        <span className="plan-title">Investigation plan</span>
        <span className="plan-counts">
          {Object.entries(counts).map(([s, n]) => (
            <span key={s} className={`plan-count ${STATUS_META[s as PlanStatus]?.cls ?? ""}`}>
              {STATUS_META[s as PlanStatus]?.icon} {n}
            </span>
          ))}
        </span>
      </div>
      <ol className="plan-list">
        {items.map((it) => {
          const meta = STATUS_META[it.status] ?? STATUS_META.open;
          return (
            <li key={it.id} className={`plan-item ${meta.cls}`}>
              <span className="plan-item-id">[{it.id}]</span>
              <span className="plan-item-status" title={meta.label}>{meta.icon}</span>
              <div className="plan-item-body">
                <div className="plan-item-hypothesis">{it.hypothesis}</div>
                {it.evidence_so_far && (
                  <div className="plan-item-evidence">{it.evidence_so_far}</div>
                )}
                {(it.next_tool || it.confidence != null) && (
                  <div className="plan-item-meta">
                    {it.next_tool && <span className="plan-next">next · <code>{it.next_tool}</code></span>}
                    {it.confidence != null && it.confidence > 0 && (
                      <span className="plan-conf">conf {Math.round(it.confidence * 100)}%</span>
                    )}
                  </div>
                )}
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
