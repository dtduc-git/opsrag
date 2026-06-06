import { useMemo, useState } from "react";
import type { InvestigationHistoryItem } from "../api";

interface Props {
  investigations: InvestigationHistoryItem[];
  activeInvestigation: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
}

type PillClass = "ok" | "warn" | "stale" | "neutral";

// Map a free-form `outcome` string onto a status pill.
function pillFor(outcome: string): { label: string; cls: PillClass } {
  const o = (outcome || "").toLowerCase();
  if (/resolved|root|validated|found|complete/.test(o)) return { label: "Resolved", cls: "ok" };
  if (/progress|running|active|pending/.test(o)) return { label: "In progress", cls: "warn" };
  if (/inconclusive|unknown|partial/.test(o)) return { label: "Inconclusive", cls: "stale" };
  return { label: outcome || "—", cls: "neutral" };
}

// Relative time from an age in seconds.
function agoSecs(ageSeconds: number): string {
  const s = Math.max(0, Math.floor(ageSeconds));
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  const days = Math.floor(s / 86400);
  if (days === 1) return "Yesterday";
  if (days < 7) return `${days}d ago`;
  return `${Math.floor(days / 7)}w ago`;
}

function truncate(text: string, max: number): string {
  if (text.length <= max) return text;
  return text.slice(0, max).trimEnd() + "…";
}

export default function InvestigationListPane({
  investigations,
  activeInvestigation,
  onSelect,
  onNew,
}: Props) {
  const [search, setSearch] = useState("");

  const q = search.trim().toLowerCase();
  const filtered = useMemo(() => {
    if (!q) return investigations;
    return investigations.filter((inv) => {
      const hay = `${inv.alert_text ?? ""} ${inv.final_root_cause ?? ""}`.toLowerCase();
      return hay.includes(q);
    });
  }, [investigations, q]);

  return (
    <div className="mdl">
      <div className="mdl-toolbar">
        <input
          className="mdl-search"
          type="text"
          placeholder="Search investigations…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      {investigations.length === 0 ? (
        <div className="mdl-empty">
          <div>No investigations yet.</div>
          <button className="mdl-empty-cta" type="button" onClick={onNew}>
            Start an investigation
          </button>
        </div>
      ) : filtered.length === 0 ? (
        <div className="mdl-empty">
          <div>No investigations match.</div>
        </div>
      ) : (
        <div className="mdl-list">
          {filtered.map((inv) => {
            const pill = pillFor(inv.outcome);
            const title = truncate(inv.alert_text || "", 60);
            return (
              <button
                key={inv.investigation_id}
                type="button"
                className={`mdl-row ${inv.investigation_id === activeInvestigation ? "active" : ""}`}
                onClick={() => onSelect(inv.investigation_id)}
              >
                <span className="mdl-row-top">
                  <span className="mdl-title">{title}</span>
                  <span className={`mdl-pill ${pill.cls}`}>{pill.label}</span>
                </span>
                {inv.final_root_cause ? (
                  <span className="mdl-preview">{inv.final_root_cause}</span>
                ) : null}
                <span className="mdl-meta">{agoSecs(inv.age_seconds)}</span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
