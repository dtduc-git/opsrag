import { useState } from "react";
import { IconCheck, IconClock, IconStack } from "./icons";

export interface ProgressStep {
  node: string;
  label: string;
  status: "active" | "done";
  startedAt: number;
  endedAt?: number;
  /** Live token stream from the reasoner LLM — appended as the model
   * generates. Rendered as a quoted block under the step label so the
   * user sees the agent's chain-of-thought unfold. */
  streamingText?: string;
}

export interface CacheHitInfo {
  similarity: number;
  ageSeconds: number;
}

interface Props {
  steps: ProgressStep[];
  cacheHit?: CacheHitInfo | null;
  /** When true, the answer has finished streaming. Component starts collapsed in that case. */
  finished: boolean;
}

/**
 * Phase 02.7 thinking-timeline. Renders the agent's per-node progress as
 * a vertical list. While streaming, the timeline is expanded so the user
 * sees what's happening; once the answer arrives, it auto-collapses to
 * a one-liner with a "show" toggle.
 */
export default function ThinkingProgress({ steps, cacheHit, finished }: Props) {
  const [expanded, setExpanded] = useState(!finished);

  const stepCount = steps.length;
  const summary = cacheHit
    ? `Returned cached answer · ${(cacheHit.similarity * 100).toFixed(0)}% match · ${formatAge(cacheHit.ageSeconds)}`
    : finished
      ? `${stepCount} step${stepCount === 1 ? "" : "s"} · done`
      : "Thinking…";

  if (steps.length === 0 && !cacheHit) return null;

  return (
    <div className={`thinking-progress ${expanded ? "expanded" : "collapsed"}`}>
      <button
        type="button"
        className="thinking-progress-head"
        onClick={() => setExpanded((v) => !v)}
      >
        <span className="tp-icon">
          {finished ? <IconCheck /> : <IconClock />}
        </span>
        <span className="tp-summary">{summary}</span>
        <span className="tp-toggle">{expanded ? "hide" : "show"}</span>
      </button>

      {expanded && (
        <ol className="thinking-progress-steps">
          {cacheHit && (
            <li className="tp-step done cache">
              <span className="tp-bullet"><IconStack /></span>
              <span className="tp-step-label">Cached answer · {(cacheHit.similarity * 100).toFixed(0)}% match</span>
              <span className="tp-step-meta">{formatAge(cacheHit.ageSeconds)}</span>
            </li>
          )}
          {steps.map((s, i) => (
            <li key={`${s.node}-${i}`} className={`tp-step ${s.status}`}>
              <div className="tp-step-row">
                <span className="tp-bullet">
                  {s.status === "done" ? <IconCheck /> : <span className="tp-spinner" />}
                </span>
                <span className="tp-step-label">{s.label}</span>
                <span className="tp-step-meta">{formatDuration(s)}</span>
              </div>
              {s.streamingText && (
                <div className="tp-step-thinking">{s.streamingText}</div>
              )}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

function formatDuration(s: ProgressStep): string {
  if (s.status === "active") {
    const ms = Date.now() - s.startedAt;
    return ms > 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
  }
  if (s.endedAt == null) return "";
  const ms = s.endedAt - s.startedAt;
  return ms > 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
}

function formatAge(seconds: number): string {
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h ago`;
  return `${Math.round(seconds / 86400)}d ago`;
}
