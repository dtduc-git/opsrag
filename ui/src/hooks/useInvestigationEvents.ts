import { useEffect, useRef, useState } from "react";

/** Event envelope the backend writes to opsrag_investigation_events.
 *  Frontend matches on `type` and dispatches into reducer state. */
export interface InvestigationEvent {
  sequence: number;
  type: string;
  payload: Record<string, unknown>;
  tags: string[];
  ts: string;
}

interface UseInvestigationEventsOpts {
  /** Investigation UUID to subscribe to. When null, the hook is a no-op
   *  — used so the page can mount before a POST creates an id. */
  investigationId: string | null;
  /** Highest sequence already applied from the snapshot fetch. The
   *  EventSource opens with `?since=initialSinceSeq` so we don't replay
   *  events the snapshot already covered. Defaults to 0. */
  initialSinceSeq?: number;
  /** Called for every event, in arrival order. Caller dispatches into
   *  state via the type. */
  onEvent: (ev: InvestigationEvent) => void;
}

/** EventSource subscription with auto-reconnect.
 *
 *  Backend recycles the SSE connection every ~30s; we reconnect with
 *  the last seen sequence so events are never replayed or dropped.
 *  Stops automatically once a terminal event arrives
 *  (`investigation_completed` / `investigation_failed`) — the page
 *  doesn't need a live connection after that. */
export function useInvestigationEvents({
  investigationId,
  initialSinceSeq = 0,
  onEvent,
}: UseInvestigationEventsOpts): { connected: boolean; latestSeq: number } {
  const [connected, setConnected] = useState(false);
  const [latestSeq, setLatestSeq] = useState(initialSinceSeq);
  const seqRef = useRef(initialSinceSeq);
  const esRef = useRef<EventSource | null>(null);
  const terminalRef = useRef(false);
  const onEventRef = useRef(onEvent);
  // Always hold the latest onEvent so the effect can use the newest closure
  // without re-subscribing every render.
  useEffect(() => {
    onEventRef.current = onEvent;
  }, [onEvent]);

  useEffect(() => {
    if (!investigationId) return undefined;
    seqRef.current = initialSinceSeq;
    setLatestSeq(initialSinceSeq);
    terminalRef.current = false;
    let cancelled = false;

    const connect = () => {
      if (cancelled || terminalRef.current) return;
      const url = `/api/investigations/${encodeURIComponent(
        investigationId,
      )}/events?since=${seqRef.current}`;
      const es = new EventSource(url);
      esRef.current = es;
      es.onopen = () => {
        if (!cancelled) setConnected(true);
      };
      es.onmessage = (msg) => {
        try {
          const parsed = JSON.parse(msg.data) as InvestigationEvent;
          if (typeof parsed.sequence !== "number") return;
          seqRef.current = Math.max(seqRef.current, parsed.sequence);
          setLatestSeq(seqRef.current);
          onEventRef.current(parsed);
          if (
            parsed.type === "investigation_completed" ||
            parsed.type === "investigation_failed"
          ) {
            terminalRef.current = true;
            es.close();
            esRef.current = null;
            setConnected(false);
          }
        } catch {
          /* keepalive / malformed — ignore */
        }
      };
      es.onerror = () => {
        es.close();
        esRef.current = null;
        if (cancelled || terminalRef.current) return;
        setConnected(false);
        // Backend recycles every ~30s; brief backoff prevents a tight
        // reconnect loop when the server is genuinely down.
        setTimeout(connect, 1500);
      };
    };

    connect();
    return () => {
      cancelled = true;
      esRef.current?.close();
      esRef.current = null;
    };
    // Re-subscribe when the investigation id changes (e.g. user clicks
    // a different one in the sidebar).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [investigationId]);

  return { connected, latestSeq };
}
