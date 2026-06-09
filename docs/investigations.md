# Investigations

The Investigate feature runs an autonomous, event-driven root-cause pipeline over an alert: three parallel triage lanes feed a hypothesis board, a reasoner calls live MCP tools to gather evidence, an evaluator grades each hypothesis, and a generator writes a cited conclusion — all streamed live to the UI from a durable Postgres event ledger.

The engine lives in `opsrag/investigations/` (`InvestigationRunner`). The HTTP surface is `opsrag/api/routes_investigations.py`. The old hypothesis-tree engine has been removed; this is the only investigation path.

## When the feature shows up (the gate)

The Investigate tab is only surfaced when the operator has enabled at least one **live-telemetry** MCP integration. There is no point running a tool-driven investigation when there are no live signals to pull.

The gate is purely config-driven (`opsrag/investigations/feature_gate.py`):

```python
_INVESTIGATION_TELEMETRY_INTEGRATIONS = (
    "datadog", "prometheus", "kubernetes", "loki",
    "grafana", "splunk", "sentry", "rootly",
)
```

`investigation_live_telemetry_enabled(cfg)` returns `True` iff any of those `mcp:` blocks is enabled. Note `code` is deliberately excluded — code search alone is not live telemetry. The result is exposed to the UI as `investigation_enabled` on `/ui-config` (`opsrag/api/routes.py` -> `UiConfig.investigation_enabled`). A vendor-neutral deployment with no telemetry enabled never sees the tab.

Wiring is gated again at startup (`opsrag/api/server.py`): the runner is only constructed when `session.provider == "postgres"` (the event ledger needs a Postgres pool). The tool registry is built from `ALL_MCP_TOOLS` after every `bind_*` call, keyed by `tool.name`. If the registry is empty, the pipeline still runs but the reasoner round is skipped and every hypothesis stays `untested`.

RBAC: the whole investigate surface (launch, snapshot, SSE) is gated on the `investigate` scope (`require_scope(Scope.INVESTIGATE)`). In `open` auth mode every user carries all scopes, so it is transparent; in `login`/`oidc` mode a chat-only member 403s. See [./auth.md](./auth.md).

## Pipeline overview

One `run_one(inv_id, alert_text)` call emits this event sequence to the ledger:

```
INVESTIGATION_STARTED
INITIAL_INVESTIGATION_STARTED
  -> 3 lanes (Flash, parallel via asyncio.gather)
       LANE_A_COMPLETED   (runbook search)
       LANE_B_COMPLETED   (historical similar investigations)
       LANE_C_COMPLETED   (live probe — Rootly payload fetch)
INSIGHT_READY             (Pro fuses A+B+C into a 4-quadrant card)
HYPOTHESES_GENERATED      (Pro structured output: 3–5 competing hypotheses)
  -> reasoner <-> evaluator loop (up to 3 rounds)
       REASONER_STEP / TOOL_CALLED / TOOL_RESULT  (per tool call)
       HYPOTHESIS_EVALUATED                       (per hypothesis, per round)
CONCLUSION_READY          (Pro prose answer)
INVESTIGATION_COMPLETED
```

Two model tiers are used throughout: a fast **Flash** model for the lanes and the evaluator, and a stronger **Pro** model for insight fusion, hypothesis generation, the reasoner, and the final answer (`opsrag/api/server.py` reads `providers.pro_llm`, falling back to the default `llm`).

### Alert hydration (pre-step)

If the user pastes only a Rootly URL, the raw `alert_text` has no keywords for the semantic lanes to match. Before fan-out, `_hydrate_alert_text` synchronously calls `rootly_get_alert` / `rootly_get_incident` and concatenates the summary, service names, labels, and annotations into a richer internal text. The hydrated text is **internal-only** — the UI's SOURCE ALERT card always shows exactly what the user pasted.

## The three lanes

All three lanes run in parallel (`asyncio.gather`) and are cheap. Each emits a `LANE_*_COMPLETED` event.

- **Lane A — runbooks** (`_lane_a`). Hybrid search over the runbook store using the embedder. A **relevance gate** is critical here: RRF over a tiny runbook corpus always ranks *something* first regardless of topical fit, and the hypothesizer treats the top hit as authoritative. So a hit is kept only if the runbook's service token literally appears in the alert, **or** the runbook's identity vocabulary (service / tags / tag fragments / title tokens, minus generic stopwords) shares >=2 words with the alert. The top hit is rendered with its full body (~1200 chars) so downstream prompts see the runbook's named failure modes verbatim, not a title.
- **Lane B — historical** (`_lane_b`). Similarity search over `investigation_cache` for past investigations resembling this alert. Returns up to 3 hits with a similarity score and summary.
- **Lane C — live probe** (`_lane_c`). When the alert is a Rootly URL, fetches the payload and extracts `env` / `target` / `namespace`. The runner consumes that `extracted` block to fill in those fields before the Pro reasoner runs — without it, a bare-URL alert would default to the prod cluster.

## Insight synthesis

`_synthesize_insight` (Pro) fuses the three lane summaries into a 4-quadrant `InsightCard`:

- `what_we_know` — from the live probe (empty when none)
- `what_weve_seen` — from historical investigations
- `what_runbook_says` — names the matched runbook and lists its failure modes verbatim
- `open_questions` — 2–3 things to verify with tools

The raw Lane A render is stashed on the insight dict (`_lane_a_raw`) **after** the event is emitted, so the hypothesizer can read the full runbook text without leaking the raw block into the UI payload.

## Hypothesis generation

`_generate_hypotheses` (Pro, Pydantic structured output) emits a `HypothesisSet` of 3–5 **mutually exclusive** root-cause hypotheses. Each `HypothesisDraft` carries:

```python
class HypothesisDraft(BaseModel):
    id: str                              # stable "h1", "h2", ...
    text: str = Field(max_length=500)
    discriminating_tools: list[str]      # MCP tools that confirm/rule-out THIS hypothesis
```

`discriminating_tools` is the load-bearing field: it tells the reasoner which evidence to pull next. The hypothesizer is prompted to pick tools that return **content** (logs, error messages, specific values) rather than **absence** (counts, list lengths) — e.g. prefer "get logs" over "list pods" when the hypothesis needs evidence of *what* went wrong.

Two hard rules shape the output:

- **Runbook priority** — when a runbook matched (the prompt key `WHAT THE RUNBOOK SAYS` carries concrete guidance), the runbook is authoritative: hypotheses must be the runbook's failure modes in the runbook's order, causal claims encoded directly, and the model must not generate a competing hypothesis that contradicts the runbook.
- The registered MCP tool names are injected into the system prompt; the model may only pick `discriminating_tools` from that live list.

On any failure the runner falls back to a small set of generic-prior hypotheses so the UI still renders cards. Each hypothesis is stamped `status: "open"` on emit.

## The reasoner <-> evaluator loop

This is the core of the engine — a multi-round loop (max 3 rounds) that mirrors the chat path's ability to *pivot* when a probe comes back empty.

Each round (`_reasoner_tool_round` + `evaluate_hypotheses`):

1. **Reasoner (Pro, function-calling)** picks 2–4 tool calls. It sees the prior rounds' tool calls and the evaluator's current verdicts, so it can change course. The tool set offered is the union of every hypothesis's `discriminating_tools` plus always-on **discovery fundamentals** (`k8s_find_workloads`, `k8s_list_pods`, `rootly_get_alert`) so the model can always *locate* the workload even if no hypothesis named those tools.
2. **Dispatch.** Each tool call is executed (max 6 per round) with a per-tool hard timeout. Errors are captured per-call so one bad tool doesn't sink the round. GitLab tools get a lazily-constructed long-lived client; all other families accept `None` and manage their own connection.
3. **Evaluator (Flash, structured output)** re-grades every hypothesis against *all* evidence collected so far and emits one `HYPOTHESIS_EVALUATED` event per hypothesis — UI cards flip live.

The loop stops when every hypothesis is decided (`confirmed`/`ruled_out`), the reasoner declines to call tools, the round cap is hit, or the budget is exhausted.

### Tool-argument discipline (anti-guessing)

A recurring failure mode is the LLM passing a guessed scope-narrowing argument (`namespace`, `project`, `region`, `cluster`, account id), the tool returning 0 results, and the evaluator misreading absence as a finding. Two scrubbers run on every tool call's args before dispatch:

- `_scrub_placeholder_args` — strips alert-payload placeholders (`"none"`, `"null"`, `"unknown"`, `"n/a"`, empty) wherever they hide, including inside PromQL/SQL/Lucene query strings (`{namespace="none"}`).
- `_drop_ungrounded_scope_args` — drops any scope-narrowing arg whose value can't be traced to the alert text, the insight context, or a **previous tool result this round**. Tool results are appended to the grounding corpus as they arrive, so a real namespace returned by call #1 legitimately grounds call #2 (chained discovery).

The reasoner prompt spells out the same rule and a worked example ("discover before assume"). The optional `OPSRAG_ENV_CLUSTER_MAP` env var lets operators bake in `env -> (cluster, project)` so the reasoner targets the right cluster for a `preprod`/`staging` alert instead of defaulting to prod.

### Evidence rules (the evaluator)

The evaluator (`opsrag/investigations/evaluator.py`) returns a `HypothesisVerdictBatch` — one `HypothesisVerdict` per hypothesis id, with `status`, an `evidence` sentence, a `confidence` in [0,1], and `supporting_tools` citations. Its system prompt enforces two hard rules:

- **Absence is not confirmation.** A tool returning 0 results / empty list / "not found" is *not* evidence the thing is broken — it usually means the filter args were wrong. Such hypotheses are marked `untested`, with the recommended fix written into `evidence` ("rerun without label or with namespace=<y>"). Absence-only or single-source evidence caps confidence at ~0.4.
- **Two independent signals or a smoking gun.** `confirmed` requires *either* a specific error/value from two different tools (e.g. logs + metrics), *or* one smoking-gun log line that names the exact failure mode ("violates not-null constraint", "OOMKilled"). Corroborated signals push confidence to 0.7+.

The four statuses are `confirmed | ruled_out | untested | open`. `untested` is preferred over `open` when uncertain — the user reads it as "still need a tool for this one".

### Per-hypothesis citations

Each verdict's `supporting_tools` carries the exact `#N [tool]` refs from the evidence pool that justify the status (e.g. `["#1 k8s_list_pods", "#3 prometheus_query"]`). The evaluator is told to cite only refs that actually appear in the pool and never invent one — this is the per-hypothesis citation trail surfaced on the UI cards.

## The hard budget

Earlier the engine had no cost/latency ceiling beyond the 3-round cap, so a pathological run (slow tools + a thrashing reasoner) could burn unbounded time. A per-run `_RunBudget` (constructed per investigation, never stored on the shared runner) enforces three circuit breakers:

```python
MAX_INVESTIGATION_WALL_CLOCK_SEC = 240.0   # hard-stop the reasoner loop past this
MAX_INVESTIGATION_TOOL_CALLS     = 40      # cumulative live tool dispatches per run
PER_TOOL_TIMEOUT_SEC             = 45.0    # a hung MCP tool can't stall a round
```

When the wall-clock or tool-call budget is hit, the loop stops, emits a `REASONER_STEP` note (`budget_exhausted`), and synthesizes the conclusion with whatever evidence exists. The per-tool timeout wraps each `tool.call` in `asyncio.wait_for`; a hung tool is recorded as a tool error and the round continues.

## Conclusion

`_generate_answer` (Pro) writes a structured markdown answer for a human reader: `### Root cause`, `### Evidence`, `### Hypotheses summary`, `### Next steps`. The per-hypothesis verdicts are authoritative and are **not** re-evaluated in prose — the UI cards already show them. The generator may only cite tool names it actually observed, and follow-up recommendations must use registered tool names verbatim (no invented `kubectl_*` variants). `_extract_root_cause` pulls the first paragraph under `### Root cause` onto the lifecycle row so the sidebar shows a one-liner.

## The durable event ledger and resumable SSE

Every step writes a typed event to Postgres via `emit_event` (`opsrag/investigations/store.py`). Two tables back the feature:

- `opsrag_investigations` — one tiny lifecycle row (alert_text, status, root_cause, outcome, timestamps). Full state is reconstructable from the events table.
- `opsrag_investigation_events` — append-only, monotonic `sequence` per event.

`emit_event` opens a fresh session per call, commits immediately, and **swallows all errors** — an observability failure must never break the runner. The event types are stable string constants the frontend filters on (`opsrag/investigations/event_types.py`); the UI is forgiving of unknown types, so adding new ones is safe.

Because state lives entirely in the ledger, the SSE stream is **resumable**: a browser refresh or network blip replays from the last seen sequence. The stream deliberately keeps every frame on the default (unnamed) SSE channel and carries the event type inside the JSON payload (`type`), so a single client `onmessage` handler catches all 10+ event types (per the WHATWG spec a named `event:` would only reach a listener registered for that exact name).

The UI hook (`ui/src/hooks/useInvestigationEvents.ts`) opens an `EventSource` with `?since=<lastSeenSeq>`, tracks the max sequence seen, and on `investigation_completed`/`investigation_failed` closes the stream. The backend recycles each connection after ~30s (60 x 0.5s polls) to keep TCP connections short-lived for load balancers; the hook reconnects with the latest sequence, so the short window is invisible to the user.

## HTTP routes

All routes are under `prefix="/investigations"` (Nginx strips `/api/` before forwarding) and gated on `Scope.INVESTIGATE`.

| Method & path | Purpose |
| --- | --- |
| `POST /investigations` | Body `{alert_text}`. Writes the lifecycle row (with `incident_target` extracted at create-time for the sidebar chip), kicks off `runner.run_one` as a detached background task, returns `{investigation_id}` immediately. |
| `GET /investigations?limit=N` | Sidebar listing — most-recent-first lifecycle rows. |
| `GET /investigations/{id}` | Full snapshot — lifecycle row + **every** event to date. The UI calls this on mount/refresh so the page renders without waiting for the SSE stream. |
| `GET /investigations/{id}/events?since=N` | SSE tail-cursor stream of events with `sequence > N`. ~30s window, then the client reconnects with the latest sequence. Unknown ids 404 instead of hanging. |

The runner catches its own errors and always emits a terminal event (`INVESTIGATION_COMPLETED` or `INVESTIGATION_FAILED`), so the UI never hangs waiting for an end state.

## See also

- [./mcp-integrations.md](./mcp-integrations.md) — the MCP tools the reasoner calls, and which ones gate the feature.
- [./multi-environment.md](./multi-environment.md) — how k8s/prometheus/elasticsearch tools select an environment.
- [./auth.md](./auth.md) — the `investigate` scope and auth modes.
- [./configuration.md](./configuration.md) — enabling MCP integrations and model tiers.
