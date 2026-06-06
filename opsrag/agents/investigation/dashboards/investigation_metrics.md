# Datadog dashboard -- OpsRAG Investigation Agent

Dashboard JSON should be created via terraform-datadog once we settle
on widget IDs. Until then, this is the spec.

## Data source

All metrics derive from one structured log per investigation:
```json
{
  "event": "opsrag.investigation.complete",
  "investigation_id": "...",
  "duration_sec": 45.2,
  "tree_size": {"total_nodes": 12, "validated": 4, "invalidated": 6, "inconclusive": 2},
  "max_depth_reached": 4,
  "tool_calls": {"retrieval": 28, "llm_query_gen": 12, "llm_judge": 12, "llm_synth": 1},
  "tokens": {"input": 45000, "output": 8000, "total": 53000},
  "circuit_breakers_hit": [],
  "outcome": "validated_root_cause",
  "service": "...",
  "env": "..."
}
```

Datadog log pipeline: JSON parser + log-to-metric on every numeric field
+ a per-`outcome` count. Recommended generated metrics
(`source:opsrag service:opsrag-investigation`):

- `opsrag.investigation.duration_sec` -- gauge
- `opsrag.investigation.tree_size.total_nodes` -- gauge
- `opsrag.investigation.tree_size.validated` -- gauge
- `opsrag.investigation.tree_size.invalidated` -- gauge
- `opsrag.investigation.tree_size.inconclusive` -- gauge
- `opsrag.investigation.max_depth_reached` -- gauge
- `opsrag.investigation.tool_calls.{retrieval,llm_query_gen,llm_judge,llm_synth,total}` -- gauge
- `opsrag.investigation.tokens.{input,output,total}` -- gauge
- `opsrag.investigation.outcome.count` -- counter, tagged `outcome:*`
- `opsrag.investigation.circuit_breaker.count` -- counter, tagged `breaker:*`
  (from the `opsrag.investigation.circuit_breaker_hit` log event)

All metrics tagged with `service`, `env`, `outcome`, plus the global
opsrag tags from the pod.

## Widgets

### Row 1 -- Volume + health

| Widget | Query | Notes |
|---|---|---|
| Investigations / hour | `sum:opsrag.investigation.outcome.count{*}.as_count()` rolled up to 1h | top-left summary |
| Outcome split | `sum:opsrag.investigation.outcome.count{*} by {outcome}` stacked area | validated vs inconclusive vs CB |
| Circuit-breaker rate | `sum:opsrag.investigation.circuit_breaker.count{*} by {breaker}.as_rate()` | alert if `> 0.05` |

### Row 2 -- Tree shape

| Widget | Query |
|---|---|
| p50 / p95 nodes per investigation | `avg:opsrag.investigation.tree_size.total_nodes{*}.rollup(percentile, 50)` and `.rollup(percentile, 95)` |
| Depth distribution | heatmap of `opsrag.investigation.max_depth_reached{*}` by 1-bucket |
| Validated vs invalidated nodes | two-series timeseries from `tree_size.validated` and `tree_size.invalidated` |

### Row 3 -- Cost + latency

| Widget | Query |
|---|---|
| p50 / p95 duration | `opsrag.investigation.duration_sec{*}.rollup(percentile, 50/95)` |
| p50 / p95 total tool calls | `opsrag.investigation.tool_calls.total{*}.rollup(percentile, 50/95)` |
| Tool-call mix | stacked area of `tool_calls.retrieval`, `llm_query_gen`, `llm_judge`, `llm_synth` |
| Token cost per investigation | `opsrag.investigation.tokens.total{*}` x Vertex Flash price ($/MTok) |

### Row 4 -- Per-service breakdown

| Widget | Query |
|---|---|
| Top services by investigation volume | `top(opsrag.investigation.outcome.count{*} by {service}, 10)` |
| Outcome quality by service | bar chart of % `outcome:validated_root_cause` per service |

## Alerts (suggested)

| Alert | Condition |
|---|---|
| `opsrag.investigation.cb_max_duration` rate | `sum:...circuit_breaker.count{breaker:circuit_breaker_max_duration}.as_rate() > 0.10` for 15m -> page |
| `opsrag.investigation.cb_max_nodes` rate | same, `breaker:circuit_breaker_max_nodes > 0.05` -> ticket |
| Inconclusive rate | `outcome:inconclusive` count / total > 0.4 for 1h -> tune the prompt |
| Token blowup | p95 `tokens.total > 500_000` for 1h -> check for runaway loops |

## Cross-link to Phoenix traces

Each investigation should produce a Phoenix trace once Phoenix is live
(Phase A is image mirror, then chart wire-up). Add a "View trace" link
column to the investigation list widget using `@investigation_id`.
