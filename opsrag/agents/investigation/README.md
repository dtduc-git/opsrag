# Hypothesis-driven Investigation Agent

A LangGraph subgraph that turns an alert (or SRE question) into a tree of
falsifiable hypotheses, tests each against targeted retrieval, and
synthesizes the deepest validated chain into a root-cause answer.

Inspired by Datadog's Bits AI SRE design -- see
[Building Bits AI SRE](https://www.datadoghq.com/blog/building-bits-ai-sre/).
Key principle we copied:

> "The agent focuses on the causal relationship between the monitor
> alert and specific telemetry data pertaining to a hypothesis, rather
> than looking at all of the available telemetry data at once."

The anti-pattern we explicitly avoid: dumping every retrieved chunk
into one summarization prompt. That's how earlier agents drifted past
the context window and silently lost signal:

> "Early SRE agents scaled by performing more tool calls and prompting
> an LLM to summarize the responses... This approach proved to have a
> notable shortcoming: incorporating additional telemetry data slowly
> degraded model performance or exceeded the context window limit."

## Phase flow

```
START
  v
bootstrap_context          fetch runbook + past-incident snippets
  v
generate_hypotheses        3-5 diverse root hypotheses (Flash)
  v
test_hypothesis  <------+
  v                    |
decide_next  ----------+   loop until tree exhausted or budget hit
  | pending  -> test_hypothesis
  | recurse  -> generate_sub_hypotheses -> test_hypothesis
  | done     -> synthesize_root_cause
  v
synthesize_root_cause      compose the validated causal chain
  v
END
```

Three-state per node, mirroring Bits AI SRE:

> "Each hypothesis is classified as **validated**, **invalidated**, or
> **inconclusive**."

Each node also carries an evidence list (Citation objects with
`source_id` + `chunk_id`) so per-hypothesis faithfulness can be
measured in eval -- not only the final answer.

## Recursive descent

> "Bits AI SRE breaks down complex hypotheses into sub-hypotheses. When
> a sub-hypothesis is supported by evidence, the agent digs deeper. If
> not, it looks elsewhere."

In our graph this is implemented inside `decide_next`: a validated
node with confidence above the depth-aware threshold spawns children;
otherwise the DFS cursor advances to the next pending sibling.

## Hard limits (circuit breakers)

Defined in `limits.py`. Any trip -> graceful termination: remaining
pending nodes marked inconclusive, partial synthesis attempted, metric
emitted.

| Limit | Value | Purpose |
|---|---|---|
| `MAX_DEPTH` | 5 | hard recursion ceiling |
| `SOFT_DEPTH` | 3 | past this, narrower fanout + stricter confidence |
| `MAX_HYPOTHESES_PER_LEVEL` | 5 | siblings at depth <= SOFT_DEPTH |
| `MAX_HYPOTHESES_DEEP` | 2 | siblings past SOFT_DEPTH |
| `MIN_CONFIDENCE_TO_RECURSE` | 0.7 | at depth <= SOFT_DEPTH |
| `MIN_CONFIDENCE_DEEP` | 0.8 | past SOFT_DEPTH |
| `MAX_TOTAL_NODES` | 50 | global tree-size kill |
| `MAX_TOTAL_TOOL_CALLS` | 300 | retrieval + LLM combined |
| `MAX_INVESTIGATION_DURATION_SEC` | 300 | 5-min wall clock |
| `MAX_LLM_TOKENS_PER_INVESTIGATION` | 1,000,000 | token safety net |
| `DUPLICATE_ANCESTOR_COSINE_THRESHOLD` | 0.9 | rephrase-loop guard |

The duplicate-ancestor check chokes the loop where an LLM rewrites the
parent under a slightly different label and the agent would otherwise
recurse forever. When triggered, the child is marked inconclusive with
`termination_reason="duplicate_ancestor"`.

## Cost expectations

| Scenario       | Nodes | Tool calls | Latency  | Cost (gemini-flash) |
|----------------|-------|------------|----------|---------------------|
| Realistic      | ~30   | ~180       | 2-5 min  | $0.10-0.30          |
| Hard cap (CB)  | 50    | 300        | 5 min    | ~$0.50              |
| WITHOUT caps   | 906   | 6,340      | 30+ min  | $5-10               |

-> Circuit breakers cut worst-case cost roughly 20x. The realistic-case
numbers come from internal pilot traces (`tests/eval/specs/phase3-*`).

## Design decisions

### Flat node store, not nested Pydantic

`InvestigationState.nodes_by_id` is a flat `dict[str, HypothesisNode]`,
and `HypothesisNode.children` is `list[str]` (IDs), not nested objects.
Two reasons:
1. **Checkpoint serialization** -- recursive Pydantic models break
   LangGraph's checkpoint reducer on tree updates.
2. **Cheap DFS cursor** -- `next_pending_id()` is just a stack walk, no
   tree mutation required.

The "tree" lives in the parent-id pointers + children-id lists; render
it when you need to (`_deepest_validated_chain`, debug print).

### Bootstrap before generation

The `bootstrap_context` node runs runbook + past-incident retrievals
**before** any hypothesis exists. This grounds the generator in real
context (concrete service names, prior failure modes) instead of
hallucinating generic SRE platitudes. Same step Bits AI SRE does.

### Strict JSON parsing with fallback

Both `generate_hypotheses` and the evidence judge use JSON output. We
strip code fences, try a strict parse, then fall back to a regex sweep
for the JSON block. If parsing still fails we return safe defaults
(empty hypothesis list, or inconclusive verdict). The agent never
hard-fails on malformed LLM output -- the worst case is the tree never
grows, which trips a graceful termination.

### LLM call accounting in one place

`_call_llm_raw` in `graph.py` is the single sink for every LLM call.
It checks budget BEFORE the call and records tokens + per-purpose
counters AFTER. Without this, per-purpose token attribution would
scatter across nodes and the Datadog dashboard would be wrong.

### Retrieval is read-only

By contract, the `RetrieveFn` callable only reads. No tools mutate
external state. This is enforced at the type-signature level (callers
must wrap their retriever; nothing in this module imports a writer).

## Wiring into OpsRAG

This subgraph is intentionally **not** connected to the existing
multi-agent live-query graph (`opsrag/agent/graph.py`). It's a
separate path the Slack bot will invoke for alert-investigation
intents.

Adapter responsibilities at the caller:

1. Build a `RetrieveFn` that wraps the existing
   `vector_store.search()` + chunk-to-dict projection.
2. Pass `embed_query=embedder.embed_query` so the duplicate-ancestor
   check works.
3. Pass an `llm` provider -- Flash is the cost-optimal default; Pro
   is overkill for hypothesis generation and judge.
4. After the run, persist `state.agent_trace` to
   `phase3-hypothesis-<timestamp>.json` for eval replay.

## Observability

`observability.py` emits two JSON log lines:

- `opsrag.investigation.complete` -- at synthesis. Datadog log-to-metric
  pipeline turns this into the dashboard counters in
  `dashboards/investigation_metrics.md`.
- `opsrag.investigation.circuit_breaker_hit` -- at WARN, tagged with
  `breaker:<name>`. Useful for an alert on `breaker:max_duration`
  rate creeping up.

## Eval surface

Each `HypothesisNode.evidence` is a list of `Citation(source_id,
chunk_id)`. The eval harness uses these to score per-hypothesis
faithfulness independently of the final-answer faithfulness. Keep
the Slack output faithfulness floor at 0.45 (the existing constraint).

## Future work

- Phoenix span integration -- every node + tool call -> child span.
- Per-source retriever specialization (Confluence-only for runbook
  questions, Rootly-only for "has this happened before").
- Adaptive fanout based on the alert's severity.
- Re-validation pass: when a child invalidates a parent's assumption,
  walk back up and consider downgrading the parent's confidence.
