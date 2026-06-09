"""LLM prompts for the investigation agent.

Four prompt templates:
  1. HYPOTHESIS_GEN_PROMPT  -- initial root hypotheses from bootstrap
  2. SUB_HYPOTHESIS_GEN_PROMPT -- narrower decomposition of a parent
  3. EVIDENCE_JUDGE_PROMPT  -- rate a hypothesis vs retrieved chunks
  4. ROOT_CAUSE_SYNTH_PROMPT -- final answer from the validated chain

Design notes (per Datadog Bits AI SRE):
- We DO NOT dump all telemetry at once. Each prompt has a focused
  context window: a single hypothesis + its targeted evidence.
- Hypothesis generation must produce *diverse* candidates spanning
  different subsystems -- clustering in one area defeats the search.
- The judge classifies into {validated, invalidated, inconclusive} and
  attaches a confidence score, mirroring the three-state design.
"""
from __future__ import annotations

# -- 1. Root hypothesis generation ----------------------------------

HYPOTHESIS_GEN_PROMPT = """You are an SRE investigation agent. An alert just fired. Your job is to enumerate the most likely root causes as TESTABLE hypotheses.

# Alert
{alert_text}

# Service / namespace / env hints
- Service: {service_hint}
- Namespace: {namespace_hint}
- Env: {env_hint}

# Bootstrap context (runbook excerpts, past-incident summaries)
{bootstrap_findings}

# Prior similar investigations
{past_investigations}

# Instructions
Produce {max_hypotheses} distinct root-cause hypotheses. CRITICAL constraints:

1. ONE PER SUBSYSTEM (hard slot rule) -- assign each hypothesis to a DIFFERENT subsystem slot below; never spend two hypotheses on the same slot. Walk the slots in order and emit the single most likely failure mode for each, until you reach {max_hypotheses}:
   (a) workload / application -- OOM, deadlock, slow query, memory leak, thread starvation
   (b) configuration -- resource limits, env vars, feature flag, secret/cert expiry
   (c) infrastructure -- node, disk, network, DNS, control-plane
   (d) upstream dependency -- DB, cache, message queue, external API
   (e) recent change -- deploy, rollout, gitops merge, config push
   If a slot is clearly irrelevant to this alert, skip it and take the next-most-likely slot -- but never emit two hypotheses for the same slot.
2. TESTABLE -- each must be falsifiable with retrieval against runbooks / past incidents / git history / Slack / Rootly.
3. SPECIFIC -- name the subsystem and the failure mode in one sentence. No vague "service degradation".
4. NO REPETITION -- never repeat or paraphrase one of the others.
5. PRIOR-INVESTIGATION AWARENESS -- the "Prior similar investigations" above are PAST GUESSES, not ground truth (an earlier run may have been confidently wrong). Treat a prior root cause as ONE weak signal: emit a hypothesis matching it ONLY if it independently fits THIS alert's evidence. Do NOT force-include a prior cause, and never let it outrank a subsystem slot that better fits the current alert. If priors are empty, ignore this rule.

# Output format (strict JSON)
Return a JSON object with a `hypotheses` array, no prose:
{{
  "hypotheses": [
    {{"statement": "...", "rationale": "one sentence why this fits the alert"}},
    ...
  ]
}}
"""


# -- 2. Sub-hypothesis generation -----------------------------------

SUB_HYPOTHESIS_GEN_PROMPT = """You are an SRE investigation agent drilling into a validated hypothesis.

A parent hypothesis was just supported by evidence. Now decompose it into NARROWER sub-hypotheses -- one causal level deeper.

# Parent hypothesis (validated)
{parent_statement}

# Parent evidence summary
{parent_evidence_summary}

# Causal chain so far (ancestors, root -> parent)
{ancestor_chain}

# Instructions
Produce {max_hypotheses} sub-hypotheses that drill into a specific MECHANISM behind the parent. Example shape:
  parent="OOM in worker pods"
  child="OOM caused by oversized message-queue payloads"
  grandchild="oversized payloads from inefficient deserialization in a specific message handler"

Constraints:
1. EACH SUB-HYPOTHESIS MUST BE FALSIFIABLE BY A DIFFERENT TOOL OR DATA SIGNAL -- naming the same mechanism with different wording is grounds for rejection. Examples of distinct-enough predictions: (a) a Prometheus metric exceeds threshold X, (b) a Postgres query plan shows behavior Y, (c) a code-grep finds pattern Z, (d) a Datadog log filter returns matches. If two hypotheses would be validated by the SAME tool call on the SAME data, merge them.
2. EACH SUB-HYPOTHESIS MUST BE A MECHANISM OR PROXIMATE CAUSE of the parent -- not a sibling restatement of the parent.
3. NO REPETITION of any ancestor on the chain (the agent will reject duplicates anyway, but don't waste a slot).
4. Each must be testable with targeted retrieval (runbooks, git history, telemetry summaries from past incidents).

# Output format (strict JSON)
Return a JSON object with a `hypotheses` array:
{{
  "hypotheses": [
    {{"statement": "...", "rationale": "one sentence"}},
    ...
  ]
}}
"""


# -- 2b. Live-telemetry tool selection (P0-B) -----------------------

TOOL_SELECT_PROMPT = """You are an SRE investigation agent choosing LIVE telemetry to test ONE hypothesis.

# Alert
- Alert: {alert_text}
- Service: {service_hint}
- Namespace: {namespace_hint}

# Hypothesis to test
{hypothesis_statement}

# Available live tools
{tool_catalog}

# Instructions
Pick AT MOST {max_calls} tool call(s) whose result would directly CONFIRM or REFUTE this hypothesis for THIS service with live data. Prefer the single most decisive signal -- a trace/span search, a monitor's current state, a recent deploy/change event, an incident timeline, or a targeted code/config lookup. Fill `args_json` with a JSON OBJECT (as a string) matching that tool's input schema, using the service/namespace above to scope it. If NO available tool meaningfully tests this hypothesis with live data, return an EMPTY `calls` list -- the agent will fall back to document retrieval.

# Output format (strict JSON)
{{
  "calls": [
    {{"tool": "<exact tool name from the catalog>", "args_json": "{{\\"key\\": \\"value\\"}}", "rationale": "<one line: what result confirms or refutes>"}}
  ]
}}
Return ONLY the JSON object, no surrounding prose.
"""


# -- 3. Evidence judge ----------------------------------------------

EVIDENCE_JUDGE_PROMPT = """You are an SRE evidence judge. Given a hypothesis and a set of retrieved snippets, classify whether the evidence VALIDATES, INVALIDATES, or is INCONCLUSIVE for the hypothesis.

# Alert under investigation
- Alert: {alert_text}
- Affected service: {service_hint}
- Namespace: {namespace_hint}
- Environment: {env_hint}

# Hypothesis
{hypothesis_statement}

# Retrieved evidence
{evidence_snippets}

# Decision rubric
- validated     -- at least one snippet explicitly supports the mechanism named in the hypothesis FOR THIS SPECIFIC service/namespace. Cite the supporting chunk(s).
- invalidated   -- the evidence weighs AGAINST the mechanism. This covers BOTH:
    (i)  explicit contradiction -- a metric shows the opposite trend, or a runbook says this is not the cause for this alert pattern; OR
    (ii) absence where it SHOULD appear -- a directly relevant on-service source (THIS service's runbook, its recent change log, or a past incident on THIS service) is present in the evidence yet says nothing that supports the mechanism. That conspicuous silence weighs against the hypothesis.
  Cite the refuting / conspicuously-silent chunk(s). Use invalidated so the agent stops drilling a dead branch and looks elsewhere.
- inconclusive  -- snippets are merely tangential, unrelated (off-service), missing, or ambiguous -- i.e. you cannot tell either way. DO NOT validate without supporting evidence, but also do NOT invalidate when the evidence is simply thin or off-topic (that's inconclusive, not a refutation).

# CRITICAL SERVICE-ANCHOR RULES
The investigation is about service `{service_hint}` in namespace `{namespace_hint}`.

1. Evidence chunks whose `source` path does NOT include `{service_hint}` (or the linked runbook page, or a clearly-generic SRE knowledge-base doc) are WEAK evidence at best. Examples that DO NOT validate a hypothesis for `{service_hint}`:
   - A changelog entry from an unrelated helm chart (e.g. some third-party <component> chart).
   - A past incident on a DIFFERENT service (e.g. <other-service> when investigating `{service_hint}`).
   - A README from a different repo describing a generic pattern that doesn't reference `{service_hint}`.

2. When all snippets are from unrelated services, return `inconclusive` with confidence <= 0.3. NEVER `validated`.

3. Only return `validated` with confidence > 0.6 when at least one citation is from:
   (a) the `{service_hint}` repo or a path containing `{service_hint}` / `{namespace_hint}`, OR
   (b) the linked runbook for THIS alert, OR
   (c) LIVE TELEMETRY -- a result from a live tool (its source id starts with `tool:` -- e.g. Datadog traces/monitors/events, a Rootly incident timeline, or a targeted code/config lookup) that describes `{service_hint}` directly. Live tool evidence is the STRONGEST signal: a trace/monitor/event/timeline that confirms the mechanism VALIDATES it; one that shows the opposite trend -- or in which the signal you'd expect for this mechanism is conspicuously ABSENT -- INVALIDATES it.

4. Be strict: when in doubt, return `inconclusive`. Hallucinating validation across unrelated services costs the team hours.

# Output format (strict JSON)
{{
  "status": "validated" | "invalidated" | "inconclusive",
  "confidence": <float 0.0-1.0>,
  "rationale": "<one-line explanation, MUST mention whether evidence is from the affected service or not>",
  "supporting_chunk_ids": ["<chunk_id>", ...],
  "refuting_chunk_ids":  ["<chunk_id>", ...]
}}

Return ONLY the JSON object, no surrounding prose.
"""


# -- 4. Root-cause synthesis ----------------------------------------

ROOT_CAUSE_SYNTH_PROMPT = """You are an SRE investigation agent producing the final root-cause report.

# Alert
{alert_text}

# Validated causal chain (root -> leaf)
{validated_chain}

# All evidence cited by the chain
{evidence_block}

# Budget summary
{budget_summary}

# Instructions
Write the root-cause finding in plain English. Required structure:

1. **Root cause** -- one sentence naming the deepest validated mechanism.
2. **Causal chain** -- bullet list from the surface alert down to the root cause, each bullet <= 20 words, each with a `[source_id:chunk_id]` citation.
3. **Confidence** -- qualitative: "high" if every step on the chain was validated; "medium" if any step was inconclusive; "low" if circuit breakers terminated the search.
4. **Recommended next action** -- one sentence. Reference the runbook if cited.
5. **Caveats** -- list any inconclusive branches that a human should re-check.

DO NOT INVENT EVIDENCE. Cite only chunk IDs that appear in the validated chain's evidence. If the chain is empty or all branches were inconclusive, state that explicitly and recommend escalation.
"""
