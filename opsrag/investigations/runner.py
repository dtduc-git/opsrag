"""InvestigationRunner -- orchestrates the Investigate-mode pipeline.

This is the new event-driven runner (Option B refactor 2026-05-27).
Replaces the LangGraph-state-based investigate flow we shipped earlier
(parse-from-text was too fragile).

Flow per `run_one()`:

    INVESTIGATION_STARTED
      -> 3 lanes (Flash, parallel-ish)
          LANE_A_COMPLETED  (runbook search via runbook_store)
          LANE_B_COMPLETED  (historical similar via investigation_cache)
          LANE_C_COMPLETED  (live probe -- skipped in this build)
      -> INSIGHT_READY        (Pro fusion of A+B+C)
      -> HYPOTHESES_GENERATED (Pro Pydantic structured output)
      -> reasoner round       (Pro with MCP tool-calling)
          TOOL_CALLED / TOOL_RESULT per call
      -> evaluator pass        (Flash Pydantic structured output)
          HYPOTHESIS_EVALUATED per hypothesis id
      -> CONCLUSION_READY     (Pro prose answer)
      -> INVESTIGATION_COMPLETED

Every event lands in opsrag_investigation_events via emit_event(). The
SSE endpoint at /investigations/{id}/events tails that table. Browser
refresh = full replay from the ledger.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from opsrag.investigations.evaluator import (
    HypothesisVerdictBatch,
    evaluate_hypotheses,
    format_evidence_pool,
)
from opsrag.investigations.event_types import EventType
from opsrag.investigations.store import (
    InvestigationEventStore,
    InvestigationStatus,
    emit_event,
)

_log = logging.getLogger("opsrag.investigations.runner")


# -- Per-investigation hard budget --------------------------------------
# Engine B previously had NO cost/latency ceiling beyond a 3-round cap, so a
# pathological run (slow tools + thrashing reasoner) could burn unbounded
# time. These mirror the hypothesis-tree engine's circuit breakers.
MAX_INVESTIGATION_WALL_CLOCK_SEC = 240.0   # hard-stop the reasoner loop past this
MAX_INVESTIGATION_TOOL_CALLS = 40          # cumulative live tool dispatches per run
PER_TOOL_TIMEOUT_SEC = 45.0                # a hung MCP tool can't stall a round


@dataclass
class _RunBudget:
    """Lightweight per-run budget. Constructed per investigation in
    `_run_pipeline` (the runner instance is shared across requests, so this
    is NEVER stored on self)."""

    started_at: float
    tool_calls: int = 0

    def wall_clock_exceeded(self) -> bool:
        return (time.monotonic() - self.started_at) >= MAX_INVESTIGATION_WALL_CLOCK_SEC

    def tool_budget_exhausted(self) -> bool:
        return self.tool_calls >= MAX_INVESTIGATION_TOOL_CALLS

    def elapsed_sec(self) -> float:
        return time.monotonic() - self.started_at


# -- Pydantic schemas for structured LLM output -------------------------

class InsightCard(BaseModel):
    """4-quadrant card the Pro synthesizer fills in from the lane data."""

    what_we_know: str = Field(default="", description="From the live probe / Lane C. Empty when probe was skipped.")
    what_weve_seen: str = Field(default="", description="From Lane B -- similar past investigations.")
    what_runbook_says: str = Field(default="", description="From Lane A -- top matching hand-authored runbook.")
    open_questions: str = Field(default="", description="What we still need to verify with tools.")


class HypothesisDraft(BaseModel):
    """One hypothesis emitted by the Pro hypothesizer."""

    id: str = Field(description="Stable short id -- 'h1', 'h2', etc.")
    # Bumped from 200 -> 500 on 2026-05-27: Pro routinely writes ~250 chars
    # when anchoring on a runbook's named failure mode (e.g. "A Kafka
    # Connect task failed due to a schema violation, causing the connector
    # to stop consuming"). 200 was strict enough to make Pydantic reject
    # the whole batch and trigger the dumb generic-prior fallback.
    text: str = Field(max_length=500)
    discriminating_tools: list[str] = Field(
        default_factory=list,
        description="MCP tool names that would confirm or rule this out. e.g. ['k8s_list_pods','cloudsql_query_insights'].",
    )


class HypothesisSet(BaseModel):
    """Output of the Pro hypothesizer -- 3-5 competing root-cause hypotheses."""

    incident_target: str | None = Field(default=None, description="The service / component the alert names.")
    hypotheses: list[HypothesisDraft] = Field(default_factory=list)


# -- Dependencies bundle ------------------------------------------------

@dataclass
class InvestigationDeps:
    """All the long-lived providers the runner needs. Built once at app
    startup and reused per investigation."""

    event_store: InvestigationEventStore
    flash_llm: Any  # Vertex Flash client (with generate_structured + generate_with_tools)
    pro_llm: Any    # Vertex Pro client
    embedder: Any | None = None
    runbook_store: Any | None = None
    investigation_cache: Any | None = None
    # MCPTool registry, keyed by tool.name. Built from ALL_MCP_TOOLS at
    # app startup (after all `bind_*` calls). When this is empty/None the
    # reasoner tool round is skipped -- evaluator runs on empty evidence
    # (cards all stay 'untested'), which is still useful for surfacing
    # the hypotheses themselves.
    tool_registry: dict[str, Any] = field(default_factory=dict)
    # Per-family clients. Most tool families manage their own connections
    # inside the handler (k8s_*, prometheus_*, rootly_*, ...) and accept
    # `None` as the client. GitLab is the historical outlier: its tools
    # expect a long-lived `GitLabClient` instance.
    gitlab_client: Any | None = None


# -- Prompts (short) ----------------------------------------------------

_SYSTEM_INSIGHT = """\
You are an SRE incident analyst. You will receive three lane summaries
(runbook hits, historical investigations, live probe) for an alert.

Synthesize a 4-quadrant Insight card:
  - what_we_know: From the live probe. Empty string if no probe data.
  - what_weve_seen: From historical investigations. Reference similarity.
  - what_runbook_says: From the runbook lane. If a runbook matched, name
    it ("Runbook '<title>' (service: <svc>) says:") and then list its
    NAMED FAILURE MODES verbatim -- bullets, do not paraphrase. The
    hypothesizer downstream anchors on these, so losing them = losing
    the on-call team's authored intelligence.
    If lane A says "No matching runbook found.", set this field to
    "No runbook matched this alert." (one sentence).
  - open_questions: 2-3 questions to resolve with live tools.

Output only the JSON object."""


_SYSTEM_HYPOTHESIZER_TEMPLATE = """\
You are an SRE incident triage agent. Given an alert + insight context,
enumerate 3-5 COMPETING root-cause hypotheses. They must be MUTUALLY
EXCLUSIVE -- each names a DIFFERENT failure mode that could plausibly
produce the alert symptoms.

Each hypothesis MUST list `discriminating_tools` -- the MCP tool names
whose results would confirm or rule out THIS specific hypothesis. The
tool selection determines which evidence the reasoner pulls next, so
pick tools that return CONTENT (logs, error messages, specific
values) -- NOT absence (counts, list lengths). Prefer a "get logs" tool
over a "list pods" tool when the hypothesis needs evidence of WHAT went
wrong, not just whether something exists.

RUNBOOK PRIORITY (HARD RULE):
If the insight context includes "WHAT THE RUNBOOK SAYS" with concrete
guidance for this alert pattern, the runbook is AUTHORITATIVE.

  * Your 3-5 hypotheses MUST be the failure modes the runbook
    enumerates, in the runbook's order -- NOT generic SRE priors layered
    on top.
  * If the runbook makes a CAUSAL CLAIM ("X is downstream symptom of
    Y", "OOM here is caused by upstream Z"), encode that claim
    directly in the hypothesis text. Do NOT generate a competing
    hypothesis that contradicts the runbook (e.g. if the runbook says
    \"local OOM is symptom, not cause\", do NOT add a separate
    hypothesis \"the local service has resource exhaustion as the
    primary cause\" -- that re-introduces the misdirection the
    runbook is specifically warning against).
  * Pick `discriminating_tools` that test the UPSTREAM cause named in
    the runbook (e.g. `k8s_find_workloads(name_contains="<upstream>")`),
    not just the local symptom.

When NO runbook matched, fall back to the GENERAL RULES below.

GENERAL RULES (only when no runbook matched):
- Avoid "pods are unhealthy" as a top-rank hypothesis unless the alert
  text explicitly names a crash/OOM/not-ready condition. For queue
  lag, error-rate, or latency alerts, the pod count is almost always
  fine -- the problem is application-level.
- Look at the alert TARGET, not just the alert TYPE. A queue-lag alert
  on topic `<x>.<service>.<table>` points at the service that
  produces/consumes that table, not at the queue itself.

REGISTERED MCP TOOL NAMES (pick discriminating_tools from this list ONLY):
{TOOL_NAMES}

Use stable ids h1, h2, h3 (lowercase). Output only the JSON object."""


def _build_hypothesizer_system_prompt(tool_registry: dict[str, Any]) -> str:
    """Render the hypothesizer system prompt with the LIVE tool registry."""
    names = sorted(tool_registry.keys()) if tool_registry else []
    if not names:
        rendered = "  (no tools registered -- emit empty discriminating_tools lists)"
    else:
        lines: list[str] = []
        cur = "  "
        for n in names:
            piece = (n + ", ")
            if len(cur) + len(piece) > 72 and cur.strip():
                lines.append(cur.rstrip(", "))
                cur = "  " + piece
            else:
                cur += piece
        if cur.strip():
            lines.append(cur.rstrip(", "))
        rendered = "\n".join(lines)
    return _SYSTEM_HYPOTHESIZER_TEMPLATE.replace("{TOOL_NAMES}", rendered)


_SYSTEM_GENERATOR_TEMPLATE = """\
You are an SRE incident analyst writing the conclusion section of an
investigation. You will receive:
  - the alert
  - the insight card
  - the hypothesis board with each hypothesis's verdict (CONFIRMED /
    RULED OUT / UNTESTED) -- these verdicts are AUTHORITATIVE; do not
    re-evaluate them in prose
  - the tool call history

Write a structured markdown answer with sections in this exact order:

### Root cause
A 1-3 sentence statement of the confirmed root cause. If no hypothesis
is confirmed, say so honestly: "Root cause not yet established --
recommend running <tool name>."

### Evidence
Bullet list of the tool results that support the named root cause.
Quote values verbatim (pod names, error messages, latency numbers).

### Hypotheses summary
DO NOT relist each hypothesis with its verdict -- the UI cards already
show that. Instead, write ONE paragraph explaining WHY the confirmed
hypothesis fits and the others don't (briefly).

### Next steps
1-3 concrete actionable bullets.

Cite tools you ACTUALLY observed in the history using `[tool_name]`
notation. Do NOT cite a tool that wasn't called.

**TOOL-NAME DISCIPLINE (HARD RULE):**
When recommending follow-up actions, you MUST use tool names from the
registered tool list below -- copy them VERBATIM. Do NOT invent variants
(no `kubectl_*`, no shell commands, no aliases the operator can't
click). If no registered tool fits, describe the action in plain
English with no fake tool name.

REGISTERED MCP TOOLS (the only names you may cite):
{TOOL_NAMES}
"""


def _build_generator_system_prompt(tool_registry: dict[str, Any]) -> str:
    """Render the generator system prompt with the LIVE tool registry.

    Avoids hardcoding `k8s_*`, `kubectl_*` etc. -- the operator's deployment
    may have a totally different set of MCP tools registered. Whatever's
    actually bound at runtime is what the generator is told it can cite.
    """
    names = sorted(tool_registry.keys()) if tool_registry else []
    if not names:
        rendered = "  (no tools registered -- recommend manual investigation only)"
    else:
        # Group into lines of ~70 chars for readability in the prompt.
        lines: list[str] = []
        cur = "  "
        for n in names:
            piece = (n + ", ")
            if len(cur) + len(piece) > 72 and cur.strip():
                lines.append(cur.rstrip(", "))
                cur = "  " + piece
            else:
                cur += piece
        if cur.strip():
            lines.append(cur.rstrip(", "))
        rendered = "\n".join(lines)
    return _SYSTEM_GENERATOR_TEMPLATE.replace("{TOOL_NAMES}", rendered)


# -- Helpers ------------------------------------------------------------


# Placeholder strings the LLM copies verbatim from alert payloads (e.g.
# Rootly / AlertManager render empty fields as the literal "none" or
# "null") and then passes to a tool as if it were a real value. Passing
# any scope-narrowing arg with one of these values is ALWAYS a bug --
# either the tool searches for a literal entity named "none" (0 results,
# misleads the evaluator) or the API errors.
_PLACEHOLDER_VALUES = {
    "none", "null", "nil", "unknown", "n/a", "na", "<none>", "<unknown>", "",
}

# Truly generic English filler -- drop from any vocab-overlap relevance
# scoring. Tenant- or domain-specific tokens (env names, SRE nouns,
# k8s terms) stay because they ARE the signal we're trying to match.
_GENERIC_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "have", "has", "had",
    "this", "that", "these", "those", "are", "was", "were", "been",
    "but", "not", "any", "all", "more", "most", "than", "then",
    "what", "which", "who", "when", "where", "why", "how",
}

# Regex to find `key="placeholder"` predicates embedded inside a query
# string (PromQL `{namespace="none"}`, SQL `WHERE region = 'none'`,
# Lucene `region:none`). Catches the three common quoting styles.
_EMBEDDED_PLACEHOLDER_RE = re.compile(
    r"""
    (?P<key>[A-Za-z_][A-Za-z0-9_]*)   # identifier
    \s*[:=]\s*                         # `=` or `:` separator
    (?P<quote>['"]?)                   # optional quote
    (?P<value>none|null|nil|unknown|n/a|na|<none>|<unknown>)
    (?P=quote)
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _scrub_placeholder_args(args: dict[str, Any]) -> dict[str, Any]:
    """Strip alert-payload placeholders from tool args, anywhere they hide.

    Three passes:
      1. Top-level (or nested-dict) string values matching the placeholder
         set -> drop the key entirely (the LLM is told to OMIT the arg
         instead).
      2. List values -> filter out placeholder elements.
      3. String values that LOOK like query strings (contain `=` or `:`
         and an identifier-style left side) -> regex out
         `key="placeholder"` predicates so we don't ship them to PromQL /
         SQL / Lucene backends that would dutifully match zero rows.

    Returns a new dict; original is not mutated.
    """
    if not isinstance(args, dict):
        return args

    def _is_placeholder(v: Any) -> bool:
        return isinstance(v, str) and v.strip().lower() in _PLACEHOLDER_VALUES

    def _scrub_query_string(s: str) -> str:
        # Only attempt query-string scrubbing on values long enough +
        # containing a `=` or `:` to be query-like -- short identifiers
        # (e.g. `service: "foo"` arg in JSON) should pass through.
        if len(s) < 8 or ("=" not in s and ":" not in s):
            return s
        new_s = _EMBEDDED_PLACEHOLDER_RE.sub("", s)
        # Clean up doubled commas / leading commas / `{ , }` artifacts
        # the regex removal can leave behind in PromQL.
        new_s = re.sub(r",\s*,", ",", new_s)
        new_s = re.sub(r"\{\s*,", "{", new_s)
        new_s = re.sub(r",\s*\}", "}", new_s)
        new_s = re.sub(r"\{\s*\}", "", new_s)
        if new_s != s:
            _log.info(
                "scrub: rewrote query string (removed placeholder predicates)",
            )
        return new_s

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            out: dict[str, Any] = {}
            for k, v in node.items():
                if _is_placeholder(v):
                    _log.info(
                        "scrub: dropping %s=%r (placeholder value)", k, v,
                    )
                    continue
                if isinstance(v, str):
                    out[k] = _scrub_query_string(v)
                elif isinstance(v, list):
                    cleaned = [
                        x for x in v if not _is_placeholder(x)
                    ]
                    cleaned = [
                        _walk(x) if isinstance(x, (dict, list)) else x
                        for x in cleaned
                    ]
                    out[k] = cleaned
                elif isinstance(v, dict):
                    out[k] = _walk(v)
                else:
                    out[k] = v
            return out
        return node

    return _walk(args)


# Scope-narrowing arg keys -- values for these MUST be grounded in the
# corpus the reasoner actually saw, or they're training-data guesses
# that produce empty / wrong results. The list is intentionally broad
# so it catches AWS (account_id, region), GCP (project), Azure
# (subscription, resource_group), K8s (namespace, cluster), and generic
# (environment, env, scope) shapes. Add to it as new clouds / tools are
# wired in.
_SCOPE_NARROWING_KEYS = {
    # k8s
    "namespace", "k8s_namespace", "ns",
    "cluster", "cluster_name", "k8s_cluster",
    # cloud account / project
    "project", "project_id", "gcp_project", "cloud_project",
    "account", "account_id", "aws_account_id", "subscription_id",
    "resource_group", "tenant_id",
    # geo
    "region", "location", "zone", "availability_zone",
    # generic
    "environment", "env", "scope",
}

# Tokens too short or too generic to require grounding for. Empty values
# already get dropped by the placeholder scrubber, so this just exempts
# obvious cases that would otherwise spam logs.
_GROUNDING_EXEMPT_VALUES = {"all", "default-region", "global"}


def _drop_ungrounded_scope_args(
    args: dict[str, Any], grounding_corpus: list[str],
) -> dict[str, Any]:
    """Drop scope-narrowing args whose values can't be traced to any
    text the reasoner actually read.

    Principle: every value the LLM puts in a scope-filter argument must
    come from one of three places:
      (a) the alert payload itself,
      (b) earlier insight context (runbook / past / probe),
      (c) the result of a previous tool call this round.
    Anything else is a guess pulled from training-data bias (e.g.
    `namespace="kube-system"` for fluentd because the model has seen
    that pairing more often than the truth). Dropping the arg forces
    the tool to run with broader scope -- usually returning the real
    value, which then grounds the NEXT call.
    """
    if not isinstance(args, dict):
        return args
    haystack = "\n".join(s for s in grounding_corpus if s).lower()
    if not haystack:
        return args
    out = dict(args)
    for k in list(out.keys()):
        if k.lower() not in _SCOPE_NARROWING_KEYS:
            continue
        v = out[k]
        if not isinstance(v, str):
            continue
        norm = v.strip().lower()
        if not norm or norm in _GROUNDING_EXEMPT_VALUES:
            continue
        if norm in haystack:
            continue
        _log.info(
            "scrub: dropping ungrounded scope arg %s=%r (not in alert/insight/prior-results)",
            k, v,
        )
        out.pop(k, None)
    return out


_TARGET_PATTERNS = [
    re.compile(
        r"\[(?:p\d+/)?(?P<env>preprod|prod|staging|shared|production|sandbox|dev)/k8s/"
        r"(?P<name>[a-z0-9][a-z0-9\-_]*)\]",
        re.IGNORECASE,
    ),
    re.compile(r"\[(?P<name>[a-z][a-z0-9\-_]{3,})\]", re.IGNORECASE),
]
_TARGET_STOPWORDS = {
    # Priority labels in common alert templates -- never a service name.
    "p1", "p2", "p3", "p4", "p5", "sev1", "sev2", "sev3", "sev4", "sev5",
    # Placeholders that alert renderers (Rootly, AlertManager, etc.) emit
    # when the service field is empty. Reading them as a literal service
    # produces hypotheses like "...in none..." and tool calls like
    # `namespace="none"`.
    *_PLACEHOLDER_VALUES,
}

# Map alert-env shorthand -> (cluster_name, gcp_project_or_cloud_account).
# Populated at import time from the `OPSRAG_ENV_CLUSTER_MAP` env var so the
# runner is tenant-neutral. Format is JSON:
#   OPSRAG_ENV_CLUSTER_MAP='{"prod":["my-prod-cluster","my-prod-gcp-project"],
#                            "staging":["my-staging-cluster","my-staging-gcp-project"]}'
# An empty / missing value means "no env->cluster mapping known" -- the
# reasoner is told nothing about clusters and figures it out from tool
# results (or asks). Operators who want the reasoner to bake in their
# cluster names just set this env var in their compose/helm config.
def _load_env_cluster_map() -> dict[str, tuple[str, str]]:
    raw = (os.environ.get("OPSRAG_ENV_CLUSTER_MAP") or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        _log.warning(
            "OPSRAG_ENV_CLUSTER_MAP is not valid JSON (%s) -- ignoring", exc,
        )
        return {}
    out: dict[str, tuple[str, str]] = {}
    if not isinstance(parsed, dict):
        return out
    for k, v in parsed.items():
        if not isinstance(k, str):
            continue
        if isinstance(v, (list, tuple)) and len(v) >= 1:
            cluster = str(v[0]) if v[0] else ""
            project = str(v[1]) if len(v) > 1 and v[1] else ""
            if cluster or project:
                out[k.lower()] = (cluster, project)
        elif isinstance(v, str) and v:
            out[k.lower()] = (v, "")
    return out


ENV_CLUSTER_MAP = _load_env_cluster_map()


def _extract_target(text: str) -> str | None:
    """Best-effort service name extraction from an alert string."""
    if not text:
        return None
    for pat in _TARGET_PATTERNS:
        for m in pat.finditer(text):
            name = (m.groupdict().get("name") or "").lower()
            if name and name not in _TARGET_STOPWORDS and len(name) >= 3:
                return name
    return None


def _extract_env(text: str) -> str | None:
    """Best-effort environment extraction from an alert string.

    Alerts typically carry the env in the `[<env>/k8s/<svc>]` bracket,
    e.g. `[P5][preprod/k8s/acme-analytics-v3]`. Returns the normalized short
    form (prod / preprod / staging / shared) -- caller looks up cluster + GCP project
    via ENV_CLUSTER_MAP. Without this, the Pro reasoner defaults to the
    prod cluster name even when the alert is for `preprod`, which silently
    pulls metrics from the wrong cluster.
    """
    if not text:
        return None
    m = _TARGET_PATTERNS[0].search(text)
    if m:
        env = (m.groupdict().get("env") or "").lower()
        if env in {"production"}:
            return "prod"
        if env in {"sandbox", "dev"}:
            return "shared"
        if env in {"prod", "preprod", "staging", "shared"}:
            return env
    # Fallback -- explicit "K8S_Namespace: <ns>" or "Environment: preprod"
    # lines that some alert payloads embed.
    m2 = re.search(r"environment\s*[:=]\s*([a-z]{2,12})", text, re.IGNORECASE)
    if m2:
        env = m2.group(1).lower()
        if env in {"production"}:
            return "prod"
        if env in {"prod", "preprod", "staging", "shared"}:
            return env
    return None


def _extract_from_rootly_payload(payload_text: str) -> dict[str, str | None]:
    """Pull env / target service / namespace from a Rootly alert payload.

    Rootly alert descriptions carry env in one of two shapes:
      - inline bracket  `[P5][preprod/k8s/acme-notes-be] Pods ...`
      - explicit field  `- Environment: preprod` / `- K8S_Namespace: acme-notes-be`

    Returns a dict with optional keys: env, target, namespace, title.
    Missing fields stay absent -- caller treats absent as "use the
    runner's other extraction passes".
    """
    out: dict[str, str | None] = {}
    if not payload_text:
        return out

    # Title -- anything between `"title":` or `Summary:` and the next quote/newline.
    m = re.search(r'"(?:title|summary)"\s*:\s*"([^"]+)"', payload_text)
    if m:
        out["title"] = m.group(1)[:200]
    else:
        m = re.search(r"^\s*-\s*Summary\s*:\s*([^\n]+)", payload_text, re.MULTILINE)
        if m:
            out["title"] = m.group(1).strip()[:200]

    # Bracket form: `[P5][preprod/k8s/acme-notes-be]`
    env = _extract_env(payload_text)
    target = _extract_target(payload_text)
    if env:
        out["env"] = env
    if target:
        out["target"] = target

    # Explicit fields the alert description embeds.
    if "env" not in out:
        m = re.search(r"environment\s*[:=]\s*([a-z]{2,12})", payload_text, re.IGNORECASE)
        if m:
            e = m.group(1).lower()
            out["env"] = "prod" if e in {"prod", "production"} else (
                "staging" if e == "staging" else (
                "shared" if e in {"sandbox", "dev"} else e
            ))
    if "namespace" not in out:
        m = re.search(r"k8s_namespace\s*[:=]\s*([a-z0-9][a-z0-9\-_]*)", payload_text, re.IGNORECASE)
        if m:
            out["namespace"] = m.group(1).lower()
        elif "target" in out:
            out["namespace"] = out["target"]
    return out


def _extract_namespace(text: str) -> str | None:
    """Namespace usually equals the target service in this deployment's
    layout. Extracted separately so it can be overridden from the alert
    payload's explicit `K8S_Namespace: <ns>` line when present."""
    if not text:
        return None
    m = re.search(r"k8s_namespace\s*[:=]\s*([a-z0-9][a-z0-9\-_]*)", text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # Fall back to the target name from the bracket form -- the
    # convention puts namespace == service name in 95% of alerts.
    return _extract_target(text)


# -- The runner ---------------------------------------------------------


class InvestigationRunner:
    """One-shot pipeline runner. Stateless across investigations --
    construct once at app startup, call `run_one(inv_id, alert_text)`
    in a background task per request."""

    def __init__(self, deps: InvestigationDeps) -> None:
        self.deps = deps

    async def run_one(self, inv_id: str, alert_text: str) -> None:
        """Run the full pipeline. Emits events at each step. Marks the
        lifecycle row completed/failed at the end. Never raises (errors
        emit INVESTIGATION_FAILED instead)."""
        try:
            await self._run_pipeline(inv_id, alert_text)
        except Exception as exc:  # noqa: BLE001
            _log.exception("investigation %s failed", inv_id)
            await emit_event(
                self.deps.event_store,
                investigation_id=inv_id,
                event_type=EventType.INVESTIGATION_FAILED,
                payload={"error": str(exc)},
            )
            try:
                await self.deps.event_store.mark_status(
                    inv_id, status=InvestigationStatus.FAILED,
                )
            except Exception:  # noqa: BLE001
                pass

    async def _hydrate_alert_text(self, alert_text: str) -> str:
        """If the input is (effectively) only a Rootly URL, fetch the
        alert payload and return a rich text that downstream lanes can
        actually semantic-search against.

        Rationale: Lane A's relevance gate, Lane A's Qdrant embedding
        lookup, and Lane B's similarity search ALL operate on the raw
        `alert_text`. A bare URL has no signal -- the runbook for
        \"acme-mobile-be 503\" can't match the string
        \"https://rootly.com/account/alerts/TFFFR1\". By concatenating
        the URL + the fetched summary + labels + annotations, we give
        every lane the real alert content to work with.

        Returns the original alert_text unchanged when:
          - no Rootly URL is in the input, OR
          - the input already has substantial keyword content (>80
            chars excluding the URL), OR
          - the rootly_get_alert tool isn't registered, OR
          - the fetch errors (we degrade gracefully -- Lane C will
            still try later).
        """
        original = alert_text or ""
        m = re.search(
            r"rootly\.com/account/(alerts|incidents)/([A-Za-z0-9_-]+)",
            original,
        )
        if not m:
            return original
        # If the user pasted the URL alongside their own context, don't
        # second-guess -- the keyword density is already high enough.
        non_url = re.sub(r"https?://\S+", "", original).strip()
        if len(non_url) > 80:
            return original

        kind, alert_id = m.group(1), m.group(2)
        tool_name = "rootly_get_alert" if kind == "alerts" else "rootly_get_incident"
        tool = (self.deps.tool_registry or {}).get(tool_name)
        if tool is None:
            return original
        try:
            result = await tool.call(None, {"short_id": alert_id})
        except Exception as exc:  # noqa: BLE001
            _log.warning("alert hydrate: %s fetch failed: %s", tool_name, exc)
            return original

        # Pull out the bits most useful for downstream semantic search.
        # We don't need every field -- just title + service names +
        # labels + annotation summary. Keeps the alert_text under a
        # couple KB so we don't blow the embedding token budget.
        try:
            r = result if isinstance(result, dict) else {}
            summary = r.get("summary") or ""
            services = ", ".join(r.get("service_names") or [])
            labels = r.get("labels") or {}
            annotations = r.get("annotations") or {}
            # Render labels/annotations as plain text -- most embedders
            # tokenise this just fine and the LLM reads it directly.
            label_lines = "\n".join(
                f"{k}: {v}" for k, v in labels.items() if isinstance(v, (str, int, float))
            )
            ann_lines = "\n".join(
                f"{k}: {v}" for k, v in annotations.items()
                if k in ("summary", "description", "runbook_url", "expr", "value", "severity")
                and isinstance(v, (str, int, float))
            )
            rich = (
                f"{original}\n"
                f"\nSummary: {summary}\n"
                f"Services: {services}\n"
                f"\n--- Labels ---\n{label_lines}\n"
                f"\n--- Annotations ---\n{ann_lines}\n"
            )
            _log.info(
                "alert hydrate: %s grew %d -> %d chars",
                tool_name, len(original), len(rich),
            )
            return rich
        except Exception as exc:  # noqa: BLE001
            _log.warning("alert hydrate: rich-text build failed: %s", exc)
            return original

    async def _run_pipeline(self, inv_id: str, alert_text: str) -> None:
        s = self.deps.event_store

        # The user's original input -- for UI display only. The SOURCE
        # ALERT card on the canvas should show exactly what they pasted
        # (typically a URL), NOT the hydrated payload below.
        original_alert_text = alert_text

        # -- 0. Hydrate the alert text --
        # When the user pastes only a Rootly URL, alert_text contains
        # zero keywords for Lane A (runbook semantic search) or Lane B
        # (historical similarity) to match against. Fetch the full
        # Rootly payload SYNCHRONOUSLY before the lanes fan out and
        # use the hydrated text everywhere downstream. Lane C is the
        # only place that originally did this, and it ran in parallel
        # with A/B -- so A/B saw only the URL string. This hydrated
        # text is INTERNAL-ONLY -- never shown in the UI.
        alert_text = await self._hydrate_alert_text(alert_text)

        target = _extract_target(alert_text)
        env = _extract_env(alert_text)
        namespace = _extract_namespace(alert_text)

        # -- 1. Lifecycle --
        # `alert_text` here = what the user typed (URL or paste). The
        # backend uses the hydrated version internally; the UI never
        # sees it.
        await emit_event(s, investigation_id=inv_id,
                         event_type=EventType.INVESTIGATION_STARTED,
                         payload={"alert_text": original_alert_text, "incident_target": target})

        # -- 2. Lanes (Flash) --
        await emit_event(s, investigation_id=inv_id,
                         event_type=EventType.INITIAL_INVESTIGATION_STARTED, payload={})

        lane_a, lane_b, lane_c = await asyncio.gather(
            self._lane_a(alert_text, target),
            self._lane_b(alert_text),
            self._lane_c(alert_text),
            return_exceptions=False,
        )
        # If Lane C pulled a Rootly payload, refine env/target/namespace
        # from its `extracted` block. This is critical when the user
        # only pasted a Rootly URL -- _extract_target/_extract_env return
        # None on a bare URL, and the Pro reasoner defaults to
        # the prod cluster without an explicit env hint.
        ext = (lane_c or {}).get("extracted") or {}
        if not env and ext.get("env"):
            env = ext["env"]
        if not target and ext.get("target"):
            target = ext["target"]
        if not namespace and ext.get("namespace"):
            namespace = ext["namespace"]
        await emit_event(s, investigation_id=inv_id,
                         event_type=EventType.LANE_A_COMPLETED, payload=lane_a)
        await emit_event(s, investigation_id=inv_id,
                         event_type=EventType.LANE_B_COMPLETED, payload=lane_b)
        await emit_event(s, investigation_id=inv_id,
                         event_type=EventType.LANE_C_COMPLETED, payload=lane_c)

        # -- 3. Insight (Pro) --
        insight = await self._synthesize_insight(alert_text, lane_a, lane_b, lane_c, target)
        await emit_event(s, investigation_id=inv_id,
                         event_type=EventType.INSIGHT_READY,
                         payload={"insight_card": insight})
        # Stash the raw runbook block on the insight dict AFTER emit so
        # the hypothesizer can read it verbatim (Pro's paraphrase loses
        # the named failure modes -- see comment in _generate_hypotheses)
        # without leaking the raw block into the UI event payload.
        insight["_lane_a_raw"] = _render_lane_a(lane_a)

        # -- 4. Hypothesizer (Pro structured output) --
        hyp_set = await self._generate_hypotheses(alert_text, insight, target)
        hypotheses = [h.model_dump() for h in hyp_set.hypotheses]
        # Stamp initial status=open so the UI cards render immediately.
        for h in hypotheses:
            h.setdefault("status", "open")

        await emit_event(s, investigation_id=inv_id,
                         event_type=EventType.HYPOTHESES_GENERATED,
                         payload={
                             "hypotheses": hypotheses,
                             "incident_target": hyp_set.incident_target or target,
                         })
        if hyp_set.incident_target:
            target = hyp_set.incident_target

        # -- 5+6. Iterative reasoner <-> evaluator loop --
        # Single-round investigation can't recover when round 1 hits an
        # empty result (wrong namespace, wrong label). The chat path's
        # multi-round loop is what lets it pivot ("0 pods with that
        # label -> try Prometheus -> metric labels reveal real namespace
        # -> list pods there"). Mirror that here. Each round:
        #   1. Reasoner picks tools, SEEING prior calls + verdicts so
        #      it can pivot.
        #   2. We dispatch (with grounding + placeholder scrubbers).
        #   3. Evaluator updates verdicts; UI cards flip live.
        # Stop when all hypotheses are decided OR reasoner declines OR
        # we hit the round cap. The cap is small (3) because each round
        # is a Pro call + <=6 tools -- past 3, we're usually thrashing.
        MAX_REASONER_ROUNDS = 3
        budget = _RunBudget(started_at=time.monotonic())
        accumulated_history: list[dict[str, Any]] = []
        verdicts: HypothesisVerdictBatch | None = None
        for round_idx in range(MAX_REASONER_ROUNDS):
            if budget.wall_clock_exceeded() or budget.tool_budget_exhausted():
                _log.warning(
                    "investigation %s budget hit (%.0fs / %d tool calls) -- "
                    "stopping reasoner loop, synthesizing with evidence so far",
                    inv_id, budget.elapsed_sec(), budget.tool_calls,
                )
                await emit_event(
                    s, investigation_id=inv_id,
                    event_type=EventType.REASONER_STEP,
                    payload={"note": "budget_exhausted",
                             "elapsed_sec": round(budget.elapsed_sec(), 1),
                             "tool_calls": budget.tool_calls},
                )
                break
            new_history = await self._reasoner_tool_round(
                inv_id, alert_text, hypotheses, target, insight, env, namespace,
                prior_history=accumulated_history,
                prior_verdicts=verdicts,
                round_idx=round_idx,
                budget=budget,
            )
            if not new_history:
                _log.info("reasoner round %d emitted no tool calls -- stopping", round_idx)
                break
            accumulated_history.extend(new_history)

            # Re-evaluate against ALL evidence collected so far.
            evidence = format_evidence_pool(accumulated_history)
            verdicts = await evaluate_hypotheses(
                llm=self.deps.flash_llm,
                hypotheses=hypotheses,
                evidence_pool=evidence,
                incident_target=target,
                prior_verdicts=[v.model_dump() for v in (verdicts.verdicts if verdicts else [])],
            )
            # Emit one HYPOTHESIS_EVALUATED per id -- UI cards flip live.
            for v in verdicts.verdicts:
                await emit_event(s, investigation_id=inv_id,
                                 event_type=EventType.HYPOTHESIS_EVALUATED,
                                 payload=v.model_dump())
            # Done if every hypothesis is decided.
            if all(v.status in ("confirmed", "ruled_out") for v in verdicts.verdicts):
                _log.info("all hypotheses decided after round %d -- stopping", round_idx)
                break

        tool_history = accumulated_history
        if verdicts is None:
            # No tool round ever fired (e.g. empty registry). Still emit
            # all-untested verdicts so the UI has something to render.
            verdicts = HypothesisVerdictBatch(verdicts=[])

        # -- 7. Generator (Pro prose) --
        answer = await self._generate_answer(
            alert_text, insight, hypotheses, verdicts, tool_history, target,
        )
        await emit_event(s, investigation_id=inv_id,
                         event_type=EventType.CONCLUSION_READY,
                         payload={"answer": answer})

        # -- 8. Done --
        root_cause = _extract_root_cause(answer)
        await s.mark_status(
            inv_id,
            status=InvestigationStatus.COMPLETED,
            root_cause=root_cause,
            outcome="completed",
        )
        await emit_event(s, investigation_id=inv_id,
                         event_type=EventType.INVESTIGATION_COMPLETED,
                         payload={"root_cause": root_cause})

    # -- Lane probes (Flash, simple) ------------------------------

    async def _lane_a(self, alert_text: str, target: str | None) -> dict[str, Any]:
        """Search the runbook store for the alert. Returns a dict
        suitable for the LANE_A_COMPLETED event payload."""
        t0 = time.perf_counter()
        if self.deps.runbook_store is None or self.deps.embedder is None:
            return {"hits": [], "elapsed_ms": 0, "skipped": "no_runbook_store"}
        try:
            # Try the runbook store's hybrid search. The store exposes
            # `search` returning list[RunbookHit]; we tolerate older
            # methods by feature-detecting.
            search_fn = getattr(self.deps.runbook_store, "search", None)
            if search_fn is None:
                return {"hits": [], "elapsed_ms": 0, "skipped": "no_search_method"}
            # Embedding is async on the Vertex provider.
            qvec = await self.deps.embedder.embed_query(alert_text)
            # NOTE: do NOT pass `service=target` here. The store treats it as
            # a hard equality filter (tsv WHERE + hydration WHERE) which
            # silently drops runbooks whose service field doesn't match the
            # alert's incident_target verbatim. Runbooks legitimately cover
            # cross-service incidents (e.g. an upstream service alert pointing
            # back to a kafka-cdc runbook) -- let RRF + embedding similarity do
            # the ranking, don't pre-exclude.
            hits = await search_fn(
                alert_text,
                query_embedding=qvec,
                top_k=5,
            )
            # Relevance gate -- RRF on a tiny corpus (2-3 runbooks) will
            # always rank SOMETHING first regardless of topical fit, and the
            # hypothesizer treats the top hit as authoritative. So before
            # we hand the hit to downstream prompts, require keyword overlap
            # between the runbook's identity (service / tags / title) and
            # the alert text. A kafka-cdc runbook surfacing for a fluentd
            # alert (observed 2026-05-27) must be filtered here.
            alert_lc = alert_text.lower()
            alert_tokens = set(re.findall(r"[a-z0-9][a-z0-9\-_]{2,}", alert_lc))

            hit_dicts = []
            for h in (hits or []):
                # RunbookHit is a Pydantic model with shape
                #   {runbook: Runbook, score: float, origin: str, ...}
                # Title/body/service live on the nested Runbook -- reading them
                # via getattr/__dict__ on the hit returns None and we render
                # "(untitled)" with empty snippet (bug observed 2026-05-27).
                rb = getattr(h, "runbook", None)
                if rb is None and isinstance(h, dict):
                    rb_obj = h.get("runbook")
                    rb = rb_obj if rb_obj is not None else h
                if rb is None:
                    continue
                def _g(name: str, default=None):
                    if isinstance(rb, dict):
                        return rb.get(name, default)
                    return getattr(rb, name, default)
                score_val = getattr(h, "score", None)
                if score_val is None and isinstance(h, dict):
                    score_val = h.get("score")

                rb_service = (_g("service") or "").lower()
                rb_tags = [str(t).lower() for t in (_g("tags") or [])]
                rb_title = (_g("title") or "").lower()
                # Build the runbook's "identity vocabulary" -- service, tags,
                # tag fragments split on hyphens (so 'consumer-lag' contributes
                # 'consumer' + 'lag'), and title tokens.
                vocab: set[str] = set()
                if rb_service:
                    vocab.add(rb_service)
                    vocab.update(rb_service.split("-"))
                for tag in rb_tags:
                    vocab.add(tag)
                    vocab.update(tag.split("-"))
                vocab.update(re.findall(r"[a-z0-9][a-z0-9\-_]{2,}", rb_title))
                # Generic English words only -- anything tenant- or
                # infrastructure-specific (env tokens, sre nouns like
                # 'pod' or 'service') stays in vocab because it's a real
                # discriminating signal when both sides mention it.
                vocab -= _GENERIC_STOPWORDS
                overlap = vocab & alert_tokens
                # Keep the hit if the service token literally appears in the
                # alert OR we share >=2 vocab words. Otherwise it's noise.
                literal_service_hit = bool(rb_service) and rb_service in alert_lc
                if not literal_service_hit and len(overlap) < 2:
                    _log.info(
                        "lane_a: dropping irrelevant runbook %r (service=%r overlap=%s)",
                        rb_title, rb_service, sorted(overlap),
                    )
                    continue

                hit_dicts.append({
                    "id": _g("id") or "",
                    "title": _g("title") or "",
                    "service": _g("service") or None,
                    "issue_kind": _g("issue_kind") or None,
                    "snippet": (_g("body_markdown") or "")[:1200],
                    "score": float(score_val) if score_val is not None else None,
                    "_overlap": sorted(overlap)[:6],
                })
            elapsed = int((time.perf_counter() - t0) * 1000)
            return {"hits": hit_dicts, "elapsed_ms": elapsed}
        except Exception as exc:  # noqa: BLE001
            _log.warning("lane_a failed: %s", exc)
            return {"hits": [], "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                    "error": str(exc)}

    async def _lane_b(self, alert_text: str) -> dict[str, Any]:
        """Search investigation_cache for similar past investigations."""
        t0 = time.perf_counter()
        cache = self.deps.investigation_cache
        if cache is None or self.deps.embedder is None:
            return {"hits": [], "elapsed_ms": 0, "skipped": "no_cache"}
        try:
            search_fn = getattr(cache, "search_similar", None) or getattr(cache, "search", None)
            if search_fn is None:
                return {"hits": [], "elapsed_ms": 0, "skipped": "no_search_method"}
            qvec = await self.deps.embedder.embed_query(alert_text)
            # InvestigationCache.search takes `embedding` positional;
            # alternative `search_similar` (older naming) typically takes
            # `query_embedding=`. Try positional first.
            try:
                hits = await search_fn(qvec, top_k=3)
            except TypeError:
                hits = await search_fn(query_embedding=qvec, top_k=3)
            hit_dicts = []
            for h in (hits or []):
                d = h if isinstance(h, dict) else getattr(h, "__dict__", {})
                hit_dicts.append({
                    "id": str(d.get("id") or d.get("investigation_id") or ""),
                    "similarity": d.get("similarity") or d.get("score") or 0.0,
                    "summary": (d.get("summary") or d.get("alert_text") or "")[:400],
                })
            elapsed = int((time.perf_counter() - t0) * 1000)
            return {"hits": hit_dicts, "elapsed_ms": elapsed}
        except Exception as exc:  # noqa: BLE001
            _log.warning("lane_b failed: %s", exc)
            return {"hits": [], "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                    "error": str(exc)}

    async def _lane_c(self, alert_text: str = "") -> dict[str, Any]:
        """Live probe -- if the alert text is a Rootly URL we fetch the
        alert payload via `rootly_get_alert`. This unlocks env / target
        extraction even when the user only pasted a URL (no inline
        `[preprod/k8s/<svc>]` bracket). The runner consumes the returned
        `extracted` block to fill env/target/namespace before handing
        anything to the Pro reasoner.
        """
        t0 = time.perf_counter()
        # Rootly URL detection -- accept both /alerts/<id> and /incidents/<id>.
        m = re.search(r"rootly\.com/account/(alerts|incidents)/([A-Za-z0-9_-]+)", alert_text or "")
        if not m:
            return {
                "summary": "No Rootly URL in alert -- live probe skipped.",
                "elapsed_ms": 0,
            }
        kind, alert_id = m.group(1), m.group(2)
        tool_name = "rootly_get_alert" if kind == "alerts" else "rootly_get_incident"
        tool = (self.deps.tool_registry or {}).get(tool_name)
        if tool is None:
            return {
                "summary": f"{tool_name} not in the tool registry.",
                "elapsed_ms": 0,
            }
        try:
            # Rootly tool accepts `short_id` (or `url`) -- not `id`.
            result = await tool.call(None, {"short_id": alert_id})
        except Exception as exc:  # noqa: BLE001
            elapsed = int((time.perf_counter() - t0) * 1000)
            _log.warning("lane_c rootly fetch failed: %s", exc)
            return {"summary": f"Rootly fetch failed: {exc}", "elapsed_ms": elapsed, "error": str(exc)}

        elapsed = int((time.perf_counter() - t0) * 1000)
        as_text = result if isinstance(result, str) else json.dumps(result)
        extracted = _extract_from_rootly_payload(as_text)
        summary_parts: list[str] = []
        if extracted.get("title"):
            summary_parts.append(extracted["title"])
        if extracted.get("env"):
            summary_parts.append(f"env={extracted['env']}")
        if extracted.get("target"):
            summary_parts.append(f"target={extracted['target']}")
        summary = " | ".join(summary_parts) or "Rootly payload fetched."
        return {
            "summary": summary,
            "elapsed_ms": elapsed,
            "rootly_id": alert_id,
            "rootly_kind": kind,
            "raw": as_text[:3000],
            "extracted": extracted,
        }

    # -- Insight synthesizer (Pro) ---------------------------------

    async def _synthesize_insight(
        self,
        alert_text: str,
        lane_a: dict[str, Any],
        lane_b: dict[str, Any],
        lane_c: dict[str, Any],
        target: str | None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        lane_a_block = _render_lane_a(lane_a)
        lane_b_block = _render_lane_b(lane_b)
        lane_c_block = lane_c.get("summary") or "(no live probe data)"
        user_msg = (
            f"ALERT:\n{alert_text[:2000]}\n\n"
            f"INCIDENT TARGET: {target or '(unspecified)'}\n\n"
            f"=== LANE A -- RUNBOOKS ===\n{lane_a_block}\n\n"
            f"=== LANE B -- HISTORICAL ===\n{lane_b_block}\n\n"
            f"=== LANE C -- LIVE PROBE ===\n{lane_c_block}\n\n"
            f"Fuse these into the 4-quadrant Insight card. Output JSON."
        )
        try:
            card = await self.deps.pro_llm.generate_structured(
                messages=[{"role": "user", "content": user_msg}],
                schema=InsightCard,
                system_prompt=_SYSTEM_INSIGHT,
                purpose="investigate_insight",
            )
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            return {**card.model_dump(), "elapsed_ms": elapsed_ms}
        except Exception as exc:  # noqa: BLE001
            _log.warning("insight synth failed: %s -- using fallback", exc)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            return {
                "what_we_know": lane_c_block,
                "what_weve_seen": lane_b_block[:300],
                "what_runbook_says": lane_a_block[:300],
                "open_questions": "What is the failing service? What tool result would confirm the symptom?",
                "elapsed_ms": elapsed_ms,
                "error": str(exc),
            }

    # -- Hypothesizer (Pro structured output) ----------------------

    async def _generate_hypotheses(
        self,
        alert_text: str,
        insight: dict[str, Any],
        target: str | None,
    ) -> HypothesisSet:
        # The hypothesizer prompt's RUNBOOK PRIORITY block keys off the
        # exact section header "WHAT THE RUNBOOK SAYS". We pass the raw
        # lane A render here -- NOT the Pro's paraphrased summary --
        # because the paraphrase compresses 4KB of authored failure
        # modes into 1-3 sentences, and the hypothesizer can't anchor
        # on what isn't in the prompt.
        runbook_raw = insight.get("_lane_a_raw") or ""
        if not runbook_raw or runbook_raw.startswith("No matching runbook"):
            runbook_section = "WHAT THE RUNBOOK SAYS: No runbook matched this alert. Use generic SRE priors below."
        else:
            runbook_section = f"WHAT THE RUNBOOK SAYS (AUTHORITATIVE -- anchor your hypotheses here):\n{runbook_raw}"

        user_msg = (
            f"ALERT:\n{alert_text[:2000]}\n\n"
            f"INCIDENT TARGET: {target or '(unspecified)'}\n\n"
            f"{runbook_section}\n\n"
            f"=== INSIGHT CONTEXT ===\n"
            f"What we know (live probe): {insight.get('what_we_know','')}\n"
            f"What we've seen (historical): {insight.get('what_weve_seen','')}\n"
            f"Open questions: {insight.get('open_questions','')}\n\n"
            f"Generate 3-5 competing root-cause hypotheses. Use stable "
            f"ids h1, h2, h3, ... and include `discriminating_tools` per "
            f"hypothesis. Output JSON."
        )
        try:
            return await self.deps.pro_llm.generate_structured(
                messages=[{"role": "user", "content": user_msg}],
                schema=HypothesisSet,
                system_prompt=_build_hypothesizer_system_prompt(self.deps.tool_registry),
                purpose="investigate_hypothesizer",
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("hypothesizer failed: %s -- using fallback", exc)
            # Generic priors so the UI still renders cards.
            return HypothesisSet(
                incident_target=target,
                hypotheses=[
                    HypothesisDraft(
                        id="h1",
                        text=f"An application-level bug in {target or 'the service'} is causing the reported alert symptoms.",
                        discriminating_tools=["elasticsearch_search_logs"],
                    ),
                    HypothesisDraft(
                        id="h2",
                        text=f"A recent deployment introduced a regression in {target or 'the service'}.",
                        discriminating_tools=["gitlab_list_deployments"],
                    ),
                    HypothesisDraft(
                        id="h3",
                        text=f"A downstream dependency of {target or 'the service'} is unavailable.",
                        discriminating_tools=["k8s_list_pods", "cloudsql_query_insights"],
                    ),
                    HypothesisDraft(
                        id="h4",
                        text=f"{target or 'The service'} pods are unhealthy (CrashLoopBackOff, OOMKilled).",
                        discriminating_tools=["k8s_list_pods"],
                    ),
                ],
            )

    # -- Reasoner tool round (Pro with function-calling) -----------

    async def _reasoner_tool_round(
        self,
        inv_id: str,
        alert_text: str,
        hypotheses: list[dict[str, Any]],
        target: str | None,
        insight: dict[str, Any],
        env: str | None = None,
        namespace: str | None = None,
        *,
        prior_history: list[dict[str, Any]] | None = None,
        prior_verdicts: HypothesisVerdictBatch | None = None,
        round_idx: int = 0,
        budget: "_RunBudget | None" = None,
    ) -> list[dict[str, Any]]:
        """One round of Pro reasoner picks discriminating tools + we
        execute them. Returns the tool call history as a list of
        `{name, args, result}` dicts.

        Iterative: caller invokes us up to N times. On round > 0 we get
        `prior_history` + `prior_verdicts` so the reasoner can PIVOT
        (e.g. "round 0 returned 0 pods with that label -- try a
        different label, or list cluster-wide, or hit Prometheus").
        """
        prior_history = prior_history or []
        if not self.deps.tool_registry:
            _log.info("reasoner: no tools registered -- skipping tool round")
            return []

        # Build the tool specs the Pro LLM sees. Two sources:
        #   1. Hypothesizer's `discriminating_tools` -- tools that would
        #      confirm/rule-out a specific hypothesis.
        #   2. Discovery fundamentals -- tools that don't discriminate
        #      between hypotheses but are required to LOCATE the
        #      workload / parse the alert / read recent state. These
        #      are always available, even if no hypothesis names them.
        wanted_tools: set[str] = set()
        for h in hypotheses:
            for t in (h.get("discriminating_tools") or []):
                if isinstance(t, str):
                    wanted_tools.add(t.strip())
        # Always-on discovery fundamentals. The reasoner can ignore any
        # that don't fit the alert, but they must be REACHABLE -- having
        # them invisible led to "0 pods with my guessed label -> give up"
        # behaviour on TFFFR1 (2026-05-27 manual investigation).
        _DISCOVERY_FUNDAMENTALS = {
            "k8s_find_workloads",      # "where does this service live?"
            "k8s_list_pods",           # then list its pods
            "rootly_get_alert",        # re-fetch the alert if labels are needed
        }
        wanted_tools |= _DISCOVERY_FUNDAMENTALS
        # If hypothesizer didn't pick any AND the fundamentals are all
        # absent (e.g. minimal registry), keep going -- _DISCOVERY_FUNDAMENTALS
        # already gives us a small core set.

        tool_specs = []
        for name in sorted(wanted_tools):
            tool = self.deps.tool_registry.get(name)
            if tool is None:
                continue
            tool_specs.append({
                "name": tool.name,
                "description": getattr(tool, "description", "") or "",
                "input_schema": getattr(tool, "input_schema", {}) or {"type": "object", "properties": {}},
            })

        if not tool_specs:
            _log.info("reasoner: no discriminating tools available in registry -- skipping")
            return []

        # Build the prompt for the Pro reasoner.
        hyp_block = "\n".join(
            f"  {h['id']}: {h.get('text','')}" for h in hypotheses
        )
        # Env-aware tool-arg discipline: spell out the exact cluster /
        # GCP project names so the reasoner doesn't default to PROD when
        # the alert is for preprod / staging / shared. Without this block the model
        # has seen the prod cluster name in training context and uses it
        # regardless of the alert env. Real failure (2026-05-27): a
        # `[P5][preprod/k8s/acme-analytics-v3]` alert triggered prometheus_query
        # with the prod cluster and cloudsql_query_insights with the
        # production project -- wrong env, no data.
        cluster_name = None
        project_name = None
        if env and env in ENV_CLUSTER_MAP:
            cluster_name, project_name = ENV_CLUSTER_MAP[env]
        env_block_lines: list[str] = []
        if env or namespace or cluster_name or project_name:
            env_block_lines.append("=== ENV DISCIPLINE (HARD RULE) ===")
            if env:
                env_block_lines.append(
                    f"This alert is for the **{env}** environment."
                )
            if cluster_name:
                env_block_lines.append(
                    f"- Tools accepting a cluster argument should use "
                    f"`cluster: \"{cluster_name}\"` for this env."
                )
            if project_name:
                env_block_lines.append(
                    f"- Tools accepting a cloud-project / account-id "
                    f"argument should use `\"{project_name}\"` for this env."
                )
            if namespace:
                env_block_lines.append(
                    f"- The alert specified namespace `\"{namespace}\"` -- "
                    f"prefer that for namespace-scoped tools."
                )
        # Universal "discover before assume" rule -- applies regardless of
        # alert shape, tenant, or cloud. Replaces the K8s-specific
        # namespace=none guardrail (which only worked for one shape of
        # alert + one tool family). Real failure pattern this targets:
        # LLM passes a placeholder/training-data-guess to a scope-narrowing
        # tool arg (namespace, project, region, account-id, cluster), the
        # tool returns 0 results, the evaluator misreads absence as
        # confirmation, the operator gets a wrong answer.
        env_block_lines.append(
            "\n=== TOOL ARGUMENT DISCIPLINE (UNIVERSAL HARD RULE) ===\n"
            "Every value you pass to a scope-narrowing argument (namespace, "
            "project, cluster, region, account-id, environment, service) "
            "MUST trace back to text you actually saw in:\n"
            "  (a) the alert payload above, OR\n"
            "  (b) the insight context (runbook / past investigations / "
            "live probe), OR\n"
            "  (c) the result of a previous tool call this round.\n"
            "\nIf the value isn't grounded in one of those three sources, "
            "you are GUESSING and the dispatcher will DROP the argument "
            "before calling the tool (with a log line).\n"
            "\nNEVER:\n"
            "  - Pass a placeholder (`\"none\"`, `\"null\"`, `\"unknown\"`, "
            "`\"n/a\"`, empty string) just because an alert field rendered "
            "empty that way.\n"
            "  - Guess from training-data bias (e.g. defaulting an unknown "
            "namespace to `kube-system`, `monitoring`, or `default`; "
            "defaulting an unknown region to `us-east-1`).\n"
            "  - Embed a guess inside a query string (PromQL, SQL, Lucene): "
            "`{namespace=\"kube-system\"}` is the same mistake.\n"
            "\n--- WORKED EXAMPLE ---\n"
            "Alert: \"[P3][prod/k8s/none] DaemonSet Fluentd Unready\"\n"
            "  (env=prod is grounded, namespace is NOT -- the slot reads "
            "`none`)\n"
            "WRONG first call: `k8s_list_pods(namespace=\"kube-system\", "
            "label_selector=\"k8s-app=fluentd-logging\")`\n"
            "  -- Both `kube-system` and the label string are training-data "
            "guesses; nothing in the alert said either.\n"
            "RIGHT first call: `k8s_list_pods(label_selector=\"name=fluentd\")`\n"
            "  -- No namespace filter. The discriminator `name=fluentd` "
            "comes from the alert verbatim. Result tells us the real "
            "namespace + label keys for the follow-up calls.\n"
            "If a tool returns empty, treat that as \"my filter was wrong\" "
            "before treating it as \"the thing doesn't exist\"."
        )
        env_block_lines.append(
            "\n- If you cite a tool but cannot match its required args to "
            "real values, SKIP the tool and report UNTESTED for the "
            "affected hypothesis."
        )
        env_block = "\n".join(env_block_lines) + "\n\n"

        # Prior-round context: every tool call the reasoner has already
        # tried this investigation, plus the evaluator's current verdicts.
        # This is what lets us pivot when round 1's results are useless --
        # the reasoner sees "round 0 returned 0 pods with that selector"
        # and tries a different selector / different tool family in
        # round 1.
        prior_block = ""
        if prior_history:
            lines = ["=== PRIOR TOOL CALLS THIS INVESTIGATION ==="]
            for i, item in enumerate(prior_history, 1):
                rn = item.get("name") or "?"
                ra = json.dumps(item.get("args") or {})[:160]
                rr = item.get("result") if "result" in item else item.get("error", "")
                if isinstance(rr, dict):
                    rr = json.dumps(rr)
                rr_str = str(rr)[:300]
                lines.append(f"#{i} [{rn}] args={ra}\n   result: {rr_str}")
            prior_block = "\n".join(lines) + "\n\n"
        verdict_block = ""
        if prior_verdicts and prior_verdicts.verdicts:
            vlines = ["=== CURRENT VERDICTS (from evaluator) ==="]
            for v in prior_verdicts.verdicts:
                vlines.append(
                    f"  {v.hypothesis_id}: {v.status} -- {v.evidence[:200] if v.evidence else '(no evidence yet)'}"
                )
            verdict_block = "\n".join(vlines) + "\n\n"

        # Round-aware directive. Round 0: pick the most discriminating
        # first probes. Round 1+: focus on hypotheses that are still
        # `untested` or `open`, and explicitly PIVOT off any prior
        # tool whose result was empty (try a different selector,
        # different tool family, or list cluster-wide).
        # Count consecutive empty results in prior_history. "Empty" =
        # tool returned a result whose serialized form has the shape
        # `count: 0`, `result: []`, `pods: []`, etc. (We can't be 100%
        # precise on every tool, so we use a substring sniff.)
        empties = 0
        for h in reversed(prior_history):
            r = h.get("result") or ""
            s = json.dumps(r) if not isinstance(r, str) else r
            if '"count": 0' in s or '"result": []' in s or '"pods": []' in s or '"items": []' in s or '"events": []' in s:
                empties += 1
            else:
                break

        if round_idx == 0:
            directive = (
                "You can call tools to gather evidence. Pick 2-4 tool calls "
                "that will most strongly discriminate between the hypotheses. "
                "\n\nIF THE ALERT NAMES A SERVICE / WORKLOAD but you don't yet "
                "know its namespace or labels, your FIRST call should be a "
                "discovery tool that scans the cluster by name substring "
                "(e.g. `k8s_find_workloads(name_contains=\"<service>\")`). "
                "The alert's service name is often NOT the pod's label or "
                "the deployment's exact name -- service `foo` may run as a "
                "Deployment named `bar-appservice-foo` in namespace `bar`. "
                "Substring search across Deployments/StatefulSets/DaemonSets "
                "finds it in one call.\n"
                "AFTER find_workloads tells you (kind, namespace, name, "
                "selector_match_labels): use those VERBATIM in the next call "
                "(e.g. k8s_list_pods with that namespace + selector). The "
                "aggregated pod fields (`pod_ready`, `max_restart_count`, "
                "`worst_termination_summary`) tell you which pods are bad -- "
                "you do NOT need to fetch logs to spot OOMKilled / "
                "CrashLoopBackOff."
            )
        elif empties >= 2:
            # Hard-broaden directive -- Pro has been narrowing on wrong
            # selectors for two rounds. Force a no-filter listing.
            directive = (
                f"This is ROUND {round_idx}. The last {empties} tool calls "
                "all returned EMPTY results (count=0, []). That is strong "
                "evidence your filters are wrong -- labels, namespaces, "
                "metric names -- not that the entity is absent.\n\n"
                "**BROADEN AGGRESSIVELY THIS ROUND:**\n"
                "  - Call the broadest listing tool you have access to "
                "(e.g. `k8s_list_daemonsets`, `k8s_list_pods`, "
                "`k8s_list_deployments`) with NO `label_selector`, NO "
                "`namespace`, NO `field_selector` -- just `cluster` + "
                "default `limit`. Read the raw list and pick out the "
                "real names / namespaces / labels.\n"
                "  - If listing a different resource type is more apt "
                "(alert says 'DaemonSet' -> use `k8s_list_daemonsets`; "
                "says 'Deployment' -> `k8s_list_deployments`), prefer that.\n"
                "  - For Prometheus, switch to a metric whose result LABELS "
                "contain the missing scope (e.g. metrics like "
                "`kube_<kind>_status_*` carry namespace/pod labels)."
            )
        else:
            directive = (
                f"This is ROUND {round_idx}. Look at the prior-round results "
                "above. For every hypothesis still marked `untested` or "
                "`open`, PIVOT: pick a DIFFERENT discriminating tool, "
                "or the SAME tool with broader/different scope, or query a "
                "different signal source (metrics vs logs vs k8s vs CI). "
                "If a prior tool returned empty, treat its filter as wrong "
                "and try a new approach -- do NOT repeat the same call.\n"
                "If a hypothesis is now decisively confirmed/ruled-out by "
                "prior results, you may skip it. Return empty tool calls "
                "ONLY when no useful next probe exists."
            )

        user_msg = (
            f"ALERT: {alert_text[:1500]}\n\n"
            f"INCIDENT TARGET: {target or '(unspecified)'}\n"
            f"ENV: {env or '(unspecified)'}  NAMESPACE: {namespace or '(unspecified)'}\n\n"
            f"{env_block}"
            f"{prior_block}"
            f"{verdict_block}"
            f"HYPOTHESIS BOARD:\n{hyp_block}\n\n"
            f"{directive}\n"
            f"Be specific with arguments -- they MUST be grounded in the "
            f"alert text or a prior tool result; ungrounded args will be "
            f"silently dropped before dispatch."
        )

        # Call the LLM with function-calling. Different providers expose
        # this with different method names; try both.
        gen_with_tools = getattr(self.deps.pro_llm, "generate_with_tools", None)
        if gen_with_tools is None:
            _log.warning("pro_llm has no generate_with_tools -- skipping tool round")
            return []

        try:
            resp = await gen_with_tools(
                messages=[{"role": "user", "content": user_msg}],
                tools=tool_specs,
                system_prompt=(
                    "You are an SRE incident triage agent. Pick discriminating "
                    "tools to test the hypothesis board. Do not write prose -- "
                    "only emit tool calls."
                ),
                temperature=0.0,
                max_tokens=2048,
                purpose="investigate_reasoner",
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("reasoner generate_with_tools failed: %s", exc)
            return []

        tool_calls = getattr(resp, "tool_calls", None) or []
        if not tool_calls:
            _log.info("reasoner declined to call tools")
            return []

        await emit_event(
            self.deps.event_store, investigation_id=inv_id,
            event_type=EventType.REASONER_STEP,
            payload={
                "thinking_text": (getattr(resp, "content", "") or "")[:800],
                "pending_tools": [tc.name for tc in tool_calls],
            },
        )

        # Lazily construct a GitLabClient ONLY if a gitlab_* call is in
        # the pending set. Mirrors the chat path's pattern in
        # multi_agent.py:1593: chat creates it as an async ctx mgr per
        # tool round. We do the same so the connection pool is reused
        # across multiple gitlab_* calls within this investigation and
        # closed cleanly afterwards.
        gitlab_client = self.deps.gitlab_client
        gitlab_ctx = None
        if gitlab_client is None and any(
            (tc.name or "").startswith("gitlab_") for tc in tool_calls
        ):
            try:
                from opsrag.mcp import GitLabClient
                gitlab_ctx = GitLabClient()
                gitlab_client = await gitlab_ctx.__aenter__()
            except Exception as exc:  # noqa: BLE001
                _log.warning("GitLabClient lazy init failed: %s", exc)
                gitlab_client = None

        # Build the grounding context bag -- the union of every text the
        # reasoner could have legitimately read a value from. Any
        # scope-narrowing arg whose value doesn't appear in this bag is
        # treated as a training-data guess and dropped at dispatch.
        # `prior_results_text` accumulates as we iterate so earlier tool
        # results ground later tool args (chained discovery).
        grounding_corpus: list[str] = [
            alert_text or "",
            env_block or "",
            json.dumps(insight or {}),
        ]
        if target:
            grounding_corpus.append(target)
        if namespace:
            grounding_corpus.append(namespace)
        prior_results_text: list[str] = []

        # Execute each tool. Errors are captured per-call so one bad
        # tool doesn't sink the round.
        history: list[dict[str, Any]] = []
        for tc in tool_calls[:6]:  # hard cap per round
            if budget is not None and budget.tool_budget_exhausted():
                _log.warning("tool-call budget exhausted -- skipping remaining tools this round")
                break
            if budget is not None:
                budget.tool_calls += 1
            name = tc.name
            args = _scrub_placeholder_args(tc.args or {})
            # Second pass: drop any scope-narrowing arg whose value isn't
            # grounded in the corpus the reasoner actually saw.
            args = _drop_ungrounded_scope_args(
                args, grounding_corpus + prior_results_text,
            )
            await emit_event(
                self.deps.event_store, investigation_id=inv_id,
                event_type=EventType.TOOL_CALLED,
                payload={"name": name, "args": args},
            )
            t0 = time.perf_counter()
            tool = self.deps.tool_registry.get(name)
            if tool is None:
                err = f"unknown tool {name!r}"
                history.append({"name": name, "args": args, "error": err, "result": ""})
                await emit_event(
                    self.deps.event_store, investigation_id=inv_id,
                    event_type=EventType.TOOL_RESULT,
                    payload={"name": name, "error": err, "latency_ms": 0},
                )
                continue
            try:
                # GitLab tools expect a long-lived client; everything
                # else manages its own connection and accepts None.
                client_for_tool = (
                    gitlab_client if name.startswith("gitlab_") else None
                )
                if name.startswith("gitlab_") and client_for_tool is None:
                    raise RuntimeError(
                        "GitLab client unavailable -- gitlab_* tool skipped"
                    )
                # Per-tool hard timeout: a hung MCP tool can't stall the whole
                # round (the except-clause below records it as a tool error).
                result = await asyncio.wait_for(
                    tool.call(client_for_tool, args), timeout=PER_TOOL_TIMEOUT_SEC,
                )
                latency_ms = int((time.perf_counter() - t0) * 1000)
                history.append({"name": name, "args": args, "result": result, "latency_ms": latency_ms})
                # Add the result to the grounding corpus so the NEXT
                # tool call in this round can use values it produced
                # (e.g. tool-1 returns a real namespace string, tool-2
                # then legitimately passes that namespace).
                try:
                    if isinstance(result, str):
                        prior_results_text.append(result)
                    else:
                        prior_results_text.append(json.dumps(result))
                except Exception:  # noqa: BLE001
                    pass
                # Don't ship the full result in the SSE -- keep payload
                # small; the evaluator reads `result` from `history` in
                # this process. UI just needs to know the tool ran.
                summary = _summarize_result(result)
                await emit_event(
                    self.deps.event_store, investigation_id=inv_id,
                    event_type=EventType.TOOL_RESULT,
                    payload={"name": name, "summary": summary, "latency_ms": latency_ms},
                )
            except Exception as exc:  # noqa: BLE001
                err = str(exc)
                latency_ms = int((time.perf_counter() - t0) * 1000)
                history.append({"name": name, "args": args, "error": err, "result": "", "latency_ms": latency_ms})
                await emit_event(
                    self.deps.event_store, investigation_id=inv_id,
                    event_type=EventType.TOOL_RESULT,
                    payload={"name": name, "error": err, "latency_ms": latency_ms},
                )
        # Release the lazy GitLab client if we opened one.
        if gitlab_ctx is not None:
            try:
                await gitlab_ctx.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
        return history

    # -- Generator (Pro prose) -------------------------------------

    async def _generate_answer(
        self,
        alert_text: str,
        insight: dict[str, Any],
        hypotheses: list[dict[str, Any]],
        verdicts: HypothesisVerdictBatch,
        tool_history: list[dict[str, Any]],
        target: str | None,
    ) -> str:
        verdict_map = {v.hypothesis_id: v for v in verdicts.verdicts}
        board_lines: list[str] = []
        for h in hypotheses:
            v = verdict_map.get(h.get("id", ""))
            status = v.status if v else "open"
            evidence = v.evidence if v else ""
            board_lines.append(
                f"  {h.get('id','?')}: {h.get('text','')} "
                f"[{status.upper()}] -- {evidence}"
            )
        evidence_block = format_evidence_pool(tool_history) or "(no tool calls)"

        user_msg = (
            f"ALERT:\n{alert_text[:1500]}\n\n"
            f"INCIDENT TARGET: {target or '(unspecified)'}\n\n"
            f"=== INSIGHT ===\n"
            f"{json.dumps(insight, indent=2)[:1500]}\n\n"
            f"=== HYPOTHESIS BOARD (verdicts are authoritative) ===\n"
            f"{chr(10).join(board_lines)}\n\n"
            f"=== TOOL CALL HISTORY ===\n{evidence_block[:6000]}\n\n"
            f"Write the structured markdown answer per the system prompt."
        )
        try:
            resp = await self.deps.pro_llm.generate(
                messages=[{"role": "user", "content": user_msg}],
                system_prompt=_build_generator_system_prompt(
                    self.deps.tool_registry,
                ),
                temperature=0.0,
                max_tokens=4096,
                purpose="investigate_generator",
            )
            return resp.content or "(empty answer)"
        except Exception as exc:  # noqa: BLE001
            _log.warning("generator failed: %s", exc)
            return f"### Root cause\nGenerator failed: {exc}"


# -- Module helpers -----------------------------------------------------


def _render_lane_a(lane_a: dict[str, Any]) -> str:
    hits = lane_a.get("hits") or []
    if not hits:
        return "No matching runbook found."
    lines = []
    # Top hit gets the FULL body (truncated to ~1200 chars). The synthesizer
    # and hypothesizer both depend on this -- if we only feed them the title,
    # the LLM has to hallucinate the failure modes. We want runbook content
    # to dominate the prompt when a runbook matches.
    for i, h in enumerate(hits[:3]):
        title = h.get("title") or "(untitled)"
        service = h.get("service") or ""
        issue_kind = h.get("issue_kind") or ""
        score = h.get("score")
        score_str = f" | score={score:.2f}" if isinstance(score, (int, float)) else ""
        meta_bits = []
        if service:
            meta_bits.append(f"service: {service}")
        if issue_kind:
            meta_bits.append(f"issue_kind: {issue_kind}")
        meta = f" ({', '.join(meta_bits)})" if meta_bits else ""
        # Top hit verbatim-ish, lower hits trimmed.
        body_cap = 1200 if i == 0 else 400
        snippet = (h.get("snippet") or "")[:body_cap]
        lines.append(f"### Runbook: {title}{meta}{score_str}\n{snippet}")
    return "\n\n".join(lines)


def _render_lane_b(lane_b: dict[str, Any]) -> str:
    hits = lane_b.get("hits") or []
    if not hits:
        return "No similar past investigation in the cache."
    lines = []
    for h in hits[:3]:
        sim = h.get("similarity") or 0.0
        summary = (h.get("summary") or "")[:200]
        pct = f"{int(float(sim) * 100)}%" if isinstance(sim, (int, float)) else "?"
        lines.append(f"- {pct} similar: {summary}")
    return "\n".join(lines)


def _summarize_result(result: Any) -> str:
    """Tiny one-liner for SSE -- full result lives in `history` for the
    evaluator. UI just needs "got 7 pods, 3 not ready"-style text."""
    if result is None:
        return "(empty)"
    s = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
    if len(s) > 200:
        s = s[:200] + "..."
    return s


def _extract_root_cause(answer: str) -> str | None:
    """Pull the first paragraph under `### Root cause` so the lifecycle
    row carries a one-liner for the sidebar."""
    if not answer:
        return None
    m = re.search(
        r"(?:^|\n)#{2,4}\s*Root\s+cause\b[^\n]*\n+([^\n][^\n]{0,500})",
        answer, re.IGNORECASE,
    )
    return m.group(1).strip() if m else None
