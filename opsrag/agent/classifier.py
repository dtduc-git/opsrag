"""Query classifier -- forensic / live / procedural.

Hybrid 3-layer pipeline:

1. **Regex hard-rules**     -- fast path; catches obvious live markers
                               ("right now", absolute IDs like
                               `pipeline 1531171`, `TICKET-7890`).
2. **Semantic router**       -- anchor-embedding cosine vs reference
                               examples per category. Reuses the query
                               embedding the retrieval path already
                               computed, so this layer is *free*.
3. **LLM fallback (Flash)**  -- only when semantic-router top-1 margin
                               is below `LLM_FALLBACK_MARGIN`. ~$5e-5
                               per call; rare in practice.

Categories drive cache strategy:

  forensic   -- query references a frozen past event (specific
               pipeline/incident ID, dated time window). Cache long
               (e.g. 90d) at high threshold.
  live       -- query asks about current state ("now", "right now").
               Bypass cache entirely.
  procedural -- query asks how something works conceptually
               ("what does X mean", "how do I rotate Y"). Cache
               medium-long (e.g. 30d).
  mixed      -- has live AND forensic markers; treat as live to be
               safe (short TTL or bypass).
  unknown    -- classifier couldn't decide; default conservative
               behaviour (use legacy threshold + TTL).

Industry references for the design:
- Redis SemanticRouter (anchor-based zero-cost classification).
- Pinecone / LangChain RouterChain (LLM fallback).
- Hugging Face multi-turn RAG canonicalization (regex first-pass).
"""
from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from opsrag.agent.prompt_render import active_deployment

_log = logging.getLogger("opsrag.agent.classifier")

# Generic, org-agnostic environment tokens. The env-discriminator regex
# is built from these PLUS whatever the active deployment declares in
# ``DeploymentContext.environments`` -- no concrete org env names are
# baked into the engine.
_GENERIC_ENVIRONMENTS: tuple[str, ...] = ("prod", "staging", "dev", "test")


class QueryCategory(str, Enum):
    FORENSIC = "forensic"
    LIVE = "live"
    PROCEDURAL = "procedural"
    MIXED = "mixed"
    # Structural infrastructure questions answerable from the
    # Cartography graph (cross-cluster RBAC, GCP asset inventory,
    # DNS lookups, WI bridging, blast-radius, etc.). Routed to the
    # cartography_* MCP family before knowledge_search / code_grep.
    INFRA_GRAPH = "infra_graph"
    # Friendly chitchat: greetings, thanks, meta-questions about the
    # bot itself ("what can you do", "who built you"). Bypasses both
    # triage and retrieval -- straight to a fast friendly generator
    # with an inline persona prompt. Sub-second latency target.
    CASUAL = "casual"
    UNKNOWN = "unknown"


# ----------------------------- Layer 1: regex -----------------------------

# Live-state markers -- bypass cache.
_LIVE_PATTERNS = re.compile(
    r"\b("
    r"current|currently|now|right now|just now|today|tonight|this hour|"
    r"recently|at the moment|present(ly)?"
    r")\b",
    re.IGNORECASE,
)

# Forensic markers -- specific past events / IDs. Cache aggressively.
# Tuned for OpsRAG corpus: GitLab, Rootly, Jira-ish ticket prefixes,
# Kubernetes resources, datestamps.
_FORENSIC_PATTERNS = [
    # Past time markers
    re.compile(
        r"\b("
        r"yesterday|last (week|month|quarter|year|night)|"
        r"on \d{4}-\d{2}-\d{2}|between \d"
        r")\b",
        re.IGNORECASE,
    ),
    # Absolute datestamp anywhere (YYYY-MM-DD)
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    # GitLab pipeline / MR / commit IDs
    re.compile(r"\b(pipeline|build|job)\s*#?\s*\d{3,}\b", re.IGNORECASE),
    re.compile(r"\b(MR|merge request|PR|pull request)\s*[!#]?\s*\d{2,}\b", re.IGNORECASE),
    re.compile(r"\bcommit\s+[0-9a-f]{7,40}\b", re.IGNORECASE),
    # Ticket / incident prefixes -- common SRE/Ops prefixes (TICKET-, INC-,
    # SRE-, OPS-) plus generic Jira-style A-Z+digit.
    re.compile(r"\b(TICKET|INC|SRE|OPS)-\d{3,}\b", re.IGNORECASE),
    re.compile(r"\bincident\s+\w{3,}\b", re.IGNORECASE),
    # NB: the env-embedded pod/instance pattern (e.g. "<name>-<env>-1") is
    # built dynamically per request from the active deployment's
    # environments -- see ``_env_pod_pattern`` -- so no concrete env names
    # are hard-coded here.
]


def _build_env_discriminator(environments: Sequence[str]) -> re.Pattern:
    """Build an env-token alternation regex from the active deployment's
    declared environments PLUS the generic defaults. De-duplicated and
    regex-escaped. Org-specific env labels live only in the runtime
    DeploymentContext, never in the engine source."""
    tokens: list[str] = []
    seen: set[str] = set()
    for env in list(environments) + list(_GENERIC_ENVIRONMENTS):
        token = (env or "").strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        tokens.append(re.escape(token))
    alternation = "|".join(tokens) if tokens else "|".join(
        re.escape(t) for t in _GENERIC_ENVIRONMENTS
    )
    return re.compile(rf"\b({alternation})\b", re.IGNORECASE)


def _env_pod_pattern() -> re.Pattern:
    """Forensic marker for specific pod / instance names that embed an
    environment token (e.g. "<service>-<env>-1"). Built at call time from
    the active deployment so it tracks operator-declared environments."""
    tokens: list[str] = []
    seen: set[str] = set()
    for env in list(active_deployment().environments) + list(_GENERIC_ENVIRONMENTS):
        token = (env or "").strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        tokens.append(re.escape(token))
    alternation = "|".join(tokens) if tokens else "|".join(
        re.escape(t) for t in _GENERIC_ENVIRONMENTS
    )
    return re.compile(rf"\b[a-z][a-z0-9]+-({alternation})(-[a-z0-9]+)+\b", re.IGNORECASE)

# Casual markers -- short greetings, thanks, simple meta-bot Qs.
# Two-step: (1) a length cap (greetings are rarely > 80 chars) +
# (2) a high-precision regex over the whole query. We deliberately
# AVOID matching mid-sentence "hi" inside a longer investigation
# ("hi team, can you investigate the payments service") -- only standalone or
# leading greetings count.
# Two pattern families. (1) Greetings = anchored ^...$ so we never
# match a "hi" mid-sentence inside an investigation. (2) Meta-bot
# questions = unanchored `\b...\b` substrings; users phrase these with
# arbitrary preambles ("Can you introduce yourself?", "Beside those,
# what can you do?") and we want to catch them all without making the
# anchor brittle.
_CASUAL_PATTERNS = [
    # -- (1) Greeting / thanks / closing -- anchored to the whole query --
    re.compile(
        r"^\s*("
        r"(hi|hello|hey|yo|sup|howdy)(\s+(there|team|man|bro|y'?all|all|opsrag|everyone))?|"
        r"good\s*(morning|afternoon|evening|night)|"
        r"thanks?(\s+(you|a\s+lot|so\s+much|man|bro))?|thank\s*you|thx|ty|"
        r"bye|goodbye|see\s*ya|cya|"
        r"ok|okay|cool|nice|great|awesome|"
        r"how\s+are\s+(you|u)(\s+(doing|going))?(\s+today)?|"
        r"how('?s| is)\s+(it|everything|things)\s+going|"
        r"what'?s\s+up|"
        r"nice\s+to\s+meet\s+(you|u)"
        r")\s*[!?.\s]*$",
        re.IGNORECASE,
    ),
    # -- (2) Meta-bot questions -- substring matches, tolerant of
    # preambles like "Can you", "Could you", "Beside those, ...". Each
    # pattern targets a question shape that's clearly about THE BOT
    # itself, not about a specific infra resource.
    re.compile(r"\bwho\s+are\s+you\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+are\s+you\b(?!\s+(doing|investigating|working\s+on))", re.IGNORECASE),
    re.compile(r"\bwhat\s+is\s+opsrag\b", re.IGNORECASE),
    re.compile(r"\b(introduce|describe)\s+yourself\b", re.IGNORECASE),
    re.compile(r"\btell\s+me\s+about\s+(yourself|opsrag)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(else\s+|other\s+|more\s+)?(can|could)\s+(you|opsrag)\s+do\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(do|are)\s+you\s+(do|capable|good\s+at|useful\s+for)\b", re.IGNORECASE),
    # Capability / source inventory -- "what sources do you ...", "what
    # tools can you ...", "what data sources ...". Require a meta-verb
    # (do/does/can/are/have) after the noun so "what sources caused the
    # outage" doesn't match.
    re.compile(
        r"\b(what|which)\s+(sources?|tools?|integrations?|systems?|data\s+sources?|capabilit(y|ies)|connectors?|features?)\s+"
        r"(do|does|can|could|are|have|you|opsrag)\b",
        re.IGNORECASE,
    ),
    # Inventory shapes -- "list your tools", "show all your integrations",
    # "list opsrag's sources". REQUIRES the possessive ("your" / "the" /
    # "opsrag" / "opsrag's") before the noun, AND the noun must be one
    # of the bot-meta nouns. So "list all pods" / "list all incidents"
    # never match (no possessive + ops noun).
    re.compile(
        r"\b(list|show)\s+(out\s+)?(all\s+)?(your|the|opsrag(?:'s)?)\s+"
        r"(sources?|tools?|integrations?|capabilit(y|ies)|features?|connectors?|systems?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdescribe\s+(a|the|your)?\s*(flow|process|workflow|architecture|design|pipeline)\s+(end\s+to\s+end|e2e)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bhow\s+(do|does)\s+(you|opsrag|this)\s+work\b", re.IGNORECASE),
    re.compile(r"\bhow\s+are\s+you\s+built\b", re.IGNORECASE),
    re.compile(
        r"\bare\s+you\s+(a\s+bot|an?\s+ai|claude|gemini|gpt|chatgpt|human|real)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bwho\s+(made|created|built|trained|developed)\s+(you|opsrag)\b", re.IGNORECASE),
]

# Procedural markers -- "how do I", "what does X mean", policy/runbook.
_PROCEDURAL_PATTERNS = [
    re.compile(
        r"\b(how (do|to|can|does|should)|what (is|does|are) .* (mean|do)|"
        r"what (is|are) the (process|procedure|step|way|steps)|"
        r"what'?s the (process|procedure|step|way) to|"
        r"the procedure to|"
        r"explain|define|definition)\b",
        re.IGNORECASE,
    ),
]


def _matches_any(query: str, patterns: Sequence[re.Pattern]) -> bool:
    return any(p.search(query) for p in patterns)


def _classify_by_regex(query: str) -> QueryCategory:
    """Fast first-pass. Returns UNKNOWN when no strong signal so the
    caller can fall through to layer 2."""
    # CASUAL gets checked FIRST: greetings/thanks/meta-bot Qs are
    # high-precision anchored matches that should bypass the heavier
    # forensic/live regex set. If the query also somehow matches a
    # forensic pattern (e.g. someone wrote "hi, what about pipeline
    # 12345?"), the anchored ^...$ won't fire -- only standalone
    # greetings count, so investigations always win.
    if _matches_any(query, _CASUAL_PATTERNS):
        return QueryCategory.CASUAL
    has_live = bool(_LIVE_PATTERNS.search(query))
    has_forensic = (
        _matches_any(query, _FORENSIC_PATTERNS)
        or bool(_env_pod_pattern().search(query))
    )
    has_procedural = _matches_any(query, _PROCEDURAL_PATTERNS)

    # Co-occurrence: live + forensic -> MIXED. Live alone -> LIVE (safest).
    if has_live and has_forensic:
        return QueryCategory.MIXED
    if has_live:
        return QueryCategory.LIVE
    if has_forensic:
        return QueryCategory.FORENSIC
    if has_procedural:
        return QueryCategory.PROCEDURAL
    return QueryCategory.UNKNOWN


# ------------------------- Layer 2: semantic router ------------------------

# Reference examples per category. These anchors are DERIVED at call
# time from the active DeploymentContext (so they reflect the operator's
# services / environments) rather than baked as a frozen module constant
# with one org's names. Cosine vs the live query embedding picks the best
# category. Keep small (~10/cat) so the dot-product loop stays trivial.
#
# Two layers:
#   1. A baseline of GENERIC, org-agnostic example shapes per category
#      (no concrete service or env names). These always anchor the
#      router even when the deployment declares nothing.
#   2. Derived examples built from ``active_deployment().services`` and
#      ``.environments`` -- the genuine "recognise THIS deployment's
#      resources" signal, supplied at runtime.
# Operators may additionally supply ``semantic_router_examples`` to add
# query shapes the templates would not produce; those are merged on top.

# Generic, org-agnostic baseline. Uses placeholder shapes ("a pipeline",
# "the deployment", "a service") -- never a real service or env name.
_GENERIC_REFERENCE_EXAMPLES: dict[QueryCategory, list[str]] = {
    QueryCategory.FORENSIC: [
        "why did pipeline 1531171 fail",
        "what was the root cause of incident TICKET-7890",
        "show me the postmortem for the outage last week",
        "what happened during the deployment yesterday at 2pm",
        "why did the deployment on 2026-04-15 break",
        "summarize the resolved incident from monday",
        "trace the failure of build #4892",
        "investigate the merged MR !567",
        "review commit a1b2c3d4 changes",
        "why did the pipeline fail to deploy",
    ],
    QueryCategory.LIVE: [
        "is the service slow right now",
        "what is the current cpu usage of the database",
        "show me current alerts",
        "are there any open incidents at the moment",
        "what services are degraded right now",
        "current memory of the pods",
        "are pods crashlooping now",
        "is the api latency high now",
        "show currently failing jobs",
        "is the service currently slow",
    ],
    QueryCategory.PROCEDURAL: [
        "how do I rotate the ssl certificate",
        "what does this metric mean",
        "explain the deployment process",
        "how to add a new helm chart",
        "what is the procedure to onboard an app",
        "how does the auth tunnel work",
        "what are the steps to escalate a P1",
        "define the term tiered cache",
        "how do I run a metrics query",
        "how do I rotate a certificate",
    ],
    # Structural infrastructure questions -- DNS, RBAC, cloud asset
    # inventory, K8s blast-radius, workload identity, diagram-the-flow.
    # Cosine anchoring picks INFRA_GRAPH whenever the query is
    # structurally about WHAT / WHERE / WHO / HOW MANY in the infra graph
    # (not "what's happening right now" -- that's LIVE). Generic shapes
    # only; concrete projects / domains come from the runtime context.
    QueryCategory.INFRA_GRAPH: [
        "who can cluster-admin in our production cluster",
        "list every cloud sql instance in the production project",
        "draw the request flow for a service",
        "what does the backend pod use -- service account, secrets, node",
        "which DNS records point at a given IP",
        "is the service account bound to a cloud identity",
        "trace workload identity for a kubernetes service account",
        "show me all object storage buckets in the project",
        "what cloud service accounts exist in production",
        "what zero-trust apps protect our domains",
        "diagram how users reach a service",
        "who has the view role across our kubernetes clusters",
        "list all kubernetes services matching a name",
        "find every DNS record containing a token",
        "what compute instances are running in the project",
        "show the RBAC blast radius for cluster-admin",
        "who has cluster-admin permission in the cluster",
        "list all secret manager secrets in the project",
    ],
    # Casual / friendly chitchat -- greetings, thanks, meta-questions
    # about the bot itself. The regex covers obvious one-liners; these
    # anchor the semantic-router for slightly longer phrasings the
    # regex doesn't catch ("can you introduce yourself", "what kind of
    # questions are you good at"). Routed to friendly_generator -- NO
    # tools, NO RAG.
    QueryCategory.CASUAL: [
        "hi there",
        "hello",
        "thanks for the help",
        "good morning",
        "how are you doing today",
        "what can you do for me",
        "tell me about yourself",
        "who built you",
        "are you a bot or a real person",
        "what kind of questions can you answer",
        "introduce yourself please",
        "nice to meet you",
        "i'm new here, what should I ask you",
        "what's your job exactly",
        "are you connected to chat as well",
        # Capability / source / "how do you work" shapes -- distinct
        # from real ops questions, because they ask about THE BOT not
        # about specific infra:
        "what sources can you connect to",
        "what data sources do you have access to",
        "list out all your integrations",
        "list all your tools",
        "what tools are available to you",
        "describe a flow end to end",
        "how do you work internally",
        "can you describe how you operate end to end",
        "besides those examples what else can you do",
        "what other capabilities do you have",
        "show me everything you can do",
        "what systems do you connect to",
    ],
}


def _derived_examples_for(category: QueryCategory) -> list[str]:
    """Build deployment-derived anchors for ``category`` from the active
    context's services and environments. Returns [] when the context
    declares nothing -- in which case only the generic baseline anchors
    the router."""
    ctx = active_deployment()
    services = [s for s in ctx.services if s and s.strip()]
    environments = [e for e in ctx.environments if e and e.strip()]
    out: list[str] = []

    if category is QueryCategory.FORENSIC:
        for svc in services:
            out.append(f"why did the {svc} deployment break yesterday")
    elif category is QueryCategory.LIVE:
        for svc in services:
            out.append(f"is {svc} slow right now")
        for env in environments:
            out.append(f"what is the current cpu usage in {env}")
    elif category is QueryCategory.PROCEDURAL:
        for svc in services:
            out.append(f"how do I deploy {svc}")
    elif category is QueryCategory.INFRA_GRAPH:
        for svc in services:
            out.append(f"diagram how users reach {svc}")
        for env in environments:
            out.append(f"who has cluster-admin in the {env} cluster")
    # CASUAL is intentionally context-free.
    return out


def _operator_examples_for(category: QueryCategory) -> list[str]:
    """Pull operator-supplied ``semantic_router_examples`` for this
    category. Keys are treated as route/category names (matched against
    ``QueryCategory`` values, case-insensitively); values may be a single
    utterance string or a list of strings."""
    raw = active_deployment().semantic_router_examples or {}
    out: list[str] = []
    for key, value in raw.items():
        if str(key).strip().lower() != category.value:
            continue
        if isinstance(value, str):
            if value.strip():
                out.append(value)
        else:
            for item in value or []:
                if isinstance(item, str) and item.strip():
                    out.append(item)
    return out


def build_reference_examples() -> dict[QueryCategory, list[str]]:
    """Compute the per-category semantic-router anchors against the active
    DeploymentContext. Combines, per category and de-duplicated:

      1. generic org-agnostic baseline shapes (always present),
      2. examples derived from the deployment's services / environments,
      3. operator-supplied ``semantic_router_examples`` overrides.

    Computed at call time so it reflects whatever context is active when
    the ``SemanticRouter`` is constructed / fitted -- never a frozen
    single-org constant.
    """
    result: dict[QueryCategory, list[str]] = {}
    for category in (
        QueryCategory.FORENSIC,
        QueryCategory.LIVE,
        QueryCategory.PROCEDURAL,
        QueryCategory.INFRA_GRAPH,
        QueryCategory.CASUAL,
    ):
        combined: list[str] = []
        seen: set[str] = set()
        for example in (
            list(_GENERIC_REFERENCE_EXAMPLES.get(category, []))
            + _derived_examples_for(category)
            + _operator_examples_for(category)
        ):
            norm = example.strip()
            if not norm or norm.lower() in seen:
                continue
            seen.add(norm.lower())
            combined.append(norm)
        result[category] = combined
    return result

LLM_FALLBACK_MARGIN = 0.05  # If top-1 - top-2 < margin, call LLM judge.
ABSTAIN_TOP_THRESHOLD = 0.55  # Below this, top-1 too weak -- abstain.


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class SemanticRouter:
    """Anchor-embedding classifier. Embed reference examples once at
    init; classify by max-cosine-per-category at request time.

    Reusing the query embedding (already computed for retrieval) makes
    this layer effectively free at runtime -- only the one-time anchor
    embedding cost during boot.
    """

    embedder: object  # any object with `await embed_query(str)`
    references: dict[QueryCategory, list[str]] = field(
        default_factory=build_reference_examples,
    )
    _anchors: dict[QueryCategory, list[list[float]]] = field(default_factory=dict)

    async def fit(self) -> None:
        """Embed the reference examples once. Idempotent -- subsequent
        calls are no-ops."""
        if self._anchors:
            return
        for cat, examples in self.references.items():
            embs: list[list[float]] = []
            for ex in examples:
                try:
                    e = await self.embedder.embed_query(ex)
                    embs.append(list(e))
                except Exception as exc:
                    _log.warning("classifier anchor embed failed %r: %s", ex, exc)
            self._anchors[cat] = embs
        _log.info("semantic router fitted: %s",
                  {k.value: len(v) for k, v in self._anchors.items()})

    def classify(
        self,
        query_embedding: Sequence[float],
    ) -> tuple[QueryCategory, float, float]:
        """Return (category, top1_score, top2_score). The caller compares
        `top1 - top2` against `LLM_FALLBACK_MARGIN` to decide whether to
        invoke the LLM judge."""
        if not self._anchors or not query_embedding:
            return QueryCategory.UNKNOWN, 0.0, 0.0
        scores: list[tuple[QueryCategory, float]] = []
        for cat, anchors in self._anchors.items():
            if not anchors:
                continue
            best = max(_cosine(query_embedding, a) for a in anchors)
            scores.append((cat, best))
        scores.sort(key=lambda x: -x[1])
        if not scores:
            return QueryCategory.UNKNOWN, 0.0, 0.0
        top1_cat, top1_score = scores[0]
        top2_score = scores[1][1] if len(scores) > 1 else 0.0
        if top1_score < ABSTAIN_TOP_THRESHOLD:
            return QueryCategory.UNKNOWN, top1_score, top2_score
        return top1_cat, top1_score, top2_score


# ----------------------- Layer 3: LLM fallback (optional) -------------------

_LLM_PROMPT = """Classify the operations query into ONE of:
- forensic: about a specific past event (pipeline ID, incident ID, dated past time window)
- live: about current state ("now", "right now", "today", live metrics)
- procedural: asks how-to, what-is, conceptual / runbook
- mixed: contains both live AND forensic markers
- infra_graph: structural infra question (RBAC, GCP assets, DNS, blast-radius, "who can X / which Y / where does Z point")
- casual: greeting, thanks, or meta-question about the bot itself (NOT an investigation)

Respond with ONLY the category word.

Query: {query}
Category:"""


async def _classify_by_llm(query: str, llm) -> QueryCategory:
    """Ask Vertex Flash. ~$5e-5/call. Returns UNKNOWN on failure.

    `llm` must implement the `LLMProvider` Protocol (`.generate(messages, ...)`).
    """
    if llm is None:
        return QueryCategory.UNKNOWN
    try:
        resp = await llm.generate(
            messages=[{"role": "user", "content": _LLM_PROMPT.format(query=query[:500])}],
            temperature=0.0,
            max_tokens=5,
            purpose="classifier",
        )
        text = (resp.content or "").strip().lower().split()[0] if resp and resp.content else ""
        for cat in QueryCategory:
            if text == cat.value:
                return cat
    except Exception as exc:
        _log.warning("LLM classifier failed: %s", exc)
    return QueryCategory.UNKNOWN


# ----------------------------- Public facade ------------------------------


@dataclass
class ClassificationResult:
    category: QueryCategory
    layer: str  # "regex" | "semantic" | "llm" | "default"
    top1_score: float = 0.0
    top2_score: float = 0.0
    margin: float = 0.0


async def classify_query(
    query: str,
    *,
    query_embedding: Sequence[float] | None = None,
    semantic_router: SemanticRouter | None = None,
    llm=None,
    enable_llm_fallback: bool = True,
) -> ClassificationResult:
    """Hybrid classification. Layer 1 (regex) -> Layer 2 (semantic
    router) -> Layer 3 (LLM, optional).

    The caller passes the already-computed `query_embedding` from the
    retrieval path so layer 2 doesn't re-embed.
    """
    # Layer 1
    cat = _classify_by_regex(query)
    if cat != QueryCategory.UNKNOWN:
        return ClassificationResult(category=cat, layer="regex")

    # Layer 2
    if semantic_router is not None and query_embedding is not None:
        cat, top1, top2 = semantic_router.classify(query_embedding)
        margin = top1 - top2
        if cat != QueryCategory.UNKNOWN and margin >= LLM_FALLBACK_MARGIN:
            return ClassificationResult(
                category=cat, layer="semantic",
                top1_score=top1, top2_score=top2, margin=margin,
            )
        # Low confidence -> fall through to LLM
        if enable_llm_fallback and llm is not None:
            llm_cat = await _classify_by_llm(query, llm)
            if llm_cat != QueryCategory.UNKNOWN:
                return ClassificationResult(
                    category=llm_cat, layer="llm",
                    top1_score=top1, top2_score=top2, margin=margin,
                )
        # Best effort -- return semantic top-1 even if margin small
        if cat != QueryCategory.UNKNOWN:
            return ClassificationResult(
                category=cat, layer="semantic-low-conf",
                top1_score=top1, top2_score=top2, margin=margin,
            )

    # Default -- unknown. Caller uses legacy behaviour.
    return ClassificationResult(category=QueryCategory.UNKNOWN, layer="default")


# -------------------------- TTL & threshold policy -------------------------

# Per-category cache policy. Negative TTL means "skip cache entirely".
# Tuned conservatively -- easier to extend TTL later than to recover
# from a bad cache leak.
CATEGORY_POLICY: dict[QueryCategory, dict] = {
    QueryCategory.FORENSIC:   {"ttl_seconds": 90 * 86400, "skip_cache": False, "qa_threshold": 0.92},
    QueryCategory.PROCEDURAL: {"ttl_seconds": 30 * 86400, "skip_cache": False, "qa_threshold": 0.93},
    QueryCategory.MIXED:      {"ttl_seconds":      300,   "skip_cache": False, "qa_threshold": 0.96},
    QueryCategory.LIVE:       {"ttl_seconds":        0,   "skip_cache": True,  "qa_threshold": 0.99},
    # INFRA_GRAPH -- cartography data refreshes daily; we cache for 4 h
    # which is well inside one ingest cycle. qa_threshold mid-tight
    # (between forensic and live) because the same question may yield
    # different answers as the graph re-ingests.
    QueryCategory.INFRA_GRAPH: {"ttl_seconds":   4 * 3600, "skip_cache": False, "qa_threshold": 0.94},
    # CASUAL -- no point caching "hi". Each greeting deserves a fresh
    # response so OpsRAG can react to the moment (time of day, recent
    # session context). skip_cache=True bypasses the QA cache entirely.
    QueryCategory.CASUAL:     {"ttl_seconds":        0,   "skip_cache": True,  "qa_threshold": 0.99},
    QueryCategory.UNKNOWN:    {"ttl_seconds": 14 * 86400, "skip_cache": False, "qa_threshold": 0.93},
}


def policy_for(category: QueryCategory) -> dict:
    return CATEGORY_POLICY.get(category, CATEGORY_POLICY[QueryCategory.UNKNOWN])
