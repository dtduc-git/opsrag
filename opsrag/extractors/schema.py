"""Shared DevOps/SRE entity-extraction schema + value sanitization.

Single source of truth for the constrained label/relation vocabulary used
by every extractor (rule-based metadata, LLM prose). Keeping it here means
``llm_extractor.py`` and the metadata-rule path agree on labels, relation
types, and -- critically -- the deterministic ``label:name:hash`` entity-ID
scheme so the same real-world thing (``Service:checkout``) merges across
files on re-ingest.

Security posture (DESIGN 3 PART 2, finding "research:graph-from-vector"):
extracted entity *values* come from chunk text, which is
attacker-influenceable (ingested repos/docs). Labels and relation types are
already constrained by the allow-lists below; this module additionally
sanitizes the free-text *values* (entity names, property values) so a
prompt-injection payload can't smuggle markup/control characters into the
graph or into the context string later rendered back to the LLM. Edges are
treated as low-trust -- the retrieval lane uses them as a soft boost only.
"""
from __future__ import annotations

import hashlib
import re

# --- constrained vocabulary -------------------------------------------------
# Entity labels the SRE graph recognizes. Anything an extractor emits outside
# this set is dropped (fail-closed) so attacker-controlled prose can't invent
# arbitrary node labels (which in Neo4j become real, queryable labels).
ALLOWED_LABELS: frozenset[str] = frozenset({
    "Service",
    "Team",
    "Runbook",
    "Incident",
    "Alert",
    "Config",
    "Database",
    "Infra",
    "Repository",
    "Environment",
    "Person",
    "Dependency",
    # Routing / topology lane (Kong, Ingress, Gateway API). Lets the graph
    # answer "what does route /abc point to?" and "which routes hit service-a?".
    "Gateway",      # Kong / Ingress controller / API gateway / LB
    "Route",        # an HTTP route (path + host) exposed by a gateway
    "Endpoint",     # a resolvable upstream target (FQDN / host:port)
    "Namespace",    # K8s namespace -- often the logical-service grouping
    "Host",         # a hostname / domain a route is served on (api.example.com)
    "Cluster",      # a k8s cluster / gitops env (prod/staging/dev, overlays/...)
    "Middleware",   # gateway plugin/middleware (auth, rate-limit, cors, jwt, ...)
})

# Relationship types the SRE graph recognizes.
ALLOWED_REL_TYPES: frozenset[str] = frozenset({
    "DEPENDS_ON",
    "USES_DATABASE",
    "DEPLOYED_ON",
    "OWNED_BY",
    "HAS_RUNBOOK",
    "HAS_ALERT",
    "CONFIGURED_BY",
    "LIVES_IN",
    "DEPLOYED_TO",
    "RUNS_IN",
    "DOCUMENTED_BY",
    "AFFECTED",
    "IMPACTED",
    "ROOT_CAUSE",
    "RESOLVED_BY",
    "MITIGATED_BY",
    "INVESTIGATED_BY",
    "TRIGGERS",
    "REFERENCES",
    "MEMBER_OF",
    "ONCALL_FOR",
    "APPLIES_TO",
    "HOSTED_ON",
    "DEFINED_IN",
    # Routing / topology lane.
    "HAS_ROUTE",       # Gateway -> Route
    "ROUTES_TO",       # Route -> Service / Endpoint (the upstream it proxies to)
    "RESOLVES_TO",     # Endpoint (FQDN) -> Service
    "IN_NAMESPACE",    # Service / Endpoint -> Namespace
    "COMPONENT_OF",    # component Service -> logical Service / Namespace
    "HAS_HOST",        # Route -> Host (the hostname it's exposed on)
    "IN_CLUSTER",      # Service / Route -> Cluster
    "USES_MIDDLEWARE", # Route -> Middleware (auth / rate-limit / ...)
})

# Cap entity-name + property-value length. Long values are almost always
# injected prose, not a real entity name, and they bloat the graph + the
# rendered context.
_MAX_VALUE_LEN = 200

# Strip anything that is not a reasonably safe identifier/name character.
# Allows letters, digits, and a small set of separators common in service /
# resource names (`checkout-api`, `tf:aws_s3_bucket.logs`, `team/payments`).
# Drops markup (`<`, `>`), backticks, braces, newlines, and other control
# characters that prompt-injection payloads rely on.
_DISALLOWED_RE = re.compile(r"[^\w\s\-./:@+]", flags=re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def sanitize_value(value: object) -> str:
    """Sanitize an attacker-influenceable entity value / name.

    - coerces to ``str``
    - strips control chars + markup the injection vector relies on
    - collapses whitespace
    - truncates to ``_MAX_VALUE_LEN``

    Returns the cleaned string (possibly empty -- callers must drop empties).
    """
    if value is None:
        return ""
    text = str(value)
    # Remove ASCII control chars (incl. newlines/tabs) outright.
    text = "".join(ch for ch in text if ch == " " or (ord(ch) >= 32 and ord(ch) != 127))
    text = _DISALLOWED_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if len(text) > _MAX_VALUE_LEN:
        text = text[:_MAX_VALUE_LEN].rstrip()
    return text


def sanitize_properties(props: dict | None) -> dict:
    """Sanitize a property dict: keys + scalar string values are cleaned.

    Non-string scalars (int/float/bool) pass through untouched; nested
    containers are dropped (the graph stores flat properties only).
    """
    if not props:
        return {}
    clean: dict = {}
    for k, v in props.items():
        key = sanitize_value(k)
        if not key:
            continue
        if isinstance(v, bool) or isinstance(v, (int, float)):
            clean[key] = v
        elif isinstance(v, str):
            cv = sanitize_value(v)
            if cv:
                clean[key] = cv
        # lists/dicts/None are intentionally dropped.
    return clean


def normalize_label(label: object) -> str | None:
    """Return the canonical label if allowed, else ``None`` (drop)."""
    s = sanitize_value(label)
    if not s:
        return None
    # Match case-insensitively against the allow-list, return canonical form.
    for allowed in ALLOWED_LABELS:
        if allowed.lower() == s.lower():
            return allowed
    return None


def normalize_rel_type(rel_type: object) -> str | None:
    """Return the canonical relation type if allowed, else ``None``."""
    s = sanitize_value(rel_type).upper().replace(" ", "_").replace("-", "_")
    if s in ALLOWED_REL_TYPES:
        return s
    return None


def make_entity_id(label: str, name: str) -> str:
    """Deterministic ``label:normalized-name:hash`` entity ID.

    Aligns with ``RuleBasedExtractor._eid`` and the original
    ``LLMEntityExtractor._make_entity_id`` so the SAME entity merges across
    chunks/files on re-ingest (shared deterministic IDs). The hash is over
    the canonical ``label:name`` so two extractors that see the same entity
    produce the same ID.
    """
    norm = name.lower().strip().replace(" ", "-")
    h = hashlib.sha1(f"{label}:{norm}".encode()).hexdigest()[:12]
    return f"{label.lower()}:{norm}:{h}"
