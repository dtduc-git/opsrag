"""Deterministic chunk-metadata enricher (no LLM).

Runs at ingest time, per chunk, after parse+chunk and before embedding.
Derives the normalized, filterable facets the parsers/chunker can't know
on their own -- `doc_type`, `environment`, `tier`/`criticality`,
`language`, `tags`, `valid` -- purely from path conventions, the section
heading, the structural doc_type, and lightweight content sniffing.

No network, no LLM, no nondeterminism: the same (path, text, source_type)
always yields the same facets. This matches the GCP `rule_based` parity
decision in the design and keeps ingest cheap.

Backward-compatible: only ADDS keys. Never overwrites a value already set
by a parser (parsers are authoritative for what they know -- e.g. a Helm
parser's `helm_file_type`, a connector's `url`). Enricher-derived keys are
written only when absent so an explicit upstream value always wins.
"""
from __future__ import annotations

import re
from typing import Any

from opsrag.ingestion.metadata import (
    DOC_TYPES,
    ENVIRONMENTS,
    LANGUAGES,
    TIERS,
)
from opsrag.interfaces.parser import DocType

# --- doc_type normalization -------------------------------------------------
# Map the structural DocType enum onto the broader normalized DOC_TYPES
# vocabulary. Most are identity; HELM values files become `helm_values`,
# and the code DocTypes collapse to `code`.
_CODE_DOC_TYPES = {
    DocType.PYTHON,
    DocType.JAVASCRIPT,
    DocType.TYPESCRIPT,
    DocType.GO,
    DocType.JAVA,
    DocType.SHELL,
}

# DocType -> language (for the code family); prose defaults to "en".
_DOCTYPE_LANGUAGE = {
    DocType.PYTHON: "python",
    DocType.JAVASCRIPT: "javascript",
    DocType.TYPESCRIPT: "typescript",
    DocType.GO: "go",
    DocType.JAVA: "java",
    DocType.SHELL: "shell",
    DocType.TERRAFORM: "hcl",
    DocType.HELM: "yaml",
    DocType.KUBERNETES: "yaml",
    DocType.YAML_CONFIG: "yaml",
    DocType.DOCKERFILE: "dockerfile",
}

# Path-segment -> environment. Checked as whole path-segments (split on
# `/`, `-`, `_`, `.`) so `prod` matches `values-prod.yaml` but not a stray
# substring inside a word. Canonical env names only; deployment-specific
# abbreviations come from `deployment.environments` in DeploymentContext.
_ENV_TOKENS: dict[str, str] = {
    "prod": "prod",
    "production": "prod",
    "live": "prod",
    "staging": "staging",
    "stage": "staging",
    # preprod is its own canonical env (see metadata.ENVIRONMENTS), NOT a
    # staging alias -- contextual.py also emits "preprod", so folding it here
    # would tag the same chunk's filterable `environment` (staging) out of
    # sync with its embedded context prefix (preprod).
    "preprod": "preprod",
    "dev": "dev",
    "develop": "dev",
    "development": "dev",
    "sandbox": "dev",
    "local": "dev",
    "test": "test",
    "testing": "test",
    "qa": "qa",
    "uat": "qa",
}

# Path-substring -> doc_type override. Convention-driven (incidents/,
# runbooks/, adr/, ...). Ordered: first match wins.
_PATH_DOC_TYPE: tuple[tuple[str, str], ...] = (
    ("postmortem", "postmortem"),
    ("post-mortem", "postmortem"),
    ("incident", "incident"),
    ("runbook", "runbook"),
    ("/sop/", "runbook"),
    ("/adr/", "adr"),
    ("/adrs/", "adr"),
    ("architecture", "architecture"),
)

# Tier tokens that may appear in a path or heading.
_TIER_TOKENS: dict[str, str] = {
    "tier0": "tier0", "tier-0": "tier0", "t0": "tier0", "p0": "tier0",
    "critical": "tier0",
    "tier1": "tier1", "tier-1": "tier1", "t1": "tier1", "p1": "tier1",
    "tier2": "tier2", "tier-2": "tier2", "t2": "tier2", "p2": "tier2",
    "tier3": "tier3", "tier-3": "tier3", "t3": "tier3", "p3": "tier3",
}

_SEGMENT_SPLIT = re.compile(r"[/\\._\-\s]+")

# Lightweight tech tags sniffed from path + text. Deterministic keyword
# presence only -- no inference. Keeps the set short and high-signal.
_TECH_TAGS: tuple[str, ...] = (
    "redis", "postgres", "postgresql", "mysql", "kafka", "rabbitmq",
    "elasticsearch", "kubernetes", "k8s", "docker", "nginx", "envoy",
    "istio", "terraform", "helm", "prometheus", "grafana", "datadog",
    "latency", "oom", "deadlock", "timeout", "throttle",
)


def _segments(path: str) -> list[str]:
    return [s for s in _SEGMENT_SPLIT.split(path.lower()) if s]


def _derive_doc_type(path: str, struct_doc_type: DocType | None, meta: dict) -> str:
    # 1. Path conventions win first (incidents/runbooks/adr/...).
    low = path.lower()
    for needle, dt in _PATH_DOC_TYPE:
        if needle in low:
            return dt
    # 2. Explicit parser flags.
    if meta.get("postmortem"):
        return "postmortem"
    if meta.get("runbook"):
        return "runbook"
    # 3. Structural DocType.
    if struct_doc_type is not None:
        if struct_doc_type in _CODE_DOC_TYPES:
            return "code"
        if struct_doc_type == DocType.HELM:
            return "helm_values" if meta.get("helm_file_type") == "values" else "helm"
        return struct_doc_type.value
    return "generic_markdown"


def _derive_environment(path: str, heading: str) -> str | None:
    tokens = _segments(path)
    # Heading tokens too -- some runbooks scope a section to an env.
    tokens += _segments(heading or "")
    for tok in tokens:
        env = _ENV_TOKENS.get(tok)
        if env:
            return env
    return None


def _derive_tier(path: str, heading: str) -> str | None:
    tokens = set(_segments(path)) | set(_segments(heading or ""))
    for tok in tokens:
        tier = _TIER_TOKENS.get(tok)
        if tier:
            return tier
    return None


def _derive_language(struct_doc_type: DocType | None, doc_type: str) -> str:
    if struct_doc_type is not None and struct_doc_type in _DOCTYPE_LANGUAGE:
        return _DOCTYPE_LANGUAGE[struct_doc_type]
    if doc_type == "code":
        return "en"  # unknown code language -> safe default
    # Prose / markdown / runbook / postmortem / wiki.
    return "en"


def _derive_tags(path: str, text: str, doc_type: str) -> list[str]:
    haystack = (path + "\n" + (text or "")).lower()
    tags: list[str] = []
    for tag in _TECH_TAGS:
        if tag in haystack:
            # Normalize a couple of aliases.
            norm = {"postgresql": "postgres", "k8s": "kubernetes"}.get(tag, tag)
            if norm not in tags:
                tags.append(norm)
    return tags


def _derive_validity(path: str, meta: dict) -> bool:
    # A doc is valid unless a path/marker says it's archived/deprecated.
    low = path.lower()
    if any(m in low for m in ("/archive/", "/archived/", "/deprecated/", ".bak")):
        return False
    return True


def enrich_metadata(
    meta: dict[str, Any],
    *,
    path: str,
    text: str,
    source_type: str,
    struct_doc_type: DocType | None = None,
) -> dict[str, Any]:
    """Add deterministic facets to `meta` in place and return it.

    Parameters
    ----------
    meta:
        The chunk's metadata dict (already carries parser + chunker keys).
    path:
        The chunk's source path (used for convention-driven derivation).
    text:
        The chunk's content (used for tag sniffing + heading fallback).
    source_type:
        Origin connector ("git", "confluence", "slack", "rootly", ...).
    struct_doc_type:
        The structural DocType from the parser, if known.

    Enricher-derived keys are written only when absent so explicit
    parser/connector values are never clobbered.
    """
    heading = str(meta.get("section_heading") or "")

    def _set_if_absent(key: str, value: Any) -> None:
        if value is None:
            return
        if meta.get(key) in (None, "", [], {}):
            meta[key] = value

    # source_system: trust explicit, else map source_type.
    _set_if_absent("source_system", source_type or "unknown")

    # doc_type (normalized): always derivable; only fill when absent.
    doc_type = _derive_doc_type(path, struct_doc_type, meta)
    if doc_type in DOC_TYPES:
        _set_if_absent("doc_type", doc_type)
    else:
        _set_if_absent("doc_type", "generic_markdown")
    doc_type = meta.get("doc_type", doc_type)

    # environment
    env = _derive_environment(path, heading)
    if env in ENVIRONMENTS:
        _set_if_absent("environment", env)

    # tier / criticality (mirror)
    tier = _derive_tier(path, heading)
    if tier in TIERS:
        _set_if_absent("tier", tier)
        _set_if_absent("criticality", tier)

    # language
    lang = _derive_language(struct_doc_type, doc_type)
    if lang in LANGUAGES:
        _set_if_absent("language", lang)

    # tags: merge (don't clobber) any parser-supplied tags.
    derived_tags = _derive_tags(path, text, doc_type)
    if derived_tags:
        existing = meta.get("tags") or []
        merged = list(existing)
        for t in derived_tags:
            if t not in merged:
                merged.append(t)
        meta["tags"] = merged

    # valid: only set when not already declared.
    if "valid" not in meta:
        meta["valid"] = _derive_validity(path, meta)

    return meta
