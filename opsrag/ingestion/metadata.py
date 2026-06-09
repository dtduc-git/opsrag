"""Generalized, source-agnostic chunk-metadata schema.

This is the *schema of record* for the dict carried on `Chunk.metadata`
(`opsrag/interfaces/chunker.py`). The field stays a plain `dict` for
backward-compat -- nothing here changes the dataclass or the Qdrant
payload shape. This module only documents the controlled key set +
vocabularies and provides small helpers so the parser / chunker /
enricher layers populate the dict consistently.

Three population tiers (see design-scratch DESIGN 3, PART 1):
  - Parser-populated: provenance + identity facets known at parse time
    (`repo`/`path`/`branch`/`sha`, `helm_file_type` + subtypes,
    `section_heading`/`section_type`/`section_level`, `runbook`/
    `postmortem` flags, `title`, `source_system`, `url`, `author`,
    `created_at`/`updated_at`, `service`, `owner_team`, `version`).
  - Chunker-populated: positional facets (`chunk_index`/`chunk_count`,
    `heading_path`, `content_hash`, `token_count`, `child_index`).
  - Enricher-populated (deterministic, no LLM -- see enrich.py):
    normalized `doc_type`, `environment`, `tier`/`criticality`,
    `tags`, `language`, `valid`.

ALL fields are optional. Absence MUST NOT break retrieval -- the Qdrant
payload defaults already cope with missing keys. New facets are additive
only; embedding/vector dimensions, `parent_chunk_id` linkage, and the
contextual prefix are untouched.

PII note: `author` is an editor/committer identifier (e.g. an email).
By default it is stored hashed (see `hash_author`); set
`OPSRAG_STORE_AUTHOR_PLAINTEXT=1` to keep the raw value. `owner_team`
is a squad label, not personal data, and is stored as-is.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, TypedDict

# ---------------------------------------------------------------------------
# Controlled vocabularies. These are the *recommended* normalized values --
# enforcement is soft (facets are soft-boost / optional filters, never hard
# rejects) to avoid empty-result over-filtering. Unknown values are allowed
# but should be rare; keeping the enum small prevents sprawl across parsers.
# ---------------------------------------------------------------------------

# Normalized content classes. Superset of DocType.value plus a few
# source-only classes (wiki/chat/incident) that have no DocType.
DOC_TYPES: frozenset[str] = frozenset({
    "runbook",
    "postmortem",
    "incident",
    "terraform",
    "helm",
    "helm_values",
    "kubernetes",
    "dockerfile",
    "alert_definition",
    "architecture",
    "adr",
    "wiki",
    "chat",
    "yaml_config",
    "code",
    "generic_markdown",
})

# Origin connector / system of record.
SOURCE_SYSTEMS: frozenset[str] = frozenset({
    "git",
    "confluence",
    "slack",
    "rootly",
    "pagerduty",
    "jira",
    "notion",
    "unknown",
})

# Deployment environment scope. `preprod` is a first-class canonical env
# (per data-model.md), NOT a fold-into-staging alias -- a preprod
# `values.yaml` carries different config than staging, so collapsing them
# would tag preprod chunks inconsistently (enrich vs contextual) and make
# the `environment` filter mismatch.
ENVIRONMENTS: frozenset[str] = frozenset({
    "prod",
    "staging",
    "preprod",
    "dev",
    "test",
    "qa",
})

# Service criticality / tier. `tier` is the primary key; `criticality`
# mirrors it for callers that prefer the word form.
TIERS: frozenset[str] = frozenset({
    "tier0",
    "tier1",
    "tier2",
    "tier3",
})

# Content language. Natural-language docs are ISO-639-1 ("en"); code/config
# use the syntax family so model-routing can split code vs prose.
LANGUAGES: frozenset[str] = frozenset({
    "en",
    "yaml",
    "hcl",
    "json",
    "dockerfile",
    "python",
    "javascript",
    "typescript",
    "go",
    "java",
    "shell",
    "markdown",
})


class ChunkMetadata(TypedDict, total=False):
    """Typed view of the chunk metadata dict. `total=False` -- every key is
    optional, matching the backward-compat "absence must not break
    retrieval" requirement.

    Existing keys (kept, unchanged semantics):
    """

    # --- Provenance / identity (parser tier) ---
    repo: str
    path: str
    branch: str
    sha: str
    title: str
    source_system: str          # SOURCE_SYSTEMS
    url: str                     # canonical deep link for citations
    author: str                 # hashed by default (PII) -- see hash_author
    author_hashed: bool         # True when `author` holds a hash, not raw
    created_at: str             # ISO-8601
    updated_at: str             # ISO-8601
    owner_team: str             # owning squad
    version: str                # chart/app/API version

    # --- Section structure (parser tier, existing) ---
    section_heading: str
    section_type: str
    section_level: int

    # --- Helm facets (parser tier, existing) ---
    # helm_file_type: "chart" | "values"
    # section_type subtypes: values_image | values_resources |
    #                        values_networking | values_config
    helm_file_type: str

    # --- Type flags (parser tier, existing) ---
    runbook: bool
    postmortem: bool

    # --- Positional facets (chunker tier) ---
    chunk_index: int
    chunk_count: int
    child_index: int            # existing
    heading_path: list[str]     # breadcrumb, e.g. ["Ops", "Checkout", "Rollback"]
    content_hash: str           # "sha256:..." -- dedup / idempotent upsert
    token_count: int

    # --- Derived facets (enricher tier, deterministic) ---
    doc_type: str               # DOC_TYPES (normalized; distinct from Chunk.doc_type)
    service: str                # scalar primary service
    services: list[str]         # multi-service chunks for graph edges
    environment: str            # ENVIRONMENTS
    tier: str                   # TIERS
    criticality: str            # mirror of tier
    language: str               # LANGUAGES
    tags: list[str]             # freeform facets (alert names, tech, PII flag)
    valid: bool                 # False to retire stale docs
    superseded_by: str          # id of the doc that replaces this one


# Facets promoted to first-class indexed Qdrant payload keys. Kept here as
# the single source of truth so the vectorstore layer and the contract can
# reference the same list. (Promotion itself lives in qdrant.ensure_collection
# and is owned by the lead -- see the returned snippet.)
INDEXED_FACETS: tuple[str, ...] = (
    "doc_type",
    "environment",
    "tier",
    "service",
    "updated_at",
    "source_system",
    "valid",
)


def _store_author_plaintext() -> bool:
    """Config gate. Default False -> author is hashed (PII-safe). Set
    OPSRAG_STORE_AUTHOR_PLAINTEXT=1 to keep raw values (e.g. for an
    internal deployment that needs author-exact citation).
    """
    return os.environ.get("OPSRAG_STORE_AUTHOR_PLAINTEXT", "").lower() in (
        "1",
        "true",
        "yes",
    )


def hash_author(author: str | None) -> str | None:
    """Return a stable, non-reversible token for an author identifier.

    sha256, truncated + prefixed so it is recognizably a hash and never
    mistaken for a raw email. Stable across runs so the same author hashes
    to the same token (enables group-by without exposing PII).
    """
    if not author:
        return None
    digest = hashlib.sha256(author.strip().lower().encode("utf-8")).hexdigest()[:16]
    return f"anon:{digest}"


def apply_author(meta: dict[str, Any], author: str | None) -> None:
    """Write `author` into `meta`, hashing unless plaintext is configured.

    Records `author_hashed` so downstream consumers know whether the value
    is reversible. No-op for falsy authors (key left absent -> backward
    compatible).
    """
    if not author:
        return
    if _store_author_plaintext():
        meta["author"] = author
        meta["author_hashed"] = False
    else:
        hashed = hash_author(author)
        if hashed:
            meta["author"] = hashed
            meta["author_hashed"] = True


def _service_from_path(path: str) -> str | None:
    """Best-effort scalar service name from a path convention.

    Common layouts put the service name as a directory under a well-known
    parent (`services/<svc>/...`, `charts/<svc>/...`, `apps/<svc>/...`).
    Deterministic and conservative -- returns None when no convention
    matches rather than guessing.
    """
    parts = [p for p in path.split("/") if p]
    parents = {"services", "service", "charts", "chart", "apps", "app", "deploy", "k8s"}
    for i, seg in enumerate(parts[:-1]):
        if seg.lower() in parents and i + 1 < len(parts):
            cand = parts[i + 1]
            if cand and not cand.startswith("."):
                return cand
    return None


def _iso(dt: Any) -> str | None:
    """Render a datetime-ish value to ISO-8601, or None."""
    if dt is None:
        return None
    iso = getattr(dt, "isoformat", None)
    if callable(iso):
        try:
            return iso()
        except Exception:
            return None
    if isinstance(dt, str):
        return dt
    return None


def apply_provenance(meta: dict[str, Any], file: Any, *, source_type: str = "git") -> None:
    """Stamp provenance/identity facets a parser can know from the file.

    Fills (only when absent) `source_system`, `created_at`/`updated_at`
    from the file's `last_modified`, a path-derived `service`, and
    `owner_team` if the source metadata carries one. `author` is applied
    via `apply_author` (hashed by default). Never clobbers values a
    parser/connector already set.
    """
    src_meta = getattr(file, "metadata", None) or {}
    st = src_meta.get("source_type") or source_type

    def _set_if_absent(key: str, value: Any) -> None:
        if value not in (None, "", [], {}) and meta.get(key) in (None, "", [], {}):
            meta[key] = value

    _set_if_absent("source_system", st)

    # Freshness from the SCM/source mtime.
    updated = _iso(getattr(file, "last_modified", None))
    _set_if_absent("updated_at", updated)

    # Identity facets that some connectors carry in their doc metadata.
    _set_if_absent("url", src_meta.get("url") or src_meta.get("page_url"))
    _set_if_absent("owner_team", src_meta.get("owner_team"))
    _set_if_absent("created_at", _iso(src_meta.get("created_at")))
    _set_if_absent("service", _service_from_path(getattr(file, "path", "") or ""))

    # author: prefer explicit source metadata; hashed unless configured.
    author = src_meta.get("author") or src_meta.get("last_editor")
    if author and "author" not in meta:
        apply_author(meta, author)


def content_hash(text: str) -> str:
    """Stable content fingerprint for dedup / idempotent upsert.

    Prefixed with the algorithm so the value is self-describing in the
    payload (`sha256:...`), matching the design example.
    """
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
