"""Repomap compass (P2) -- service catalog + key repo inventory.

Per 2025-2026 industry consensus, code-aware agents need a *compass*
in their system context so they can plan tool calls without a prior
retrieval pass. Aider does this via tree-sitter + personalized
PageRank; Cody via SCIP; Cursor via directory enumeration. This is
the OpsRAG equivalent -- deliberately scoped to what the operating
organization's LLM needs most:

  1. A **service catalog** listing every service declared in
     gitops (`values/saas/*/`) plus, when available,
     a hint of which source repo backs it (`saas/<name>`).
  2. A **key-repo inventory** summarising the top-level directory
     layout of the most-touched repos. The set of "key" repos is NOT
     baked into the engine: it comes from the operator-supplied
     DeploymentContext (``key_repos``) at call time. When the operator
     names no key repos, this section is simply omitted -- the engine
     carries no organization-specific repo facts of its own.

The card is built from the local repo cache (`/tmp/opsrag-repos`).
Pure filesystem + git ls-files, no LLM, no embedding. ~5K char
budget -- small enough to fit in every reasoner prompt without
crowding tool calls.

Refresh: built lazily on first call after process start (cached for
30 minutes). The indexer's daily clone pass keeps the underlying
cache current; if a new repo shows up between refreshes the card
will lag by up to 30 min, which is fine for the failure modes this
addresses (LLM not knowing a service name exists at all).
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from opsrag.agent.prompt_render import active_deployment

_log = logging.getLogger("opsrag.agent.repomap")

_CACHE_ROOT = Path(os.environ.get("OPSRAG_REPO_CACHE", "/tmp/opsrag-repos"))
_REPO_SEP = "__"
_TTL_SECONDS = 30 * 60

# Module-level cache.
_cache: dict[str, object] = {"value": None, "built_at": 0.0}


def _human_name(flat: str) -> str:
    return flat.replace(_REPO_SEP, "/")


def _key_repo_hints() -> list[str]:
    """Repos whose directory layouts are surfaced as a tree.

    Sourced from the operator-supplied DeploymentContext (``key_repos``)
    and computed at call time so it reflects the active context. When the
    operator names no key repos the result is empty, and the layout
    section degrades gracefully to "no specific repo hints" rather than
    falling back to any baked-in organization's repos.
    """
    return list(active_deployment().key_repos)


def _list_cached_repos() -> list[str]:
    if not _CACHE_ROOT.is_dir():
        return []
    out: list[str] = []
    for child in sorted(_CACHE_ROOT.iterdir()):
        if not child.is_dir() or not (child / ".git").exists():
            continue
        out.append(_human_name(child.name))
    return out


def _service_catalog(cached_repos: list[str]) -> list[str]:
    """Enumerate services from `gitops/values/saas/*/`.

    Each `values/saas/<name>/` directory is one declared service
    (`<service-a>`, `<service-b>`, ...). Returns sorted service names;
    empty when the gitops repo isn't cached yet.
    """
    flat = "devops__gitops"
    repo_root = _CACHE_ROOT / flat
    saas_dir = repo_root / "values" / "saas"
    if not saas_dir.is_dir():
        return []
    services: list[str] = []
    for child in sorted(saas_dir.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            services.append(child.name)
    return services


def _saas_source_repos(cached_repos: list[str]) -> set[str]:
    """Set of service names that ALSO have a `saas/<name>` source repo
    cached. Used to annotate the service catalog.

    The cached repos include all groups (devops/, saas/, ...). For each
    `saas/<x>` repo we pull just `<x>` (and similarly for nested groups
    like `saas/<group>/backend/<x>`). The match is lenient: we record
    just the LAST path segment.
    """
    out: set[str] = set()
    for r in cached_repos:
        if not r.startswith("saas/"):
            continue
        leaf = r.rsplit("/", 1)[-1]
        out.add(leaf)
    return out


def _top_dirs(repo: str, max_depth: int = 2, limit: int = 12) -> list[str]:
    """Return up to `limit` top-level (and one level deeper) dirs of `repo`."""
    flat = repo.replace("/", _REPO_SEP)
    root = _CACHE_ROOT / flat
    if not root.is_dir():
        return []
    out: list[str] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            out.append(child.name + "/")
            if len(out) >= limit:
                break
    return out


def _build_card() -> str:
    """Build the full repomap compass card. ~3-5K chars."""
    cached = _list_cached_repos()
    if not cached:
        return ""

    services = _service_catalog(cached)
    saas_sources = _saas_source_repos(cached)

    sections: list[str] = []

    if services:
        sections.append("=== Service catalog (from gitops/values/saas/) ===")
        sections.append(
            "Every name below is a real service with a Helm chart in "
            "`devops/gitops/values/saas/<name>/`. When a service ALSO has a "
            "dedicated source repo, it appears with `[src: saas/<name>]`. Use these names "
            "VERBATIM when the user asks about a service -- do not invent variants."
        )
        lines: list[str] = []
        for s in services:
            tag = f" [src: saas/{s}]" if s in saas_sources else ""
            lines.append(f"  - {s}{tag}")
        sections.append("\n".join(lines))

    cached_set = set(cached)
    key_repos = _key_repo_hints()
    layout_lines: list[str] = []
    for r in key_repos:
        if r not in cached_set:
            continue
        top = _top_dirs(r)
        if top:
            layout_lines.append(f"  {r}/  ->  " + "  ".join(top))
    if layout_lines:
        sections.append("=== Key repo top-level layout ===")
        sections.append(
            "Quick directory shape of the most-touched repos. Use as a starting "
            "point for `code_glob` / `code_grep` path scoping."
        )
        sections.append("\n".join(layout_lines))

    sections.append("=== Other repos available via code_* tools ===")
    key_in_cache = {h for h in key_repos if h in cached_set}
    other = [r for r in cached if r not in key_in_cache]
    if other:
        # Keep it compact -- one-line list.
        sections.append("  " + ", ".join(other))
    else:
        sections.append("  (none beyond key repos listed above)")

    return "\n\n".join(sections)


def get_repomap_card(force_refresh: bool = False) -> str:
    """Return the cached repomap card (rebuild every 30 min by default)."""
    now = time.time()
    built = float(_cache.get("built_at") or 0.0)
    if force_refresh or (now - built) > _TTL_SECONDS or _cache.get("value") is None:
        try:
            card = _build_card()
        except Exception as exc:
            _log.warning("repomap rebuild failed: %s", exc)
            card = str(_cache.get("value") or "")
        _cache["value"] = card
        _cache["built_at"] = now
    return str(_cache.get("value") or "")
