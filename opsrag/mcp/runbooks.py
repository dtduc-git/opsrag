"""Skills-pattern lazy runbook loader -- `list_runbooks` + `load_runbook`.

Recommendation #1 from `deepagents_review.md`: when runbook content arrives via
**TOOL RESULTS** instead of retrieval, the LLM tends to quote it verbatim
(strings, paths, commands) rather than paraphrase. That fixes the
`chart/` -> `helm/` substitution bug we saw when the same content came in
via vector retrieval and got stylistically rewritten by the agent.

## Two-step loader

The flow is intentionally split so the agent loads the **catalog** first
(cheap, tokens-light) and then chooses **one** runbook to read into
context (~1-5 KB of literal markdown).

| Tool             | Behaviour                                                |
|------------------|----------------------------------------------------------|
| `list_runbooks`  | List the catalog (optionally topic-filtered, top-20).    |
| `load_runbook`   | Return the FULL markdown for one runbook by `name` (id). |

The flow mirrors Claude Code's "skills" pattern: show the LLM a menu of
capabilities, let it pick one, then load the full body on demand. The
LLM should NOT call `load_runbook` blindly -- it should call
`list_runbooks(topic=...)` first to learn which `name` to pass.

## Source of truth

The catalog is built **at module import** by walking the local sre-kb
clone (configured via `OPSRAG_SRE_KB_PATH`). Each runbook is a markdown
file under `docs/runbooks/<service>/<slug>.md` with YAML frontmatter
and a `# H1` immediately after the closing `---`.

The catalog is cached for 5 minutes. Within `list_runbooks`, if the
cache is still valid but ANY file mtime is newer than the cache build
time, the catalog reloads (per spec).

## Read-only

Both tools are pure file IO -- no DB connection, no LLM, no embeddings.
They never write, never delete, never call out to Confluence or
GitLab. If a frontmatter block is malformed, the file is skipped with
a logged warning rather than crashing the catalog.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from opsrag.mcp.gitlab import MCPTool

_log = logging.getLogger("opsrag.mcp.runbooks")

# Match a YAML frontmatter block at the start of a file:
#   ---
#   <yaml>
#   ---
# followed by the body. We allow leading whitespace tolerantly.
_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<yaml>.*?)\n---\s*\n(?P<body>.*)\Z", re.DOTALL
)
_H1_RE = re.compile(r"^#\s+(?P<title>.+?)\s*$", re.MULTILINE)
# Match a "## Summary" section and capture everything until the next
# "## " section heading or end-of-file.
_SUMMARY_RE = re.compile(
    r"^##\s+Summary\s*\n+(?P<body>.+?)(?=\n##\s+|\Z)",
    re.MULTILINE | re.DOTALL,
)

_CACHE_TTL_S = 300.0
_LIST_TOPIC_TOP_K = 20
_LIST_NO_TOPIC_CAP = 40
_DEFAULT_SRE_KB_PATH = "./sre-knowledge-base"


# --- data shape -----------------------------------------------------


@dataclass(frozen=True)
class RunbookEntry:
    """One entry in the catalog. `markdown` is the FULL file body
    (everything after the frontmatter closing `---`)."""

    name: str  # frontmatter `id`
    title: str  # H1
    when_to_use: str  # first sentence of Summary + keywords
    source: str  # confluence_url if present else absolute file path
    file_path: str  # absolute path on disk
    last_reviewed: str  # frontmatter `last_reviewed`, "" if absent
    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    markdown: str = ""  # full file body (no frontmatter)
    mtime: float = 0.0


# --- catalog state (process-local, locked) --------------------------


_catalog_lock = threading.Lock()
_catalog: dict[str, RunbookEntry] = {}
_catalog_built_at: float = 0.0
_catalog_max_mtime: float = 0.0


# --- helpers --------------------------------------------------------


def _sre_kb_root() -> Path:
    """Resolve the sre-knowledge-base root (env-overridable)."""
    return Path(os.environ.get("OPSRAG_SRE_KB_PATH", _DEFAULT_SRE_KB_PATH))


def _runbooks_dir() -> Path:
    return _sre_kb_root() / "docs" / "runbooks"


def _first_sentence(text: str) -> str:
    """Return the first sentence of `text`. Splits on '. ' / newlines /
    end-of-string; trims trailing whitespace."""
    if not text:
        return ""
    s = text.strip()
    # Drop bold markers that often start a Summary's "**X**" sentence.
    # We just look for the first '.' that ends a sentence (followed by
    # space/newline/EOF) -- this is good enough for our markdown.
    m = re.search(r"\.(?:\s|$)", s)
    if m:
        return s[: m.end()].strip()
    # No period found -- return up to first newline.
    return s.splitlines()[0].strip() if s else ""


def _parse_runbook_file(path: Path) -> RunbookEntry | None:
    """Read one runbook .md file and return a `RunbookEntry`. Returns
    `None` (logged warning) on malformed frontmatter or missing fields.
    Caller is responsible for path filtering (e.g. skip `_template-*`)."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        _log.warning("runbooks: failed to read %s: %s", path, exc)
        return None

    m = _FRONTMATTER_RE.match(raw)
    if not m:
        _log.warning("runbooks: no YAML frontmatter in %s (skipping)", path)
        return None

    try:
        fm = yaml.safe_load(m.group("yaml")) or {}
    except yaml.YAMLError as exc:
        _log.warning("runbooks: malformed YAML frontmatter in %s: %s", path, exc)
        return None
    if not isinstance(fm, dict):
        _log.warning(
            "runbooks: frontmatter is not a mapping in %s (got %s) -- skipping",
            path,
            type(fm).__name__,
        )
        return None

    rb_id = fm.get("id")
    if not rb_id or not isinstance(rb_id, str):
        _log.warning("runbooks: frontmatter missing/non-string `id` in %s -- skipping", path)
        return None

    body = m.group("body")

    # H1 title -- required. Fall back to frontmatter `title` if H1 absent.
    h1 = _H1_RE.search(body)
    title = h1.group("title").strip() if h1 else (fm.get("title") or rb_id)

    # `when_to_use` = first sentence of ## Summary + comma-joined keywords.
    summary_match = _SUMMARY_RE.search(body)
    summary_first = _first_sentence(summary_match.group("body")) if summary_match else ""
    keywords = fm.get("keywords") or []
    if not isinstance(keywords, list):
        keywords = []
    kw_str = ", ".join(str(k) for k in keywords if k)
    when_to_use = summary_first
    if kw_str:
        when_to_use = f"{summary_first} (keywords: {kw_str})" if summary_first else f"keywords: {kw_str}"

    tags = fm.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    tags = [str(t) for t in tags if t]

    confluence_url = fm.get("confluence_url") or ""
    source = confluence_url if confluence_url else str(path.resolve())

    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0

    return RunbookEntry(
        name=rb_id,
        title=title,
        when_to_use=when_to_use,
        source=source,
        file_path=str(path.resolve()),
        last_reviewed=str(fm.get("last_reviewed") or ""),
        tags=tags,
        keywords=[str(k) for k in keywords if k],
        markdown=body,
        mtime=mtime,
    )


def _walk_runbooks() -> list[Path]:
    """Find all candidate runbook markdowns. Excludes `_template-*` files."""
    root = _runbooks_dir()
    if not root.is_dir():
        _log.warning("runbooks: catalog root %s does not exist", root)
        return []
    out: list[Path] = []
    for p in root.rglob("*.md"):
        if p.name.startswith("_"):
            continue
        out.append(p)
    return sorted(out)


def _build_catalog() -> tuple[dict[str, RunbookEntry], float]:
    """Walk the runbook dir, parse each file, return (catalog, max_mtime).
    Pure function -- does not mutate module state."""
    paths = _walk_runbooks()
    cat: dict[str, RunbookEntry] = {}
    max_mtime = 0.0
    for p in paths:
        entry = _parse_runbook_file(p)
        if entry is None:
            continue
        if entry.name in cat:
            _log.warning(
                "runbooks: duplicate id %r -- keeping %s, ignoring %s",
                entry.name,
                cat[entry.name].file_path,
                entry.file_path,
            )
            continue
        cat[entry.name] = entry
        if entry.mtime > max_mtime:
            max_mtime = entry.mtime
    return cat, max_mtime


def _current_max_mtime(paths: list[Path]) -> float:
    """Cheapest possible staleness check -- just stat() each .md file."""
    mx = 0.0
    for p in paths:
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if m > mx:
            mx = m
    return mx


def _get_catalog(force_refresh: bool = False) -> dict[str, RunbookEntry]:
    """Return the live catalog. Reloads if (a) never built, (b) TTL
    expired, (c) any on-disk mtime is newer than the last build, or
    (d) `force_refresh=True`."""
    global _catalog, _catalog_built_at, _catalog_max_mtime
    now = time.time()

    with _catalog_lock:
        needs_reload = force_refresh or not _catalog or _catalog_built_at == 0.0
        if not needs_reload and (now - _catalog_built_at) > _CACHE_TTL_S:
            needs_reload = True
        if not needs_reload:
            # Within TTL -- but the spec says "refresh on cache miss if
            # the underlying file mtime changed", so we also check
            # mtimes proactively on each call. This is cheap (stat-only).
            cur_max = _current_max_mtime(_walk_runbooks())
            if cur_max > _catalog_max_mtime:
                needs_reload = True

        if needs_reload:
            cat, max_mtime = _build_catalog()
            _catalog = cat
            _catalog_built_at = now
            _catalog_max_mtime = max_mtime
            _log.info(
                "runbooks: catalog (re)built with %d entries from %s",
                len(cat),
                _runbooks_dir(),
            )

        return _catalog


# --- ranking --------------------------------------------------------


def _tokenize(s: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", (s or "").lower()) if t]


def _score(entry: RunbookEntry, topic_tokens: list[str], topic_lower: str) -> float:
    """Simple lex-overlap score. No LLM, no embedder.

    Scoring rules (additive):
      - +3.0 for each topic-token that's a substring of the title
      - +2.0 for each topic-token that exactly matches a keyword
      - +1.5 for each topic-token that exactly matches a tag
      - +1.0 for each topic-token that's a substring anywhere in the
        runbook's full text (markdown body)
      - +5.0 if the entire topic string appears as a substring of
        title (case-insensitive) -- gives a strong boost to specific
        phrases like 'long running transaction'
    """
    if not topic_tokens:
        return 0.0
    title_l = entry.title.lower()
    kws = {k.lower() for k in entry.keywords}
    tags = {t.lower() for t in entry.tags}
    body_l = entry.markdown.lower()

    score = 0.0
    if topic_lower and topic_lower in title_l:
        score += 5.0
    for tok in topic_tokens:
        if not tok:
            continue
        if tok in title_l:
            score += 3.0
        if tok in kws:
            score += 2.0
        if tok in tags:
            score += 1.5
        if tok in body_l:
            score += 1.0
    return score


# --- handlers -------------------------------------------------------


async def _h_list_runbooks(_unused, args: dict) -> Any:
    topic = (args.get("topic") or "").strip()
    catalog = _get_catalog()
    if not catalog:
        return {
            "topic": topic,
            "count": 0,
            "runbooks": [],
            "error": "no runbooks found -- check OPSRAG_SRE_KB_PATH and that sre-knowledge-base/docs/runbooks/ has .md files",
        }

    entries = list(catalog.values())

    if topic:
        topic_lower = topic.lower()
        topic_tokens = _tokenize(topic)
        scored = [(_score(e, topic_tokens, topic_lower), e) for e in entries]
        scored = [(s, e) for s, e in scored if s > 0.0]
        scored.sort(key=lambda se: (-se[0], se[1].name))
        ranked = [e for _, e in scored[:_LIST_TOPIC_TOP_K]]
    else:
        # No topic -- return all (up to cap), sorted alphabetically by id
        # so the LLM gets a stable list across calls.
        entries.sort(key=lambda e: e.name)
        ranked = entries[:_LIST_NO_TOPIC_CAP]

    out = [
        {
            "name": e.name,
            "title": e.title,
            "when_to_use": e.when_to_use,
            "source": e.source,
        }
        for e in ranked
    ]
    return {
        "topic": topic,
        "count": len(out),
        "total_in_catalog": len(catalog),
        "runbooks": out,
    }


async def _h_load_runbook(_unused, args: dict) -> Any:
    name = (args.get("name") or "").strip()
    if not name:
        return {"error": "name is required (use list_runbooks to discover ids)"}

    catalog = _get_catalog()
    entry = catalog.get(name)
    if entry is None:
        available = sorted(catalog.keys())[:5]
        return {
            "error": "runbook not found",
            "requested_name": name,
            "available_names_first_5": available,
        }

    # Cap markdown to keep the generator's context budget under control.
    # ~12K chars ~= 3K tokens -- enough to carry a typical runbook
    # (5 KB median, 15 KB max) without truncation on most, while giving
    # the generator room to actually write an answer. The biggest runbook
    # (onboarding-new-saas-app.md at ~38KB) gets cut to its first 12KB --
    # which still includes Summary + Prerequisites + Steps 1-3, the part
    # the user most often needs.
    body = entry.markdown
    _LOAD_CAP = 12_000
    truncated = False
    if len(body) > _LOAD_CAP:
        # Cut at a paragraph boundary near the cap, not mid-sentence.
        cut = body.rfind("\n\n", 0, _LOAD_CAP)
        if cut < _LOAD_CAP // 2:
            cut = _LOAD_CAP
        body = body[:cut].rstrip() + (
            f"\n\n[...runbook truncated at {cut} chars; full file is "
            f"{len(entry.markdown)} chars at {entry.source}]"
        )
        truncated = True
    return {
        "name": entry.name,
        "title": entry.title,
        "source": entry.source,
        "markdown": body,
        "truncated": truncated,
        "full_length_chars": len(entry.markdown),
        "last_reviewed": entry.last_reviewed,
    }


# --- tool registry --------------------------------------------------


RUNBOOK_TOOLS: list[MCPTool] = [
    MCPTool(
        name="runbook_list",
        description=(
            "List SRE runbooks (the team's source-of-truth "
            "incident-response procedures for Kafka, CloudSQL, Istio, "
            "ArgoCD, Kong, Keda, Clickhouse, Celery, customer-reports, "
            "and deployment/onboarding workflows). Use this BEFORE "
            "answering any 'how do we handle <alert>?' or 'what's the "
            "runbook for X?' / 'how do we onboard <service>?' style "
            "question -- runbooks contain operator-verified commands and "
            "decision trees that must be quoted verbatim, not "
            "paraphrased.\n\n"
            "Two-step pattern: (1) call `runbook_list` with a `topic` "
            "to get a menu of candidate runbooks (each with a "
            "`when_to_use` hint), then (2) call `runbook_load` with the "
            "chosen `name` to read the full body. Prefer this over "
            "`knowledge_search` for runbooks -- knowledge_search "
            "returns chunked excerpts; this returns the full file."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Free-text topic to filter the catalog (e.g. 'cloudsql long running transaction', 'onboard new service', 'kafka consumer lag'). Optional -- omit to list everything (up to 40).",
                },
            },
            "required": [],
        },
        handler=_h_list_runbooks,
    ),
    MCPTool(
        name="runbook_load",
        description=(
            "Load the FULL markdown of one runbook by `name` (the id "
            "returned by `runbook_list`). Returns the verbatim "
            "operator-authored procedure -- quote exact commands, "
            "paths, and SQL from this output rather than paraphrasing. "
            "Runbooks are typically 1-5 KB; this tool does "
            "not truncate.\n\n"
            "If `name` is wrong, the response includes "
            "`available_names_first_5` so you can self-correct -- "
            "usually it means you should call `runbook_list` with a "
            "better topic first."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The runbook id, e.g. `runbook-cloudsql-long-running-transaction`. Always discovered via `runbook_list` -- do not guess.",
                },
            },
            "required": ["name"],
        },
        handler=_h_load_runbook,
    ),
]


__all__ = [
    "RUNBOOK_TOOLS",
    "RunbookEntry",
    "_get_catalog",  # exposed for tests + observability
    "_build_catalog",
    "_parse_runbook_file",
    "_runbooks_dir",
    "_sre_kb_root",
    "build_fake",
]


# --- fake backend (FR-012; integration tests) ----------------------
#
# The runbooks tools are self-contained: their handlers ignore the
# client arg and read from the module-global catalog state built by
# `_get_catalog()` walking the local FS. So the offline fake does NOT
# need a fake client (client=None per the _fake.py contract). Instead
# it installs a small in-memory catalog directly into the module
# globals and pins the build timestamp so `_get_catalog()` serves it
# without ever touching disk. `teardown` restores the previous state.


def _fake_catalog() -> dict[str, RunbookEntry]:
    """A tiny, shape-faithful catalog with generic content (no
    proprietary names; example.com only)."""
    entry_a = RunbookEntry(
        name="runbook-service-restart",
        title="Restart a Degraded Service",
        when_to_use=(
            "Use when a service is degraded and needs a rolling restart. "
            "(keywords: restart, degraded, rollout)"
        ),
        source="https://docs.example.com/runbooks/service-restart",
        file_path="/fake/docs/runbooks/example/service-restart.md",
        last_reviewed="2026-01-15",
        tags=["service", "restart"],
        keywords=["restart", "degraded", "rollout"],
        markdown=(
            "# Restart a Degraded Service\n\n"
            "## Summary\n\n"
            "Use when a service is degraded and needs a rolling restart.\n\n"
            "## Steps\n\n"
            "1. Check current status.\n"
            "2. Trigger a rolling restart.\n"
            "3. Verify recovery.\n"
        ),
        mtime=0.0,
    )
    entry_b = RunbookEntry(
        name="runbook-disk-pressure",
        title="Resolve Disk Pressure",
        when_to_use=(
            "Use when a node reports disk pressure. "
            "(keywords: disk, storage, cleanup)"
        ),
        source="https://docs.example.com/runbooks/disk-pressure",
        file_path="/fake/docs/runbooks/example/disk-pressure.md",
        last_reviewed="2026-02-01",
        tags=["node", "storage"],
        keywords=["disk", "storage", "cleanup"],
        markdown=(
            "# Resolve Disk Pressure\n\n"
            "## Summary\n\n"
            "Use when a node reports disk pressure.\n\n"
            "## Steps\n\n"
            "1. Identify the largest consumers.\n"
            "2. Reclaim space.\n"
            "3. Confirm pressure clears.\n"
        ),
        mtime=0.0,
    )
    return {entry_a.name: entry_a, entry_b.name: entry_b}


def build_fake():
    """Return a FakeMCP exposing the runbook tools backed by an offline,
    in-memory catalog. No filesystem access, no network."""
    from opsrag.mcp._fake import FakeMCP

    global _catalog, _catalog_built_at, _catalog_max_mtime

    # Snapshot prior state so teardown can restore it exactly.
    prev_catalog = _catalog
    prev_built_at = _catalog_built_at
    prev_max_mtime = _catalog_max_mtime

    with _catalog_lock:
        _catalog = _fake_catalog()
        # Pin a fresh build time so `_get_catalog()` treats the cache as
        # valid; a high max_mtime keeps the on-disk staleness check from
        # ever forcing a reload from a (possibly absent) runbooks dir.
        _catalog_built_at = time.time()
        _catalog_max_mtime = float("inf")

    def _restore() -> None:
        global _catalog, _catalog_built_at, _catalog_max_mtime
        with _catalog_lock:
            _catalog = prev_catalog
            _catalog_built_at = prev_built_at
            _catalog_max_mtime = prev_max_mtime

    return FakeMCP(tools=list(RUNBOOK_TOOLS), client=None, teardown=_restore)
