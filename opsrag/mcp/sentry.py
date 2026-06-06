"""Sentry MCP-style tools for OpsRAG (read-only).

Read-only async tools over the Sentry REST API (`/api/0`). Reuses
`SENTRY_TOKEN` (bearer) from env. Base host is configurable via
`SENTRY_HOST` (default `sentry.io`) to support regional / self-hosted
deployments; a default org slug can be set via `SENTRY_ORG`.

## Read-only enforcement

Every tool issues an HTTP `GET` against a read endpoint -- list /
search / fetch only. No `POST` / `PUT` / `DELETE` / `PATCH` anywhere:
no issue resolution, no project mutation, no event ingestion.

## Tool list (8 read-only)

| Tool                       | Endpoint                                            |
|----------------------------|------------------------------------------------------|
| `sentry_list_projects`     | GET `/organizations/{org}/projects/`                 |
| `sentry_search_issues`     | GET `/organizations/{org}/issues/`                   |
| `sentry_get_issue`         | GET `/organizations/{org}/issues/{issue_id}/`        |
| `sentry_get_latest_event`  | GET `/organizations/{org}/issues/{id}/events/latest/`|
| `sentry_search_events`     | GET `/organizations/{org}/events/`                   |
| `sentry_get_event`         | GET `/projects/{org}/{project}/events/{event_id}/`   |
| `sentry_get_trace`         | GET `/organizations/{org}/events-trace/{trace_id}/`  |
| `sentry_list_releases`     | GET `/organizations/{org}/releases/`                 |
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from opsrag.mcp.gitlab import MCPTool

_log = logging.getLogger("opsrag.mcp.sentry")

DEFAULT_SENTRY_HOST = "sentry.io"
_DEFAULT_TIMEOUT_S = 30.0
_DEFAULT_LIMIT = 25
_MAX_LIMIT = 100
_RESULT_TRUNCATE_CHARS = 8000
_MAX_STACK_FRAMES = 30


# Secrets can leak into event tags / messages / stack frames -- redact.
_REDACT_PATTERNS = [
    (re.compile(r"\bxoxb-[A-Za-z0-9-]{20,}"), "[REDACTED:slack_bot_token]"),
    (re.compile(r"\bxoxp-[A-Za-z0-9-]{20,}"), "[REDACTED:slack_user_token]"),
    (re.compile(r"\bghp_[A-Za-z0-9]{30,}"), "[REDACTED:github_token]"),
    (re.compile(r"\bglpat-[A-Za-z0-9_]{20,}"), "[REDACTED:gitlab_token]"),
    (re.compile(r"\bAKIA[A-Z0-9]{16}\b"), "[REDACTED:aws_access_key]"),
    (re.compile(r"\bAIza[A-Za-z0-9_-]{35}"), "[REDACTED:google_api_key]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]+"), "[REDACTED:jwt]"),
    # Sentry auth tokens (sntrys_/sntryu_) -- never echo back into errors.
    (re.compile(r"\bsntry[a-z]_[A-Za-z0-9_=+/-]{20,}"), "[REDACTED:sentry_token]"),
]


def _redact(text: str) -> str:
    for pat, repl in _REDACT_PATTERNS:
        text = pat.sub(repl, text)
    return text


class SentryMCPError(Exception):
    """Raised on Sentry API errors. Wraps the upstream status + body."""

    def __init__(self, status: int, body: str, *, tool: str | None = None):
        self.status = status
        self.body = _redact(body or "")
        self.tool = tool
        super().__init__(f"[{tool or 'sentry'}] {status}: {self.body[:300]}")


@dataclass
class _Config:
    token: str
    api_url: str
    org: str | None


def _config() -> _Config:
    token = (os.environ.get("SENTRY_TOKEN") or "").strip().strip('"').strip("'")
    host = (os.environ.get("SENTRY_HOST") or DEFAULT_SENTRY_HOST).strip().rstrip("/")
    # Allow callers to pass either a bare host or a full scheme+host.
    if host.startswith("http://") or host.startswith("https://"):
        base = f"{host}/api/0"
    else:
        base = f"https://{host}/api/0"
    org = (os.environ.get("SENTRY_ORG") or "").strip() or None
    if not token:
        raise SentryMCPError(
            0,
            "Sentry credentials not set. Set SENTRY_TOKEN (a bearer "
            "auth token with read scopes: project:read, event:read, "
            "org:read). Optionally set SENTRY_HOST (region/self-hosted) "
            "and SENTRY_ORG (default org slug).",
        )
    return _Config(token=token, api_url=base, org=org)


def _headers() -> dict:
    cfg = _config()
    return {
        "Authorization": f"Bearer {cfg.token}",
        "Content-Type": "application/json",
    }


async def _get(path: str, params: dict | None = None, *, tool: str = "sentry") -> Any:
    cfg = _config()
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    async with httpx.AsyncClient(headers=_headers(), timeout=_DEFAULT_TIMEOUT_S) as http:
        resp = await http.get(f"{cfg.api_url}{path}", params=clean)
    if resp.status_code >= 400:
        raise SentryMCPError(resp.status_code, resp.text, tool=tool)
    return resp.json() if resp.text else {}


def _resolve_org(args: dict, *, tool: str) -> str:
    org = (args.get("org") or _config().org or "").strip()
    if not org:
        raise SentryMCPError(
            0,
            "No organization slug. Pass `org` or set SENTRY_ORG.",
            tool=tool,
        )
    return org


def _clamp(n: int | None, default: int = _DEFAULT_LIMIT, *, max_n: int = _MAX_LIMIT) -> int:
    if n is None:
        return default
    return max(1, min(int(n), max_n))


def _truncate(text: str, limit: int = _RESULT_TRUNCATE_CHARS) -> str:
    if not text:
        return ""
    text = _redact(text)
    if len(text) > limit:
        return text[:limit] + " ...[truncated]"
    return text


def _trim_frames(stacktrace: dict | None) -> list[dict]:
    """Trim a Sentry stacktrace to a bounded list of high-signal frames."""
    if not isinstance(stacktrace, dict):
        return []
    frames = stacktrace.get("frames") or []
    # Sentry orders frames oldest-first; the crashing frame is last. Keep
    # the tail (closest to the error) and trim the rest.
    tail = frames[-_MAX_STACK_FRAMES:] if len(frames) > _MAX_STACK_FRAMES else frames
    out = []
    for f in tail:
        if not isinstance(f, dict):
            continue
        out.append({
            "function": f.get("function"),
            "filename": f.get("filename") or f.get("absPath"),
            "lineno": f.get("lineNo") or f.get("lineno"),
            "module": f.get("module"),
            "in_app": f.get("inApp"),
            "context": _truncate(str(f.get("context") or "")[:500], 500) if f.get("context") else None,
        })
    return out


def _extract_stacktrace(event: dict) -> list[dict]:
    """Pull stack frames out of a Sentry event's `exception`/`stacktrace`
    entries (lives under the event's `entries` list)."""
    entries = event.get("entries") or []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        etype = entry.get("type")
        data = entry.get("data") or {}
        if etype == "exception":
            values = data.get("values") or []
            for v in values:
                st = (v or {}).get("stacktrace")
                frames = _trim_frames(st)
                if frames:
                    return frames
        if etype == "stacktrace":
            frames = _trim_frames(data)
            if frames:
                return frames
    return []


# --- handlers -------------------------------------------------------


async def _h_list_projects(_unused, args: dict) -> Any:
    """`GET /organizations/{org}/projects/` -- list projects in the org."""
    org = _resolve_org(args, tool="sentry_list_projects")
    params = {"per_page": _clamp(args.get("limit"))}
    resp = await _get(
        f"/organizations/{quote(org, safe='')}/projects/",
        params=params, tool="sentry_list_projects",
    )
    items = resp if isinstance(resp, list) else (resp.get("data") or [])
    out = []
    for p in items[: _clamp(args.get("limit"))]:
        out.append({
            "id": p.get("id"),
            "slug": p.get("slug"),
            "name": p.get("name"),
            "platform": p.get("platform"),
            "team": (p.get("team") or {}).get("slug") if isinstance(p.get("team"), dict) else None,
            "status": p.get("status"),
            "date_created": p.get("dateCreated"),
        })
    return {"org": org, "count": len(out), "projects": out}


async def _h_search_issues(_unused, args: dict) -> Any:
    """`GET /organizations/{org}/issues/` -- search issues with Sentry
    search syntax. Default query `is:unresolved`."""
    org = _resolve_org(args, tool="sentry_search_issues")
    params = {
        "query": args.get("query") or "is:unresolved",
        "statsPeriod": args.get("statsPeriod") or "24h",
        "project": args.get("project"),
        "environment": args.get("environment"),
        "sort": args.get("sort"),
        "per_page": _clamp(args.get("limit")),
    }
    resp = await _get(
        f"/organizations/{quote(org, safe='')}/issues/",
        params=params, tool="sentry_search_issues",
    )
    items = resp if isinstance(resp, list) else (resp.get("data") or [])
    out = []
    for x in items[: _clamp(args.get("limit"))]:
        meta = x.get("metadata") or {}
        out.append({
            "id": x.get("id"),
            "short_id": x.get("shortId"),
            "title": _truncate(x.get("title") or "", 500),
            "culprit": _truncate(x.get("culprit") or "", 500),
            "level": x.get("level"),
            "status": x.get("status"),
            "count": x.get("count"),
            "user_count": x.get("userCount"),
            "first_seen": x.get("firstSeen"),
            "last_seen": x.get("lastSeen"),
            "project": (x.get("project") or {}).get("slug") if isinstance(x.get("project"), dict) else None,
            "type": meta.get("type"),
            "permalink": x.get("permalink"),
        })
    return {
        "org": org,
        "query": params["query"],
        "statsPeriod": params["statsPeriod"],
        "count": len(out),
        "issues": out,
    }


async def _h_get_issue(_unused, args: dict) -> Any:
    """`GET /organizations/{org}/issues/{issue_id}/` -- one issue's detail."""
    org = _resolve_org(args, tool="sentry_get_issue")
    issue_id = quote(str(args["issue_id"]), safe="")
    resp = await _get(
        f"/organizations/{quote(org, safe='')}/issues/{issue_id}/",
        tool="sentry_get_issue",
    )
    meta = resp.get("metadata") or {}
    return {
        "id": resp.get("id"),
        "short_id": resp.get("shortId"),
        "title": _truncate(resp.get("title") or "", 500),
        "culprit": _truncate(resp.get("culprit") or "", 500),
        "level": resp.get("level"),
        "status": resp.get("status"),
        "count": resp.get("count"),
        "user_count": resp.get("userCount"),
        "first_seen": resp.get("firstSeen"),
        "last_seen": resp.get("lastSeen"),
        "type": meta.get("type"),
        "value": _truncate(meta.get("value") or "", 1000),
        "project": (resp.get("project") or {}).get("slug") if isinstance(resp.get("project"), dict) else None,
        "permalink": resp.get("permalink"),
    }


async def _h_get_latest_event(_unused, args: dict) -> Any:
    """`GET /organizations/{org}/issues/{issue_id}/events/latest/` -- the
    most recent event for an issue, with stacktrace / culprit / tags."""
    org = _resolve_org(args, tool="sentry_get_latest_event")
    issue_id = quote(str(args["issue_id"]), safe="")
    resp = await _get(
        f"/organizations/{quote(org, safe='')}/issues/{issue_id}/events/latest/",
        tool="sentry_get_latest_event",
    )
    tags = [
        {"key": t.get("key"), "value": _truncate(str(t.get("value") or ""), 300)}
        for t in (resp.get("tags") or [])[:30]
        if isinstance(t, dict)
    ]
    return {
        "event_id": resp.get("eventID") or resp.get("id"),
        "issue_id": resp.get("groupID") or args.get("issue_id"),
        "title": _truncate(resp.get("title") or resp.get("message") or "", 500),
        "culprit": _truncate(resp.get("culprit") or "", 500),
        "level": (resp.get("tags") and next((t.get("value") for t in resp["tags"] if isinstance(t, dict) and t.get("key") == "level"), None)) or resp.get("level"),
        "platform": resp.get("platform"),
        "date_created": resp.get("dateCreated"),
        "tags": tags,
        "stacktrace": _extract_stacktrace(resp),
    }


async def _h_search_events(_unused, args: dict) -> Any:
    """`GET /organizations/{org}/events/` -- discover-style event search
    over selected `field`s with Sentry query syntax."""
    org = _resolve_org(args, tool="sentry_search_events")
    fields = args.get("field")
    if isinstance(fields, str):
        fields = [fields]
    if not fields:
        fields = ["title", "project", "timestamp", "level", "id"]
    params: dict[str, Any] = {
        "query": args.get("query") or "",
        "statsPeriod": args.get("statsPeriod") or "24h",
        "project": args.get("project"),
        "environment": args.get("environment"),
        "sort": args.get("sort"),
        "per_page": _clamp(args.get("limit")),
        "field": fields,  # multi-value query param
    }
    resp = await _get(
        f"/organizations/{quote(org, safe='')}/events/",
        params=params, tool="sentry_search_events",
    )
    items = resp.get("data") if isinstance(resp, dict) else (resp if isinstance(resp, list) else [])
    items = items or []
    out = []
    for row in items[: _clamp(args.get("limit"))]:
        if not isinstance(row, dict):
            continue
        trimmed = {}
        for k, v in row.items():
            trimmed[k] = _truncate(v, 500) if isinstance(v, str) else v
        out.append(trimmed)
    return {
        "org": org,
        "query": params["query"],
        "statsPeriod": params["statsPeriod"],
        "fields": fields,
        "count": len(out),
        "events": out,
    }


async def _h_get_event(_unused, args: dict) -> Any:
    """`GET /projects/{org}/{project}/events/{event_id}/` -- a single event
    by ID within a project, with stacktrace / tags."""
    org = _resolve_org(args, tool="sentry_get_event")
    project = quote(str(args["project"]), safe="")
    event_id = quote(str(args["event_id"]), safe="")
    resp = await _get(
        f"/projects/{quote(org, safe='')}/{project}/events/{event_id}/",
        tool="sentry_get_event",
    )
    tags = [
        {"key": t.get("key"), "value": _truncate(str(t.get("value") or ""), 300)}
        for t in (resp.get("tags") or [])[:30]
        if isinstance(t, dict)
    ]
    return {
        "event_id": resp.get("eventID") or resp.get("id"),
        "issue_id": resp.get("groupID"),
        "project": args.get("project"),
        "title": _truncate(resp.get("title") or resp.get("message") or "", 500),
        "culprit": _truncate(resp.get("culprit") or "", 500),
        "platform": resp.get("platform"),
        "date_created": resp.get("dateCreated"),
        "tags": tags,
        "stacktrace": _extract_stacktrace(resp),
    }


async def _h_get_trace(_unused, args: dict) -> Any:
    """`GET /organizations/{org}/events-trace/{trace_id}/` -- the span tree
    for a distributed trace by `trace_id`."""
    org = _resolve_org(args, tool="sentry_get_trace")
    trace_id = quote(str(args["trace_id"]), safe="")
    params = {
        "statsPeriod": args.get("statsPeriod") or "24h",
        "project": args.get("project"),
        "limit": _clamp(args.get("limit"), default=100),
    }
    resp = await _get(
        f"/organizations/{quote(org, safe='')}/events-trace/{trace_id}/",
        params=params, tool="sentry_get_trace",
    )
    items = resp if isinstance(resp, list) else (resp.get("data") or resp.get("transactions") or [])
    spans = []
    services_seen: set[str] = set()
    errors_seen = 0
    for x in items[: _clamp(args.get("limit"), default=100)]:
        if not isinstance(x, dict):
            continue
        svc = x.get("project_slug") or x.get("project")
        if svc:
            services_seen.add(svc)
        errs = x.get("errors")
        if isinstance(errs, list):
            errors_seen += len(errs)
        elif isinstance(errs, int):
            errors_seen += errs
        spans.append({
            "event_id": x.get("event_id"),
            "span_id": x.get("span_id"),
            "parent_span_id": x.get("parent_span_id"),
            "transaction": _truncate(x.get("transaction") or "", 300),
            "op": x.get("transaction.op") or x.get("op"),
            "project": svc,
            "duration_ms": x.get("transaction.duration") or x.get("duration"),
            "start_timestamp": x.get("start_timestamp"),
            "timestamp": x.get("timestamp"),
        })
    return {
        "org": org,
        "trace_id": args.get("trace_id"),
        "span_count": len(spans),
        "services_seen": sorted(services_seen),
        "errors": errors_seen,
        "spans": spans,
    }


async def _h_list_releases(_unused, args: dict) -> Any:
    """`GET /organizations/{org}/releases/` -- list releases for the org."""
    org = _resolve_org(args, tool="sentry_list_releases")
    params = {
        "query": args.get("query"),
        "project": args.get("project"),
        "per_page": _clamp(args.get("limit")),
    }
    resp = await _get(
        f"/organizations/{quote(org, safe='')}/releases/",
        params=params, tool="sentry_list_releases",
    )
    items = resp if isinstance(resp, list) else (resp.get("data") or [])
    out = []
    for r in items[: _clamp(args.get("limit"))]:
        projects = r.get("projects") or []
        out.append({
            "version": r.get("version"),
            "short_version": r.get("shortVersion"),
            "ref": r.get("ref"),
            "date_created": r.get("dateCreated"),
            "date_released": r.get("dateReleased"),
            "new_groups": r.get("newGroups"),
            "commit_count": r.get("commitCount"),
            "projects": [p.get("slug") for p in projects if isinstance(p, dict)][:20],
            "last_event": r.get("lastEvent"),
        })
    return {"org": org, "count": len(out), "releases": out}


# --- tool registry --------------------------------------------------


SENTRY_TOOLS: list[MCPTool] = [
    MCPTool(
        name="sentry_list_projects",
        description=(
            "List Sentry projects in an organization (id, slug, name, "
            "platform, team, status). Use this FIRST to find the project "
            "slug needed by `sentry_get_event` and to scope issue/event "
            "searches with the `project` filter."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "org": {"type": "string", "description": "Org slug. Defaults to SENTRY_ORG."},
                "limit": {"type": "number", "description": f"Max {_MAX_LIMIT}, default {_DEFAULT_LIMIT}"},
            },
        },
        handler=_h_list_projects,
    ),
    MCPTool(
        name="sentry_search_issues",
        description=(
            "Search Sentry issues (grouped errors) with Sentry search "
            "syntax. Default query `is:unresolved`. Returns trimmed "
            "high-signal fields: title, culprit, level, count, "
            "firstSeen/lastSeen. Common queries:\n"
            "- Unresolved errors: `query='is:unresolved level:error'`\n"
            "- One service: scope with `project='<slug>'`\n"
            "- Recent regressions: `query='is:regressed'`\n"
            "Each issue has an `id`; chain with `sentry_get_issue` or "
            "`sentry_get_latest_event` to drill into the stacktrace."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "org": {"type": "string", "description": "Org slug. Defaults to SENTRY_ORG."},
                "query": {"type": "string", "description": "Sentry search query. Default 'is:unresolved'."},
                "statsPeriod": {"type": "string", "description": "e.g. '24h', '14d'. Default '24h'."},
                "project": {"type": "string", "description": "Project ID or slug to scope to."},
                "environment": {"type": "string"},
                "sort": {"type": "string", "enum": ["date", "new", "freq", "user", "priority"]},
                "limit": {"type": "number"},
            },
        },
        handler=_h_search_issues,
    ),
    MCPTool(
        name="sentry_get_issue",
        description=(
            "Get one Sentry issue's detail by `issue_id` -- title, culprit, "
            "level, status, event count, user count, first/last seen. For "
            "the actual stacktrace, call `sentry_get_latest_event`."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "org": {"type": "string"},
                "issue_id": {"type": "string", "description": "Numeric issue ID or short ID."},
            },
            "required": ["issue_id"],
        },
        handler=_h_get_issue,
    ),
    MCPTool(
        name="sentry_get_latest_event",
        description=(
            "Get the most recent event for an issue, including the trimmed "
            "stacktrace frames (function/filename/lineno/in_app), culprit, "
            "platform, and tags. This is the primary 'show me the error' "
            "tool after `sentry_search_issues`."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "org": {"type": "string"},
                "issue_id": {"type": "string"},
            },
            "required": ["issue_id"],
        },
        handler=_h_get_latest_event,
    ),
    MCPTool(
        name="sentry_search_events",
        description=(
            "Discover-style event search across the org. Returns one row "
            "per selected `field` (e.g. title, project, timestamp, level, "
            "id, trace). Use Sentry query syntax in `query`. Provide "
            "`field` as a string or list of column names."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "org": {"type": "string"},
                "query": {"type": "string"},
                "statsPeriod": {"type": "string", "description": "e.g. '24h'. Default '24h'."},
                "project": {"type": "string"},
                "environment": {"type": "string"},
                "field": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "Column(s) to return. Default title/project/timestamp/level/id.",
                },
                "sort": {"type": "string"},
                "limit": {"type": "number"},
            },
        },
        handler=_h_search_events,
    ),
    MCPTool(
        name="sentry_get_event",
        description=(
            "Get a single event by `event_id` within a `project`, with "
            "trimmed stacktrace and tags. Use when you have a specific "
            "event ID (e.g. from `sentry_search_events`) rather than an "
            "issue. Requires the project slug -- get it from "
            "`sentry_list_projects`."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "org": {"type": "string"},
                "project": {"type": "string", "description": "Project slug."},
                "event_id": {"type": "string", "description": "32-char event ID."},
            },
            "required": ["project", "event_id"],
        },
        handler=_h_get_event,
    ),
    MCPTool(
        name="sentry_get_trace",
        description=(
            "Pull the span/transaction tree for a distributed trace by "
            "`trace_id`. Returns per-span transaction name, op, project, "
            "duration, and parent linkage, plus the set of services seen "
            "and an error count. Use to follow a request across services "
            "when an issue or event carries a `trace` id."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "org": {"type": "string"},
                "trace_id": {"type": "string"},
                "statsPeriod": {"type": "string", "description": "e.g. '24h'. Default '24h'."},
                "project": {"type": "string"},
                "limit": {"type": "number", "description": "Max spans (default 100)."},
            },
            "required": ["trace_id"],
        },
        handler=_h_get_trace,
    ),
    MCPTool(
        name="sentry_list_releases",
        description=(
            "List releases for the org -- version, ref, created/released "
            "dates, new-issue count, commit count, associated projects. "
            "Useful for 'what shipped right before this error spiked' and "
            "correlating an issue's firstSeen with a deploy."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "org": {"type": "string"},
                "query": {"type": "string", "description": "Filter on version substring."},
                "project": {"type": "string"},
                "limit": {"type": "number"},
            },
        },
        handler=_h_list_releases,
    ),
]


def get_tool(name: str) -> MCPTool:
    for t in SENTRY_TOOLS:
        if t.name == name:
            return t
    raise KeyError(f"unknown sentry tool: {name}")


# --- fake backend (FR-012; integration tests) ----------------------
#
# Sentry handlers ignore their first (`_unused`) arg and reach the
# module-level `_get`, which builds an httpx client from env via
# `_config()`. The offline fake replaces `_get` with a canned,
# shape-faithful responder -- no network, no SENTRY_TOKEN. build_fake()
# returns client=None (handlers discard it) plus a teardown that
# restores the real `_get`.


async def _fake_get(path: str, params: dict | None = None, *, tool: str = "sentry") -> Any:
    """Canned stand-in for the module-level GET. Routes by path to a
    response shaped like the real Sentry `/api/0` endpoint the handler
    parses."""
    # /organizations/{org}/projects/
    if path.endswith("/projects/") and path.startswith("/organizations/"):
        return [
            {
                "id": "1",
                "slug": "acme-notes-be",
                "name": "acme-notes-be",
                "platform": "python",
                "team": {"slug": "platform"},
                "status": "active",
                "dateCreated": "2026-01-01T00:00:00Z",
            }
        ]
    # /organizations/{org}/issues/{id}/events/latest/
    if path.endswith("/events/latest/"):
        return {
            "eventID": "ev-deadbeef",
            "id": "ev-deadbeef",
            "groupID": "1001",
            "title": "RuntimeError: boom",
            "message": "RuntimeError: boom",
            "culprit": "app.views in get_notes",
            "platform": "python",
            "dateCreated": "2026-05-20T00:00:00Z",
            "tags": [
                {"key": "level", "value": "error"},
                {"key": "environment", "value": "prod"},
            ],
            "entries": [
                {
                    "type": "exception",
                    "data": {
                        "values": [
                            {
                                "type": "RuntimeError",
                                "value": "boom",
                                "stacktrace": {
                                    "frames": [
                                        {
                                            "function": "get_notes",
                                            "filename": "app/views.py",
                                            "lineNo": 42,
                                            "module": "app.views",
                                            "inApp": True,
                                        }
                                    ]
                                },
                            }
                        ]
                    },
                }
            ],
        }
    # /organizations/{org}/issues/  (search) -- check BEFORE the detail
    # branch so the bare collection path doesn't fall into issue-by-id.
    if path.endswith("/issues/") and path.startswith("/organizations/"):
        return [
            {
                "id": "1001",
                "shortId": "ACME-1",
                "title": "RuntimeError: boom",
                "culprit": "app.views in get_notes",
                "level": "error",
                "status": "unresolved",
                "count": "57",
                "userCount": 12,
                "firstSeen": "2026-05-19T00:00:00Z",
                "lastSeen": "2026-05-20T00:00:00Z",
                "metadata": {"type": "RuntimeError"},
                "project": {"slug": "acme-notes-be"},
                "permalink": "https://sentry.io/organizations/acme/issues/1001/",
            }
        ]
    # /organizations/{org}/issues/{id}/
    if "/issues/" in path and path.startswith("/organizations/") and path.endswith("/"):
        return {
            "id": "1001",
            "shortId": "ACME-1",
            "title": "RuntimeError: boom",
            "culprit": "app.views in get_notes",
            "level": "error",
            "status": "unresolved",
            "count": "57",
            "userCount": 12,
            "firstSeen": "2026-05-19T00:00:00Z",
            "lastSeen": "2026-05-20T00:00:00Z",
            "metadata": {"type": "RuntimeError", "value": "boom"},
            "project": {"slug": "acme-notes-be"},
            "permalink": "https://sentry.io/organizations/acme/issues/1001/",
        }
    # /organizations/{org}/events/  (discover search)
    if path.endswith("/events/") and path.startswith("/organizations/"):
        return {
            "data": [
                {
                    "id": "ev-deadbeef",
                    "title": "RuntimeError: boom",
                    "project": "acme-notes-be",
                    "timestamp": "2026-05-20T00:00:00Z",
                    "level": "error",
                }
            ],
            "meta": {},
        }
    # /projects/{org}/{project}/events/{event_id}/
    if path.startswith("/projects/") and "/events/" in path and path.endswith("/"):
        return {
            "eventID": "ev-deadbeef",
            "id": "ev-deadbeef",
            "groupID": "1001",
            "title": "RuntimeError: boom",
            "message": "RuntimeError: boom",
            "culprit": "app.views in get_notes",
            "platform": "python",
            "dateCreated": "2026-05-20T00:00:00Z",
            "tags": [{"key": "level", "value": "error"}],
            "entries": [
                {
                    "type": "exception",
                    "data": {
                        "values": [
                            {
                                "type": "RuntimeError",
                                "value": "boom",
                                "stacktrace": {
                                    "frames": [
                                        {
                                            "function": "get_notes",
                                            "filename": "app/views.py",
                                            "lineNo": 42,
                                            "module": "app.views",
                                            "inApp": True,
                                        }
                                    ]
                                },
                            }
                        ]
                    },
                }
            ],
        }
    # /organizations/{org}/events-trace/{trace_id}/
    if "/events-trace/" in path:
        return [
            {
                "event_id": "ev-deadbeef",
                "span_id": "span-1",
                "parent_span_id": None,
                "transaction": "GET /notes",
                "transaction.op": "http.server",
                "project_slug": "acme-notes-be",
                "transaction.duration": 123.4,
                "start_timestamp": "2026-05-20T00:00:00Z",
                "timestamp": "2026-05-20T00:00:01Z",
                "errors": [{"event_id": "err-1"}],
            }
        ]
    # /organizations/{org}/releases/
    if path.endswith("/releases/") and path.startswith("/organizations/"):
        return [
            {
                "version": "acme-notes-be@1.2.3",
                "shortVersion": "1.2.3",
                "ref": "deadbeef",
                "dateCreated": "2026-05-19T00:00:00Z",
                "dateReleased": "2026-05-19T01:00:00Z",
                "newGroups": 3,
                "commitCount": 12,
                "projects": [{"slug": "acme-notes-be"}],
                "lastEvent": "2026-05-20T00:00:00Z",
            }
        ]
    return {}


def build_fake():
    """Return a FakeMCP exposing the Sentry tools wired to an offline
    backend. Needs NO SENTRY_TOKEN / network: the module-level `_get` is
    swapped for a canned responder and restored by `teardown`."""
    import opsrag.mcp.sentry as _mod
    from opsrag.mcp._fake import FakeMCP

    _orig_get = _mod._get
    _orig_config = _mod._config
    _mod._get = _fake_get

    # _resolve_org() calls _config() for the default org; stub it so the
    # fake needs no SENTRY_TOKEN / SENTRY_ORG in the environment.
    def _fake_config() -> _Config:
        return _Config(token="fake", api_url="https://sentry.io/api/0", org="acme")

    _mod._config = _fake_config

    def _restore() -> None:
        _mod._get = _orig_get
        _mod._config = _orig_config

    return FakeMCP(tools=list(SENTRY_TOOLS), client=None, teardown=_restore)
