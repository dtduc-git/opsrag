"""Rootly MCP-style tools for OpsRAG (Sub-sprint 4).

Read-only async tools over the Rootly REST API
(`https://api.rootly.com/v1`). Reuses `ROOTLY_API_TOKEN` from `.env`,
the same service-account key the daily Rootly indexing source uses.

## Read-only enforcement

Every tool issues `httpx.AsyncClient.get`. No POST/PUT/PATCH/DELETE
anywhere. The token has read scope only per the deploying organization's
Rootly config.

## Tool list (7 read-only)

| Tool                              | Endpoint(s)                             |
|-----------------------------------|------------------------------------------|
| `rootly_list_incidents`           | GET `/v1/incidents`                      |
| `rootly_get_incident`             | GET `/v1/incidents/<id>` or filter[sequential_id] |
| `rootly_get_incident_timeline`    | GET `/v1/incidents/<id>/incident_events` |
| `rootly_list_post_mortems`        | GET `/v1/post_mortems`                   |
| `rootly_get_post_mortem`          | GET `/v1/post_mortems?filter[incident_id]=<id>` |
| `rootly_search`                   | parallel: incidents + post_mortems       |
| `rootly_list_services`            | GET `/v1/services`                       |

## URL convention for citations

Incidents:    `https://rootly.com/account/incidents/<sequential_id>-<slug>`
Post-mortems: `https://rootly.com/account/incidents/<sequential_id>-<slug>/post_mortem`
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from opsrag.mcp.gitlab import MCPTool

_log = logging.getLogger("opsrag.mcp.rootly")

DEFAULT_API_URL = "https://api.rootly.com/v1"
ROOTLY_WEB_BASE = "https://rootly.com/account"
_DEFAULT_TIMEOUT_S = 30.0
_DEFAULT_PER_PAGE = 20
_MAX_PER_PAGE = 100
_RESULT_TRUNCATE_CHARS = 32000
_DEFAULT_SINCE_DAYS = 90  # default lookback when caller doesn't specify

_TOKEN_ENV_KEYS = ("OPSRAG_ROOTLY_TOKEN", "ROOTLY_API_TOKEN")


class RootlyMCPError(Exception):
    """Raised on Rootly API errors. Wraps the upstream status + body."""

    def __init__(self, status: int, body: str, *, tool: str | None = None):
        self.status = status
        self.body = body
        self.tool = tool
        super().__init__(f"[{tool or 'rootly'}] {status}: {body[:300]}")


def _resolve_token() -> str | None:
    for key in _TOKEN_ENV_KEYS:
        v = os.environ.get(key)
        if v and v.strip():
            return v.strip().strip('"').strip("'")
    return None


@dataclass
class _ClientConfig:
    token: str
    api_url: str


def _config() -> _ClientConfig:
    token = _resolve_token()
    if not token:
        raise RuntimeError(
            "Rootly token not set. Set one of: " + ", ".join(_TOKEN_ENV_KEYS)
        )
    api_url = (
        os.environ.get("OPSRAG_ROOTLY_API_URL") or DEFAULT_API_URL
    ).rstrip("/")
    return _ClientConfig(token=token, api_url=api_url)


async def _get(path: str, params: dict | None = None, *, tool: str = "rootly") -> Any:
    cfg = _config()
    clean = {}
    for k, v in (params or {}).items():
        if v is None:
            continue
        if isinstance(v, list):
            clean[k] = ",".join(str(x) for x in v)
        else:
            clean[k] = v
    url = f"{cfg.api_url}{path}"
    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {cfg.token}"},
        timeout=_DEFAULT_TIMEOUT_S,
    ) as http:
        resp = await http.get(url, params=clean)
    if resp.status_code >= 400:
        raise RootlyMCPError(resp.status_code, resp.text, tool=tool)
    if not resp.text:
        return {}
    return resp.json()


def _clamp_per_page(value: int | None) -> int:
    if value is None:
        return _DEFAULT_PER_PAGE
    return max(1, min(int(value), _MAX_PER_PAGE))


def _default_since() -> str:
    return (datetime.now(UTC) - timedelta(days=_DEFAULT_SINCE_DAYS)).strftime("%Y-%m-%d")


def _incident_url(sequential_id: Any, slug: str | None) -> str:
    """Web URL for an incident in Rootly UI."""
    sid = str(sequential_id or "").strip("#")
    s = slug or ""
    return f"{ROOTLY_WEB_BASE}/incidents/{sid}-{s}" if sid and s else ""


def _post_mortem_url(sequential_id: Any, slug: str | None) -> str:
    sid = str(sequential_id or "").strip("#")
    s = slug or ""
    return f"{ROOTLY_WEB_BASE}/incidents/{sid}-{s}/post_mortem" if sid and s else ""


# Heuristic: treat all-digit ids as `sequential_id`; UUID strings as direct id.
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


async def _resolve_incident_uuid(incident_id: str) -> tuple[str, dict | None]:
    """Return (uuid, raw_attributes_or_None). Accepts either a Rootly
    UUID or a `sequential_id` like 292 / "#292"."""
    raw = str(incident_id).strip().lstrip("#")
    if _UUID_RE.match(raw):
        return raw, None
    if raw.isdigit():
        # Look up by sequential_id (filter[sequential_id]=N)
        out = await _get(
            "/incidents",
            params={"filter[sequential_id]": int(raw), "page[size]": 1},
            tool="rootly_get_incident",
        )
        items = out.get("data") or []
        if not items:
            raise RootlyMCPError(
                404, f"no incident found with sequential_id={raw}",
                tool="rootly_get_incident",
            )
        first = items[0]
        return first["id"], first.get("attributes") or {}
    raise RootlyMCPError(
        400,
        f"unrecognized incident_id {incident_id!r}; expected UUID or sequential_id",
        tool="rootly_get_incident",
    )


def _summarize_incident(raw: dict) -> dict:
    a = raw.get("attributes") or {}
    rels = raw.get("relationships") or {}

    def _names(rel_key: str) -> list[str]:
        rel = rels.get(rel_key) or {}
        data = rel.get("data") or []
        # Without `?include=`, relationships only have id+type; surface the ids.
        return [d.get("id") for d in data if d.get("id")]

    seq = a.get("sequential_id")
    slug = a.get("slug")
    return {
        "id": raw.get("id"),
        "sequential_id": seq,
        "title": a.get("title"),
        "summary": (a.get("summary") or "")[:1000],
        "status": a.get("status"),
        "severity": (a.get("severity") or {}).get("name") if isinstance(a.get("severity"), dict) else a.get("severity"),
        "kind": a.get("kind"),
        "started_at": a.get("started_at"),
        "detected_at": a.get("detected_at"),
        "mitigated_at": a.get("mitigated_at"),
        "resolved_at": a.get("resolved_at"),
        "duration_in_minutes": a.get("duration_in_minutes"),
        "url": a.get("url") or _incident_url(seq, slug),
        "labels": a.get("labels"),
        "service_ids": _names("services"),
        "team_ids": _names("teams"),
    }


def _summarize_post_mortem(raw: dict, incident_summary: dict | None = None) -> dict:
    a = raw.get("attributes") or {}
    seq = (incident_summary or {}).get("sequential_id")
    slug = (incident_summary or {}).get("slug") or a.get("slug")
    body = a.get("content_markdown") or a.get("content") or ""
    if len(body) > _RESULT_TRUNCATE_CHARS:
        body = body[:_RESULT_TRUNCATE_CHARS] + " ...[truncated]"
    return {
        "id": raw.get("id"),
        "incident_id": a.get("incident_id"),
        "title": a.get("title"),
        "status": a.get("status"),
        "url": a.get("url") or _post_mortem_url(seq, slug),
        "started_at": a.get("started_at"),
        "mitigated_at": a.get("mitigated_at"),
        "published_at": a.get("published_at"),
        "content": body,
        "content_chars": len(body),
    }


def _summarize_event(raw: dict) -> dict:
    a = raw.get("attributes") or {}
    return {
        "id": raw.get("id"),
        "kind": a.get("kind"),
        "event": a.get("event"),
        "occurred_at": a.get("occurred_at"),
        "source": a.get("source"),
        "user": (a.get("user") or {}).get("name") if isinstance(a.get("user"), dict) else a.get("user"),
        "summary": (a.get("event") or a.get("description") or "")[:300],
    }


# --- handlers -------------------------------------------------------


async def _h_list_incidents(_unused, args: dict) -> Any:
    params = {
        "filter[status]": args.get("status"),
        "filter[severity]": args.get("severity"),
        "filter[service]": args.get("service"),
        "filter[search]": args.get("search"),
        "filter[started_at][gte]": args.get("since") or _default_since(),
        "filter[started_at][lte]": args.get("until"),
        "page[size]": _clamp_per_page(args.get("limit")),
        "page[number]": int(args.get("page") or 1),
        "sort": args.get("sort") or "-started_at",
    }
    out = await _get("/incidents", params=params, tool="rootly_list_incidents")
    items = out.get("data") or []
    summarized = [_summarize_incident(x) for x in items]
    return {
        "count": len(summarized),
        "total": (out.get("meta") or {}).get("total_count"),
        "incidents": summarized,
    }


async def _h_get_incident(_unused, args: dict) -> Any:
    uuid, _ = await _resolve_incident_uuid(args["incident_id"])
    out = await _get(f"/incidents/{uuid}", tool="rootly_get_incident")
    raw = out.get("data") or {}
    return {"incident": _summarize_incident(raw)}


async def _h_get_incident_timeline(_unused, args: dict) -> Any:
    uuid, _ = await _resolve_incident_uuid(args["incident_id"])
    params = {
        "page[size]": _clamp_per_page(args.get("limit")),
        "sort": args.get("sort") or "occurred_at",
    }
    out = await _get(
        f"/incidents/{uuid}/incident_events",
        params=params, tool="rootly_get_incident_timeline",
    )
    items = out.get("data") or []
    return {
        "incident_id": uuid,
        "count": len(items),
        "events": [_summarize_event(e) for e in items],
    }


async def _h_list_post_mortems(_unused, args: dict) -> Any:
    params = {
        "filter[status]": args.get("status"),
        "filter[search]": args.get("search"),
        "filter[created_at][gte]": args.get("since") or _default_since(),
        "page[size]": _clamp_per_page(args.get("limit")),
        "page[number]": int(args.get("page") or 1),
        "sort": args.get("sort") or "-created_at",
    }
    out = await _get("/post_mortems", params=params, tool="rootly_list_post_mortems")
    items = out.get("data") or []
    summarized = []
    for x in items:
        a = x.get("attributes") or {}
        summarized.append({
            "id": x.get("id"),
            "incident_id": a.get("incident_id"),
            "title": a.get("title"),
            "status": a.get("status"),
            "url": a.get("url"),
            "created_at": a.get("created_at"),
            "published_at": a.get("published_at"),
        })
    return {
        "count": len(summarized),
        "total": (out.get("meta") or {}).get("total_count"),
        "post_mortems": summarized,
    }


async def _h_get_post_mortem(_unused, args: dict) -> Any:
    """Find a post-mortem for a specific incident.

    Caveat (2026-05-09): Rootly API silently ignores
    `filter[incident_id]` on `/v1/post_mortems` -- it returns the full
    collection regardless. Fixing server-side requires Rootly support;
    workaround is to scan pages client-side until we find a match.
    With a typical tenant (~100-200 post-mortems, sorted newest-first),
    this is 1-2 page-fetches in practice. Hard-cap at 10 pages x 100
    = 1000 to avoid runaway."""
    uuid, inc_attrs = await _resolve_incident_uuid(args["incident_id"])
    incident_summary = None
    if inc_attrs:
        incident_summary = {
            "sequential_id": inc_attrs.get("sequential_id"),
            "slug": inc_attrs.get("slug"),
        }
    pages_scanned = 0
    items_scanned = 0
    for page in range(1, 11):
        out = await _get(
            "/post_mortems",
            params={
                "page[size]": 100,
                "page[number]": page,
                "sort": "-created_at",
            },
            tool="rootly_get_post_mortem",
        )
        items = out.get("data") or []
        pages_scanned += 1
        items_scanned += len(items)
        for x in items:
            attrs = x.get("attributes") or {}
            if attrs.get("incident_id") == uuid:
                return {
                    "incident_id": uuid,
                    "post_mortem": _summarize_post_mortem(x, incident_summary),
                    "_pages_scanned": pages_scanned,
                }
        if len(items) < 100:
            break
    return {
        "incident_id": uuid,
        "post_mortem": None,
        "error": (
            f"no post-mortem found for incident_id={uuid} "
            f"(scanned {items_scanned} post-mortems across {pages_scanned} pages)"
        ),
    }


_ROOTLY_ALERT_URL_RE = re.compile(
    r"https?://(?:[a-z0-9-]+\.)?rootly\.com/(?:account/)?alerts/([A-Za-z0-9_-]{4,32})\b"
)


async def _h_list_alerts(_unused, args: dict) -> Any:
    """List Rootly alerts (paginated). Confirmed working shape
    (curl-verified 2026-05-15): GET /v1/alerts with optional
    `filter[status]` and `page[size]` / `page[number]`.

    Use this for 'last 10 alerts', 'unresolved alerts since yesterday',
    'alerts from #sysalerts source'. Returns a flattened list of
    {short_id, status, source, summary, urls, started_at, resolved_at,
    services} -- enough to triage without an extra get_alert per row.
    """
    params: dict[str, Any] = {
        "page[size]": int(args.get("per_page") or 20),
        "page[number]": int(args.get("page") or 1),
    }
    if status := args.get("status"):
        params["filter[status]"] = status
    if source := args.get("source"):
        params["filter[source]"] = source
    if created_after := args.get("created_after"):
        params["filter[created_at][gte]"] = created_after
    if created_before := args.get("created_before"):
        params["filter[created_at][lte]"] = created_before
    try:
        resp = await _get("/alerts", params=params, tool="rootly_list_alerts")
    except Exception as exc:  # noqa: BLE001
        return {"error": f"rootly fetch failed: {exc}"}
    out: list[dict] = []
    for row in (resp.get("data") or []):
        attrs = row.get("attributes") or {}
        out.append({
            "short_id": attrs.get("short_id"),
            "status": attrs.get("status"),
            "source": attrs.get("source"),
            "summary": attrs.get("summary"),
            "url": attrs.get("url"),
            "external_url": attrs.get("external_url"),
            "started_at": attrs.get("started_at"),
            "resolved_at": attrs.get("resolved_at"),
            "service_names": [
                s.get("name") for s in (attrs.get("services") or []) if s.get("name")
            ],
        })
    return {
        "count": len(out),
        "filter": {k: v for k, v in {
            "status": args.get("status"),
            "source": args.get("source"),
            "created_after": args.get("created_after"),
            "created_before": args.get("created_before"),
        }.items() if v},
        "alerts": out,
    }


async def _h_get_alert(_unused, args: dict) -> Any:
    """Fetch a single Rootly ALERT (not incident) by short_id or URL.

    Slack pages may link alerts as `https://rootly.com/account/alerts/<short_id>`
    (e.g. `xHRRRD`). This tool resolves the URL or raw short_id to the
    full alert payload -- title, severity, status, labels, the underlying
    AlertManager `expr`, the linked runbook URL, the firing value.

    Use this BEFORE `rootly_list_incidents` when the user pastes a
    Rootly alert URL -- an alert may not have an associated incident,
    so listing incidents returns empty and misleads.
    """
    raw = (args.get("short_id") or args.get("url") or "").strip()
    if not raw:
        return {"error": "short_id or url is required"}
    m = _ROOTLY_ALERT_URL_RE.search(raw)
    short_id = m.group(1) if m else raw
    if not re.fullmatch(r"[A-Za-z0-9_-]{4,32}", short_id):
        return {"error": f"could not parse short_id from {raw!r}"}
    try:
        resp = await _get(f"/alerts/{short_id}", tool="rootly_get_alert")
    except Exception as exc:  # noqa: BLE001
        return {"error": f"rootly fetch failed: {exc}", "short_id": short_id}
    data = (resp.get("data") or {})
    attrs = (data.get("attributes") or {})
    nested = (attrs.get("data") or {}).get("alerts") or []
    labels: dict = (nested[0].get("labels") if nested else {}) or {}
    annotations: dict = (nested[0].get("annotations") if nested else {}) or {}
    return {
        "short_id": attrs.get("short_id") or short_id,
        "summary": attrs.get("summary") or "",
        "status": attrs.get("status"),
        "source": attrs.get("source"),
        "url": attrs.get("url"),
        "external_url": attrs.get("external_url"),
        "service_names": [s.get("name") for s in (attrs.get("services") or []) if s.get("name")],
        "labels": labels,
        "annotations": annotations,
        "runbook_url": annotations.get("runbook_url") or labels.get("alertrule_runbook_url"),
        "promql_expression": annotations.get("expr"),
        "firing_value": annotations.get("value"),
        "severity": labels.get("severity") or annotations.get("severity"),
    }


async def _h_search(_unused, args: dict) -> Any:
    """Parallel cross-search: incidents and post-mortems both filtered
    by the query string. Useful for 'have we seen this before?' work."""
    q = args["q"]
    limit = _clamp_per_page(args.get("limit"))

    async def _incidents() -> list[dict]:
        try:
            out = await _get(
                "/incidents",
                params={
                    "filter[search]": q,
                    "page[size]": limit,
                    "filter[started_at][gte]": args.get("since") or _default_since(),
                    "sort": "-started_at",
                },
                tool="rootly_search",
            )
            return [_summarize_incident(x) for x in (out.get("data") or [])]
        except RootlyMCPError as exc:
            return [{"_error": str(exc)}]

    async def _post_mortems() -> list[dict]:
        try:
            out = await _get(
                "/post_mortems",
                params={
                    "filter[search]": q,
                    "page[size]": limit,
                    "sort": "-created_at",
                },
                tool="rootly_search",
            )
            results = []
            for x in (out.get("data") or []):
                a = x.get("attributes") or {}
                results.append({
                    "id": x.get("id"),
                    "incident_id": a.get("incident_id"),
                    "title": a.get("title"),
                    "status": a.get("status"),
                    "url": a.get("url"),
                    "created_at": a.get("created_at"),
                })
            return results
        except RootlyMCPError as exc:
            return [{"_error": str(exc)}]

    incidents, post_mortems = await asyncio.gather(_incidents(), _post_mortems())
    return {
        "query": q,
        "incidents": incidents,
        "post_mortems": post_mortems,
    }


async def _h_list_services(_unused, args: dict) -> Any:
    params = {
        "filter[search]": args.get("search"),
        "page[size]": _clamp_per_page(args.get("limit")),
        "page[number]": int(args.get("page") or 1),
        "sort": args.get("sort") or "name",
    }
    out = await _get("/services", params=params, tool="rootly_list_services")
    items = out.get("data") or []
    summarized = [
        {
            "id": x.get("id"),
            "name": (x.get("attributes") or {}).get("name"),
            "description": ((x.get("attributes") or {}).get("description") or "")[:200],
            "status": (x.get("attributes") or {}).get("status"),
            "service_tier": (x.get("attributes") or {}).get("service_tier"),
        }
        for x in items
    ]
    return {
        "count": len(summarized),
        "total": (out.get("meta") or {}).get("total_count"),
        "services": summarized,
    }


# --- tool registry --------------------------------------------------


_INCIDENT_ID_PROP = {
    "type": "string",
    "description": (
        "Incident UUID OR sequential id (e.g. '292' or '#292'). "
        "Sequential ids are looked up via `filter[sequential_id]`."
    ),
}


ROOTLY_TOOLS: list[MCPTool] = [
    MCPTool(
        name="rootly_list_incidents",
        description=(
            "List incidents. Default lookback 90 days unless `since` is "
            "specified. Use `search` for keyword filter, `severity` to "
            "scope (e.g. 'sev1'), `service` for one-service drilldown."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "started, mitigated, resolved, etc."},
                "severity": {"type": "string", "description": "Severity name or id"},
                "service": {"type": "string", "description": "Service id or name"},
                "search": {"type": "string", "description": "Free-text search across title + summary"},
                "since": {"type": "string", "description": "ISO date or datetime; default 90 days ago"},
                "until": {"type": "string", "description": "ISO date or datetime; default now"},
                "limit": {"type": "number", "description": f"Max {_MAX_PER_PAGE}, default {_DEFAULT_PER_PAGE}"},
                "page": {"type": "number"},
                "sort": {"type": "string", "description": "Default '-started_at' (newest first)"},
            },
        },
        handler=_h_list_incidents,
    ),
    MCPTool(
        name="rootly_get_incident",
        description="Get one incident's details (status, severity, summary, time markers, services).",
        input_schema={
            "type": "object",
            "properties": {"incident_id": _INCIDENT_ID_PROP},
            "required": ["incident_id"],
        },
        handler=_h_get_incident,
    ),
    MCPTool(
        name="rootly_get_incident_timeline",
        description=(
            "Chronological timeline of events for one incident -- status "
            "changes, action items, slack messages, runbook actions. "
            "Use to reconstruct what happened during the incident."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "incident_id": _INCIDENT_ID_PROP,
                "limit": {"type": "number"},
                "sort": {"type": "string", "description": "Default 'occurred_at' (chronological)"},
            },
            "required": ["incident_id"],
        },
        handler=_h_get_incident_timeline,
    ),
    MCPTool(
        name="rootly_list_post_mortems",
        description=(
            "List post-mortems / retrospectives. Default lookback 90 "
            "days. Use `search` for topic-keyword filter."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "draft, published, etc."},
                "search": {"type": "string"},
                "since": {"type": "string"},
                "limit": {"type": "number"},
                "page": {"type": "number"},
                "sort": {"type": "string", "description": "Default '-created_at'"},
            },
        },
        handler=_h_list_post_mortems,
    ),
    MCPTool(
        name="rootly_get_post_mortem",
        description=(
            "Get the post-mortem for a specific incident -- full markdown body, "
            "truncated at 32KB. The killer tool for 'have we seen this before' "
            "drilldowns: list+search incidents -> grab the post-mortem of the matching one."
        ),
        input_schema={
            "type": "object",
            "properties": {"incident_id": _INCIDENT_ID_PROP},
            "required": ["incident_id"],
        },
        handler=_h_get_post_mortem,
    ),
    MCPTool(
        name="rootly_list_alerts",
        description=(
            "List Rootly alerts (paginated). Use this for 'last N alerts', "
            "'unresolved alerts in the last day', 'alerts from source X'. "
            "Optional `filter[status]`: open | acknowledged | resolved | closed. "
            "Use `rootly_get_alert` to drill into one row's full detail. "
            "When the user pastes a Rootly alert URL, prefer `rootly_get_alert` "
            "instead -- listing returns recent alerts, not a specific URL."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["open", "acknowledged", "resolved", "closed"],
                },
                "source": {
                    "type": "string",
                    "description": "Filter by alert source, e.g. `alertmanager`, `datadog`.",
                },
                "created_after": {
                    "type": "string",
                    "description": "ISO-8601 timestamp lower bound, e.g. `2026-05-14T00:00:00Z`.",
                },
                "created_before": {
                    "type": "string",
                    "description": "ISO-8601 timestamp upper bound.",
                },
                "per_page": {"type": "number"},
                "page": {"type": "number"},
            },
        },
        handler=_h_list_alerts,
    ),
    MCPTool(
        name="rootly_get_alert",
        description=(
            "Fetch a Rootly ALERT (not incident) by short_id or URL. "
            "Slack pages link alerts as `https://rootly.com/account/alerts/<short_id>` "
            "(e.g. `xHRRRD`). Returns title, severity, status, services, "
            "labels (env, namespace, alertrule_appname, datname, instance), "
            "annotations (PromQL `expr`, `value`, `runbook_url`), and the live "
            "Prometheus generator URL. PREFER this over `rootly_list_incidents` "
            "when the user pasted a Rootly URL -- many alerts have no associated "
            "incident, so list_incidents would return empty and mislead you."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "short_id": {"type": "string", "description": "Rootly alert short_id (e.g. 'xHRRRD' or 'lxFr7T'). Either this OR `url`."},
                "url": {"type": "string", "description": "Full Rootly alert URL like https://rootly.com/account/alerts/<short_id>. Parsed to extract short_id."},
            },
        },
        handler=_h_get_alert,
    ),
    MCPTool(
        name="rootly_search",
        description=(
            "Cross-search incidents AND post-mortems in parallel for the "
            "same query -- best when user asks 'have we seen this pattern "
            "before?'. Returns up to N incidents + N post-mortems."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "q": {"type": "string", "description": "Free-text query"},
                "since": {"type": "string", "description": "Default 90 days ago"},
                "limit": {"type": "number", "description": f"Per-stream, max {_MAX_PER_PAGE}"},
            },
            "required": ["q"],
        },
        handler=_h_search,
    ),
    MCPTool(
        name="rootly_list_services",
        description="Service catalog -- useful when the agent needs to validate a service name before filtering.",
        input_schema={
            "type": "object",
            "properties": {
                "search": {"type": "string"},
                "limit": {"type": "number"},
                "page": {"type": "number"},
                "sort": {"type": "string", "description": "Default 'name'"},
            },
        },
        handler=_h_list_services,
    ),
]


def get_tool(name: str) -> MCPTool:
    for t in ROOTLY_TOOLS:
        if t.name == name:
            return t
    raise KeyError(f"unknown rootly tool: {name}")


# --- fake backend (FR-012; integration tests) ----------------------
#
# Data path: every handler discards its first arg (`_unused`) and reaches
# the module-level `_get`, which builds an httpx client from `_config()`
# (token + base URL). `build_fake()` swaps `_get` for a canned responder
# routed by path, returning Rootly JSON:API-style payloads with NO network
# and NO token, then restores the real `_get` via `teardown`. Mirrors the
# Datadog fake pattern; the GitLab fake instead swaps the client object
# because its handlers receive a live client.

_FAKE_INCIDENT_UUID = "11111111-1111-4111-8111-111111111111"


def _fake_incident_record() -> dict:
    """A Rootly JSON:API incident resource object."""
    return {
        "id": _FAKE_INCIDENT_UUID,
        "type": "incidents",
        "attributes": {
            "sequential_id": 292,
            "slug": "database-connection-pool-exhausted",
            "title": "Database connection pool exhausted",
            "summary": "Primary DB hit max connections; API 500s for 12 min.",
            "status": "resolved",
            "severity": {"name": "sev1"},
            "kind": "normal",
            "started_at": "2026-05-20T10:00:00Z",
            "detected_at": "2026-05-20T10:02:00Z",
            "mitigated_at": "2026-05-20T10:10:00Z",
            "resolved_at": "2026-05-20T10:12:00Z",
            "duration_in_minutes": 12,
            "url": "https://rootly.com/account/incidents/292-database-connection-pool-exhausted",
            "labels": ["database", "capacity"],
        },
        "relationships": {
            "services": {"data": [{"id": "svc-1", "type": "services"}]},
            "teams": {"data": [{"id": "team-1", "type": "teams"}]},
        },
    }


def _fake_post_mortem_record() -> dict:
    return {
        "id": "pm-1",
        "type": "post_mortems",
        "attributes": {
            "incident_id": _FAKE_INCIDENT_UUID,
            "title": "Post-mortem: Database connection pool exhausted",
            "status": "published",
            "url": "https://rootly.com/account/incidents/292-database-connection-pool-exhausted/post_mortem",
            "content_markdown": "## Summary\nPool exhausted under load.\n## Action items\n- Raise max_connections.",
            "created_at": "2026-05-21T00:00:00Z",
            "published_at": "2026-05-22T00:00:00Z",
            "started_at": "2026-05-20T10:00:00Z",
            "mitigated_at": "2026-05-20T10:10:00Z",
        },
    }


async def _fake_get(path: str, params: dict | None = None, *, tool: str = "rootly") -> Any:
    """Canned stand-in for the module-level GET. Routes by path to a
    Rootly JSON:API-shaped response the handler parses. No network."""
    params = params or {}
    # Single incident by UUID: GET /incidents/<uuid>
    if path.startswith("/incidents/") and path.endswith("/incident_events"):
        return {
            "data": [
                {
                    "id": "evt-1",
                    "type": "incident_events",
                    "attributes": {
                        "kind": "status_change",
                        "event": "Incident marked resolved",
                        "occurred_at": "2026-05-20T10:12:00Z",
                        "source": "web",
                        "user": {"name": "On-call SRE"},
                    },
                }
            ],
            "meta": {"total_count": 1},
        }
    if path.startswith("/incidents/"):
        return {"data": _fake_incident_record()}
    # Incident collection: list, search, or sequential_id resolution.
    if path == "/incidents":
        return {"data": [_fake_incident_record()], "meta": {"total_count": 1}}
    if path == "/post_mortems":
        return {"data": [_fake_post_mortem_record()], "meta": {"total_count": 1}}
    if path.startswith("/alerts/"):
        return {
            "data": {
                "id": "alert-1",
                "type": "alerts",
                "attributes": {
                    "short_id": "xHRRRD",
                    "summary": "HighErrorRate firing for acme-notes-be",
                    "status": "open",
                    "source": "alertmanager",
                    "url": "https://rootly.com/account/alerts/xHRRRD",
                    "external_url": "https://prometheus.example.com/graph",
                    "services": [{"name": "acme-notes-be"}],
                    "data": {
                        "alerts": [
                            {
                                "labels": {"severity": "critical", "namespace": "prod"},
                                "annotations": {
                                    "expr": "rate(http_errors[5m]) > 0.1",
                                    "value": "0.42",
                                    "runbook_url": "https://example.com/runbooks/errors",
                                },
                            }
                        ]
                    },
                },
            }
        }
    if path == "/alerts":
        return {
            "data": [
                {
                    "id": "alert-1",
                    "type": "alerts",
                    "attributes": {
                        "short_id": "xHRRRD",
                        "status": "open",
                        "source": "alertmanager",
                        "summary": "HighErrorRate firing for acme-notes-be",
                        "url": "https://rootly.com/account/alerts/xHRRRD",
                        "external_url": "https://prometheus.example.com/graph",
                        "started_at": "2026-05-20T10:00:00Z",
                        "resolved_at": None,
                        "services": [{"name": "acme-notes-be"}],
                    },
                }
            ],
            "meta": {"total_count": 1},
        }
    if path == "/services":
        return {
            "data": [
                {
                    "id": "svc-1",
                    "type": "services",
                    "attributes": {
                        "name": "acme-notes-be",
                        "description": "Notes backend service.",
                        "status": "operational",
                        "service_tier": "1",
                    },
                }
            ],
            "meta": {"total_count": 1},
        }
    return {}


def build_fake():
    """Return a FakeMCP exposing the Rootly tools wired to an offline
    backend. Needs NO Rootly token / network: the module-level `_get` is
    swapped for a canned responder and restored by `teardown`."""
    import opsrag.mcp.rootly as _mod
    from opsrag.mcp._fake import FakeMCP

    _orig_get = _mod._get
    _mod._get = _fake_get

    def _restore() -> None:
        _mod._get = _orig_get

    return FakeMCP(tools=list(ROOTLY_TOOLS), client=None, teardown=_restore)
