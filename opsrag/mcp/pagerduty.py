"""PagerDuty MCP-style tools for OpsRAG.

Read-only async tools over the PagerDuty REST API v2
(`https://api.pagerduty.com`). Auth is the classic API token header
(`Authorization: Token token=<token>`), resolved from one of
`OPSRAG_PAGERDUTY_TOKEN` / `PAGERDUTY_API_TOKEN`.

## Read-only enforcement

Every tool issues `httpx.AsyncClient.get`. No POST/PUT/PATCH/DELETE
anywhere -- no acknowledge, no resolve, no create/update/delete. The
token's role should be read-only per the deploying organization's
PagerDuty config.

## Tool list (5 read-only)

| Tool                                   | Endpoint(s)                            |
|----------------------------------------|----------------------------------------|
| `pagerduty_list_incidents`             | GET `/incidents`                       |
| `pagerduty_get_incident`               | GET `/incidents/<id>`                  |
| `pagerduty_list_services`              | GET `/services`                        |
| `pagerduty_list_oncalls`               | GET `/oncalls`                         |
| `pagerduty_get_incident_log_entries`   | GET `/incidents/<id>/log_entries`      |
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

from opsrag.mcp.gitlab import MCPTool

_log = logging.getLogger("opsrag.mcp.pagerduty")

DEFAULT_API_URL = "https://api.pagerduty.com"
_DEFAULT_TIMEOUT_S = 30.0
_DEFAULT_LIMIT = 20
_MAX_LIMIT = 100
_RESULT_TRUNCATE_CHARS = 32000

_TOKEN_ENV_KEYS = ("OPSRAG_PAGERDUTY_TOKEN", "PAGERDUTY_API_TOKEN")


class PagerDutyMCPError(Exception):
    """Raised on PagerDuty API errors. Wraps the upstream status + body."""

    def __init__(self, status: int, body: str, *, tool: str | None = None):
        self.status = status
        self.body = body
        self.tool = tool
        super().__init__(f"[{tool or 'pagerduty'}] {status}: {body[:300]}")


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
            "PagerDuty token not set. Set one of: " + ", ".join(_TOKEN_ENV_KEYS)
        )
    api_url = (
        os.environ.get("OPSRAG_PAGERDUTY_API_URL") or DEFAULT_API_URL
    ).rstrip("/")
    return _ClientConfig(token=token, api_url=api_url)


async def _get(path: str, params: dict | None = None, *, tool: str = "pagerduty") -> Any:
    cfg = _config()
    # PagerDuty repeats list params (e.g. statuses[]=triggered&statuses[]=acknowledged),
    # so list values stay as lists for httpx to expand; scalars pass through.
    clean: dict[str, Any] = {}
    for k, v in (params or {}).items():
        if v is None:
            continue
        clean[k] = v
    url = f"{cfg.api_url}{path}"
    async with httpx.AsyncClient(
        headers={
            "Authorization": f"Token token={cfg.token}",
            "Accept": "application/vnd.pagerduty+json;version=2",
        },
        timeout=_DEFAULT_TIMEOUT_S,
    ) as http:
        resp = await http.get(url, params=clean)
    if resp.status_code >= 400:
        raise PagerDutyMCPError(resp.status_code, resp.text, tool=tool)
    if not resp.text:
        return {}
    return resp.json()


def _clamp_limit(value: int | None) -> int:
    if value is None:
        return _DEFAULT_LIMIT
    return max(1, min(int(value), _MAX_LIMIT))


def _truncate(text: str) -> str:
    if len(text) > _RESULT_TRUNCATE_CHARS:
        return text[:_RESULT_TRUNCATE_CHARS] + " ...[truncated]"
    return text


def _summarize_incident(raw: dict) -> dict:
    svc = raw.get("service") or {}
    ep = raw.get("escalation_policy") or {}
    assignments = raw.get("assignments") or []
    assignees = [
        (a.get("assignee") or {}).get("summary")
        for a in assignments
        if (a.get("assignee") or {}).get("summary")
    ]
    return {
        "id": raw.get("id"),
        "incident_number": raw.get("incident_number"),
        "title": raw.get("title") or raw.get("summary"),
        "description": _truncate(raw.get("description") or ""),
        "status": raw.get("status"),
        "urgency": raw.get("urgency"),
        "priority": (raw.get("priority") or {}).get("summary") if isinstance(raw.get("priority"), dict) else raw.get("priority"),
        "created_at": raw.get("created_at"),
        "updated_at": raw.get("updated_at"),
        "resolved_at": raw.get("resolved_at"),
        "service": svc.get("summary"),
        "service_id": svc.get("id"),
        "escalation_policy": ep.get("summary"),
        "assignees": assignees,
        "url": raw.get("html_url"),
    }


def _summarize_service(raw: dict) -> dict:
    ep = raw.get("escalation_policy") or {}
    return {
        "id": raw.get("id"),
        "name": raw.get("name") or raw.get("summary"),
        "description": (raw.get("description") or "")[:200],
        "status": raw.get("status"),
        "created_at": raw.get("created_at"),
        "escalation_policy": ep.get("summary"),
        "url": raw.get("html_url"),
    }


def _summarize_oncall(raw: dict) -> dict:
    user = raw.get("user") or {}
    ep = raw.get("escalation_policy") or {}
    sched = raw.get("schedule") or {}
    return {
        "user": user.get("summary"),
        "user_id": user.get("id"),
        "escalation_policy": ep.get("summary"),
        "escalation_policy_id": ep.get("id"),
        "escalation_level": raw.get("escalation_level"),
        "schedule": sched.get("summary"),
        "schedule_id": sched.get("id"),
        "start": raw.get("start"),
        "end": raw.get("end"),
    }


def _summarize_log_entry(raw: dict) -> dict:
    agent = raw.get("agent") or {}
    chan = raw.get("channel") or {}
    return {
        "id": raw.get("id"),
        "type": raw.get("type"),
        "summary": (raw.get("summary") or "")[:500],
        "created_at": raw.get("created_at"),
        "agent": agent.get("summary"),
        "channel": chan.get("type"),
        "note": (chan.get("summary") or "")[:500] if isinstance(chan, dict) else None,
    }


# --- handlers -------------------------------------------------------


async def _h_list_incidents(_unused, args: dict) -> Any:
    params: dict[str, Any] = {
        "limit": _clamp_limit(args.get("limit")),
        "sort_by": args.get("sort_by") or "created_at:desc",
    }
    if statuses := args.get("statuses"):
        params["statuses[]"] = statuses if isinstance(statuses, list) else [statuses]
    if urgency := args.get("urgency"):
        params["urgencies[]"] = [urgency]
    if since := args.get("since"):
        params["since"] = since
    if until := args.get("until"):
        params["until"] = until
    out = await _get("/incidents", params=params, tool="pagerduty_list_incidents")
    items = out.get("incidents") or []
    summarized = [_summarize_incident(x) for x in items]
    return {
        "count": len(summarized),
        "more": out.get("more"),
        "incidents": summarized,
    }


async def _h_get_incident(_unused, args: dict) -> Any:
    incident_id = args["incident_id"]
    out = await _get(f"/incidents/{incident_id}", tool="pagerduty_get_incident")
    raw = out.get("incident") or {}
    return {"incident": _summarize_incident(raw)}


async def _h_list_services(_unused, args: dict) -> Any:
    params: dict[str, Any] = {
        "limit": _clamp_limit(args.get("limit")),
    }
    if query := args.get("query"):
        params["query"] = query
    out = await _get("/services", params=params, tool="pagerduty_list_services")
    items = out.get("services") or []
    summarized = [_summarize_service(x) for x in items]
    return {
        "count": len(summarized),
        "more": out.get("more"),
        "services": summarized,
    }


async def _h_list_oncalls(_unused, args: dict) -> Any:
    params: dict[str, Any] = {
        "limit": _clamp_limit(args.get("limit")),
    }
    if ep_ids := args.get("escalation_policy_ids"):
        params["escalation_policy_ids[]"] = ep_ids if isinstance(ep_ids, list) else [ep_ids]
    if sched_ids := args.get("schedule_ids"):
        params["schedule_ids[]"] = sched_ids if isinstance(sched_ids, list) else [sched_ids]
    if since := args.get("since"):
        params["since"] = since
    if until := args.get("until"):
        params["until"] = until
    out = await _get("/oncalls", params=params, tool="pagerduty_list_oncalls")
    items = out.get("oncalls") or []
    summarized = [_summarize_oncall(x) for x in items]
    return {
        "count": len(summarized),
        "more": out.get("more"),
        "oncalls": summarized,
    }


async def _h_get_incident_log_entries(_unused, args: dict) -> Any:
    incident_id = args["incident_id"]
    params: dict[str, Any] = {
        "limit": _clamp_limit(args.get("limit")),
    }
    if is_overview := args.get("is_overview"):
        params["is_overview"] = is_overview
    out = await _get(
        f"/incidents/{incident_id}/log_entries",
        params=params,
        tool="pagerduty_get_incident_log_entries",
    )
    items = out.get("log_entries") or []
    return {
        "incident_id": incident_id,
        "count": len(items),
        "more": out.get("more"),
        "log_entries": [_summarize_log_entry(e) for e in items],
    }


# --- tool registry --------------------------------------------------


PAGERDUTY_TOOLS: list[MCPTool] = [
    MCPTool(
        name="pagerduty_list_incidents",
        description=(
            "List PagerDuty incidents. Filter by `statuses` "
            "(triggered, acknowledged, resolved), `urgency` (high|low), "
            "and a `since`/`until` time window (ISO-8601). Returns a "
            "summarized list (status, urgency, service, assignees, URL)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "statuses": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["triggered", "acknowledged", "resolved"]},
                    "description": "Filter by one or more incident statuses.",
                },
                "urgency": {"type": "string", "enum": ["high", "low"]},
                "since": {"type": "string", "description": "ISO-8601 lower time bound."},
                "until": {"type": "string", "description": "ISO-8601 upper time bound."},
                "limit": {"type": "number", "description": f"Max {_MAX_LIMIT}, default {_DEFAULT_LIMIT}"},
                "sort_by": {"type": "string", "description": "Default 'created_at:desc'."},
            },
        },
        handler=_h_list_incidents,
    ),
    MCPTool(
        name="pagerduty_get_incident",
        description=(
            "Get one incident's full details (status, urgency, priority, "
            "service, escalation policy, assignees, timestamps, URL)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "incident_id": {"type": "string", "description": "PagerDuty incident id (e.g. 'PT4KHLK')."},
            },
            "required": ["incident_id"],
        },
        handler=_h_get_incident,
    ),
    MCPTool(
        name="pagerduty_list_services",
        description=(
            "List PagerDuty services (technical services / applications). "
            "Use `query` to filter by name. Useful to validate a service "
            "name or grab its id before drilling into incidents."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Free-text filter on service name."},
                "limit": {"type": "number", "description": f"Max {_MAX_LIMIT}, default {_DEFAULT_LIMIT}"},
            },
        },
        handler=_h_list_services,
    ),
    MCPTool(
        name="pagerduty_list_oncalls",
        description=(
            "Who is on call right now (or in a `since`/`until` window). "
            "Filter by `escalation_policy_ids` and/or `schedule_ids`. "
            "Returns the on-call user, escalation level, and schedule."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "escalation_policy_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter to these escalation policy ids.",
                },
                "schedule_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter to these schedule ids.",
                },
                "since": {"type": "string", "description": "ISO-8601 lower time bound."},
                "until": {"type": "string", "description": "ISO-8601 upper time bound."},
                "limit": {"type": "number", "description": f"Max {_MAX_LIMIT}, default {_DEFAULT_LIMIT}"},
            },
        },
        handler=_h_list_oncalls,
    ),
    MCPTool(
        name="pagerduty_get_incident_log_entries",
        description=(
            "Chronological timeline (log entries) for one incident -- "
            "trigger, acknowledge, escalate, notify, resolve, and notes. "
            "Use to reconstruct what happened during the incident."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "incident_id": {"type": "string", "description": "PagerDuty incident id."},
                "limit": {"type": "number", "description": f"Max {_MAX_LIMIT}, default {_DEFAULT_LIMIT}"},
                "is_overview": {"type": "boolean", "description": "Return only the most important log entries."},
            },
            "required": ["incident_id"],
        },
        handler=_h_get_incident_log_entries,
    ),
]


def get_tool(name: str) -> MCPTool:
    for t in PAGERDUTY_TOOLS:
        if t.name == name:
            return t
    raise KeyError(f"unknown pagerduty tool: {name}")


# --- fake backend (FR-012; integration tests) ----------------------
#
# Data path: every handler discards its first arg (`_unused`) and reaches
# the module-level `_get`, which builds an httpx client from `_config()`
# (token + base URL). `build_fake()` swaps `_get` for a canned responder
# routed by path, returning PagerDuty REST v2-shaped payloads with NO
# network and NO token, then restores the real `_get` via `teardown`.
# Mirrors the Rootly fake pattern.

_FAKE_INCIDENT_ID = "PT4KHLK"


def _fake_incident_record() -> dict:
    return {
        "id": _FAKE_INCIDENT_ID,
        "type": "incident",
        "incident_number": 1234,
        "title": "Database connection pool exhausted",
        "description": "Primary DB hit max connections; API 500s for 12 min.",
        "status": "resolved",
        "urgency": "high",
        "priority": {"summary": "P1"},
        "created_at": "2026-05-20T10:00:00Z",
        "updated_at": "2026-05-20T10:12:00Z",
        "resolved_at": "2026-05-20T10:12:00Z",
        "html_url": "https://acme.pagerduty.com/incidents/PT4KHLK",
        "service": {"id": "PSVC001", "summary": "acme-notes-be"},
        "escalation_policy": {"id": "PEP001", "summary": "Platform on-call"},
        "assignments": [
            {"assignee": {"id": "PUSER01", "summary": "On-call SRE"}}
        ],
    }


def _fake_service_record() -> dict:
    return {
        "id": "PSVC001",
        "type": "service",
        "name": "acme-notes-be",
        "summary": "acme-notes-be",
        "description": "Notes backend service.",
        "status": "active",
        "created_at": "2025-01-01T00:00:00Z",
        "html_url": "https://acme.pagerduty.com/services/PSVC001",
        "escalation_policy": {"id": "PEP001", "summary": "Platform on-call"},
    }


def _fake_oncall_record() -> dict:
    return {
        "user": {"id": "PUSER01", "summary": "On-call SRE"},
        "escalation_policy": {"id": "PEP001", "summary": "Platform on-call"},
        "escalation_level": 1,
        "schedule": {"id": "PSCH01", "summary": "Primary rotation"},
        "start": "2026-05-20T00:00:00Z",
        "end": "2026-05-27T00:00:00Z",
    }


def _fake_log_entry_record() -> dict:
    return {
        "id": "PLOG01",
        "type": "resolve_log_entry",
        "summary": "Resolved by On-call SRE",
        "created_at": "2026-05-20T10:12:00Z",
        "agent": {"id": "PUSER01", "summary": "On-call SRE"},
        "channel": {"type": "website", "summary": "Resolved via UI"},
    }


async def _fake_get(path: str, params: dict | None = None, *, tool: str = "pagerduty") -> Any:
    """Canned stand-in for the module-level GET. Routes by path to a
    PagerDuty REST v2-shaped response the handler parses. No network."""
    params = params or {}
    if path.startswith("/incidents/") and path.endswith("/log_entries"):
        return {"log_entries": [_fake_log_entry_record()], "more": False}
    if path.startswith("/incidents/"):
        return {"incident": _fake_incident_record()}
    if path == "/incidents":
        return {"incidents": [_fake_incident_record()], "more": False}
    if path == "/services":
        return {"services": [_fake_service_record()], "more": False}
    if path == "/oncalls":
        return {"oncalls": [_fake_oncall_record()], "more": False}
    return {}


def build_fake():
    """Return a FakeMCP exposing the PagerDuty tools wired to an offline
    backend. Needs NO PagerDuty token / network: the module-level `_get`
    is swapped for a canned responder and restored by `teardown`."""
    import opsrag.mcp.pagerduty as _mod
    from opsrag.mcp._fake import FakeMCP

    _orig_get = _mod._get
    _mod._get = _fake_get

    def _restore() -> None:
        _mod._get = _orig_get

    return FakeMCP(tools=list(PAGERDUTY_TOOLS), client=None, teardown=_restore)
