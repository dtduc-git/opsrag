"""Splunk MCP-style tools for OpsRAG (read-only search).

Read-only async tools over the Splunk management REST API (default port
8089). Reuses `SPLUNK_URL` (mgmt API base, e.g. https://host:8089) +
`SPLUNK_TOKEN` (bearer) from env. `SPLUNK_VERIFY_SSL` (default true)
controls TLS verification for self-hosted certs.

## Read-only enforcement

Every tool issues a GET or a read-only search POST (oneshot search,
saved-search dispatch+results, export stream). The SPL guardrail
rejects any search pipeline containing a mutating command
(`| delete`, `| collect`, `| outputlookup`, `| sendemail`, `| script`)
so the agent can never mutate data through a crafted query. No
configuration writes, no index edits, no alert acks.

## Tool list (6 read-only)

| Tool                          | Endpoint                                          |
|-------------------------------|---------------------------------------------------|
| `splunk_run_search`           | POST `/services/search/v2/jobs` (oneshot)         |
| `splunk_export_search`        | POST `/services/search/v2/jobs/export`            |
| `splunk_list_saved_searches`  | GET  `/services/saved/searches`                   |
| `splunk_run_saved_search`     | POST `/services/saved/searches/{name}/dispatch`   |
| `splunk_list_indexes`         | GET  `/services/data/indexes`                     |
| `splunk_fired_alerts`         | GET  `/services/alerts/fired_alerts`              |
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

_log = logging.getLogger("opsrag.mcp.splunk")

_DEFAULT_TIMEOUT_S = 60.0
_DEFAULT_LIMIT = 100
_MAX_LIMIT = 1000
_RESULT_TRUNCATE_CHARS = 32000

# Mutating SPL commands -- reject any search whose pipeline contains one.
# Matched after a pipe `|` so a literal field value can't trip it.
_MUTATING_SPL = re.compile(
    r"\|\s*(delete|collect|outputlookup|outputcsv|sendemail|script|tscollect|sendalert)\b",
    re.IGNORECASE,
)

# Redact credentials that might surface in error bodies / search results.
_REDACT_PATTERNS = [
    (re.compile(r"\bxoxb-[A-Za-z0-9-]{20,}"), "[REDACTED:slack_bot_token]"),
    (re.compile(r"\bghp_[A-Za-z0-9]{30,}"), "[REDACTED:github_token]"),
    (re.compile(r"\bglpat-[A-Za-z0-9_]{20,}"), "[REDACTED:gitlab_token]"),
    (re.compile(r"\bAKIA[A-Z0-9]{16}\b"), "[REDACTED:aws_access_key]"),
    (re.compile(r"\bAIza[A-Za-z0-9_-]{35}"), "[REDACTED:google_api_key]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]+"), "[REDACTED:jwt]"),
    (re.compile(r"(?i)\bSplunk\s+[A-Za-z0-9._-]{20,}"), "[REDACTED:splunk_token]"),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._-]{20,}"), "[REDACTED:bearer_token]"),
]


def _redact(text: str) -> str:
    for pat, repl in _REDACT_PATTERNS:
        text = pat.sub(repl, text)
    return text


class SplunkMCPError(Exception):
    """Raised on Splunk API errors or guardrail violations. Token-redacted."""

    def __init__(self, status: int, body: str, *, tool: str | None = None):
        self.status = status
        self.body = _redact(body or "")
        self.tool = tool
        super().__init__(f"[{tool or 'splunk'}] {status}: {self.body[:300]}")


@dataclass
class _Config:
    base_url: str
    token: str
    verify_ssl: bool


def _config() -> _Config:
    base = (os.environ.get("SPLUNK_URL") or "").strip().rstrip("/")
    token = (os.environ.get("SPLUNK_TOKEN") or "").strip()
    verify_raw = (os.environ.get("SPLUNK_VERIFY_SSL") or "true").strip().lower()
    verify_ssl = verify_raw not in ("0", "false", "no", "off")
    if not base:
        raise SplunkMCPError(
            0,
            "SPLUNK_URL not set. Need the Splunk management API base "
            "(e.g. https://splunk.example.com:8089).",
            tool="splunk",
        )
    if not token:
        raise SplunkMCPError(
            0,
            "SPLUNK_TOKEN not set. Need a Splunk bearer token with read scope.",
            tool="splunk",
        )
    return _Config(base_url=base, token=token, verify_ssl=verify_ssl)


def _headers() -> dict:
    cfg = _config()
    return {"Authorization": f"Bearer {cfg.token}"}


async def _get(path: str, params: dict | None = None, *, tool: str = "splunk") -> Any:
    cfg = _config()
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    clean.setdefault("output_mode", "json")
    async with httpx.AsyncClient(
        headers=_headers(), timeout=_DEFAULT_TIMEOUT_S, verify=cfg.verify_ssl
    ) as http:
        resp = await http.get(f"{cfg.base_url}{path}", params=clean)
    if resp.status_code >= 400:
        raise SplunkMCPError(resp.status_code, resp.text, tool=tool)
    if not resp.text:
        return {}
    try:
        return resp.json()
    except ValueError:
        # export stream is newline-delimited JSON, not a single object.
        return _parse_ndjson(resp.text)


async def _post(path: str, data: dict, *, tool: str = "splunk") -> Any:
    cfg = _config()
    clean = {k: v for k, v in (data or {}).items() if v is not None}
    clean.setdefault("output_mode", "json")
    async with httpx.AsyncClient(
        headers=_headers(), timeout=_DEFAULT_TIMEOUT_S, verify=cfg.verify_ssl
    ) as http:
        resp = await http.post(f"{cfg.base_url}{path}", data=clean)
    if resp.status_code >= 400:
        raise SplunkMCPError(resp.status_code, resp.text, tool=tool)
    if not resp.text:
        return {}
    try:
        return resp.json()
    except ValueError:
        return _parse_ndjson(resp.text)


def _parse_ndjson(text: str) -> dict:
    """Splunk export returns newline-delimited JSON objects (one per result).
    Normalize to the `{results: [...]}` shape the oneshot endpoint returns."""
    results = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            import json

            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict) and "result" in obj:
            results.append(obj["result"])
        elif isinstance(obj, dict):
            results.append(obj)
    return {"results": results}


def _clamp(n: int | None, default: int = _DEFAULT_LIMIT, max: int = _MAX_LIMIT) -> int:
    if n is None:
        return default
    n = int(n)
    if n < 1:
        return 1
    if n > max:
        return max
    return n


def _truncate(text: str, limit: int = _RESULT_TRUNCATE_CHARS) -> str:
    if not text:
        return ""
    text = _redact(text)
    if len(text) > limit:
        return text[:limit] + " ...[truncated]"
    return text


def _normalize_spl(spl: str) -> str:
    """Reject mutating pipes; prepend `search ` when the query is bare SPL.

    Splunk's search endpoint requires the query to start with a generating
    command. A user-supplied `index=foo error` lacks the leading `search`,
    so we add it. Queries already beginning with `search ` or a `|`
    generating command are left as-is.
    """
    spl = (spl or "").strip()
    if not spl:
        raise SplunkMCPError(0, "empty SPL query", tool="splunk")
    if _MUTATING_SPL.search(spl):
        raise SplunkMCPError(
            0,
            "refused: SPL contains a mutating command "
            "(| delete | collect | outputlookup | sendemail | script). "
            "This integration is read-only.",
            tool="splunk",
        )
    lowered = spl.lstrip().lower()
    if lowered.startswith("search ") or spl.lstrip().startswith("|"):
        return spl
    return f"search {spl}"


def _extract_results(resp: Any) -> list[dict]:
    """Splunk oneshot/results returns `{results: [...]}`. Guard non-dict."""
    if isinstance(resp, dict):
        rows = resp.get("results") or []
    elif isinstance(resp, list):
        rows = resp
    else:
        rows = []
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        clean = {}
        for k, v in r.items():
            if isinstance(v, str):
                clean[k] = _truncate(v, 4000)
            else:
                clean[k] = v
        out.append(clean)
    return out


# --- handlers -------------------------------------------------------


async def _h_run_search(_unused, args: dict) -> Any:
    """`POST /services/search/v2/jobs` with `exec_mode=oneshot` -- run an
    ad-hoc SPL search and return rows synchronously. SPL is guardrailed
    (read-only) and `search ` is prepended when missing."""
    spl = _normalize_spl(args.get("search") or args.get("query") or "")
    count = _clamp(args.get("count") or args.get("limit"))
    data = {
        "search": spl,
        "exec_mode": "oneshot",
        "earliest_time": args.get("earliest_time") or "-15m",
        "latest_time": args.get("latest_time") or "now",
        "count": count,
        "output_mode": "json",
    }
    resp = await _post("/services/search/v2/jobs", data, tool="splunk_run_search")
    rows = _extract_results(resp)
    return {
        "search": spl,
        "earliest_time": data["earliest_time"],
        "latest_time": data["latest_time"],
        "count": len(rows),
        "results": rows[:count],
    }


async def _h_export_search(_unused, args: dict) -> Any:
    """`POST /services/search/v2/jobs/export` -- streaming export of search
    results (NDJSON). Same SPL guardrail + result caps as run_search."""
    spl = _normalize_spl(args.get("search") or args.get("query") or "")
    count = _clamp(args.get("count") or args.get("limit"))
    data = {
        "search": spl,
        "earliest_time": args.get("earliest_time") or "-15m",
        "latest_time": args.get("latest_time") or "now",
        "count": count,
        "output_mode": "json",
    }
    resp = await _post(
        "/services/search/v2/jobs/export", data, tool="splunk_export_search"
    )
    rows = _extract_results(resp)
    return {
        "search": spl,
        "earliest_time": data["earliest_time"],
        "latest_time": data["latest_time"],
        "count": len(rows),
        "results": rows[:count],
    }


async def _h_list_saved_searches(_unused, args: dict) -> Any:
    """`GET /services/saved/searches` -- list saved searches / reports."""
    count = _clamp(args.get("count") or args.get("limit"))
    params = {"count": count, "search": args.get("filter")}
    resp = await _get("/services/saved/searches", params=params, tool="splunk_list_saved_searches")
    entries = resp.get("entry") or [] if isinstance(resp, dict) else []
    out = []
    for e in entries:
        content = e.get("content") or {}
        out.append({
            "name": e.get("name"),
            "search": _truncate(content.get("search") or "", 2000),
            "is_scheduled": content.get("is_scheduled"),
            "cron_schedule": content.get("cron_schedule"),
            "disabled": content.get("disabled"),
            "app": (e.get("acl") or {}).get("app"),
            "owner": (e.get("acl") or {}).get("owner"),
        })
    return {"count": len(out), "saved_searches": out[:count]}


async def _h_run_saved_search(_unused, args: dict) -> Any:
    """Read-only dispatch of a saved search: `POST .../dispatch` then GET
    the resulting job's `/results`. Does not modify the saved search."""
    name = args.get("name")
    if not name:
        raise SplunkMCPError(0, "`name` (saved search name) is required", tool="splunk_run_saved_search")
    enc = quote(str(name), safe="")
    dispatch = await _post(
        f"/services/saved/searches/{enc}/dispatch",
        {
            "trigger_actions": "0",
            "dispatch.earliest_time": args.get("earliest_time"),
            "dispatch.latest_time": args.get("latest_time"),
            "output_mode": "json",
        },
        tool="splunk_run_saved_search",
    )
    # Splunk returns the created job sid.
    sid = None
    if isinstance(dispatch, dict):
        sid = dispatch.get("sid") or (dispatch.get("entry") or [{}])[0].get("name")
    count = _clamp(args.get("count") or args.get("limit"))
    results_resp: Any = {}
    if sid:
        results_resp = await _get(
            f"/services/search/v2/jobs/{quote(str(sid), safe='')}/results",
            params={"count": count, "output_mode": "json"},
            tool="splunk_run_saved_search",
        )
    rows = _extract_results(results_resp)
    return {
        "name": name,
        "sid": sid,
        "count": len(rows),
        "results": rows[:count],
    }


async def _h_list_indexes(_unused, args: dict) -> Any:
    """`GET /services/data/indexes` -- list indexes with event counts /
    size so the agent can pick the right index for a search."""
    count = _clamp(args.get("count") or args.get("limit"))
    resp = await _get("/services/data/indexes", params={"count": count}, tool="splunk_list_indexes")
    entries = resp.get("entry") or [] if isinstance(resp, dict) else []
    out = []
    for e in entries:
        content = e.get("content") or {}
        out.append({
            "name": e.get("name"),
            "total_event_count": content.get("totalEventCount"),
            "current_db_size_mb": content.get("currentDBSizeMB"),
            "max_data_size": content.get("maxDataSize"),
            "min_time": content.get("minTime"),
            "max_time": content.get("maxTime"),
            "disabled": content.get("disabled"),
        })
    return {"count": len(out), "indexes": out[:count]}


async def _h_fired_alerts(_unused, args: dict) -> Any:
    """`GET /services/alerts/fired_alerts` -- recently triggered alerts."""
    count = _clamp(args.get("count") or args.get("limit"))
    resp = await _get("/services/alerts/fired_alerts", params={"count": count}, tool="splunk_fired_alerts")
    entries = resp.get("entry") or [] if isinstance(resp, dict) else []
    out = []
    for e in entries:
        content = e.get("content") or {}
        out.append({
            "name": e.get("name"),
            "savedsearch_name": content.get("savedsearch_name"),
            "severity": content.get("severity"),
            "trigger_time": content.get("trigger_time"),
            "triggered_alerts": content.get("triggered_alert_count") or content.get("triggered_alerts"),
            "sid": content.get("sid"),
            "app": (e.get("acl") or {}).get("app"),
        })
    return {"count": len(out), "fired_alerts": out[:count]}


# --- tool registry --------------------------------------------------


SPLUNK_TOOLS: list[MCPTool] = [
    MCPTool(
        name="splunk_run_search",
        description=(
            "Run an ad-hoc SPL search (oneshot) and return matching events "
            "synchronously. Read-only: mutating SPL (| delete, | collect, "
            "| outputlookup, | sendemail, | script) is rejected. `search ` "
            "is prepended automatically if your SPL doesn't start with it. "
            "Time window defaults to -15m..now. Example search: "
            "`index=main sourcetype=access_combined status=500`."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "search": {"type": "string", "description": "SPL query (without leading 'search ' is fine)"},
                "earliest_time": {"type": "string", "description": "e.g. '-15m', '-1h', '-1d@d'. Default -15m."},
                "latest_time": {"type": "string", "description": "e.g. 'now'. Default now."},
                "count": {"type": "number", "description": f"Max rows (cap {_MAX_LIMIT}, default {_DEFAULT_LIMIT})."},
            },
            "required": ["search"],
        },
        handler=_h_run_search,
    ),
    MCPTool(
        name="splunk_export_search",
        description=(
            "Stream-export results of an SPL search (NDJSON), useful for "
            "larger result sets. Same read-only SPL guardrail and row caps "
            "as splunk_run_search."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "search": {"type": "string"},
                "earliest_time": {"type": "string"},
                "latest_time": {"type": "string"},
                "count": {"type": "number"},
            },
            "required": ["search"],
        },
        handler=_h_export_search,
    ),
    MCPTool(
        name="splunk_list_saved_searches",
        description=(
            "List saved searches / scheduled reports (name, SPL, schedule, "
            "owner, app). Use to find a saved search to run with "
            "splunk_run_saved_search."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "count": {"type": "number"},
                "filter": {"type": "string", "description": "Optional name filter."},
            },
        },
        handler=_h_list_saved_searches,
    ),
    MCPTool(
        name="splunk_run_saved_search",
        description=(
            "Dispatch a saved search by name (read-only; trigger_actions "
            "disabled) and return its results. Does NOT modify the saved "
            "search definition."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Saved search name."},
                "earliest_time": {"type": "string"},
                "latest_time": {"type": "string"},
                "count": {"type": "number"},
            },
            "required": ["name"],
        },
        handler=_h_run_saved_search,
    ),
    MCPTool(
        name="splunk_list_indexes",
        description=(
            "List Splunk indexes with event counts, size, and time bounds. "
            "Use to discover which index holds the data before searching."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "count": {"type": "number"},
            },
        },
        handler=_h_list_indexes,
    ),
    MCPTool(
        name="splunk_fired_alerts",
        description=(
            "List recently fired (triggered) alerts -- savedsearch name, "
            "severity, trigger time, count. Use to see what alerting "
            "conditions have recently tripped."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "count": {"type": "number"},
            },
        },
        handler=_h_fired_alerts,
    ),
]


def get_tool(name: str) -> MCPTool:
    for t in SPLUNK_TOOLS:
        if t.name == name:
            return t
    raise KeyError(f"unknown splunk tool: {name}")


# --- fake backend (FR-012; integration tests) ----------------------
#
# Splunk handlers ignore their first (`_unused`) arg and reach the
# module-level `_get` / `_post`, which build httpx clients from env via
# `_config()`. The offline fake swaps those two for canned responders --
# no network, no SPLUNK_URL / SPLUNK_TOKEN. `build_fake()` returns
# client=None plus a teardown that restores the real `_get` / `_post`.


async def _fake_get(path: str, params: dict | None = None, *, tool: str = "splunk") -> Any:
    """Canned stand-in for the module-level GET, shaped like the real
    Splunk mgmt REST JSON (`{entry: [...], ...}` or `{results: [...]}`)."""
    if path == "/services/saved/searches":
        return {
            "entry": [
                {
                    "name": "High 5xx rate",
                    "content": {
                        "search": "index=web status>=500 | stats count",
                        "is_scheduled": True,
                        "cron_schedule": "*/5 * * * *",
                        "disabled": False,
                    },
                    "acl": {"app": "search", "owner": "sre"},
                }
            ]
        }
    if path == "/services/data/indexes":
        return {
            "entry": [
                {
                    "name": "main",
                    "content": {
                        "totalEventCount": 123456,
                        "currentDBSizeMB": 2048,
                        "maxDataSize": "auto",
                        "minTime": "2026-05-01T00:00:00Z",
                        "maxTime": "2026-06-01T00:00:00Z",
                        "disabled": False,
                    },
                }
            ]
        }
    if path == "/services/alerts/fired_alerts":
        return {
            "entry": [
                {
                    "name": "High 5xx rate - 2026-06-01",
                    "content": {
                        "savedsearch_name": "High 5xx rate",
                        "severity": "5",
                        "trigger_time": 1716000000,
                        "triggered_alert_count": 3,
                        "sid": "scheduler__sre__search__abc",
                    },
                    "acl": {"app": "search"},
                }
            ]
        }
    if "/results" in path:
        return {
            "results": [
                {"_time": "2026-06-01T00:00:00Z", "status": "500", "count": "42"},
            ]
        }
    return {}


async def _fake_post(path: str, data: dict, *, tool: str = "splunk") -> Any:
    """Canned stand-in for the module-level POST (oneshot search, export,
    saved-search dispatch)."""
    if path == "/services/search/v2/jobs":
        return {
            "results": [
                {
                    "_time": "2026-06-01T00:00:01Z",
                    "host": "web-1",
                    "source": "/var/log/nginx/access.log",
                    "status": "500",
                    "_raw": "GET /notes 500 12ms",
                },
                {
                    "_time": "2026-06-01T00:00:02Z",
                    "host": "web-2",
                    "source": "/var/log/nginx/access.log",
                    "status": "500",
                    "_raw": "POST /notes 500 30ms",
                },
            ]
        }
    if path == "/services/search/v2/jobs/export":
        # Real export is NDJSON; the module's _post already normalizes to
        # {results: [...]} when it can't parse a single JSON doc, but the
        # fake returns the normalized shape directly.
        return {
            "results": [
                {"_time": "2026-06-01T00:00:03Z", "host": "web-3", "status": "503"},
            ]
        }
    if "/dispatch" in path:
        return {"sid": "scheduler__sre__search__dispatched1"}
    return {}


def build_fake():
    """Return a FakeMCP exposing the Splunk tools wired to an offline
    backend. Needs NO Splunk creds / network: the module-level
    `_get` / `_post` are swapped for canned responders and restored by
    `teardown`."""
    import opsrag.mcp.splunk as _mod
    from opsrag.mcp._fake import FakeMCP

    _orig_get = _mod._get
    _orig_post = _mod._post
    _mod._get = _fake_get
    _mod._post = _fake_post

    def _restore() -> None:
        _mod._get = _orig_get
        _mod._post = _orig_post

    return FakeMCP(tools=list(SPLUNK_TOOLS), client=None, teardown=_restore)
