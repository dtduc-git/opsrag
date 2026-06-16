"""Google Cloud Platform MCP-style tools for OpsRAG.

Read-only async tools over GCP REST APIs (GKE, Cloud Run, Cloud Asset
Inventory). Auth uses Application Default Credentials (ADC) via
``google.auth.default()`` to mint a short-lived bearer token, then issues
plain ``httpx`` calls. The google libraries are LAZY-imported inside
``_config()`` so the module imports fine without the SDK installed (the
offline fake swaps the module-level ``_get`` and needs no google libs at
all).

Cloud Monitoring (metrics + alert policies) and Cloud Logging now live in
the dedicated ``stackdriver`` connector (``opsrag/mcp/stackdriver.py``).

## Read-only enforcement

Every tool is an HTTP GET. No create / update / delete / patch / set
anywhere.

## Tool list (3 read-only)

| Tool                              | Endpoint                                                    |
|-----------------------------------|------------------------------------------------------------|
| `gcp_gke_list_clusters`           | GET  `container.googleapis.com/v1/.../clusters`            |
| `gcp_run_list_services`           | GET  `run.googleapis.com/v2/.../services`                  |
| `gcp_asset_search`                | GET  `cloudasset.googleapis.com/v1/...:searchAllResources` |
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx

from opsrag.mcp.gitlab import MCPTool

_log = logging.getLogger("opsrag.mcp.gcp")

_DEFAULT_TIMEOUT_S = 30.0
_DEFAULT_LIMIT = 25
_MAX_LIMIT = 100
_RESULT_TRUNCATE_CHARS = 8000

# Scope: cloud-platform.read-only covers logging/monitoring/container/run/asset reads.
_SCOPES = ["https://www.googleapis.com/auth/cloud-platform.read-only"]

# Redact secrets that can leak through log payloads / resource metadata.
_REDACT_PATTERNS = [
    (re.compile(r"\bxoxb-[A-Za-z0-9-]{20,}"), "[REDACTED:slack_bot_token]"),
    (re.compile(r"\bghp_[A-Za-z0-9]{30,}"), "[REDACTED:github_token]"),
    (re.compile(r"\bglpat-[A-Za-z0-9_]{20,}"), "[REDACTED:gitlab_token]"),
    (re.compile(r"\bAKIA[A-Z0-9]{16}\b"), "[REDACTED:aws_access_key]"),
    (re.compile(r"\bAIza[A-Za-z0-9_-]{35}"), "[REDACTED:google_api_key]"),
    (re.compile(r"\bya29\.[A-Za-z0-9_-]{20,}"), "[REDACTED:google_oauth_token]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]+"), "[REDACTED:jwt]"),
]


def _redact(text: str) -> str:
    for pat, repl in _REDACT_PATTERNS:
        text = pat.sub(repl, text)
    return text


class GCPMCPError(Exception):
    """Raised on GCP API errors / missing config. Redacts bearer tokens."""

    def __init__(self, status: int, body: str, *, tool: str | None = None):
        self.status = status
        self.body = _redact(body or "")
        self.tool = tool
        super().__init__(f"[{tool or 'gcp'}] {status}: {self.body[:300]}")


@dataclass
class _Config:
    project: str
    token: str


def _config(tool: str = "gcp") -> _Config:
    """Resolve default project + a bearer token via ADC.

    LAZY-imports ``google.auth`` so the module imports without the SDK.
    Raises a clear :class:`GCPMCPError` when project or credentials are
    missing.
    """
    project = (os.environ.get("GOOGLE_CLOUD_PROJECT") or "").strip()
    try:
        import google.auth  # lazy -- never import at module top
        import google.auth.transport.requests as _grequests
    except Exception as exc:  # noqa: BLE001 - SDK not installed
        raise GCPMCPError(
            0,
            "google-auth not installed. Install `google-auth` and provide "
            "ADC (GOOGLE_APPLICATION_CREDENTIALS or workload identity).",
            tool=tool,
        ) from exc

    try:
        creds, adc_project = google.auth.default(scopes=_SCOPES)
        creds.refresh(_grequests.Request())
    except Exception as exc:  # noqa: BLE001 - auth failure
        raise GCPMCPError(
            0,
            "Could not obtain Application Default Credentials. Set "
            "GOOGLE_APPLICATION_CREDENTIALS or run on a workload-identity "
            "enabled environment.",
            tool=tool,
        ) from exc

    project = project or (adc_project or "").strip()
    if not project:
        raise GCPMCPError(
            0,
            "No GCP project. Set GOOGLE_CLOUD_PROJECT or pass `project`.",
            tool=tool,
        )
    token = getattr(creds, "token", None)
    if not token:
        raise GCPMCPError(0, "ADC returned no access token.", tool=tool)
    return _Config(project=project, token=token)


def _project(args: dict, tool: str) -> str:
    """Project from args override, else config/env (which may trigger ADC)."""
    explicit = (args.get("project") or "").strip()
    if explicit:
        return explicit
    return _config(tool=tool).project


def _headers(tool: str) -> dict:
    cfg = _config(tool=tool)
    return {
        "Authorization": f"Bearer {cfg.token}",
        "Content-Type": "application/json",
    }


async def _get(url: str, params: dict | None = None, *, tool: str = "gcp") -> Any:
    """Module-level GET helper. The offline fake swaps this out."""
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    async with httpx.AsyncClient(headers=_headers(tool), timeout=_DEFAULT_TIMEOUT_S) as http:
        resp = await http.get(url, params=clean)
    if resp.status_code >= 400:
        raise GCPMCPError(resp.status_code, resp.text, tool=tool)
    return resp.json() if resp.text else {}


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


# --- handlers -------------------------------------------------------


async def _h_gke_list_clusters(_unused, args: dict) -> Any:
    """`container.googleapis.com/v1/projects/{p}/locations/-/clusters`
    -- all GKE clusters across regions/zones (location `-`)."""
    tool = "gcp_gke_list_clusters"
    project = _project(args, tool)
    location = (args.get("location") or "-").strip() or "-"
    url = (
        f"https://container.googleapis.com/v1/projects/{project}"
        f"/locations/{location}/clusters"
    )
    resp = await _get(url, tool=tool)
    clusters = resp.get("clusters") or []
    out = []
    for c in clusters[:_clamp(args.get("limit"))]:
        out.append({
            "name": c.get("name"),
            "location": c.get("location"),
            "status": c.get("status"),
            "current_node_count": c.get("currentNodeCount"),
            "node_version": c.get("currentNodeVersion"),
            "master_version": c.get("currentMasterVersion"),
            "endpoint": c.get("endpoint"),
            "network": c.get("network"),
            "create_time": c.get("createTime"),
        })
    return {
        "project": project,
        "location": location,
        "count": len(out),
        "missing_zones": resp.get("missingZones") or [],
        "clusters": out,
    }


async def _h_run_list_services(_unused, args: dict) -> Any:
    """`run.googleapis.com/v2/projects/{p}/locations/{loc}/services`
    -- Cloud Run services. `location` arg defaults to `-` (all regions)."""
    tool = "gcp_run_list_services"
    project = _project(args, tool)
    location = (args.get("location") or "-").strip() or "-"
    params = {
        "pageSize": _clamp(args.get("limit")),
        "pageToken": args.get("page_token"),
    }
    url = (
        f"https://run.googleapis.com/v2/projects/{project}"
        f"/locations/{location}/services"
    )
    resp = await _get(url, params=params, tool=tool)
    services = resp.get("services") or []
    out = []
    for s in services:
        tmpl = s.get("template") or {}
        containers = tmpl.get("containers") or []
        images = [c.get("image") for c in containers if c.get("image")]
        out.append({
            "name": s.get("name"),
            "uri": s.get("uri"),
            "uid": s.get("uid"),
            "ingress": s.get("ingress"),
            "images": images[:10],
            "latest_ready_revision": s.get("latestReadyRevision"),
            "create_time": s.get("createTime"),
            "update_time": s.get("updateTime"),
            "labels": s.get("labels") or {},
        })
    return {
        "project": project,
        "location": location,
        "count": len(out),
        "next_page_token": resp.get("nextPageToken"),
        "services": out,
    }


async def _h_asset_search(_unused, args: dict) -> Any:
    """`cloudasset.googleapis.com/v1/projects/{p}:searchAllResources`
    -- escape-hatch read across ALL GCP resource types in the project's
    scope. `query` is Cloud Asset query syntax; `asset_types` is a
    comma-separated list filter."""
    tool = "gcp_asset_search"
    project = _project(args, tool)
    asset_types = args.get("asset_types")
    if isinstance(asset_types, list):
        asset_types = ",".join(asset_types)
    params = {
        "query": args.get("query"),
        "assetTypes": asset_types,
        "pageSize": _clamp(args.get("limit")),
        "pageToken": args.get("page_token"),
        "orderBy": args.get("order_by"),
    }
    url = (
        f"https://cloudasset.googleapis.com/v1/projects/{project}"
        f":searchAllResources"
    )
    resp = await _get(url, params=params, tool=tool)
    results = resp.get("results") or []
    out = []
    for r in results:
        out.append({
            "name": r.get("name"),
            "asset_type": r.get("assetType"),
            "display_name": r.get("displayName"),
            "location": r.get("location"),
            "project": r.get("project"),
            "state": r.get("state"),
            "labels": r.get("labels") or {},
            "description": _truncate(r.get("description") or "", 500),
        })
    return {
        "project": project,
        "query": args.get("query"),
        "asset_types": asset_types,
        "count": len(out),
        "next_page_token": resp.get("nextPageToken"),
        "results": out,
    }


# --- tool registry --------------------------------------------------


GCP_TOOLS: list[MCPTool] = [
    MCPTool(
        name="gcp_gke_list_clusters",
        description=(
            "List GKE clusters across all locations (default location `-`) "
            "with status, node count, versions, endpoint, network. Use for "
            "'what GKE clusters do we have', 'cluster versions'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "project": {"type": "string"},
                "location": {"type": "string", "description": "region/zone, default '-' (all)"},
                "limit": {"type": "number"},
            },
        },
        handler=_h_gke_list_clusters,
    ),
    MCPTool(
        name="gcp_run_list_services",
        description=(
            "List Cloud Run services in a location (default `-` = all "
            "regions) with image, URI, ingress, latest ready revision. Use "
            "for 'what Cloud Run services are deployed', 'which image is "
            "service X running'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "project": {"type": "string"},
                "location": {"type": "string", "description": "region, default '-' (all)"},
                "limit": {"type": "number"},
                "page_token": {"type": "string"},
            },
        },
        handler=_h_run_list_services,
    ),
    MCPTool(
        name="gcp_asset_search",
        description=(
            "Escape-hatch READ across ALL GCP resource types via Cloud Asset "
            "Inventory (searchAllResources). `query` uses Cloud Asset query "
            "syntax (e.g. `state:RUNNING`, `name:prod*`); `asset_types` "
            "filters by type (e.g. "
            "`compute.googleapis.com/Instance,storage.googleapis.com/Bucket`). "
            "Use when no specific tool fits -- 'find any resource named X', "
            "'list all buckets / instances / service accounts'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "project": {"type": "string"},
                "query": {"type": "string", "description": "Cloud Asset query syntax"},
                "asset_types": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "Comma-separated or list of asset type names",
                },
                "order_by": {"type": "string"},
                "limit": {"type": "number"},
                "page_token": {"type": "string"},
            },
        },
        handler=_h_asset_search,
    ),
]


def get_tool(name: str) -> MCPTool:
    for t in GCP_TOOLS:
        if t.name == name:
            return t
    raise KeyError(f"unknown gcp tool: {name}")


# --- fake backend (FR-012; integration tests) ----------------------
#
# GCP handlers ignore their first (`_unused`) arg and reach the
# module-level `_get`, which builds httpx clients + ADC tokens via
# `_config()`. The offline fake replaces that module function with a
# canned, shape-faithful responder -- no network, no google libs, no GCP
# credentials. `build_fake()` returns client=None plus a teardown that
# restores the real `_get`.


async def _fake_get(url: str, params: dict | None = None, *, tool: str = "gcp") -> Any:
    """Canned GET stand-in. Routes by URL substring to a response shaped
    like the real GCP REST endpoint the handler parses."""
    if "/clusters" in url:
        return {
            "clusters": [
                {
                    "name": "prod-gke",
                    "location": "us-central1",
                    "status": "RUNNING",
                    "currentNodeCount": 6,
                    "currentNodeVersion": "1.30.2-gke.100",
                    "currentMasterVersion": "1.30.2-gke.100",
                    "endpoint": "10.0.0.1",
                    "network": "default",
                    "createTime": "2026-01-01T00:00:00Z",
                }
            ],
            "missingZones": [],
        }
    if "/services" in url:
        return {
            "services": [
                {
                    "name": "projects/demo/locations/us-central1/services/api",
                    "uri": "https://api-xyz.a.run.app",
                    "uid": "uid-1",
                    "ingress": "INGRESS_TRAFFIC_ALL",
                    "template": {
                        "containers": [
                            {"image": "gcr.io/demo/api:1.2.3"}
                        ]
                    },
                    "latestReadyRevision": "projects/demo/locations/us-central1/services/api/revisions/api-001",
                    "createTime": "2026-01-01T00:00:00Z",
                    "updateTime": "2026-05-20T00:00:00Z",
                    "labels": {"team": "platform"},
                }
            ]
        }
    if ":searchAllResources" in url:
        return {
            "results": [
                {
                    "name": "//compute.googleapis.com/projects/demo/zones/us-central1-a/instances/vm-1",
                    "assetType": "compute.googleapis.com/Instance",
                    "displayName": "vm-1",
                    "location": "us-central1-a",
                    "project": "projects/demo",
                    "state": "RUNNING",
                    "labels": {"env": "prod"},
                    "description": "demo instance",
                }
            ]
        }
    return {}


def build_fake():
    """Return a FakeMCP exposing the GCP tools wired to an offline backend.

    Needs NO google libs / network / GCP credentials: the module-level
    `_get` is swapped for a canned responder and restored by `teardown`."""
    import opsrag.mcp.gcp as _mod
    from opsrag.mcp._fake import FakeMCP

    _orig_get = _mod._get
    _mod._get = _fake_get

    def _restore() -> None:
        _mod._get = _orig_get

    return FakeMCP(tools=list(GCP_TOOLS), client=None, teardown=_restore)
