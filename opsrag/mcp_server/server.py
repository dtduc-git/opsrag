"""MCP server (JSON-RPC 2.0 dispatch).

This module is transport-agnostic. The HTTP wiring (SSE event stream
and JSON-RPC inbox) lives in `opsrag.api.mcp_routes`; this module
implements the protocol layer that route handlers call into.

## Methods implemented (subset of the MCP spec)

| Method                       | Behaviour                                       |
|------------------------------|-------------------------------------------------|
| `initialize`                 | Server info + capability map (tools only).      |
| `notifications/initialized`  | Client->server handshake ack. No response sent.  |
| `tools/list`                 | Filtered registry from `registry.build_*`.      |
| `tools/call`                 | Rate-check -> dispatch -> audit -> JSON-RPC envelope. |
| `ping`                       | `{}` (idle keep-alive).                         |

Per JSON-RPC 2.0, **requests** carry an `"id"` and expect a response;
**notifications** omit `"id"` and do not. We handle both.

## Error model

JSON-RPC error codes used:
  -32700  parse error (caller's JSON is malformed)
  -32600  invalid request (missing "method", wrong jsonrpc version)
  -32601  method not found
  -32602  invalid params (unknown tool, bad args)
  -32603  internal error (rate-limit, handler crash)

`tools/call` returns a **success** envelope even when the underlying
tool returned `{"error": ...}`. That matches the MCP spec: the tool
*result* can carry error semantics (via the `isError` flag on
`content` items); only protocol-level failures use JSON-RPC errors.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from opsrag.mcp_server.audit import AuditLogger
from opsrag.mcp_server.rate_limit import TokenRateLimiter
from opsrag.mcp_server.registry import build_external_registry
from opsrag.mcp_server.registry_loader import active_enabled_tool_names

_log = logging.getLogger("opsrag.mcp_server.server")


# --- server metadata ------------------------------------------------

# Bump this on protocol-affecting changes. Independent of the OpsRAG
# package version because external clients pin against this surface.
MCP_SERVER_NAME = "opsrag-mcp-proxy"
MCP_SERVER_VERSION = "0.1.0"

# MCP protocol version the server speaks. We accept the client's
# requested version in `initialize.params.protocolVersion` and echo
# back this string -- clients negotiate by checking equality / range.
MCP_PROTOCOL_VERSION = "2024-11-05"


# --- Server instructions (initialize.result.instructions) ------------
#
# Top-level guidance shown to the consuming LLM ON CONNECT. Per the MCP
# spec, this is short + high-level -- per-tool detail belongs in each
# tool's `description`. The goal here: route the client to the RIGHT
# tool family in one shot, instead of letting it grope through 90+
# tools by name.
#
# Keep this under ~2 KB so it doesn't dominate the system-prompt token
# budget when many MCP servers are mounted side-by-side in a Claude
# Code session.
_SERVER_INSTRUCTIONS = """\
OpsRAG MCP -- SRE/security knowledge + live infra tools.

WHEN TO USE WHICH TOOL FAMILY (pick FIRST before searching by name):

* **Live K8s state** (pods/services/deployments running NOW, logs,
  metrics, recent events): `k8s_*` tools. Multi-cluster -- pass
  `env=prod|staging|preprod|dev`.

* **Live Elasticsearch / Kibana logs** (application logs, error
  traces, structured queries): `elasticsearch_*` tools. Per-env API
  keys; tools emit Kibana deep-links for click-through.

* **Live Datadog traces / spans / monitors / SLOs** (NOT logs -- logs
  are in Elasticsearch in this deployment): `datadog_*` tools.

* **Cartography graph queries** (cross-cluster, RBAC, blast-radius,
  GCP/K8s/Cloudflare relationships -- daily snapshot): `cartography_*`.
  6 tools. Best for "who can X / which Y touches Z" structural
  questions. Schema is read-only Cypher templates.

* **Live Cloudflare API** (zones, DNS, Zero-Trust apps, firewall/page
  rules -- fresher than the daily Cartography snapshot, and covers
  Zero-Trust which Cartography 0.136.0 doesn't ingest):
  `cloudflare_*` tools.

* **GitLab** (pipelines, MRs, commits, branches, deployments) ->
  `gitlab_*` tools. Pipeline-debugging drill: pipeline -> failed jobs
  -> trace per job.

* **Rootly incidents + post-mortems**: `rootly_*` tools.

* **Slack message + thread fetch**: `slack_*` tools. URL-based fetch
  for a single message / thread; not a broad search.

* **CloudSQL / Prometheus / runbooks / code-search / knowledge-base
  retrieval**: `cloudsql_*`, `prometheus_*`, `runbook_*`, `code_*`,
  `knowledge_search`.

DECISION RULES:

1. "What's happening RIGHT NOW" -> live tool (k8s/datadog/cloudflare/
   elasticsearch). NOT cartography (24h-old snapshot).

2. "Show me the RELATIONSHIPS / who can / blast-radius / cross-cluster"
   -> cartography_* (graph traversal is one query vs many live calls).

3. "Cloudflare Zero-Trust / Access apps / Page Rules" -> cloudflare_*
   live (Cartography doesn't have these).

4. "DNS lookup for value/IP across all zones" -> cartography_dns_for_value
   (graph traversal). For LIVE DNS on ONE zone ->
   cloudflare_list_dns_records.

5. "Pod blast-radius" -> cartography_pod_blast_radius (one call returns
   SA + secrets + node + exposing services). For pod RUNNING STATE +
   logs -> k8s_get_pod / k8s_get_pod_logs.

6. "Who can cluster-admin" / "which SA has role X" ->
   cartography_who_holds_role.

7. "K8s SA -> GCP identity bridging" -> cartography_workload_identity_chain
   (string-bridges via the gcp_service_account annotation property).

If a tool returns `{"reason": "not_bound"}`, that subsystem isn't
configured here -- surface that honestly + fall back. Never fabricate
infrastructure facts.
"""


# --- JSON-RPC error envelope helpers --------------------------------


class MCPProtocolError(Exception):
    """Internal: raise this to short-circuit dispatch with a JSON-RPC
    error code. The dispatcher catches it and emits the wire envelope.
    """

    def __init__(self, code: int, message: str, *, data: Any = None) -> None:
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"jsonrpc error {code}: {message}")


def _success(req_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error(req_id: Any, code: int, message: str, *, data: Any = None) -> dict:
    err: dict = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


# --- canonical args hash for audit ----------------------------------


def canonical_args_hash(args: dict | None) -> str:
    """Hex sha256 of canonical JSON for `args`.

    Canonical = sorted keys, no whitespace, ensure_ascii=False. This is
    stable across reordered dicts, which is what an audit query like
    "how many times was (tool, args) called?" needs.
    """
    if not args:
        return hashlib.sha256(b"{}").hexdigest()
    try:
        body = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        body = repr(args)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


# --- server ---------------------------------------------------------


@dataclass
class _MCPCallCtx:
    """Per-request context the dispatcher hands to `tools/call`."""

    user_oid: str | None
    token_id: str | None


class MCPServer:
    """JSON-RPC 2.0 dispatcher for MCP-over-HTTP.

    Stateless w.r.t. transport. Build once at startup, stash on
    `app.state.mcp_server`, then call `dispatch(envelope, ctx)` from
    each `POST /api/mcp/messages` handler.
    """

    def __init__(
        self,
        *,
        rate_limiter: TokenRateLimiter,
        audit: AuditLogger | None,
    ) -> None:
        self._rate_limiter = rate_limiter
        self._audit = audit
        # Snapshot the external registry at construction. If the agent
        # ever needs hot-reload it can call `refresh_registry()`.
        self._tools = {t.name: t for t in build_external_registry()}

    def refresh_registry(self) -> None:
        """Re-snapshot the external tool registry. Cheap; for tests."""
        self._tools = {t.name: t for t in build_external_registry()}

    # --- public dispatch ------------------------------------------

    async def dispatch(
        self, envelope: dict | list, ctx: _MCPCallCtx
    ) -> dict | list | None:
        """Dispatch a single JSON-RPC envelope (or batch).

        Returns the response envelope. For notifications (no `id`),
        returns ``None`` per spec. For batches, returns a list of
        responses excluding notification entries; an empty list
        collapses to ``None``.
        """
        if isinstance(envelope, list):
            responses: list[dict] = []
            for item in envelope:
                resp = await self._dispatch_one(item, ctx)
                if resp is not None:
                    responses.append(resp)
            return responses or None
        return await self._dispatch_one(envelope, ctx)

    async def _dispatch_one(
        self, env: Any, ctx: _MCPCallCtx
    ) -> dict | None:
        if not isinstance(env, dict):
            return _error(None, -32600, "invalid request: envelope must be an object")
        # JSON-RPC version pin. The spec is 2.0; we don't accept 1.x.
        if env.get("jsonrpc") != "2.0":
            return _error(env.get("id"), -32600, "invalid request: jsonrpc must be '2.0'")
        method = env.get("method")
        if not isinstance(method, str) or not method:
            return _error(env.get("id"), -32600, "invalid request: missing method")
        req_id = env.get("id")  # None for notifications
        params = env.get("params") or {}
        is_notification = "id" not in env

        try:
            result = await self._handle(method, params, ctx)
        except MCPProtocolError as exc:
            if is_notification:
                return None
            return _error(req_id, exc.code, exc.message, data=exc.data)
        except Exception as exc:  # noqa: BLE001 -- last-line defence
            _log.exception("mcp dispatch failed for method=%s", method)
            if is_notification:
                return None
            return _error(req_id, -32603, f"internal error: {exc}")

        # Notifications: no response.
        if is_notification:
            return None
        return _success(req_id, result)

    # --- method handlers ------------------------------------------

    async def _handle(self, method: str, params: dict, ctx: _MCPCallCtx) -> Any:
        if method == "initialize":
            return self._handle_initialize(params)
        if method == "notifications/initialized":
            # Spec: client sends this after `initialize` succeeds. No
            # result needed (it's a notification anyway).
            return None
        if method == "tools/list":
            return self._handle_tools_list()
        if method == "tools/call":
            return await self._handle_tools_call(params, ctx)
        if method == "ping":
            return {}
        raise MCPProtocolError(-32601, f"method not found: {method!r}")

    def _handle_initialize(self, params: dict) -> dict:
        # The spec asks the server to echo the highest protocol version
        # it supports. We pin to MCP_PROTOCOL_VERSION; client decides
        # whether that's acceptable.
        return {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "serverInfo": {
                "name": MCP_SERVER_NAME,
                "version": MCP_SERVER_VERSION,
            },
            # Only `tools` capability is exposed today. Resources /
            # prompts / sampling are not on the proxy path.
            "capabilities": {
                "tools": {
                    # We don't broadcast `tools/list_changed` events,
                    # so this is `false`.
                    "listChanged": False,
                },
            },
            # MCP spec optional field -- surfaced to the consuming LLM as
            # system-level guidance for the whole tool catalog. Per the
            # spec this should be short + high-level; per-tool detail
            # lives in each tool's `description`.
            "instructions": _SERVER_INSTRUCTIONS,
        }

    def _handle_tools_list(self) -> dict:
        # MCP `tools/list` returns `{ tools: Tool[] }` where each Tool
        # is `{ name, description, inputSchema }`. The internal
        # registry uses `input_schema` (snake_case); transform to the
        # wire-shape here.
        #
        # Enabled-gate (dynamic): self._tools is the static SAFE_FOR_EXTERNAL
        # allow-list snapshot built at construction. Intersect it with the
        # operator's *enabled* integrations at request time (set_active_enabled
        # runs AFTER this server is constructed, so a build-time snapshot would
        # miss it) -- else tools of DISABLED integrations (stackdriver, cloudwatch,
        # ...) leak into the catalog. None = gating off (dev) -> expose all safe.
        active = active_enabled_tool_names()
        return {
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.input_schema,
                }
                for t in self._tools.values()
                if active is None or t.name in active
            ],
        }

    async def _handle_tools_call(self, params: dict, ctx: _MCPCallCtx) -> dict:
        name = params.get("name")
        if not isinstance(name, str) or not name:
            raise MCPProtocolError(-32602, "tools/call requires `name` (string)")
        args = params.get("arguments") or {}
        if not isinstance(args, dict):
            raise MCPProtocolError(-32602, "tools/call: `arguments` must be an object")

        tool = self._tools.get(name)
        # Enabled-gate (same as tools/list): a tool in the static allow-list
        # whose integration the operator did NOT enable must be un-callable --
        # otherwise a disabled connector (e.g. stackdriver over ambient GCP ADC)
        # is reachable by name even though it's hidden from tools/list.
        active = active_enabled_tool_names()
        if tool is None or (active is not None and name not in active):
            # Unknown / un-exposed / integration-disabled tool. Audit-log the
            # attempt -- useful for spotting probe traffic and stale clients.
            self._audit_log(
                ctx=ctx, tool_name=name, args=args, status="denied",
                error=f"tool not exposed: {name!r}", latency_ms=0,
            )
            raise MCPProtocolError(-32602, f"unknown tool: {name!r}")

        # Rate-limit gate. We hash args BEFORE the check so we can audit
        # the denial with the same hash a successful call would carry.
        args_hash = canonical_args_hash(args)
        if ctx.token_id is None:
            # Should never happen -- the bearer-token dependency ensures
            # ctx.token_id is set before dispatch -- but guard defensively.
            raise MCPProtocolError(-32603, "missing token context")
        ok, reason = self._rate_limiter.allow(ctx.token_id, name)
        if not ok:
            self._audit_log(
                ctx=ctx, tool_name=name, args_hash=args_hash,
                status="denied", error=reason or "rate limited",
                latency_ms=0,
            )
            # Return a *successful* JSON-RPC envelope carrying an MCP
            # tool error. This matches how MCP clients (Claude Code)
            # surface tool failures to the LLM -- they see the message
            # rather than a protocol error.
            return {
                "content": [
                    {"type": "text", "text": reason or "rate limited"},
                ],
                "isError": True,
            }

        # Dispatch the handler. The handler has been wrapped by the
        # registry layer to convert exceptions into `{"error": ...}`
        # dicts -- so a "broken" call still returns normally here.
        # `client` is None for non-gitlab handlers; gitlab handlers
        # need a `GitLabClient` instance. For the external MCP path we
        # build a per-call client lazily, mirroring `multi_agent.py`'s
        # tool_caller. Cost: 1 httpx.AsyncClient per gitlab call ~= a
        # few ms of TLS setup; the cluster keeps connection pools warm
        # via keep-alive across the parent app's other GitLab traffic.
        client = None
        if name.startswith("gitlab_"):
            client = await self._build_gitlab_client()
        t0 = time.perf_counter()
        status = "ok"
        error: str | None = None
        try:
            result = await tool.handler(client, args)
            # The safe-handler wrapper turns exceptions into
            # `{"error": ...}` dicts. Reflect that into the audit
            # `status` field so SREs can grep for upstream failures.
            if isinstance(result, dict) and result.get("error"):
                status = "error"
                error = str(result.get("error"))[:1000]
        finally:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            if client is not None:
                # GitLabClient owns an httpx.AsyncClient; close it if
                # it was minted just for this call.
                close = getattr(client, "aclose", None)
                if close is not None:
                    try:
                        await close()
                    except Exception:
                        pass
            self._audit_log(
                ctx=ctx, tool_name=name, args_hash=args_hash,
                status=status, error=error, latency_ms=latency_ms,
            )

        # MCP wire shape for a tool result is
        #   { content: ContentItem[], isError?: bool, structuredContent?: any }
        # We emit one `text` item (canonical JSON of the result) AND
        # OPTIONALLY the raw value on `structuredContent` -- but ONLY
        # when the result is a JSON object. Claude Code's MCP SDK
        # validates `structuredContent` as a record/object and rejects
        # arrays/primitives outright ("expected record, received array"),
        # which is exactly what `gitlab_list_commits` etc. return.
        #
        # When the tool returns an array, we drop `structuredContent`
        # and rely on the text payload (which contains the same JSON,
        # valid for any shape). Tested 2026-05-15 after the dev hit
        # the schema-validation error on list_commits.
        text_payload = json.dumps(result, default=str)
        envelope: dict[str, Any] = {
            "content": [{"type": "text", "text": text_payload}],
        }
        if isinstance(result, dict):
            envelope["structuredContent"] = result
        if status == "error":
            envelope["isError"] = True
        return envelope

    # --- helpers --------------------------------------------------

    async def _build_gitlab_client(self) -> Any | None:
        """Mint a per-call GitLabClient. Returns None if token isn't
        configured -- the wrapper will surface a clean error dict.
        """
        try:
            from opsrag.mcp.gitlab import GitLabClient
            return GitLabClient()
        except Exception as exc:
            _log.warning("gitlab client init failed in mcp_server: %s", exc)
            return None

    def _audit_log(
        self,
        *,
        ctx: _MCPCallCtx,
        tool_name: str,
        args: dict | None = None,
        args_hash: str | None = None,
        latency_ms: int,
        status: str,
        error: str | None = None,
    ) -> None:
        if self._audit is None:
            return
        if args_hash is None:
            args_hash = canonical_args_hash(args)
        try:
            self._audit.log(
                occurred_at=datetime.now(UTC),
                user_oid=ctx.user_oid,
                token_id=ctx.token_id,
                tool_name=tool_name,
                args_hash=args_hash,
                latency_ms=latency_ms,
                status=status,
                error=error,
            )
        except Exception as exc:  # noqa: BLE001
            # Audit must never break the hot path. Demote to debug.
            _log.debug("audit enqueue failed for %s: %s", tool_name, exc)
