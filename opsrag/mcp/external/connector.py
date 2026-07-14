"""Discover + wrap an upstream MCP server's tools as native MCPTools, and
register them into the live registry at RUNTIME (inside the app lifespan)."""
from __future__ import annotations

import logging

from opsrag.mcp import MCPTool
from opsrag.mcp.external.client import ExternalMCPClient

_log = logging.getLogger("opsrag.mcp.external.connector")

_DEFAULT_SCHEMA = {"type": "object", "properties": {}}

# Write-verb backstop (defense-in-depth): even a correctly-configured
# allowlist/denylist is operator-authored and can be wrong. This drops any
# surviving upstream tool whose name contains a write verb at a token
# boundary (split on "_"), so a misconfigured allowlist can't expose a
# mutation to the agent surface. MUST NOT include "search"/"get"/"find"/
# "list" -- those are the pilot's own read tools (search_events,
# search_issues, find_projects, find_organizations, find_monitors,
# get_sentry_resource) and must survive.
_WRITE_VERBS = frozenset({
    "create", "update", "delete", "remove", "add", "set", "write", "drop",
    "patch", "put", "post", "edit", "modify", "deactivate", "disable",
    "enable", "revoke", "grant", "execute",
})


def _is_write_tool(upstream_name: str) -> bool:
    return any(tok in _WRITE_VERBS for tok in upstream_name.lower().split("_"))


# JSON-Schema keys Vertex's function-declaration parser ignores or chokes on;
# dropped during sanitization. (`$ref`/`$defs`/`definitions` in particular abort
# translation.)
_SCHEMA_DROP_KEYS = frozenset({
    "$schema", "$id", "$ref", "$defs", "$comment", "definitions",
    "title", "default", "examples", "additionalProperties",
})
# Recognized JSON-Schema type names -- used to pick the meaningful branch of a
# `type: [..., "null"]` list without guessing.
_JSON_TYPES = ("object", "array", "string", "number", "integer", "boolean")


def _sanitize_schema(node):
    """Make an upstream MCP tool's JSON-Schema safe for the Vertex/Gemini
    tool-spec adapter.

    Community MCP servers emit JSON-Schema constructs Vertex's function
    declaration parser rejects -- most notably `anyOf`/`oneOf` unions that
    include a `null` branch (an optional field as `anyOf: [{type: string},
    {type: null}]`) and list-valued `type` like `["string", "null"]`. Left
    verbatim these silently abort tool-spec translation, so the reasoner never
    gets the tool (observed: Sentry triage dies in ~23ms and falls back to
    retrieval-only). This collapses unions to their first non-null branch, drops
    `null` from `type` lists, recurses through `properties`/`items`, and strips
    schema-draft metadata keys. Total: any unexpected shape degrades to a
    permissive object schema instead of raising.
    """
    if not isinstance(node, dict):
        return node

    # Collapse a union (anyOf/oneOf/allOf) to a single non-null branch, merging
    # the parent's sibling keys (description, etc.) onto it.
    for union_key in ("anyOf", "oneOf", "allOf"):
        options = node.get(union_key)
        if isinstance(options, list) and options:
            branches = [
                b for b in options
                if isinstance(b, dict) and b.get("type") != "null"
            ]
            chosen = dict(branches[0]) if branches else {"type": "string"}
            for k, v in node.items():
                if k in ("anyOf", "oneOf", "allOf"):
                    continue
                chosen.setdefault(k, v)
            return _sanitize_schema(chosen)

    out: dict = {}
    for k, v in node.items():
        if k in _SCHEMA_DROP_KEYS:
            continue
        if k == "type" and isinstance(v, list):
            non_null = [t for t in v if t != "null"]
            chosen = next((t for t in _JSON_TYPES if t in non_null), None)
            out["type"] = chosen or (non_null[0] if non_null else "string")
        elif k == "properties" and isinstance(v, dict):
            out["properties"] = {pk: _sanitize_schema(pv) for pk, pv in v.items()}
        elif k == "items" and isinstance(v, dict):
            out["items"] = _sanitize_schema(v)
        else:
            out[k] = v
    return out


def _make_handler(client: ExternalMCPClient, upstream_name: str):
    async def handler(_unused, args: dict):  # arg1 ignored (proxy passes None)
        return await client.call_tool(upstream_name, args or {})
    return handler


async def build_external_connector(
    name: str,
    cfg_block,
    client: ExternalMCPClient,
) -> tuple[list[MCPTool], list[str]]:
    """Return (wrapped MCPTools, upstream names kept). Allowlist-first, then
    denylist. Wrapped name = f"{name}_{upstream}"; handler closes over client."""
    allow = set(cfg_block.tool_allowlist or [])
    deny = set(cfg_block.tool_denylist or [])
    upstream = await client.list_tools()
    wrapped: list[MCPTool] = []
    kept: list[str] = []
    for spec in upstream:
        uname = spec.get("name")
        if uname not in allow or uname in deny:
            continue
        if _is_write_tool(uname):
            _log.debug(
                "external connector %s: dropping %r -- write-verb backstop",
                name, uname,
            )
            continue
        schema = spec.get("inputSchema") or spec.get("input_schema") or _DEFAULT_SCHEMA
        if not isinstance(schema, dict) or not schema:
            schema = _DEFAULT_SCHEMA
        # Upstream schemas are third-party -- sanitize for the Vertex/Gemini
        # tool-spec adapter (strips anyOf/null unions that silently break triage).
        schema = _sanitize_schema(schema)
        if not isinstance(schema, dict) or not schema.get("type"):
            schema = _DEFAULT_SCHEMA
        wrapped.append(MCPTool(
            name=f"{name}_{uname}",
            description=str(spec.get("description") or uname),
            input_schema=schema,
            handler=_make_handler(client, uname),
        ))
        kept.append(uname)
    _log.info("external connector %s: %d/%d tools admitted", name, len(kept), len(upstream))
    return wrapped, kept


async def _default_client_factory(name: str, blk) -> ExternalMCPClient:
    import os
    token = (os.environ.get(blk.auth_env) or "").strip() if blk.auth_env else ""
    auth = (("Authorization", f"{blk.auth_scheme} {token}") if token else None)
    client = ExternalMCPClient(blk.url, auth_header=auth)
    try:
        await client.initialize()
    except Exception:
        await client.aclose()
        raise
    return client


async def register_external_connectors(settings, *, client_factory=None) -> list[str]:
    """Discover + register every ENABLED external MCP server at RUNTIME.

    Mutates the import-frozen structures IN PLACE (never rebind ALL_MCP_TOOLS).
    Graceful-degrade: a server whose discovery fails registers zero tools and is
    skipped -- never blocks boot. Returns the registered connector names."""
    from opsrag import mcp as mcp_pkg
    from opsrag.mcp.registry import REGISTRY, MCPIntegration
    from opsrag.mcp_server.registry_loader import (
        enabled_integration_names,
        rebuild_tool_to_connector,
        set_active_enabled,
    )

    factory = client_factory or _default_client_factory
    registered: list[str] = []
    external = getattr(settings, "external_mcp", {}) or {}
    for name, blk in external.items():
        if not getattr(blk, "enabled", False):
            continue
        if name in REGISTRY:  # idempotency guard (re-entrant lifespan)
            continue
        try:
            client = await factory(name, blk)
            tools, kept = await build_external_connector(name, blk, client)
        except Exception as exc:  # noqa: BLE001 -- degrade, never block boot
            _log.warning("external MCP %s discovery failed: %s -- skipping", name, exc)
            continue
        if not tools:
            _log.warning("external MCP %s produced 0 tools -- skipping", name)
            continue
        mcp_pkg.ALL_MCP_TOOLS.extend(tools)  # EXTEND in place
        auth_env = getattr(blk, "auth_env", None)
        REGISTRY[name] = MCPIntegration(
            name=name,
            # Friendly label for the Integrations UI (falls back to the config
            # key). Avoids the "sentry_mcp / sentry_mcp" double-name.
            display_name=(getattr(blk, "display_name", None) or name),
            config_type=type(blk),
            category=getattr(blk, "category", "Integrations"),
            # Surface the token env so the UI shows the requirement instead of
            # "no env required". Not validated at boot (validate_enabled_mcps
            # only iterates settings.mcp), so this is display-only.
            required_env=((auth_env,) if auth_env else ()),
            tool_names=tuple(t.name for t in tools),
        )
        registered.append(name)
        _log.info("external MCP %s registered with %d tools", name, len(tools))

    if registered:
        rebuild_tool_to_connector()
        # REPLACE semantics -> pass the UNION (native + external). Compute the
        # external half from the LIVE REGISTRY (not just `registered`), so a
        # later widening call never drops an earlier external connector that
        # was skipped this time by the idempotency guard above.
        from opsrag.config_mcp import ExternalMCPConfigBlock
        all_external = tuple(
            n for n, integ in REGISTRY.items()
            if isinstance(integ.config_type, type)
            and issubclass(integ.config_type, ExternalMCPConfigBlock)
        )
        set_active_enabled(tuple(enabled_integration_names(settings)) + all_external)
    return registered
