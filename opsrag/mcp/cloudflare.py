"""Cloudflare MCP-style tools for OpsRAG.

LIVE Cloudflare API surface -- complements the daily Cartography snapshot
which lives in cartography-neo4j. Use this MCP when:

* You need **fresh** state (Cartography ingest is daily; Cloudflare
  changes by the minute -- DNS edits, firewall rule additions).
* You need data that Cartography 0.136.0 **doesn't ingest**: Zero-Trust
  Access apps + policies, Page Rules, Cache Rules, Origin Rules,
  Transform Rules, Zone WAF custom rules, Firewall Services (legacy +
  filters).
* You're investigating a specific zone in detail (DNS records on
  `example.com`, the access policies attached to `app-X`, etc.).

For graph-shaped / cross-zone queries (e.g. "find every DNS record
pointing at IP X" -- joins across all zones), prefer the
`cartography_dns_for_value` tool which queries the pre-ingested Neo4j
graph in one hop.

Token + scopes
--------------
Auth: bearer token from env var `CLOUDFLARE_API_KEY` (same env the
cartography-runner uses for its Cloudflare ingest, set by
ExternalSecret from GSM `opsrag-cloudflare-api-key`). The token must
have Read scope on, at minimum:

    Zone:Zone, Zone:DNS, Zone:Page Rules, Zone:Firewall Services,
    Account:Access: Apps and Policies, Account:Account Settings

Optional extra scopes the tools below DO NOT use but a future expansion
could:

    Account:Analytics:Read   -> top traffic, threat counts (NOT yet)
    Zone:Analytics:Read      -> per-zone traffic stats     (NOT yet)

If you add those scopes later, mint new tools (e.g.
`cloudflare_top_traffic`, `cloudflare_security_events`); the scope-gate
fence below should still hold.

Defense-in-depth
----------------
1. **Read-only token scope.** No write/edit perms granted -- even if the
   handler had a typo and called a write endpoint, Cloudflare 403's it.
2. **No raw account_id in args.** Account-scoped calls resolve the
   default account via `_default_account()`; account_id arg only honored
   if explicitly passed (avoids LLM hallucinating an account ID).
3. **All errors round-tripped through `MCPCloudflareError` with stable
   `reason`** (`not_bound`, `bad_args`, `upstream_error`, `http_4xx`,
   `http_5xx`) -- never raw API responses leaked to the caller.

Tools (5 in v1)
---------------

| Tool                            | Question it answers                                |
|---------------------------------|----------------------------------------------------|
| cloudflare_list_zones           | What zones do we own + their status?               |
| cloudflare_list_dns_records     | What's the LIVE DNS for zone X right now?          |
| cloudflare_list_firewall_rules  | What firewall rules + custom WAF rules are live?   |
| cloudflare_list_page_rules      | What page rules + cache rules apply to zone X?     |
| cloudflare_list_access_apps     | What Zero-Trust apps + their policies live?        |
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from opsrag.mcp.gitlab import MCPTool

_log = logging.getLogger("opsrag.mcp.cloudflare")

# --- module-level state ----------------------------------------------
_config: _BoundConfig | None = None
_bound: bool = False

# Cloudflare API base. Versioned v4 since 2015; OK to hardcode.
_CF_API_BASE = "https://api.cloudflare.com/client/v4"

# Per-call HTTP timeout. CF zone-list can be slow on large accounts;
# 15s is a comfortable upper bound for a single read.
_DEFAULT_TIMEOUT_SECONDS = 15

# Bounded result sizes -- large CF responses (1000s of DNS records on a
# busy zone) would blow the agent's context window if shoved in raw.
_DEFAULT_MAX_ROWS = 200


class MCPCloudflareError(Exception):
    """Raised for Cloudflare MCP-tool errors.

    `reason` is a short machine code:
      - `not_bound`     -- token env var unset / module disabled
      - `bad_args`      -- caller passed bad input
      - `forbidden`     -- CF API returned 403 (insufficient token scope)
      - `not_found`     -- CF API returned 404 (zone/account missing)
      - `http_4xx`      -- other 4xx
      - `http_5xx`      -- CF-side outage
      - `upstream_error`-- network / parse / unexpected exception
    """

    def __init__(self, message: str, *, reason: str = "error"):
        self.reason = reason
        super().__init__(message)


@dataclass(frozen=True)
class _BoundConfig:
    token: str
    default_account_id: str
    timeout_seconds: int
    max_rows: int


# --- bind ------------------------------------------------------------


def bind(
    *,
    token: str | None = None,
    default_account_id: str = "",
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    max_rows: int = _DEFAULT_MAX_ROWS,
) -> None:
    """Register Cloudflare MCP config.

    The factory passes a `token` value (read from `CLOUDFLARE_API_KEY`
    env var, populated by ExternalSecret from GSM
    `opsrag-cloudflare-api-key`). If `token` is empty or None, this is
    a no-op -- every tool will return `reason="not_bound"`.

    `default_account_id` is optional. If empty, account-scoped tools
    (e.g. `list_access_apps`) auto-resolve via `_default_account()` on
    first call. Pass explicitly when a non-default CF account should be
    targeted (if the token can see multiple accounts; we want the
    org-account, not the personal one).
    """
    global _config, _bound

    if not token or not token.strip():
        _log.info("cloudflare mcp: bind called without token -- no-op (tools -> not_bound)")
        _config = None
        _bound = False
        return

    _config = _BoundConfig(
        token=token.strip(),
        default_account_id=(default_account_id or "").strip(),
        timeout_seconds=int(timeout_seconds),
        max_rows=int(max_rows),
    )
    _bound = True
    _log.info(
        "cloudflare mcp: bound timeout=%ds max_rows=%d account=%s",
        _config.timeout_seconds,
        _config.max_rows,
        _config.default_account_id or "(auto-resolve)",
    )


def _require_bound() -> _BoundConfig:
    if not _bound or _config is None:
        raise MCPCloudflareError(
            "Cloudflare MCP not configured -- set CLOUDFLARE_API_KEY env var "
            "(populated by ExternalSecret from GSM `opsrag-cloudflare-api-key`).",
            reason="not_bound",
        )
    return _config


# --- HTTP plumbing ---------------------------------------------------


async def _cf_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """GET against CF v4 API. Returns the full JSON envelope.

    CF responses have shape:
      `{"success": bool, "errors": [...], "messages": [...], "result": <data>, "result_info": {...}}`

    We raise `MCPCloudflareError` for transport errors, 4xx/5xx HTTP,
    and `success: false` responses. Caller deals with the `result`
    payload directly.
    """
    cfg = _require_bound()
    url = f"{_CF_API_BASE}{path}"
    headers = {
        "Authorization": f"Bearer {cfg.token}",
        "Accept": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=cfg.timeout_seconds) as client:
            r = await client.get(url, headers=headers, params=params or {})
    except httpx.RequestError as exc:
        raise MCPCloudflareError(
            f"cloudflare network error: {exc!r}", reason="upstream_error",
        ) from exc

    if r.status_code == 401 or r.status_code == 403:
        raise MCPCloudflareError(
            f"cloudflare {r.status_code} -- token missing the required scope "
            f"for path {path!r}. Add the corresponding Read scope to the "
            f"opsrag-cloudflare-api-key token and re-bind.",
            reason="forbidden",
        )
    if r.status_code == 404:
        raise MCPCloudflareError(
            f"cloudflare 404 -- resource at {path!r} not found.",
            reason="not_found",
        )
    if 400 <= r.status_code < 500:
        raise MCPCloudflareError(
            f"cloudflare {r.status_code}: {r.text[:300]!r}", reason="http_4xx",
        )
    if r.status_code >= 500:
        raise MCPCloudflareError(
            f"cloudflare {r.status_code} (server error): {r.text[:300]!r}",
            reason="http_5xx",
        )

    try:
        envelope = r.json()
    except ValueError as exc:
        raise MCPCloudflareError(
            f"cloudflare returned non-JSON: {r.text[:300]!r}",
            reason="upstream_error",
        ) from exc

    if not envelope.get("success", False):
        errs = envelope.get("errors") or []
        raise MCPCloudflareError(
            f"cloudflare api error: {errs!r}", reason="upstream_error",
        )

    return envelope


async def _default_account(cfg: _BoundConfig) -> str:
    """Return the default account ID. Prefers `cfg.default_account_id`,
    else picks the first account the token can see EXCLUDING personal
    accounts (those whose name contains an `@`). Cached at module load
    via the bound config; this is just the auto-resolve fallback.
    """
    if cfg.default_account_id:
        return cfg.default_account_id

    env = await _cf_get("/accounts", params={"per_page": 50})
    accounts = env.get("result") or []
    # Filter out personal accounts (e.g. "user@example.com's Account").
    non_personal = [a for a in accounts if "@" not in (a.get("name") or "")]
    if not non_personal:
        # Fall back to whatever we have.
        non_personal = accounts
    if not non_personal:
        raise MCPCloudflareError(
            "cloudflare: token can't see any account -- check scope.",
            reason="forbidden",
        )
    return non_personal[0]["id"]


# --- tool handlers ---------------------------------------------------


async def _h_list_zones(_unused, args: dict) -> dict:
    """List Cloudflare zones the token can see.

    Returns name, status, plan, account, paused. Use to disambiguate
    between multiple owned zones (e.g. `example.com` vs `example.net`)
    before calling per-zone tools.
    """
    cfg = _require_bound()
    name_contains = (args.get("name_contains") or "").strip().lower()
    per_page = min(int(args.get("per_page") or 50), cfg.max_rows)

    params: dict[str, Any] = {"per_page": per_page}
    env = await _cf_get("/zones", params=params)
    zones = env.get("result") or []

    rows: list[dict[str, Any]] = []
    for z in zones:
        name = z.get("name") or ""
        if name_contains and name_contains not in name.lower():
            continue
        rows.append({
            "id": z.get("id") or "",
            "name": name,
            "status": z.get("status") or "",
            "paused": bool(z.get("paused")),
            "plan": (z.get("plan") or {}).get("name") or "",
            "account": (z.get("account") or {}).get("name") or "",
        })
    return {"zones": rows, "count": len(rows)}


async def _resolve_zone_id(zone_arg: str) -> str:
    """Accept either a zone ID (32-char hex) or a zone NAME (e.g.
    `example.com`) and return the canonical zone ID. Saves callers
    a round trip when they only know the human name."""
    z = zone_arg.strip()
    if not z:
        raise MCPCloudflareError("zone is required (id or name).", reason="bad_args")
    # Zone IDs are 32-char lowercase hex. Names contain a `.`.
    if len(z) == 32 and all(c in "0123456789abcdef" for c in z.lower()):
        return z
    # Look up by name. CF supports `?name=foo.com`.
    env = await _cf_get("/zones", params={"name": z, "per_page": 5})
    hits = env.get("result") or []
    if not hits:
        raise MCPCloudflareError(
            f"cloudflare: zone {z!r} not found (or not visible to this token).",
            reason="not_found",
        )
    return hits[0]["id"]


async def _h_list_dns_records(_unused, args: dict) -> dict:
    """List LIVE DNS records on a zone. Use when you need fresher data
    than the daily Cartography snapshot -- DNS changes often.
    """
    # Defensive arg parsing: accept the canonical `zone` arg name AND
    # the common LLM-paraphrase variants (`zone_id`, `zone_name`,
    # `zoneName`). Agents observed in production occasionally pass
    # `zone_id` despite the schema; rather than 400 + force a retry,
    # we honour whichever variant arrived first.
    zone_arg = ""
    for k in ("zone", "zone_id", "zone_name", "zoneName"):
        v = (args.get(k) or "").strip()
        if v:
            zone_arg = v
            break
    if not zone_arg:
        raise MCPCloudflareError(
            "zone is required (name or id). If you don't know which "
            "zone, call `cloudflare_list_zones` first to enumerate them -- "
            "DO NOT fabricate a zone id.",
            reason="bad_args",
        )
    # Reject obvious hallucinated zone ids (all-same-digit strings, or
    # not 32-char hex). Real CF zone ids are 32-char lowercase hex.
    if len(zone_arg) == 32 and not all(c in "0123456789abcdef" for c in zone_arg.lower()):
        raise MCPCloudflareError(
            f"zone={zone_arg!r} doesn't look like a valid CF zone id "
            "(must be 32-char lowercase hex). If you meant the zone "
            "NAME (e.g. 'example.com'), pass it directly. "
            "Otherwise call `cloudflare_list_zones` to get real ids.",
            reason="bad_args",
        )
    record_type = (args.get("type") or "").strip().upper() or None
    name_contains = (args.get("name_contains") or "").strip().lower()

    cfg = _require_bound()
    zone_id = await _resolve_zone_id(zone_arg)

    params: dict[str, Any] = {"per_page": cfg.max_rows}
    if record_type:
        params["type"] = record_type

    env = await _cf_get(f"/zones/{zone_id}/dns_records", params=params)
    records = env.get("result") or []

    rows: list[dict[str, Any]] = []
    for r in records:
        name = r.get("name") or ""
        if name_contains and name_contains not in name.lower():
            continue
        rows.append({
            "id": r.get("id") or "",
            "name": name,
            "type": r.get("type") or "",
            "content": r.get("content") or "",
            "ttl": r.get("ttl"),
            "proxied": bool(r.get("proxied")),
            "comment": r.get("comment") or "",
        })
    return {"zone": zone_arg, "records": rows, "count": len(rows)}


async def _h_list_firewall_rules(_unused, args: dict) -> dict:
    """List firewall + custom-WAF rules on a zone.

    Cloudflare moved firewall_rules -> Rulesets (HTTP request firewall
    custom). We hit both endpoints + merge -- old `firewall/rules` first
    for legacy compatibility, then the Rulesets API for modern custom
    rules. If the legacy endpoint 404s (some zones have it disabled),
    we keep going with just the Rulesets result rather than failing
    the whole call.
    """
    zone_arg = (args.get("zone") or "").strip()
    if not zone_arg:
        raise MCPCloudflareError("zone is required.", reason="bad_args")
    zone_id = await _resolve_zone_id(zone_arg)

    legacy: list[dict[str, Any]] = []
    modern: list[dict[str, Any]] = []

    # Legacy firewall/rules (deprecated but still exposed on some zones)
    try:
        env = await _cf_get(f"/zones/{zone_id}/firewall/rules", params={"per_page": 50})
        for r in (env.get("result") or []):
            legacy.append({
                "kind": "legacy_firewall_rule",
                "id": r.get("id") or "",
                "action": r.get("action") or "",
                "description": r.get("description") or "",
                "paused": bool(r.get("paused")),
                "filter_expression": ((r.get("filter") or {}).get("expression") or ""),
            })
    except MCPCloudflareError as exc:
        # Tolerate forbidden / not_found on the legacy endpoint -- the
        # zone may have been migrated to Rulesets-only.
        if exc.reason not in {"forbidden", "not_found", "http_4xx"}:
            raise

    # Modern: Rulesets API -> http_request_firewall_custom phase
    try:
        env = await _cf_get(f"/zones/{zone_id}/rulesets/phases/http_request_firewall_custom/entrypoint")
        ruleset = env.get("result") or {}
        for r in (ruleset.get("rules") or []):
            modern.append({
                "kind": "ruleset_firewall_custom",
                "id": r.get("id") or "",
                "action": r.get("action") or "",
                "description": r.get("description") or "",
                "enabled": bool(r.get("enabled")),
                "expression": r.get("expression") or "",
            })
    except MCPCloudflareError as exc:
        if exc.reason not in {"forbidden", "not_found", "http_4xx"}:
            raise

    return {
        "zone": zone_arg,
        "rules": legacy + modern,
        "count": len(legacy) + len(modern),
    }


async def _h_list_page_rules(_unused, args: dict) -> dict:
    """List Page Rules + Cache Rules on a zone.

    Page Rules are the legacy URL-pattern config (cache_level,
    forwarding_url, etc.); Cache Rules are the modern Rulesets-backed
    equivalent. Surface both so the user sees the complete picture --
    Some zones still have both shapes coexisting.
    """
    zone_arg = (args.get("zone") or "").strip()
    if not zone_arg:
        raise MCPCloudflareError("zone is required.", reason="bad_args")
    zone_id = await _resolve_zone_id(zone_arg)

    pagerules: list[dict[str, Any]] = []
    cache_rules: list[dict[str, Any]] = []

    # Legacy Page Rules
    try:
        env = await _cf_get(f"/zones/{zone_id}/pagerules")
        for r in (env.get("result") or []):
            targets = r.get("targets") or []
            url_pattern = ""
            if targets:
                url_pattern = ((targets[0].get("constraint") or {}).get("value") or "")
            actions = r.get("actions") or []
            action_names = [a.get("id") for a in actions if a.get("id")]
            pagerules.append({
                "kind": "page_rule",
                "id": r.get("id") or "",
                "url_pattern": url_pattern,
                "actions": action_names,
                "status": r.get("status") or "",
                "priority": r.get("priority"),
            })
    except MCPCloudflareError as exc:
        if exc.reason not in {"forbidden", "not_found", "http_4xx"}:
            raise

    # Modern Cache Rules
    try:
        env = await _cf_get(f"/zones/{zone_id}/rulesets/phases/http_request_cache_settings/entrypoint")
        rs = env.get("result") or {}
        for r in (rs.get("rules") or []):
            cache_rules.append({
                "kind": "cache_rule",
                "id": r.get("id") or "",
                "description": r.get("description") or "",
                "expression": r.get("expression") or "",
                "action": r.get("action") or "",
                "enabled": bool(r.get("enabled")),
            })
    except MCPCloudflareError as exc:
        if exc.reason not in {"forbidden", "not_found", "http_4xx"}:
            raise

    return {
        "zone": zone_arg,
        "page_rules": pagerules,
        "cache_rules": cache_rules,
        "count": len(pagerules) + len(cache_rules),
    }


async def _h_list_access_apps(_unused, args: dict) -> dict:
    """List Cloudflare Zero-Trust Access apps + their policies.

    This is what Cartography 0.136.0 does NOT ingest (only the basic
    cloudflare module is present; no Zero-Trust). Useful for "what's
    behind Pomerium / Access SSO" questions -- the apps protect each
    internal hostname.
    """
    cfg = _require_bound()
    account_id = (args.get("account_id") or "").strip() or await _default_account(cfg)
    name_contains = (args.get("name_contains") or "").strip().lower()

    env = await _cf_get(
        f"/accounts/{account_id}/access/apps",
        params={"per_page": cfg.max_rows},
    )
    apps = env.get("result") or []

    rows: list[dict[str, Any]] = []
    for a in apps:
        name = a.get("name") or ""
        if name_contains and name_contains not in name.lower():
            continue
        rows.append({
            "id": a.get("id") or "",
            "name": name,
            "type": a.get("type") or "",
            "domain": a.get("domain") or "",
            "session_duration": a.get("session_duration") or "",
            "aud": a.get("aud") or "",
            # Don't fetch per-app policies here -- that's N+1; let the
            # agent call `cloudflare_get_access_app_policies` if it
            # actually needs them for a specific app.
        })
    return {
        "account_id": account_id,
        "apps": rows,
        "count": len(rows),
        "note": (
            "policies omitted -- call cloudflare_get_access_app_policies "
            "with the app id for per-app policy detail."
        ),
    }


async def _h_get_access_app_policies(_unused, args: dict) -> dict:
    """Get the policies (include/exclude/require groups) for a specific
    Zero-Trust Access app. Use after `list_access_apps` returns the
    matching app id -- this is the second hop for "who is allowed to
    reach app X" questions.
    """
    cfg = _require_bound()
    app_id = (args.get("app_id") or "").strip()
    if not app_id:
        raise MCPCloudflareError(
            "app_id is required (from cloudflare_list_access_apps).",
            reason="bad_args",
        )
    account_id = (args.get("account_id") or "").strip() or await _default_account(cfg)

    env = await _cf_get(
        f"/accounts/{account_id}/access/apps/{app_id}/policies"
    )
    policies = env.get("result") or []

    rows: list[dict[str, Any]] = []
    for p in policies:
        rows.append({
            "id": p.get("id") or "",
            "name": p.get("name") or "",
            "decision": p.get("decision") or "",
            "precedence": p.get("precedence"),
            # `include`, `exclude`, `require` are arrays of rule objects;
            # we keep the raw shape so the agent can describe them
            # naturally. JSON-stringify to keep the dict simple.
            "include": p.get("include") or [],
            "exclude": p.get("exclude") or [],
            "require": p.get("require") or [],
        })
    return {
        "account_id": account_id,
        "app_id": app_id,
        "policies": rows,
        "count": len(rows),
    }


# --- tool registry ---------------------------------------------------


def _clean_schema(schema: dict) -> dict:
    """Strip None-valued schema entries (mirrors helper in other MCPs)."""
    out = dict(schema)
    if "properties" in out:
        cleaned = {}
        for k, v in out["properties"].items():
            if isinstance(v, dict):
                cleaned[k] = {kk: vv for kk, vv in v.items() if vv is not None}
            else:
                cleaned[k] = v
        out["properties"] = cleaned
    return out


def _safe_wrap(handler):
    """Wrap so `MCPCloudflareError` returns a structured dict, not raises."""
    async def _wrapped(client: Any, args: dict) -> dict:
        try:
            return await handler(client, args)
        except MCPCloudflareError as exc:
            return {"error": str(exc), "reason": exc.reason}
    _wrapped.__name__ = handler.__name__
    _wrapped.__doc__ = handler.__doc__
    return _wrapped


CLOUDFLARE_TOOLS: list[MCPTool] = [
    MCPTool(
        name="cloudflare_list_zones",
        description=(
            "List Cloudflare zones the token can see, with status, plan, "
            "and owning account. LIVE -- pulled from CF API on each call. "
            "Use to disambiguate zone names before calling per-zone "
            "tools. Optional `name_contains` substring filter (case-"
            "insensitive). Returns `{zones: [{id, name, status, paused, "
            "plan, account}], count}`."
        ),
        input_schema=_clean_schema({
            "type": "object",
            "properties": {
                "name_contains": {
                    "type": "string",
                    "description": "Optional case-insensitive substring filter on zone name.",
                },
            },
            "required": [],
        }),
        handler=_safe_wrap(_h_list_zones),
    ),
    MCPTool(
        name="cloudflare_list_dns_records",
        description=(
            "USE WHEN you need LIVE DNS records on ONE specific zone "
            "(name or id you already know). Returns `{zone, records: "
            "[{name, type, content, ttl, proxied, comment}], count}`. "
            "PREFER this over `cartography_dns_for_value` when you "
            "need data fresher than 24h (DNS was edited recently). "
            "DO NOT FABRICATE zone IDs -- if you don't already know "
            "the zone, call `cloudflare_list_zones` FIRST and pass the "
            "returned id (or name like 'example.com') back here. "
            "For cross-zone search (e.g. 'records matching X across "
            "all zones'), use `cartography_dns_for_value(pattern=X)` "
            "instead -- one Cypher hop spans every zone. For 'all "
            "unproxied records across all zones', iterate "
            "`cloudflare_list_zones` -> per-zone "
            "`cloudflare_list_dns_records(zone=name)` -> filter "
            "`proxied=false` client-side."
        ),
        input_schema=_clean_schema({
            "type": "object",
            "properties": {
                "zone": {
                    "type": "string",
                    "description": "Zone name (e.g. 'example.com') or zone id (32-char hex).",
                },
                "type": {
                    "type": "string",
                    "description": "Optional record-type filter (A, AAAA, CNAME, TXT, MX, NS, ...).",
                },
                "name_contains": {
                    "type": "string",
                    "description": "Optional case-insensitive substring filter on record name.",
                },
            },
            "required": ["zone"],
        }),
        handler=_safe_wrap(_h_list_dns_records),
    ),
    MCPTool(
        name="cloudflare_list_firewall_rules",
        description=(
            "List firewall + custom-WAF rules on a Cloudflare zone. "
            "Surfaces BOTH the legacy `firewall/rules` API and the "
            "modern Rulesets `http_request_firewall_custom` phase. "
            "Some zones still have both shapes coexisting. Returns "
            "`{zone, rules: [{kind, id, action, "
            "description, expression/filter_expression, enabled|paused}], "
            "count}`. `kind` is `legacy_firewall_rule` or "
            "`ruleset_firewall_custom`."
        ),
        input_schema=_clean_schema({
            "type": "object",
            "properties": {
                "zone": {
                    "type": "string",
                    "description": "Zone name or id.",
                },
            },
            "required": ["zone"],
        }),
        handler=_safe_wrap(_h_list_firewall_rules),
    ),
    MCPTool(
        name="cloudflare_list_page_rules",
        description=(
            "List Page Rules + Cache Rules on a Cloudflare zone. Page "
            "Rules are the legacy URL-pattern config (cache_level, "
            "forwarding_url, etc.); Cache Rules are the modern "
            "Rulesets-backed equivalent. Returns `{zone, page_rules, "
            "cache_rules, count}`. Use for 'how is /api/foo cached?' "
            "or 'is there a redirect on /old/path' questions."
        ),
        input_schema=_clean_schema({
            "type": "object",
            "properties": {
                "zone": {
                    "type": "string",
                    "description": "Zone name or id.",
                },
            },
            "required": ["zone"],
        }),
        handler=_safe_wrap(_h_list_page_rules),
    ),
    MCPTool(
        name="cloudflare_list_access_apps",
        description=(
            "List Cloudflare Zero-Trust Access apps in the account. "
            "Cartography 0.136.0 does NOT ingest Zero-Trust -- this is "
            "the ONLY way to surface Access app metadata in OpsRAG. "
            "Use for 'what's behind Pomerium / Access SSO', 'which app "
            "protects hostname X'. Per-app policies (groups allowed/"
            "denied) are omitted by default to keep payload small -- "
            "call `cloudflare_get_access_app_policies` with the app id "
            "to drill in. Returns `{account_id, apps: [{id, name, "
            "type, domain, session_duration, aud}], count, note}`."
        ),
        input_schema=_clean_schema({
            "type": "object",
            "properties": {
                "name_contains": {
                    "type": "string",
                    "description": "Optional case-insensitive substring filter on app name.",
                },
                "account_id": {
                    "type": "string",
                    "description": (
                        "Optional account override. Defaults to the "
                        "first non-personal account the token sees "
                        "(the configured org-account)."
                    ),
                },
            },
            "required": [],
        }),
        handler=_safe_wrap(_h_list_access_apps),
    ),
    MCPTool(
        name="cloudflare_get_access_app_policies",
        description=(
            "Get the include/exclude/require policies for a specific "
            "Zero-Trust Access app. Second hop after `list_access_apps` "
            "returns the matching app id. Returns `{account_id, app_id, "
            "policies: [{id, name, decision, precedence, include, "
            "exclude, require}], count}`. `decision` is one of `allow`, "
            "`deny`, `non_identity`, `bypass`; `include/exclude/require` "
            "are arrays of rule objects (groups, emails, IPs, etc.)."
        ),
        input_schema=_clean_schema({
            "type": "object",
            "properties": {
                "app_id": {
                    "type": "string",
                    "description": "Access app id (from cloudflare_list_access_apps).",
                },
                "account_id": {
                    "type": "string",
                    "description": "Optional account override.",
                },
            },
            "required": ["app_id"],
        }),
        handler=_safe_wrap(_h_get_access_app_policies),
    ),
]


def get_tool(name: str) -> MCPTool:
    for t in CLOUDFLARE_TOOLS:
        if t.name == name:
            return t
    raise KeyError(f"unknown cloudflare tool: {name}")


# --- fake backend (FR-012; integration tests) ----------------------
#
# Data path (b): the handlers ignore the `client` arg they are handed and
# instead reach module-internal state -- the bound `_config` plus the
# `_cf_get()` / `_default_account()` helpers that wrap httpx. To run them
# offline we (1) install a synthetic bound config so `_require_bound()`
# passes, and (2) monkeypatch `_cf_get` + `_default_account` with canned,
# shape-faithful stand-ins that return Cloudflare v4 "result" payloads
# without any network or token. `build_fake().teardown` restores the
# originals so the module is left untouched after the test.


# Canned account id used by the fake. 32-char lowercase hex, like a real
# Cloudflare account/zone id, but entirely synthetic.
_FAKE_ACCOUNT_ID = "00000000000000000000000000000acc"
_FAKE_ZONE_ID = "0000000000000000000000000000z0ne"


async def _fake_default_account(_cfg: _BoundConfig) -> str:
    return _FAKE_ACCOUNT_ID


async def _fake_cf_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Offline stand-in for `_cf_get`. Returns canned CF v4 envelopes keyed
    by the request path. No network, no token. Shapes mirror the subset of
    fields the handlers actually read."""
    result: Any

    # Order matters: most specific path fragments first.
    if path == "/accounts":
        result = [{"id": _FAKE_ACCOUNT_ID, "name": "Example Org"}]
    elif path == "/zones":
        result = [
            {
                "id": _FAKE_ZONE_ID,
                "name": "example.com",
                "status": "active",
                "paused": False,
                "plan": {"name": "Pro"},
                "account": {"name": "Example Org"},
            },
        ]
    elif path.endswith("/dns_records"):
        result = [
            {
                "id": "dns0000000000000000000000000000a",
                "name": "www.example.com",
                "type": "A",
                "content": "192.0.2.10",
                "ttl": 1,
                "proxied": True,
                "comment": "apex web",
            },
            {
                "id": "dns0000000000000000000000000000b",
                "name": "mail.example.com",
                "type": "MX",
                "content": "mail.example.com",
                "ttl": 3600,
                "proxied": False,
                "comment": "",
            },
        ]
    elif path.endswith("/firewall/rules"):
        result = [
            {
                "id": "fw00000000000000000000000000000a",
                "action": "block",
                "description": "block bad bots",
                "paused": False,
                "filter": {"expression": '(http.user_agent contains "badbot")'},
            },
        ]
    elif "/rulesets/phases/http_request_firewall_custom" in path:
        result = {
            "rules": [
                {
                    "id": "rs00000000000000000000000000000a",
                    "action": "managed_challenge",
                    "description": "challenge anonymizers",
                    "enabled": True,
                    "expression": "(ip.geoip.is_in_european_union)",
                },
            ],
        }
    elif "/rulesets/phases/http_request_cache_settings" in path:
        result = {
            "rules": [
                {
                    "id": "cr00000000000000000000000000000a",
                    "description": "cache api responses",
                    "expression": '(http.request.uri.path contains "/api/")',
                    "action": "set_cache_settings",
                    "enabled": True,
                },
            ],
        }
    elif path.endswith("/pagerules"):
        result = [
            {
                "id": "pr00000000000000000000000000000a",
                "targets": [
                    {"constraint": {"value": "example.com/old/*"}},
                ],
                "actions": [{"id": "forwarding_url"}],
                "status": "active",
                "priority": 1,
            },
        ]
    elif path.endswith("/access/apps"):
        result = [
            {
                "id": "app0000000000000000000000000000a",
                "name": "Internal Dashboard",
                "type": "self_hosted",
                "domain": "dash.example.com",
                "session_duration": "24h",
                "aud": "aud0000000000000000000000000000a",
            },
        ]
    elif "/access/apps/" in path and path.endswith("/policies"):
        result = [
            {
                "id": "pol0000000000000000000000000000a",
                "name": "Allow staff",
                "decision": "allow",
                "precedence": 1,
                "include": [{"email_domain": {"domain": "example.com"}}],
                "exclude": [],
                "require": [],
            },
        ]
    else:
        result = []

    return {
        "success": True,
        "errors": [],
        "messages": [],
        "result": result,
        "result_info": {},
    }


def build_fake():
    """Return a FakeMCP exposing the Cloudflare tools wired to an offline
    backend.

    Needs NO credentials and makes NO network calls. Installs a synthetic
    bound config and monkeypatches the HTTP helpers; the returned
    `FakeMCP.teardown` restores the prior module state.
    """
    from opsrag.mcp import cloudflare as _mod
    from opsrag.mcp._fake import FakeMCP

    prev_config = _mod._config
    prev_bound = _mod._bound
    prev_cf_get = _mod._cf_get
    prev_default_account = _mod._default_account

    _mod._config = _BoundConfig(
        token="fake-token",
        default_account_id=_FAKE_ACCOUNT_ID,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        max_rows=_DEFAULT_MAX_ROWS,
    )
    _mod._bound = True
    _mod._cf_get = _fake_cf_get
    _mod._default_account = _fake_default_account

    def _restore() -> None:
        _mod._config = prev_config
        _mod._bound = prev_bound
        _mod._cf_get = prev_cf_get
        _mod._default_account = prev_default_account

    # Handlers ignore the client arg for this family -> client=None.
    return FakeMCP(tools=list(CLOUDFLARE_TOOLS), client=None, teardown=_restore)
