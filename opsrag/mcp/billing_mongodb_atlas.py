"""MongoDB Atlas billing MCP connector — Billing category, read-only.

Read-only FinOps tools over the **MongoDB Atlas Administration API v2**
billing endpoints (org invoices + running cost). Atlas exposes actual
charges programmatically through `/orgs/{orgId}/invoices*` — no BigQuery
export required. All amounts come back from Atlas in **cents**; every
handler converts to dollars (`*Cents` / 100) and surfaces
`groupName` / `clusterName` / `sku` where the line item carries them.

Base URL: ``https://cloud.mongodb.com/api/atlas/v2``.
Versioned media type header (required on the invoice endpoints):
``Accept: application/vnd.atlas.2023-01-01+json``.

## Auth (both supported; OAuth2 preferred) — read at call time

- **OAuth2 Service Account (preferred).** ``OPSRAG_ATLAS_CLIENT_ID`` +
  ``OPSRAG_ATLAS_CLIENT_SECRET`` → POST
  ``https://cloud.mongodb.com/api/oauth/token`` with
  ``grant_type=client_credentials`` and HTTP Basic ``client_id:client_secret``.
  The returned bearer token is cached module-level for ~1h (minus a
  60s safety margin) and sent as ``Authorization: Bearer <tok>``.
- **API keys (fallback).** ``OPSRAG_ATLAS_PUBLIC_KEY`` +
  ``OPSRAG_ATLAS_PRIVATE_KEY`` → HTTP **Digest** auth
  (``httpx.DigestAuth``).

``OPSRAG_ATLAS_ORG_ID`` is **REQUIRED** (the organization the invoices
belong to). If neither credential pair is present, a ``bad_config``
error is raised.

## Read-only enforcement

Every tool issues an HTTP ``GET``. No ``POST`` (except the OAuth token
exchange, which mutates nothing), ``PUT``, ``PATCH`` or ``DELETE`` — no
invoice, org, or cluster mutation anywhere.

## Tool list (4 read-only)

| Tool                            | Endpoint                                   |
|---------------------------------|--------------------------------------------|
| `billing_atlas_list_invoices`   | GET `/orgs/{orgId}/invoices`               |
| `billing_atlas_get_invoice`     | GET `/orgs/{orgId}/invoices/{invoiceId}`   |
| `billing_atlas_pending_invoice` | GET `/orgs/{orgId}/invoices/pending`       |
| `billing_atlas_cost_per_project`| derived from a (pending/given) invoice     |
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from opsrag.mcp.gitlab import MCPTool

_log = logging.getLogger("opsrag.mcp.billing_mongodb_atlas")

BASE_URL = "https://cloud.mongodb.com/api/atlas/v2"
OAUTH_TOKEN_URL = "https://cloud.mongodb.com/api/oauth/token"
# Versioned media type — Atlas API v2 requires an explicit resource version.
_ACCEPT = "application/vnd.atlas.2023-01-01+json"

_DEFAULT_TIMEOUT_S = 30.0
_DEFAULT_LIMIT = 25
_MAX_ITEMS_PER_PAGE = 100
# Refresh the OAuth token this many seconds before its stated expiry.
_TOKEN_SKEW_S = 60.0

# Module-level bearer-token cache (OAuth2 service-account flow). Reset by
# build_fake() so offline tests never touch it.
_token_cache: dict[str, Any] = {"token": None, "expires_at": 0.0}


class AtlasBillingMCPError(Exception):
    """Read-only Atlas billing tool failure. Carries a short ``reason`` code
    (``bad_config`` / ``bad_args`` / ``auth`` / ``http``) and, where relevant,
    the upstream HTTP ``status``."""

    def __init__(self, message: str, *, reason: str = "error", status: int | None = None) -> None:
        super().__init__(message)
        self.reason = reason
        self.status = status


# --- config (Helm values via the bound config block; env vars are fallback) --
#
# `org_id` + the credential env-var NAMES come from the config block (Helm
# values -> config.yaml). The credential VALUES are always read from those env
# vars (never from values), so secrets stay in secrets.
_BOUND: Any | None = None


def bind(cfg: Any | None = None) -> None:
    """Register the billing_mongodb_atlas config block (or None to clear)."""
    global _BOUND
    _BOUND = cfg


def _env_name(field: str, default: str) -> str:
    return str(getattr(_BOUND, field, None) or default) if _BOUND is not None else default


def _org_id() -> str:
    org = (str(getattr(_BOUND, "org_id", None) or "").strip()
           if _BOUND is not None else "")
    if not org:
        org = (os.environ.get("OPSRAG_ATLAS_ORG_ID") or "").strip()
    if not org:
        raise AtlasBillingMCPError(
            "billing_mongodb_atlas org_id is not configured. Set "
            "`mcp.billing_mongodb_atlas.org_id` in Helm values (or "
            "OPSRAG_ATLAS_ORG_ID) to the Atlas organization id.",
            reason="bad_config",
        )
    return org


def _oauth_creds() -> tuple[str, str] | None:
    cid = (os.environ.get(_env_name("client_id_env", "OPSRAG_ATLAS_CLIENT_ID")) or "").strip()
    csec = (os.environ.get(_env_name("client_secret_env", "OPSRAG_ATLAS_CLIENT_SECRET")) or "").strip()
    return (cid, csec) if cid and csec else None


def _api_keys() -> tuple[str, str] | None:
    pub = (os.environ.get(_env_name("public_key_env", "OPSRAG_ATLAS_PUBLIC_KEY")) or "").strip()
    priv = (os.environ.get(_env_name("private_key_env", "OPSRAG_ATLAS_PRIVATE_KEY")) or "").strip()
    return (pub, priv) if pub and priv else None


# --- pure helpers (unit-testable) ------------------------------------------

def _clamp_items(n: Any, default: int = _DEFAULT_LIMIT) -> int:
    """Clamp an items-per-page value to [1, 100] (Atlas' hard page ceiling)."""
    try:
        v = int(n)
    except (TypeError, ValueError):
        return default
    return max(1, min(v, _MAX_ITEMS_PER_PAGE))


def _cents_to_dollars(cents: Any) -> float | None:
    """Atlas returns money in integer cents. Convert to rounded dollars."""
    if cents is None:
        return None
    try:
        return round(int(cents) / 100.0, 2)
    except (TypeError, ValueError):
        return None


def _compact_line_item(li: dict) -> dict:
    """Trim one invoice line item to the fields agents reason over. Surfaces
    `sku`, `groupName`/`clusterName`, and cents→dollars price."""
    return {
        "sku": li.get("sku"),
        "groupName": li.get("groupName"),
        "clusterName": li.get("clusterName"),
        "totalPriceDollars": _cents_to_dollars(li.get("totalPriceCents")),
        "quantity": li.get("quantity"),
        "unit": li.get("unit"),
    }


def _compact_invoice(inv: dict) -> dict:
    """Compact a full invoice object (header fields + compacted line items)."""
    line_items = [_compact_line_item(li) for li in (inv.get("lineItems") or [])]
    return {
        "id": inv.get("id"),
        "statusName": inv.get("statusName"),
        "amountBilledDollars": _cents_to_dollars(inv.get("amountBilledCents")),
        "amountPaidDollars": _cents_to_dollars(inv.get("amountPaidCents")),
        "startDate": inv.get("startDate"),
        "endDate": inv.get("endDate"),
        "lineItemCount": len(line_items),
        "lineItems": line_items,
    }


# --- OAuth token (module-level; swappable seam) ----------------------------

async def _fetch_oauth_token(client_id: str, client_secret: str) -> tuple[str, float]:
    """POST the OAuth client-credentials exchange. Returns
    ``(access_token, expires_in_seconds)``. Network seam — build_fake swaps it."""
    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT_S) as http:
        resp = await http.post(
            OAUTH_TOKEN_URL,
            auth=(client_id, client_secret),  # HTTP Basic client_id:client_secret
            data={"grant_type": "client_credentials"},
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
        )
    if resp.status_code >= 400:
        raise AtlasBillingMCPError(
            f"Atlas OAuth token request failed: {resp.status_code}: {resp.text[:300]}",
            reason="auth",
            status=resp.status_code,
        )
    data = resp.json() if resp.text else {}
    token = data.get("access_token")
    if not token:
        raise AtlasBillingMCPError(
            "Atlas OAuth token response had no access_token.", reason="auth"
        )
    return token, float(data.get("expires_in", 3600) or 3600)


async def _bearer_token(client_id: str, client_secret: str) -> str:
    """Return a cached OAuth bearer token, refreshing when near expiry."""
    now = time.time()
    if _token_cache["token"] and _token_cache["expires_at"] > now:
        return _token_cache["token"]
    token, expires_in = await _fetch_oauth_token(client_id, client_secret)
    _token_cache["token"] = token
    _token_cache["expires_at"] = now + max(0.0, expires_in - _TOKEN_SKEW_S)
    return token


# --- request choke-point (swapped by build_fake) ---------------------------

async def _request(
    method: str, path: str, *, params: dict | None = None, tool: str = "billing_atlas"
) -> Any:
    """Issue one read-only Atlas Admin API request. THE single network seam —
    build_fake swaps it (and `_get`). Resolves auth per call: OAuth2 bearer
    (preferred) else API-key Digest; raises ``bad_config`` if neither is set."""
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    oauth = _oauth_creds()
    keys = _api_keys()
    if not oauth and not keys:
        raise AtlasBillingMCPError(
            "No MongoDB Atlas credentials set. Provide OPSRAG_ATLAS_CLIENT_ID + "
            "OPSRAG_ATLAS_CLIENT_SECRET (OAuth2 service account, preferred), or "
            "OPSRAG_ATLAS_PUBLIC_KEY + OPSRAG_ATLAS_PRIVATE_KEY (API-key Digest).",
            reason="bad_config",
        )

    headers = {"Accept": _ACCEPT}
    auth: httpx.Auth | None = None
    if oauth:
        token = await _bearer_token(*oauth)
        headers["Authorization"] = f"Bearer {token}"
    else:
        auth = httpx.DigestAuth(*keys)

    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT_S, headers=headers, auth=auth) as http:
        resp = await http.request(method, f"{BASE_URL}{path}", params=clean)
    if resp.status_code >= 400:
        raise AtlasBillingMCPError(
            f"[{tool}] {resp.status_code}: {resp.text[:300]}",
            reason="http",
            status=resp.status_code,
        )
    return resp.json() if resp.text else {}


async def _get(path: str, params: dict | None = None, *, tool: str = "billing_atlas") -> Any:
    return await _request("GET", path, params=params, tool=tool)


# --- handlers --------------------------------------------------------------

async def _h_list_invoices(_unused, args: dict) -> Any:
    """List the org's invoices (most recent first as Atlas returns them),
    compacted to id / status / billed dollars / period."""
    org = _org_id()
    limit = _clamp_items(args.get("limit"))
    params = {
        "pageNum": 1,
        "itemsPerPage": limit,
        # Don't fan out into linked (cross-org) invoices — keep it to this org.
        "viewLinkedInvoices": "false",
    }
    resp = await _get(f"/orgs/{org}/invoices", params=params, tool="billing_atlas_list_invoices")
    results = resp.get("results") or []
    out = [
        {
            "id": inv.get("id"),
            "statusName": inv.get("statusName"),
            "amountBilledDollars": _cents_to_dollars(inv.get("amountBilledCents")),
            "startDate": inv.get("startDate"),
            "endDate": inv.get("endDate"),
        }
        for inv in results[:limit]
    ]
    return {
        "orgId": org,
        "count": len(out),
        "invoices": out,
        "currency": "USD",
        "note": "Amounts converted from Atlas cents. Read-only.",
    }


async def _h_get_invoice(_unused, args: dict) -> Any:
    """One invoice in full — header + compacted line items (sku, group/cluster,
    price in dollars, quantity, unit)."""
    org = _org_id()
    invoice_id = (args.get("invoice_id") or "").strip()
    if not invoice_id:
        raise AtlasBillingMCPError("invoice_id is required.", reason="bad_args")
    resp = await _get(
        f"/orgs/{org}/invoices/{invoice_id}", tool="billing_atlas_get_invoice"
    )
    out = _compact_invoice(resp)
    out["orgId"] = org
    out["currency"] = "USD"
    return out


async def _h_pending_invoice(_unused, args: dict) -> Any:
    """The current unbilled ("pending") invoice — the running cost for the
    period in progress. Same compact shape as `billing_atlas_get_invoice`."""
    org = _org_id()
    resp = await _get(
        f"/orgs/{org}/invoices/pending", tool="billing_atlas_pending_invoice"
    )
    out = _compact_invoice(resp)
    out["orgId"] = org
    out["currency"] = "USD"
    out["note"] = "Pending (unbilled) invoice — running cost for the current period."
    return out


async def _h_cost_per_project(_unused, args: dict) -> Any:
    """Per-project (Atlas *group*) cost, derived in Python from an invoice's
    line items grouped by `groupName`. Defaults to the pending invoice; pass
    `invoice_id` to break down a specific one instead."""
    org = _org_id()
    invoice_id = (args.get("invoice_id") or "").strip()
    if invoice_id:
        resp = await _get(
            f"/orgs/{org}/invoices/{invoice_id}", tool="billing_atlas_cost_per_project"
        )
    else:
        resp = await _get(
            f"/orgs/{org}/invoices/pending", tool="billing_atlas_cost_per_project"
        )

    totals: dict[str, int] = {}
    for li in resp.get("lineItems") or []:
        name = li.get("groupName") or "(no project)"
        try:
            cents = int(li.get("totalPriceCents") or 0)
        except (TypeError, ValueError):
            cents = 0
        totals[name] = totals.get(name, 0) + cents

    by_project = [
        {"groupName": name, "cost_dollars": round(cents / 100.0, 2)}
        for name, cents in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    ]
    return {
        "orgId": org,
        "invoiceId": resp.get("id"),
        "statusName": resp.get("statusName"),
        "count": len(by_project),
        "by_project": by_project,
        "currency": "USD",
        "note": "Per-project cost grouped from invoice line items (Atlas cents → USD).",
    }


# --- tool specs ------------------------------------------------------------

BILLING_ATLAS_TOOLS: list[MCPTool] = [
    MCPTool(
        name="billing_atlas_list_invoices",
        description=(
            "List MongoDB Atlas organization invoices (id, status, amount billed "
            "in USD, billing period). Read-only. `limit` clamps items-per-page "
            "(max 100). Use `billing_atlas_get_invoice` to drill into line items."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max invoices to return (default 25, max 100).",
                },
            },
        },
        handler=_h_list_invoices,
    ),
    MCPTool(
        name="billing_atlas_get_invoice",
        description=(
            "One Atlas invoice in full: header plus line items (sku, "
            "group/cluster, price in USD, quantity, unit). Amounts converted "
            "from Atlas cents. Read-only."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "invoice_id": {
                    "type": "string",
                    "description": "Atlas invoice id (from billing_atlas_list_invoices).",
                },
            },
            "required": ["invoice_id"],
        },
        handler=_h_get_invoice,
    ),
    MCPTool(
        name="billing_atlas_pending_invoice",
        description=(
            "The current unbilled (pending) Atlas invoice — running cost for the "
            "in-progress billing period, with line items in USD. Use this for "
            "'how much are we spending on Atlas this month so far?'. Read-only."
        ),
        input_schema={"type": "object", "properties": {}},
        handler=_h_pending_invoice,
    ),
    MCPTool(
        name="billing_atlas_cost_per_project",
        description=(
            "Per-project (Atlas group) cost for an invoice, grouped from its line "
            "items by `groupName` and summed in USD. Defaults to the pending "
            "invoice; pass `invoice_id` to break down a specific finalized one. "
            "Read-only."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "invoice_id": {
                    "type": "string",
                    "description": "Optional Atlas invoice id. Defaults to the pending invoice.",
                },
            },
        },
        handler=_h_cost_per_project,
    ),
]


def get_tool(name: str) -> MCPTool:
    for t in BILLING_ATLAS_TOOLS:
        if t.name == name:
            return t
    raise KeyError(f"unknown billing_mongodb_atlas tool: {name}")


# --- fake backend (FR-012; offline tests) ----------------------------------
#
# Atlas handlers reach the module-level `_get` (which calls `_request`, which
# resolves OAuth/Digest auth from env). The offline fake swaps `_get` /
# `_request` for a canned path-keyed dispatcher AND short-circuits the OAuth
# token fetch — so tests need NO Atlas creds and touch NO network. The bearer
# cache is cleared so a stale token can't leak between runs.

# Two line items across two projects; cents sum to amountBilledCents (166000).
_FAKE_LINE_ITEMS = [
    {
        "sku": "ATLAS_AWS_INSTANCE_M30",
        "groupName": "prod",
        "clusterName": "prod-cluster-0",
        "totalPriceCents": 120000,
        "quantity": 720,
        "unit": "hours",
    },
    {
        "sku": "ATLAS_AWS_DATA_TRANSFER",
        "groupName": "staging",
        "clusterName": "stg-cluster-0",
        "totalPriceCents": 46000,
        "quantity": 100,
        "unit": "GB",
    },
]


def _fake_invoice() -> dict:
    return {
        "id": "inv1",
        "statusName": "PENDING",
        "amountBilledCents": 166000,
        "startDate": "2026-07-01",
        "endDate": "2026-08-01",
        "lineItems": [dict(li) for li in _FAKE_LINE_ITEMS],
    }


async def _fake_get(path: str, params: dict | None = None, *, tool: str = "billing_atlas") -> Any:
    """Canned stand-in for the module-level GET, routed by path fragment to a
    response shaped like the real Atlas invoice endpoints."""
    if path.endswith("/invoices"):
        return {"results": [_fake_invoice()], "totalCount": 1}
    if path.endswith("/invoices/pending"):
        return _fake_invoice()
    if "/invoices/" in path:  # /invoices/{invoiceId}
        return _fake_invoice()
    return {}


def build_fake():
    """Return a FakeMCP exposing the Atlas billing tools wired to an offline
    backend. Needs NO Atlas creds / network: `_get` / `_request` are swapped
    for a canned dispatcher, the OAuth token fetch is short-circuited, and the
    bearer cache is reset. `teardown` restores everything."""
    import opsrag.mcp.billing_mongodb_atlas as _mod
    from opsrag.mcp._fake import FakeMCP

    _orig_get = _mod._get
    _orig_request = _mod._request
    _orig_fetch = _mod._fetch_oauth_token
    _orig_cache = dict(_mod._token_cache)

    async def _fake_request(method, path, *, params=None, tool="billing_atlas"):
        return await _fake_get(path, params, tool=tool)

    async def _fake_fetch_oauth_token(client_id, client_secret):
        return ("fake-atlas-token", 3600.0)

    _mod._get = _fake_get
    _mod._request = _fake_request
    _mod._fetch_oauth_token = _fake_fetch_oauth_token
    _mod._token_cache["token"] = None
    _mod._token_cache["expires_at"] = 0.0

    def _restore() -> None:
        _mod._get = _orig_get
        _mod._request = _orig_request
        _mod._fetch_oauth_token = _orig_fetch
        _mod._token_cache.clear()
        _mod._token_cache.update(_orig_cache)

    return FakeMCP(tools=list(BILLING_ATLAS_TOOLS), client=None, teardown=_restore)
