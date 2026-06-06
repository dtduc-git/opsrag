"""Rootly Web API client -- async, paginated, JSON:API style.

Read-only. Endpoints used:
  - GET /v1/incidents       -- list with filters + relationship includes
  - GET /v1/incidents/{id}  -- detail (only when relationships are needed
                              that listing didn't return)
  - GET /v1/post_mortems    -- list, joined to incidents by incident_id

Cloudflare quirk: requests with the default Python user-agent get a
1010 challenge. We always send a curl-style UA.

Pagination: JSON:API standard -- `meta.total_pages` + `links.next`.
We walk via `page[number]` because `links.next` is sometimes returned
as a relative path that needs base-URL handling.

Permission quirk: Rootly returns 404 (not 401/403) when a token is
authenticated but lacks scope for an endpoint. Caller handles 404 as
"resource missing OR no permission" -- usually fatal during startup
checks.

Rate limits: 3000 req/min/account per the X-RateLimit-Limit header.
With per_page=100 and 168 incidents in this workspace, a full backfill
is ~3-5 requests -- well under the ceiling.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

_log = logging.getLogger("opsrag.rootly.client")

_BASE = "https://api.rootly.com/v1"


@dataclass(frozen=True)
class Incident:
    id: str
    sequential_id: int | None
    title: str
    status: str
    summary: str
    slug: str
    url: str
    started_at: datetime | None
    detected_at: datetime | None
    acknowledged_at: datetime | None
    mitigated_at: datetime | None
    resolved_at: datetime | None
    closed_at: datetime | None
    cancellation_message: str
    mitigation_message: str
    resolution_message: str
    slack_channel_name: str
    severity_name: str
    private: bool
    services: tuple[str, ...]
    environments: tuple[str, ...]
    teams: tuple[str, ...]
    incident_types: tuple[str, ...]
    causes: tuple[str, ...]
    labels: tuple[str, ...]
    updated_at: datetime | None


@dataclass(frozen=True)
class PostMortem:
    id: str
    incident_id: str
    title: str
    status: str           # "draft" | "published" | "in_review" | ...
    content_html: str
    url: str
    published_at: datetime | None
    updated_at: datetime | None


class RootlyClient:
    def __init__(
        self,
        api_token: str,
        *,
        base_url: str = _BASE,
        max_retries: int = 3,
        retry_base_seconds: float = 2.0,
    ):
        if not api_token or not api_token.startswith("rootly_"):
            raise ValueError("ROOTLY_API_TOKEN must start with 'rootly_'")
        self._token = api_token
        self._max_retries = max_retries
        self._retry_base = retry_base_seconds
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_token}",
                "Accept": "application/json",
                # Cloudflare blocks default Python/httpx UAs on Rootly's
                # edge. A curl-style UA is the lowest-friction fix.
                "User-Agent": "opsrag-connector/0.1 (+https://example.com/opsrag)",
            },
            timeout=httpx.Timeout(45.0, connect=15.0),
        )

    async def close(self) -> None:
        await self._http.aclose()

    # -- core request ------------------------------------------------
    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = await self._http.get(path, params=params or {})
            except httpx.RequestError as exc:
                if attempt > self._max_retries:
                    raise
                wait = self._retry_base * (2 ** (attempt - 1))
                _log.warning("rootly %s transport error: %s -- retry %d in %.1fs", path, exc, attempt, wait)
                await asyncio.sleep(wait)
                continue

            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", self._retry_base * (2 ** (attempt - 1))))
                _log.warning("rootly %s rate-limited -- sleeping %.1fs", path, wait)
                await asyncio.sleep(wait)
                if attempt > self._max_retries:
                    resp.raise_for_status()
                continue

            if resp.status_code >= 500 and attempt <= self._max_retries:
                wait = self._retry_base * (2 ** (attempt - 1))
                _log.warning("rootly %s status=%d -- retry %d in %.1fs", path, resp.status_code, attempt, wait)
                await asyncio.sleep(wait)
                continue

            if resp.status_code == 404:
                # Rootly's signal for "no permission OR genuinely
                # missing". Surface verbatim so factory init can
                # warn-and-skip rather than crash startup.
                raise RuntimeError(
                    f"rootly {path} returned 404 -- token may lack scope "
                    f"(incidents:read / post_mortems:read / etc.)"
                )

            resp.raise_for_status()
            return resp.json()

    # -- auth / probe ------------------------------------------------
    async def whoami(self) -> dict[str, Any]:
        return await self._get("/users/me")

    # -- incidents ---------------------------------------------------
    async def list_incidents(
        self,
        *,
        statuses: tuple[str, ...] = ("resolved", "mitigated"),
        updated_since: datetime | None = None,
    ) -> AsyncIterator[Incident]:
        """Yield incidents matching `statuses`. Filtering is done client
        side because Rootly's multi-value filter syntax is inconsistent
        across endpoints -- and the volume is tiny (~hundreds total).

        `updated_since` enables daily delta runs. Server-side filter
        keeps the response small.
        """
        keep = set(statuses)
        page = 1
        per_page = 100
        params: dict[str, Any] = {
            "page[size]": per_page,
            # `include` lets us pull severity / services / teams /
            # environments inline as `included[]`, avoiding N+1 follow-
            # up calls per incident.
            "include": "severity,services,teams,environments,incident_types,causes",
            "sort": "-updated_at",
        }
        if updated_since is not None:
            # Rootly accepts ISO-8601 with timezone here.
            params["filter[updated_at_gte]"] = updated_since.astimezone(UTC).isoformat()

        while True:
            params["page[number]"] = page
            data = await self._get("/incidents", params)
            included = _index_included(data.get("included") or [])
            rows = data.get("data") or []
            for raw in rows:
                inc = _to_incident(raw, included)
                if inc.status not in keep:
                    continue
                yield inc
            total_pages = (data.get("meta") or {}).get("total_pages") or 1
            if page >= total_pages or not rows:
                break
            page += 1

    async def list_post_mortems(
        self,
        *,
        statuses: tuple[str, ...] = ("published",),
        updated_since: datetime | None = None,
    ) -> AsyncIterator[PostMortem]:
        """Yield post-mortems matching `statuses`. Default is published
        only; drafts are still in flight and would noise up retrieval.
        Pass `("published", "draft", "in_review")` for full coverage.
        """
        keep = set(statuses)
        page = 1
        per_page = 100
        params: dict[str, Any] = {
            "page[size]": per_page,
            # `/v1/post_mortems` rejects `sort=-updated_at` (400) but
            # accepts `created_at`. We use it descending so the
            # newest-superseding-older behaviour in `_load_post_mortems`
            # still picks the right one when multiple exist per incident.
            "sort": "-created_at",
        }
        if updated_since is not None:
            params["filter[updated_at_gte]"] = updated_since.astimezone(UTC).isoformat()
        while True:
            params["page[number]"] = page
            data = await self._get("/post_mortems", params)
            rows = data.get("data") or []
            for raw in rows:
                pm = _to_post_mortem(raw)
                if pm.status not in keep:
                    continue
                yield pm
            total_pages = (data.get("meta") or {}).get("total_pages") or 1
            if page >= total_pages or not rows:
                break
            page += 1


# -- helpers -----------------------------------------------------------
def _parse_dt(val: Any) -> datetime | None:
    if not val:
        return None
    try:
        return datetime.fromisoformat(str(val).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _index_included(included: list[dict]) -> dict[tuple[str, str], dict]:
    """Build a `(type, id) -> attributes` lookup from JSON:API `included`."""
    out: dict[tuple[str, str], dict] = {}
    for item in included:
        t = item.get("type")
        i = item.get("id")
        if t and i:
            out[(t, i)] = item.get("attributes") or {}
    return out


def _names_from_relationships(
    rels: dict | None,
    rel_name: str,
    rel_type: str,
    included: dict[tuple[str, str], dict],
    field: str = "name",
) -> tuple[str, ...]:
    """Resolve a JSON:API relationship to a tuple of names.

    `rels[rel_name].data` is either a single dict, list of dicts, or
    None. Each dict has `id` + `type`; we look up `(type, id)` in the
    included index and pull `attributes[field]`.
    """
    if not rels:
        return ()
    block = rels.get(rel_name) or {}
    data = block.get("data")
    if data is None:
        return ()
    items = data if isinstance(data, list) else [data]
    out: list[str] = []
    for it in items:
        rid = it.get("id")
        attrs = included.get((rel_type, rid), {}) if rid else {}
        name = attrs.get(field)
        if name:
            out.append(name)
    return tuple(out)


def _to_incident(raw: dict, included: dict[tuple[str, str], dict]) -> Incident:
    a = raw.get("attributes") or {}
    rels = raw.get("relationships") or {}
    sev_names = _names_from_relationships(rels, "severity", "severities", included)
    return Incident(
        id=raw.get("id", ""),
        sequential_id=a.get("sequential_id"),
        title=a.get("title") or "",
        status=a.get("status") or "",
        summary=a.get("summary") or "",
        slug=a.get("slug") or "",
        url=a.get("url") or "",
        started_at=_parse_dt(a.get("started_at")),
        detected_at=_parse_dt(a.get("detected_at")),
        acknowledged_at=_parse_dt(a.get("acknowledged_at")),
        mitigated_at=_parse_dt(a.get("mitigated_at")),
        resolved_at=_parse_dt(a.get("resolved_at")),
        closed_at=_parse_dt(a.get("closed_at")),
        cancellation_message=a.get("cancellation_message") or "",
        mitigation_message=a.get("mitigation_message") or "",
        resolution_message=a.get("resolution_message") or "",
        slack_channel_name=a.get("slack_channel_name") or "",
        severity_name=sev_names[0] if sev_names else "",
        private=bool(a.get("private")),
        services=_names_from_relationships(rels, "services", "services", included),
        environments=_names_from_relationships(rels, "environments", "environments", included),
        teams=_names_from_relationships(rels, "teams", "teams", included),
        incident_types=_names_from_relationships(rels, "incident_types", "incident_types", included),
        causes=_names_from_relationships(rels, "causes", "causes", included),
        labels=tuple(a.get("labels") or ()),
        updated_at=_parse_dt(a.get("updated_at")),
    )


def _to_post_mortem(raw: dict) -> PostMortem:
    a = raw.get("attributes") or {}
    rels = raw.get("relationships") or {}
    incident_id = ((rels.get("incident") or {}).get("data") or {}).get("id") \
        or a.get("incident_id") or ""
    return PostMortem(
        id=raw.get("id", ""),
        incident_id=incident_id,
        title=a.get("title") or "",
        status=a.get("status") or "",
        content_html=a.get("content") or "",
        url=a.get("url") or "",
        published_at=_parse_dt(a.get("published_at")),
        updated_at=_parse_dt(a.get("updated_at")),
    )
