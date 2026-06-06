"""Async Confluence Cloud REST client.

Talks directly to `/wiki/api/v2/...` (and a couple of v1 fallbacks where
v2 doesn't expose a needed field). No LangChain wrapper, no
mcp-atlassian dependency -- same project rule as the rest of OpsRAG.

Auth: HTTP basic with `email + api_token`. Atlassian doesn't offer a
machine OAuth flow that's simpler than this for batch ingestion, so we
use a service-account token. The token must NEVER be logged -- only the
email is exposed in info logs.

Rate limiting: Atlassian Cloud caps at ~5000 req/hr/IP. With our
`fetch_concurrency=5` default and ~500 pages/space, a full crawl uses
~600 requests (1 list-spaces + N list-pages-pages + N get-page) so
we're an order of magnitude under the ceiling. Retry-on-429 with
exponential backoff handles transient spikes regardless.
"""
from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

_log = logging.getLogger("opsrag.confluence.client")

# v2 surface -- paginated, ADF-native, returns clean JSON.
_V2_BASE = "/wiki/api/v2"


@dataclass
class Space:
    """A Confluence space -- a top-level grouping of pages."""

    id: str
    key: str           # e.g. "SRE" -- used as `scope` in DocRef
    name: str
    type: str          # "global" | "personal" | "collaboration"
    homepage_id: str | None = None


@dataclass
class PageRef:
    """Lightweight pointer to a page returned by `list_pages`.

    Body is NOT included -- fetch via `get_page`. Pagination yields
    these cheaply so the indexing pipeline can collect a full list
    before deciding what to fetch.
    """

    id: str
    title: str
    space_id: str
    space_key: str | None = None    # may be unset until resolved
    status: str = "current"
    parent_id: str | None = None
    last_modified: datetime | None = None
    version: int | None = None


@dataclass
class Page:
    """A fully-fetched page with body + metadata."""

    id: str
    title: str
    space_id: str
    space_key: str
    status: str
    version: int
    last_modified: datetime
    body_adf: dict                  # the raw Atlassian Document Format JSON
    url: str
    labels: list[str] = field(default_factory=list)
    ancestors: list[str] = field(default_factory=list)
    parent_id: str | None = None


class ConfluenceClient:
    """Lightweight async client for Atlassian Cloud Confluence v2.

    Constructed via factory `from_config(cfg.confluence)` so the
    auth resolution from env vars happens in one place.
    """

    def __init__(
        self,
        base_url: str,
        email: str,
        api_token: str,
        *,
        max_retries: int = 3,
        retry_base_seconds: float = 2.0,
        timeout_seconds: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._max_retries = max(1, max_retries)
        self._retry_base = max(0.5, retry_base_seconds)
        # Allow tests to inject a mocked AsyncClient.
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url,
            auth=(email, api_token),
            timeout=timeout_seconds,
            headers={"Accept": "application/json"},
        )
        self._owns_client = client is None

    @classmethod
    def from_config(cls, cfg: Any) -> ConfluenceClient:
        """Build from a `ConfluenceConfig` -- resolves email + token from env."""
        email = cfg.email or os.environ.get(cfg.email_env, "")
        token = cfg.api_token or os.environ.get(cfg.api_token_env, "")
        if not email or not token:
            raise RuntimeError(
                "Confluence auth missing: set "
                f"{cfg.email_env} and {cfg.api_token_env}, or "
                "ConfluenceConfig.email / .api_token"
            )
        return cls(
            base_url=cfg.base_url,
            email=email,
            api_token=token,
            max_retries=cfg.max_retries,
            retry_base_seconds=cfg.retry_base_seconds,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> ConfluenceClient:
        return self

    async def __aexit__(self, *_):
        await self.close()

    # -- HTTP layer with retry-on-429 --

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json_body: dict | None = None,
    ) -> dict:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = await self._client.request(
                    method, path, params=params, json=json_body
                )
            except httpx.RequestError as exc:
                last_exc = exc
                # Network-level error -- retry with backoff.
                await asyncio.sleep(self._retry_base * (2 ** attempt))
                continue

            if resp.status_code == 429:
                retry_after = float(
                    resp.headers.get("Retry-After")
                    or self._retry_base * (2 ** attempt)
                )
                _log.info(
                    "confluence rate-limited, retry_after=%.1fs attempt=%d/%d path=%s",
                    retry_after, attempt + 1, self._max_retries, path,
                )
                await asyncio.sleep(retry_after)
                continue

            if resp.status_code >= 400:
                # 4xx error messages from Atlassian are useful for triage
                # (`unknown query parameter`, `invalid sort`, etc.) and
                # don't carry page content. Surface them, but cap length
                # so a runaway HTML error page doesn't flood logs.
                detail = (resp.text or "")[:400]
                _log.warning(
                    "confluence %s %s -> %d: %s",
                    method, path, resp.status_code, detail,
                )
                resp.raise_for_status()

            return resp.json() if resp.content else {}

        # Exhausted retries.
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(
            f"confluence: exhausted {self._max_retries} retries on {method} {path}"
        )

    # -- Spaces --

    async def list_spaces(
        self,
        *,
        types: tuple[str, ...] = ("global",),
    ) -> list[Space]:
        """All spaces visible to the authenticated user, optionally filtered
        by type. Defaults to `("global",)` -- explicitly excludes personal
        `~username` spaces, which we never want to ingest.
        """
        spaces: list[Space] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"limit": 100}
            if types:
                params["type"] = ",".join(types)
            if cursor:
                params["cursor"] = cursor
            data = await self._request("GET", f"{_V2_BASE}/spaces", params=params)
            for s in data.get("results", []):
                spaces.append(Space(
                    id=str(s.get("id", "")),
                    key=s.get("key", ""),
                    name=s.get("name", ""),
                    type=s.get("type", ""),
                    homepage_id=str(s["homepageId"]) if s.get("homepageId") else None,
                ))
            cursor = (data.get("_links") or {}).get("next")
            if not cursor:
                break
            # Atlassian sometimes returns the cursor as a full URL -- keep
            # only the query-param value if so.
            if "cursor=" in cursor:
                cursor = cursor.split("cursor=", 1)[1].split("&", 1)[0]
            else:
                # Unexpected format -- bail to avoid an infinite loop.
                break
        return spaces

    async def list_pages(
        self,
        space_id: str,
        *,
        space_key: str | None = None,
        updated_since: datetime | None = None,
        status: str = "current",
        limit_per_page: int = 100,
    ) -> AsyncIterator[PageRef]:
        """Stream `PageRef`s in a space, paginated.

        `updated_since` filters server-side via the v2 API's `body-format=none`
        + client-side comparison (v2 has no native `updatedSince` for page
        listing as of writing, so we compare `version.createdAt` here).
        """
        cursor: str | None = None
        while True:
            # NB: v2 page-listing has no `body-format=none` value -- the
            # endpoint just doesn't return a body unless body-format is
            # specified. Don't pass it.
            params: dict[str, Any] = {
                "limit": limit_per_page,
                "status": status,
                "sort": "-modified-date",
            }
            if cursor:
                params["cursor"] = cursor
            data = await self._request(
                "GET",
                f"{_V2_BASE}/spaces/{space_id}/pages",
                params=params,
            )
            stop = False
            for p in data.get("results", []):
                version = (p.get("version") or {}).get("number")
                modified_str = (p.get("version") or {}).get("createdAt") or ""
                modified = _parse_iso8601(modified_str)
                if updated_since and modified and modified <= updated_since:
                    # Sort is descending by modified; once we hit a page
                    # older than the cutoff, every following page is too.
                    stop = True
                    break
                yield PageRef(
                    id=str(p.get("id", "")),
                    title=p.get("title", ""),
                    space_id=space_id,
                    space_key=space_key,
                    status=p.get("status", "current"),
                    parent_id=str(p["parentId"]) if p.get("parentId") else None,
                    last_modified=modified,
                    version=int(version) if version is not None else None,
                )
            if stop:
                return
            cursor = (data.get("_links") or {}).get("next")
            if not cursor:
                return
            if "cursor=" in cursor:
                cursor = cursor.split("cursor=", 1)[1].split("&", 1)[0]
            else:
                return

    # -- Page body --

    async def get_page(
        self,
        page_id: str,
        *,
        body_format: str = "atlas_doc_format",
    ) -> Page:
        """Fetch one page including body + metadata.

        `body_format`:
          - `atlas_doc_format` (default) -- ADF JSON, our target.
          - `storage` -- XHTML-ish; harder to render to clean Markdown.
          - `view` -- already-rendered HTML; lossy for code/tables.
        """
        data = await self._request(
            "GET",
            f"{_V2_BASE}/pages/{page_id}",
            params={
                "body-format": body_format,
                "include-labels": "true",
            },
        )
        body = data.get("body") or {}
        adf_value = (body.get(body_format) or {}).get("value", "")
        # The ADF body comes through as a JSON-encoded string in v2 --
        # parse it once here so callers always see a dict.
        adf: dict
        if isinstance(adf_value, str):
            import json
            try:
                adf = json.loads(adf_value) if adf_value else {}
            except json.JSONDecodeError:
                _log.warning(
                    "confluence: unparseable ADF body page_id=%s len=%d",
                    page_id, len(adf_value),
                )
                adf = {}
        else:
            adf = adf_value or {}

        version = (data.get("version") or {}).get("number") or 0
        modified = _parse_iso8601(
            (data.get("version") or {}).get("createdAt") or ""
        )
        labels_block = data.get("labels") or {}
        labels = [
            str(lb.get("name", ""))
            for lb in (labels_block.get("results") or [])
            if lb.get("name")
        ]
        space_id = str(data.get("spaceId", ""))
        # v2 returns `_links.webui` as the path; combine with base URL.
        webui = ((data.get("_links") or {}).get("webui")) or ""
        url = f"{self._base_url}{webui}" if webui else ""

        return Page(
            id=str(data.get("id", page_id)),
            title=data.get("title", ""),
            space_id=space_id,
            space_key="",  # caller can fill from list_spaces lookup
            status=data.get("status", "current"),
            version=int(version),
            last_modified=modified or datetime.utcnow(),
            body_adf=adf,
            url=url,
            labels=labels,
            ancestors=[],   # Filled separately if needed (extra round-trip)
            parent_id=str(data["parentId"]) if data.get("parentId") else None,
        )

    async def resolve_page_id_from_url(self, url: str) -> str | None:
        """Map an Atlassian Confluence URL to a numeric page_id.

        Handles two formats the organization's runbook URLs use:
          - Canonical: `.../wiki/spaces/<SPACE>/pages/<ID>/<slug>` -- id
            parsed straight from the path.
          - Short:     `.../wiki/x/<short_code>` -- Confluence chains
            redirects (`/wiki/x/...` -> `/wiki/pages/tinyurl.action?...`
            -> `/wiki/spaces/<SPACE>/pages/<ID>/...`). We use the
            Confluence-authenticated client (httpx auth is bearer/basic
            via header -- never echoed into Location, so safe) and
            follow up to 5 hops to capture the final canonical URL.

        Returns None on any failure (404, malformed URL, unexpected
        redirect target) so callers can skip the runbook hypothesis
        path without raising.
        """
        try:
            import re as _re
            from urllib.parse import urljoin, urlparse

            parsed = urlparse(url)
            path = parsed.path or ""

            # Canonical form: parse id straight from path.
            m = _re.search(r"/pages/(\d+)(?:/|$)", path)
            if m:
                return m.group(1)

            # Short form: /wiki/x/<code> -- must follow auth'd redirect.
            if "/wiki/x/" not in path and "tinyurl.action" not in path:
                return None

            # Follow the redirect chain with the same auth as get_page.
            # httpx passes basic-auth via the Authorization header on
            # the initial request; on follow_redirects it does NOT
            # auto-re-add it for cross-origin hops, which is fine here
            # since we stay on `<tenant>.atlassian.net` throughout.
            current = url
            for _ in range(5):
                resp = await self._client.get(current, follow_redirects=False)
                if resp.status_code in (200, 204):
                    break
                if resp.status_code not in (301, 302, 303, 307, 308):
                    break
                loc = resp.headers.get("location") or ""
                if not loc:
                    break
                m = _re.search(r"/pages/(\d+)(?:/|$)", loc)
                if m:
                    return m.group(1)
                # Resolve relative redirects against the previous URL.
                current = urljoin(current, loc)
            return None
        except Exception as exc:  # noqa: BLE001
            _log.warning("resolve_page_id_from_url(%s) failed: %s", url, exc)
            return None

    async def get_page_by_url(self, url: str) -> Page | None:
        """Resolve a Confluence URL to a numeric page_id, then fetch.
        Returns None on any failure (URL doesn't resolve, page deleted,
        permission denied, etc.) -- callers should treat this as
        "runbook unavailable" and proceed without it.
        """
        page_id = await self.resolve_page_id_from_url(url)
        if not page_id:
            return None
        try:
            return await self.get_page(page_id)
        except Exception as exc:  # noqa: BLE001
            _log.warning("get_page_by_url(%s) -> fetch page_id=%s failed: %s", url, page_id, exc)
            return None


def _parse_iso8601(value: str) -> datetime | None:
    """Tolerant ISO-8601 parser for Atlassian timestamps.

    Atlassian returns `2024-08-12T03:30:00.000Z` or `...+0000`; both forms
    are handled. Returns None for malformed input rather than raising --
    a missing modified date isn't worth failing ingestion over.
    """
    if not value:
        return None
    try:
        # Python 3.11+ handles 'Z' suffix in fromisoformat.
        v = value.replace("Z", "+00:00")
        return datetime.fromisoformat(v)
    except (ValueError, TypeError):
        return None
