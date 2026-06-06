"""ConfluenceSource -- bridges `ConfluenceClient` to `SourceProtocol`.

Wires together:
- `ConfluenceClient.list_pages` -> yields `DocRef` per page.
- `ConfluenceClient.get_page` -> fetches body.
- `render_page` (ADF -> Markdown + frontmatter) -> produces a
  `SourceDocument` shaped for the existing parser/chunker pipeline.

Filtering applied at this layer:
- Spaces: hard allowlist + denylist from `ConfluenceConfig`.
- Pages: `status: "trashed"` skipped; `restrictions.read` skipped (set
  by the API server-side, not enforced here yet -- Step 5 hardening).
- Labels: pages carrying any label in `label_denylist` are dropped.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from opsrag.interfaces.source import DocRef, SourceDocument
from opsrag.sources.confluence.adf import render_page
from opsrag.sources.confluence.client import ConfluenceClient, Space

_log = logging.getLogger("opsrag.confluence.source")


def _slugify(title: str) -> str:
    """Cosmetic slug for the dedup `path` field. Stable enough to be
    readable in logs; we never key on the slug, only on `page_id`."""
    out = []
    for ch in (title or "").lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_"):
            out.append("-")
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug[:60]


class ConfluenceSource:
    """SourceProtocol implementation for Atlassian Cloud Confluence."""

    source_type = "confluence"

    def __init__(
        self,
        client: ConfluenceClient,
        *,
        spaces_allowlist: list[str] | None = None,
        spaces_denylist: list[str] | None = None,
        label_denylist: list[str] | None = None,
    ):
        self._client = client
        # Normalize for case-insensitive comparison.
        self._allowlist = {s.upper() for s in (spaces_allowlist or [])}
        self._denylist = {s.upper() for s in (spaces_denylist or [])}
        self._label_denylist = {s.lower() for s in (label_denylist or [])}
        # Cached space-key -> space-id resolution. Populated on first
        # list_documents call so we don't refetch per page.
        self._spaces_by_key: dict[str, Space] = {}
        # Per-process cache of `(space_key, page_id) -> last_modified`
        # so callers can pass `updated_since` for incremental runs.
        self._last_modified: dict[tuple[str, str], datetime] = {}

    async def _resolve_space(self, space_key: str) -> Space | None:
        """Look up the Space by key, caching after first call."""
        if not self._spaces_by_key:
            spaces = await self._client.list_spaces()
            self._spaces_by_key = {s.key: s for s in spaces}
        return self._spaces_by_key.get(space_key)

    def _is_space_allowed(self, key: str) -> bool:
        norm = key.upper()
        if key.startswith("~"):
            # Personal spaces are never allowed -- hard rule.
            return False
        if self._denylist and norm in self._denylist:
            return False
        if self._allowlist and norm not in self._allowlist:
            return False
        return True

    async def list_documents(
        self,
        scope: str,
        *,
        updated_since: datetime | None = None,
    ) -> AsyncIterator[DocRef]:
        """Yield a `DocRef` per page in the space `scope` (a space key).

        Pages with `status != "current"` (trashed / archived / draft) are
        skipped. ACL-blocked pages don't appear in v2 API responses for
        the authenticated user, so no extra filter needed here.
        """
        if not self._is_space_allowed(scope):
            _log.warning(
                "confluence: space=%s rejected by allowlist/denylist -- skipping",
                scope,
            )
            return

        space = await self._resolve_space(scope)
        if space is None:
            _log.warning("confluence: space=%s not visible to service account", scope)
            return

        async for page_ref in self._client.list_pages(
            space_id=space.id,
            space_key=space.key,
            updated_since=updated_since,
            status="current",
        ):
            if page_ref.last_modified:
                self._last_modified[(scope, page_ref.id)] = page_ref.last_modified
            yield DocRef(
                source_type=self.source_type,
                scope=scope,
                doc_id=f"{page_ref.id}:{_slugify(page_ref.title)}",
            )

    async def fetch_document(self, ref: DocRef) -> SourceDocument:
        """Fetch + render one page as Markdown wrapped in `SourceDocument`."""
        # The doc_id is `<page_id>:<slug>`; the page_id alone is the
        # Confluence-side identifier.
        page_id = ref.doc_id.split(":", 1)[0]
        page = await self._client.get_page(page_id)
        # Backfill space_key on the Page from our cached space lookup
        # (the v2 page endpoint returns space_id but not the key).
        if not page.space_key:
            for k, s in self._spaces_by_key.items():
                if s.id == page.space_id:
                    page.space_key = k
                    break
            else:
                page.space_key = ref.scope

        # Drop pages whose labels match the denylist BEFORE rendering --
        # cheap to check, expensive to chunk + embed unnecessarily.
        if self._label_denylist:
            page_labels = {lb.lower() for lb in page.labels or []}
            if page_labels & self._label_denylist:
                _log.info(
                    "confluence: page id=%s labels=%s blocked by label_denylist",
                    page.id, sorted(page_labels & self._label_denylist),
                )
                # Returning a SourceDocument with empty content makes the
                # downstream parser pipeline a no-op -- same effect as
                # skipping but keeps the iteration contract simple.
                content = ""
            else:
                content = render_page(
                    page,
                    last_reviewed=datetime.now(UTC).date().isoformat(),
                )
        else:
            content = render_page(
                page,
                last_reviewed=datetime.now(UTC).date().isoformat(),
            )

        # The dedup `(repo, branch, path)` triple in indexed_files: see
        # opsrag.interfaces.source for the convention.
        # Append `.md` so the existing markdown parser's extension-based
        # `supports()` check matches -- render_page produces well-formed
        # Markdown with YAML frontmatter, so this is honest, not a hack.
        doc_path = f"{ref.doc_id}.md" if not ref.doc_id.endswith(".md") else ref.doc_id
        doc = SourceDocument(
            path=doc_path,
            content=content,
            sha=f"v{page.version}",
            last_modified=page.last_modified,
            repo=f"{self.source_type}:{ref.scope}",
            branch=self.source_type,
            metadata={
                "source_type": self.source_type,
                "space_key": page.space_key,
                "page_id": page.id,
                "page_title": page.title,
                "page_url": page.url,
                "page_version": page.version,
                "labels": list(page.labels or []),
                "ancestors": list(page.ancestors or []),
            },
        )
        _log.info(
            "confluence ingested space=%s page_id=%s title=%r labels=%s",
            ref.scope, page.id, page.title[:80], page.labels,
        )
        return doc
