"""RootlySource -- bridges `RootlyClient` to `SourceProtocol`.

Document model: one incident = one Markdown document. Post-mortems
attach to their parent incident at fetch time so retrieval surfaces
both the incident summary and the deep root-cause analysis as a
single unit.

Status filter (defaults): only `resolved` and `mitigated` incidents
are indexed. `cancelled` / `scheduled` / `planning` carry no
post-incident knowledge -- skipping them is a quality decision, not a
permission one.

Privacy posture:
- Incidents marked `private: true` are skipped entirely.
- Token-shaped strings get redacted in the rendered Markdown
  (formatter handles this -- the source layer just feeds it).

Scope: `rootly:` is single-tenant -- there's one Rootly account per
organization. `scope` is fixed at `default` (any value works,
the connector ignores it on listing). Pipeline still passes a scope
for tracker-key consistency.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from datetime import datetime

from opsrag.interfaces.source import DocRef, SourceDocument
from opsrag.sources.rootly.client import Incident, PostMortem, RootlyClient
from opsrag.sources.rootly.formatter import render_incident

_log = logging.getLogger("opsrag.rootly.source")


class RootlySource:
    """SourceProtocol implementation for Rootly incidents + post-mortems."""

    source_type = "rootly"

    def __init__(
        self,
        client: RootlyClient,
        *,
        statuses: tuple[str, ...] = ("resolved", "mitigated"),
        post_mortem_statuses: tuple[str, ...] = ("published",),
        skip_private: bool = True,
    ):
        self._client = client
        self._statuses = statuses
        self._post_mortem_statuses = post_mortem_statuses
        self._skip_private = skip_private
        # Cache populated during list_documents so fetch_document can
        # join post-mortems and serialize without a second listing.
        self._incident_cache: dict[str, Incident] = {}
        self._post_mortem_by_incident: dict[str, PostMortem] = {}
        self._post_mortems_loaded = False

    async def _load_post_mortems(self) -> None:
        if self._post_mortems_loaded:
            return
        async for pm in self._client.list_post_mortems(
            statuses=self._post_mortem_statuses,
        ):
            if pm.incident_id:
                # Multiple post-mortems on one incident is rare but
                # possible (e.g., draft superseded by published). Keep
                # the first one we see -- list is sorted by -updated_at,
                # so it's the most recent.
                self._post_mortem_by_incident.setdefault(pm.incident_id, pm)
        self._post_mortems_loaded = True
        _log.info(
            "rootly: loaded %d post-mortems",
            len(self._post_mortem_by_incident),
        )

    async def list_documents(
        self,
        scope: str,
        *,
        updated_since: datetime | None = None,
    ) -> AsyncIterator[DocRef]:
        """Yield a `DocRef` per qualifying incident.

        `scope` is ignored for filtering (Rootly is single-tenant per
        token) but flows through the pipeline as the tracker key.
        """
        # Pre-load post-mortems once so fetch_document stays cheap.
        await self._load_post_mortems()

        async for inc in self._client.list_incidents(
            statuses=self._statuses,
            updated_since=updated_since,
        ):
            if self._skip_private and inc.private:
                continue
            self._incident_cache[inc.id] = inc
            yield DocRef(
                source_type=self.source_type,
                scope=scope,
                doc_id=inc.id,
            )

    async def fetch_document(self, ref: DocRef) -> SourceDocument:
        """Render one incident (+ joined post-mortem) as Markdown."""
        inc = self._incident_cache.get(ref.doc_id)
        if inc is None:
            # Listing was bypassed (e.g., direct fetch from a saved ref).
            # Re-listing one item costs roughly the same as listing all
            # in this volume, but doing a single targeted call is the
            # right shape -- Rootly supports GET /v1/incidents/{id}.
            data = await self._client._get(
                f"/incidents/{ref.doc_id}",
                {"include": "severity,services,teams,environments,incident_types,causes"},
            )
            from opsrag.sources.rootly.client import _index_included, _to_incident
            included = _index_included(data.get("included") or [])
            inc = _to_incident(data["data"], included)
            self._incident_cache[ref.doc_id] = inc

        post_mortem = self._post_mortem_by_incident.get(inc.id)
        content = render_incident(inc, post_mortem=post_mortem)

        # Path: human-readable. `incident-<seq>-<slug>.md` so citations
        # surface useful context. Falls back to UUID if seq/slug missing.
        if inc.sequential_id and inc.slug:
            path = f"incidents/incident-{inc.sequential_id}-{inc.slug}.md"
        elif inc.sequential_id:
            path = f"incidents/incident-{inc.sequential_id}.md"
        else:
            path = f"incidents/incident-{inc.id}.md"

        # SHA changes when either the incident or its post-mortem is
        # updated -- so daily delta correctly re-embeds amended writeups.
        sha_parts = [inc.id, inc.updated_at.isoformat() if inc.updated_at else ""]
        if post_mortem:
            sha_parts.append(post_mortem.id)
            sha_parts.append(post_mortem.updated_at.isoformat() if post_mortem.updated_at else "")
        sha = "|".join(sha_parts)

        return SourceDocument(
            path=path,
            content=content,
            sha=sha,
            last_modified=inc.updated_at,
            repo=f"{self.source_type}:{ref.scope}",
            branch=self.source_type,
            metadata={
                "source_type": self.source_type,
                "incident_id": inc.id,
                "sequential_id": inc.sequential_id,
                "title": inc.title,
                "status": inc.status,
                "severity": inc.severity_name,
                "services": list(inc.services),
                "teams": list(inc.teams),
                "environments": list(inc.environments),
                "labels": list(inc.labels),
                "has_post_mortem": post_mortem is not None,
                "post_mortem_id": post_mortem.id if post_mortem else None,
                "url": inc.url,
                "slack_channel": inc.slack_channel_name,
            },
        )
