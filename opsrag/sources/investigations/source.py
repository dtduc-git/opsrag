"""InvestigationsSource -- bridges `InvestigationCache` to `SourceProtocol`.

Every settled past investigation becomes one Markdown document indexed
into the main Qdrant corpus under `repo="investigation-history"`. The
generator/reasoner sees them through retrieval the same way it sees
Confluence pages or runbooks -- but the chunk format makes the
historical-snapshot nature obvious so the model can reason about
freshness rather than treating it as canonical.

Filtering applied:

  - `created_at` newer than `max_age_days` ago (older = stale; default 90d)
  - `created_at` older than `min_age_days` ago (younger = unsettled; default 7d)
  - thumbs-down feedback excluded outright
  - thumbs-up feedback prioritized (we still index neutral, but rank
    isn't our concern -- the corpus retrieval handles relevance)

Document shape (per investigation):

    repo:           "investigation-history"
    branch:         "investigation"          (sentinel)
    path:           "<investigation-id>:<short-slug>.md"
    sha:            investigation point id (immutable)
    last_modified:  created_at
    content:        rendered Markdown (see _render_markdown)
    metadata:
      source_type:    "investigation-history"
      investigation_id: <uuid>
      created_at:    <ISO8601>
      thumbs_up:     int
      thumbs_down:   int
      tools_used:    [tool names]
      original_url:  None  (no external URL -- this IS the source)

The header of every Markdown body explicitly states the snapshot
timestamp + a "VERIFY-IF-STILL-TRUE" warning so the LLM treats the
content as historical context, not authority.
"""
from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from opsrag.interfaces.source import DocRef, SourceDocument

_log = logging.getLogger("opsrag.sources.investigations")

DEFAULT_MIN_AGE_DAYS = 7    # below this, investigation may still be evolving
DEFAULT_MAX_AGE_DAYS = 90   # above this, snapshot likely stale (kafka 3->5 brokers scenario)
DEFAULT_MAX_DOCS = 500      # safety cap on per-run yield


def _slugify(s: str) -> str:
    out = []
    for ch in (s or "").lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_"):
            out.append("-")
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug[:60]


def _render_markdown(
    *,
    investigation_id: str,
    question: str,
    answer: str,
    created_at: datetime,
    thumbs_up: int,
    thumbs_down: int,
    tool_call_audit: list[dict],
) -> str:
    """Build the Markdown document for one past investigation.

    The header is intentionally heavy on freshness cues so retrieval
    snippets surface the timestamp + "verify if still true" warning to
    the generator regardless of which chunk gets retrieved.
    """
    age_days = max(0, (datetime.now(UTC) - created_at).days)
    iso = created_at.strftime("%Y-%m-%d %H:%M UTC")
    feedback = ""
    if thumbs_up or thumbs_down:
        feedback = f" | up {thumbs_up} / down {thumbs_down}"
    tools = ", ".join(sorted({t.get("name", "") for t in tool_call_audit if t.get("name")}))[:200]
    tools_line = f"\n**Tools used:** {tools}" if tools else ""

    body = (
        f"# Past investigation snapshot -- {iso} ({age_days} days ago){feedback}\n"
        "\n"
        "> **REFERENCE ONLY -- verify if still true with current tools.** "
        "This is a historical record of how a similar question was answered "
        "in the past; live state may have changed (cluster sizes, configs, "
        "owners, alert thresholds). Use it as a hint about what to look at, "
        "not as authority.\n"
        "\n"
        f"## Original question\n{question.strip()}\n"
        "\n"
        f"## Answer (as of {iso})\n{answer.strip()}\n"
        f"{tools_line}\n"
        "\n"
        f"<!-- investigation-id: {investigation_id} -->\n"
    )
    return body


class InvestigationsSource:
    """SourceProtocol implementation that walks `InvestigationCache`.

    Invariants:
    - Documents are yielded in newest-first order so `last_modified`
      filtering by the ingestion pipeline gives stable behavior.
    - The same `investigation_id` ALWAYS produces the same `path`
      (so re-runs idempotently overwrite the chunk via the
      `indexed_files` dedup table -- no duplicates).
    - We do NOT delete corpus chunks for expired investigations here;
      a separate `prune_expired()` method handles that.
    """

    source_type = "investigation-history"

    def __init__(
        self,
        investigation_cache,  # InvestigationCache instance (avoids cyclic import)
        *,
        min_age_days: int = DEFAULT_MIN_AGE_DAYS,
        max_age_days: int = DEFAULT_MAX_AGE_DAYS,
        max_docs: int = DEFAULT_MAX_DOCS,
        skip_thumbs_down: bool = True,
    ):
        self._cache = investigation_cache
        self._min_age = max(0, int(min_age_days))
        self._max_age = max(self._min_age + 1, int(max_age_days))
        self._max_docs = max_docs
        self._skip_thumbs_down = skip_thumbs_down
        # In-memory map of (id -> payload) populated by list_documents,
        # consumed by fetch_document. Keeps fetch_document a fast lookup
        # without re-querying Qdrant for every doc.
        self._payload_cache: dict[str, dict] = {}

    async def list_documents(self, scope: str) -> AsyncIterator[DocRef]:
        """`scope` is unused (single global investigation collection)
        but kept for SourceProtocol compatibility."""
        cache = self._cache
        if cache is None:
            return
        # Direct Qdrant scroll on the investigation collection -- bypasses
        # the search-by-vector path since we need to enumerate everything,
        # not just close matches.
        await cache._ensure_collection()
        if not cache._ensured:
            return
        try:
            offset = None
            yielded = 0
            now = time.time()
            while yielded < self._max_docs:
                points, offset = await cache._qdrant.scroll(
                    collection_name=cache._collection,
                    limit=128,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                if not points:
                    break
                for p in points:
                    payload = p.payload or {}
                    created_at = float(payload.get("created_at") or 0.0)
                    if created_at <= 0:
                        continue
                    age_days = (now - created_at) / 86400.0
                    if age_days < self._min_age:
                        continue  # too fresh -- may still be in flux
                    if age_days > self._max_age:
                        continue  # too old -- snapshot likely wrong now
                    feedback = payload.get("feedback") or {}
                    if self._skip_thumbs_down and int(feedback.get("down", 0) or 0) > 0:
                        continue
                    inv_id = str(p.id)
                    self._payload_cache[inv_id] = payload
                    yield DocRef(
                        source_type=self.source_type,
                        scope="opsrag",
                        doc_id=inv_id,
                    )
                    yielded += 1
                    if yielded >= self._max_docs:
                        break
                if offset is None:
                    break
        except Exception as exc:
            _log.warning("investigations list_documents failed: %s", exc)

    async def fetch_document(self, ref: DocRef) -> SourceDocument:
        """Render the cached payload as a Markdown SourceDocument."""
        payload = self._payload_cache.get(ref.doc_id)
        if payload is None:
            # Refresh from Qdrant -- list_documents wasn't called or
            # the cache was cleared. Best-effort.
            try:
                points = await self._cache._qdrant.retrieve(
                    collection_name=self._cache._collection,
                    ids=[ref.doc_id],
                    with_payload=True,
                )
                if points:
                    payload = points[0].payload or {}
            except Exception:
                payload = {}
            payload = payload or {}

        question = payload.get("question", "") or ""
        answer = payload.get("answer", "") or ""
        created_at_ts = float(payload.get("created_at") or 0.0)
        created_at = (
            datetime.fromtimestamp(created_at_ts, tz=UTC)
            if created_at_ts > 0 else datetime.now(UTC)
        )
        feedback = payload.get("feedback") or {}
        thumbs_up = int(feedback.get("up", 0) or 0)
        thumbs_down = int(feedback.get("down", 0) or 0)
        tool_audit = payload.get("tool_call_audit") or []

        slug = _slugify(question[:60]) or "untitled"
        path = f"{ref.doc_id}:{slug}.md"
        content = _render_markdown(
            investigation_id=ref.doc_id,
            question=question,
            answer=answer,
            created_at=created_at,
            thumbs_up=thumbs_up,
            thumbs_down=thumbs_down,
            tool_call_audit=tool_audit,
        )

        return SourceDocument(
            path=path,
            content=content,
            sha=ref.doc_id,  # immutable, makes a fine sha
            last_modified=created_at,
            repo="investigation-history",
            branch="investigation",
            metadata={
                "source_type": self.source_type,
                "investigation_id": ref.doc_id,
                "created_at": created_at.isoformat(),
                "thumbs_up": thumbs_up,
                "thumbs_down": thumbs_down,
                "tools_used": sorted({t.get("name", "") for t in tool_audit if t.get("name")}),
                "snapshot_age_days": max(0, (datetime.now(UTC) - created_at).days),
            },
        )

    async def prune_expired(self, vector_store) -> int:
        """Delete corpus chunks whose backing investigation has aged out.

        Walks the main Qdrant vector store filtering for our source-type
        + last_modified older than `max_age_days`. Used by the daily job
        to keep the kafka-3-broker-snapshot from haunting future Q&A.

        Returns count attempted (Qdrant doesn't return count).
        """
        if not hasattr(vector_store, "_client"):
            return 0
        try:
            from qdrant_client import models as qm
            cutoff_ts = time.time() - (self._max_age * 86400)
            cutoff_iso = datetime.fromtimestamp(cutoff_ts, tz=UTC).isoformat()
            await vector_store._client.delete(
                collection_name=vector_store._collection,
                points_selector=qm.FilterSelector(
                    filter=qm.Filter(must=[
                        qm.FieldCondition(
                            key="repo",
                            match=qm.MatchValue(value="investigation-history"),
                        ),
                        qm.FieldCondition(
                            key="last_modified",
                            range=qm.Range(lt=cutoff_iso),
                        ),
                    ])
                ),
                wait=False,
            )
            _log.info("investigations prune cutoff=%s ok", cutoff_iso)
            return -1  # Qdrant doesn't return count
        except Exception as exc:
            _log.warning("investigations prune failed: %s", exc)
            return 0
