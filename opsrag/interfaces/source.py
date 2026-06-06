"""Source provider interface -- generalizes ingestion beyond git.

Phase 2 introduces non-git sources (Confluence, Rootly, Slack, etc).
The git-specific `SCMProvider` in `opsrag.interfaces.scm` stays as the
entry point for git repos (its semantics -- branches, working trees,
file paths -- don't generalize cleanly). For everything else, providers
implement `SourceProtocol` and ingest through
`IngestionPipeline.index_source(source_type, scope)`.

A `SourceDocument` is a content-bearing unit:
- For Confluence: one wiki page (rendered to Markdown).
- For Rootly: one incident (rendered to a structured doc).
- For Slack: one thread summary.

Internally `SourceDocument` is a type alias for `RepoFile` so the
existing parser / chunker / embedder / vector-store path doesn't need
changes -- only the field interpretation widens:

| RepoFile field | git interpretation | confluence interpretation |
|---|---|---|
| `path` | "src/foo.py" | "<page_id>:<slug>" |
| `repo` | "saas/acme-notes-be" | "confluence:SRE" |
| `branch` | "master" | "confluence" (sentinel) |
| `sha` | git blob sha | page version (e.g. "v21") |
| `last_modified` | git commit date | page lastModifiedDate |
| `metadata` | freeform | source_type, page_id, page_url, labels, ancestors |

The dedup `(repo, branch, path)` triple in the `indexed_files` Postgres
table works unchanged with this convention.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# Re-export so callers can `from opsrag.interfaces.source import SourceDocument`
# without also having to know about scm.py. The alias is intentional --
# we generalize meaning, not structure.
from opsrag.interfaces.scm import RepoFile as SourceDocument

__all__ = ["DocRef", "SourceDocument", "SourceProtocol"]


@dataclass(frozen=True)
class DocRef:
    """A pointer to a document in a source -- fetch it via SourceProtocol.fetch_document.

    The pair (source_type, doc_id) is globally unique within OpsRAG.
    `scope` is the source-specific grouping (Confluence space key,
    Rootly project id, etc.) and lets the indexer dedup at the
    grouping level.

    Frozen + hashable on purpose so callers can use a `set[DocRef]` to
    de-duplicate during pagination. Keep this small -- payload metadata
    belongs on the `SourceDocument` returned by `fetch_document`.
    """

    source_type: str
    scope: str
    doc_id: str


@runtime_checkable
class SourceProtocol(Protocol):
    """Provider that yields documents from a non-git source.

    Implementations must be safe to call concurrently (the ingestion
    pipeline fans out fetches under a semaphore). Network errors should
    be retried internally -- the pipeline expects best-effort iteration.
    """

    source_type: str   # "confluence" | "rootly" | "slack" | ...

    async def list_documents(self, scope: str) -> AsyncIterator[DocRef]:
        """Stream `DocRef`s within `scope` (e.g. a Confluence space key).

        Should be paginated internally; callers iterate without
        knowing the page size.
        """
        ...

    async def fetch_document(self, ref: DocRef) -> SourceDocument:
        """Fetch + render one document. Must populate at minimum:

        - `content` -- already in the format the parser expects (e.g.
          Markdown for Confluence -- ADF rendering happens here, not in
          the parser layer).
        - `path` -- stable id (e.g. `<page_id>:<slug>`).
        - `repo` -- `<source_type>:<scope>` convention.
        - `branch` -- source-specific sentinel (e.g. `"confluence"`).
        - `sha` -- version identifier sufficient for content-hash dedup.
        - `last_modified`.
        - `metadata` -- extra fields downstream consumers can use,
          including at least `source_type` and any source-native
          identifiers (page_id, incident_id, channel_id, etc).
        """
        ...
