"""Unit test: _reaugment_docs deletes through the CONFIGURED vector store.

The Confluence re-augment admin loop used to delete via the raw Qdrant client
with a hardcoded collection_name="opsrag" -- wrong on any deployment using a
non-default (renamed) collection, where it silently failed
every doc into summary["failed"]. The reroute sends the delete through
`pipeline.vector_store.delete_by_filter`, which targets the configured
collection and picks up the keyword file_key fast path when enabled.

Regression pin for the scope bug the reroute initially shipped with: the
helper referenced a `vector_store` name that only existed in the CALLER's
scope -> NameError on every doc, swallowed by the per-doc `except Exception`,
so the endpoint no-op'd with failed == len(affected_list).
"""
from __future__ import annotations

from opsrag.api.routes import _reaugment_docs


class _FakeStore:
    def __init__(self) -> None:
        self.delete_filters: list[dict] = []

    async def delete_by_filter(self, filters: dict) -> int:
        self.delete_filters.append(filters)
        return -1


class _FakeSource:
    async def fetch_document(self, ref):
        return {"doc": ref.doc_id}


class _FakePipeline:
    def __init__(self, store: _FakeStore) -> None:
        self.vector_store = store

    async def _process_file(self, doc) -> int:
        return 3  # pretend chunks were produced


async def test_reaugment_docs_deletes_through_pipeline_vector_store():
    store = _FakeStore()
    pipeline = _FakePipeline(store)
    summary = {"processed": 0, "failed": 0, "failures": []}

    await _reaugment_docs(
        ["123:page-a.md", "456:page-b.md"],
        "OPS",
        "confluence:OPS",
        _FakeSource(),
        pipeline,
        summary,
    )

    assert store.delete_filters == [
        {"repo": "confluence:OPS", "source_path": "123:page-a.md"},
        {"repo": "confluence:OPS", "source_path": "456:page-b.md"},
    ]
    assert summary["processed"] == 2
    assert summary["failed"] == 0, f"unexpected failures: {summary['failures']}"
