"""F13: cross-user scoping for the investigation cache.

``InvestigationCache.store`` persists ``user_id`` on every point, but
``search`` historically returned ALL high-cosine hits to whoever asked --
so a memory-influenced investigation (full answer + tool_call_audit) cached
for user A leaked into user B's reasoning context on a near-match.

Decision (mirrors ``opsrag.qa_cache``): investigations stay SHARED by
default (the documented shared / scope-gated authz model), with one
carve-out -- when an answer wove in per-user Mem0 memories it is stamped
with ``user_scope`` and returned ONLY to its author.

These tests drive ``InvestigationCache`` against a tiny fake Qdrant that
honours the ``query_filter`` (IsEmpty(user_scope) OR user_scope == me),
asserting:
  * a user-scoped entry is NOT returned to a different user,
  * a user-scoped entry IS returned to its author,
  * a shared (unscoped) entry is returned to everyone.
"""
from __future__ import annotations

import asyncio

from qdrant_client import models as qm

from opsrag.agent.cache.investigation_cache import InvestigationCache


class _FakePoint:
    def __init__(self, point_id, score, payload):
        self.id = point_id
        self.score = score
        self.payload = payload


class _FakeResult:
    def __init__(self, points):
        self.points = points


class _FakeQdrant:
    """Minimal in-memory AsyncQdrantClient honouring the scope filter."""

    def __init__(self):
        self._points: dict[str, dict] = {}

    async def get_collection(self, collection_name):  # noqa: ANN001
        return object()  # collection always "exists" -> no create path

    async def create_collection(self, *a, **k):  # noqa: ANN001, ANN002
        return None

    async def create_payload_index(self, *a, **k):  # noqa: ANN001, ANN002
        return None

    async def upsert(self, collection_name, points):  # noqa: ANN001
        for p in points:
            self._points[p.id] = dict(p.payload)
        return None

    @staticmethod
    def _matches(payload: dict, query_filter) -> bool:
        """Evaluate the shared-or-mine filter the cache builds."""
        if query_filter is None:
            return True
        # must=[IsEmpty(user_scope)]  -> shared-only (anonymous caller)
        for cond in (query_filter.must or []):
            if isinstance(cond, qm.IsEmptyCondition):
                if payload.get(cond.is_empty.key) not in (None, ""):
                    return False
        # should=[IsEmpty(user_scope), user_scope == me] -> shared OR mine
        shoulds = query_filter.should or []
        if shoulds:
            ok = False
            for cond in shoulds:
                if isinstance(cond, qm.IsEmptyCondition):
                    if payload.get(cond.is_empty.key) in (None, ""):
                        ok = True
                elif isinstance(cond, qm.FieldCondition):
                    if payload.get(cond.key) == cond.match.value:
                        ok = True
            if not ok:
                return False
        return True

    async def query_points(self, collection_name, query, limit,  # noqa: ANN001
                           with_payload=True, query_filter=None):
        pts = [
            _FakePoint(pid, 0.99, payload)
            for pid, payload in self._points.items()
            if self._matches(payload, query_filter)
        ]
        return _FakeResult(pts[:limit])


def _store(cache: InvestigationCache, *, question, user_id, user_scope=None):
    return asyncio.run(cache.store(
        question=question,
        embedding=[0.0, 1.0, 0.0],
        answer=f"answer to {question}",
        tool_call_audit=[{"name": "search_logs"}],
        thread_id=f"{user_id}_t",
        user_id=user_id,
        user_scope=user_scope,
    ))


def _search(cache: InvestigationCache, *, user_id=None):
    return asyncio.run(cache.search(
        [0.0, 1.0, 0.0], top_k=5, user_id=user_id,
    ))


def _new_cache() -> InvestigationCache:
    # threshold 0.0 so the 0.99 fake score always clears regardless of decay.
    return InvestigationCache(_FakeQdrant(), threshold=0.0)


def test_user_scoped_entry_not_returned_to_other_user():
    cache = _new_cache()
    _store(cache, question="my prod secret rotation", user_id="alice",
           user_scope="alice")  # wove in alice's Mem0 memories
    hits = _search(cache, user_id="bob")
    assert hits == [], "user-scoped investigation leaked to a different user"


def test_user_scoped_entry_returned_to_its_author():
    cache = _new_cache()
    _store(cache, question="my prod secret rotation", user_id="alice",
           user_scope="alice")
    hits = _search(cache, user_id="alice")
    assert len(hits) == 1
    assert hits[0].user_scope == "alice"
    assert hits[0].user_id == "alice"


def test_shared_entry_returned_to_everyone():
    cache = _new_cache()
    _store(cache, question="how does the deploy pipeline work", user_id="alice")
    # No user_scope stamped -> shared.
    assert _search(cache, user_id="bob")[0].user_scope is None
    assert _search(cache, user_id="alice")[0].user_scope is None
    # anonymous (user_id=None) still gets shared entries
    assert len(_search(cache, user_id=None)) == 1


def test_anonymous_caller_excluded_from_scoped_entries():
    cache = _new_cache()
    _store(cache, question="my prod secret rotation", user_id="alice",
           user_scope="alice")
    assert _search(cache, user_id=None) == []
