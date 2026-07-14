"""Unit tests: investigation-cache feedback — thread-id resolution + runbook ids.

Prod bug: the chat UI posts feedback with a THREAD-shaped id
(`<uuid>_<8hex>`) whenever the real investigation UUID never reached it
(cache-hit answers, replayed sessions). `record_feedback` fed that id
straight into `qdrant.retrieve(ids=[...])` -> 400 Bad Request (not a valid
point id) -> silent False, so the Qdrant-side feedback flag (thumbs-down
purge, cache audit) never recorded anything.

Fix under test (runs against the REAL InvestigationCache on an in-memory
fake Qdrant that mirrors the real client's retrieve/scroll/set_payload):
  - `record_feedback` accepts UUID point ids (unchanged fast path) AND
    thread-shaped ids, resolved server-side by payload `thread_id` filter;
    disambiguation: answer_snippet substring match first, newest otherwise.
  - Returns a truthy/falsy `FeedbackResult` carrying the resolved point id
    and the rb-ids of every runbook the stored answer loaded
    (`tool_call_audit` rows: name==runbook_load, no error, args.name=rb-*)
    so the route can bump runbook thumbs (Fix A).
"""
from __future__ import annotations

import uuid as _uuid

from qdrant_client import models as qm

from opsrag.agent.cache.investigation_cache import (
    InvestigationCache,
    extract_loaded_runbook_ids,
)

RB = "3f0af711-9f7e-4a56-8f21-9c1d2e3f4a5b"
THREAD = "7a1b2c3d-4e5f-6a7b-8c9d-0e1f2a3b4c5d_1a2b3c4d"

AUDIT_WITH_RUNBOOK = [
    {"name": "prometheus_query", "args": {"query": "up"}},
    {"name": "runbook_load", "args": {"name": f"rb-{RB}"}},
    {"name": "runbook_load", "args": {"name": f"rb-{RB}"}},          # dup -> once
    {"name": "runbook_load", "args": {"name": "runbook-file-one"}},  # file catalog -> skip
    {"name": "runbook_load", "args": {"name": "rb-dead"}, "error": "not found"},
]


class _FakePoint:
    def __init__(self, pid, payload):
        self.id = pid
        self.payload = payload


class _FakeQdrant:
    """In-memory stand-in mirroring the real AsyncQdrantClient calls the
    cache makes: retrieve raises on non-UUID ids exactly like the server."""

    def __init__(self):
        self._points: dict[str, dict] = {}
        self.index_fields: list[str] = []
        self.set_payload_calls: list[tuple[str, dict]] = []

    async def get_collection(self, collection_name):
        return object()

    async def create_collection(self, *a, **k):
        return None

    async def create_payload_index(self, collection_name, field_name, field_schema):
        self.index_fields.append(field_name)

    async def upsert(self, collection_name, points):
        for p in points:
            self._points[p.id] = dict(p.payload)

    async def retrieve(self, collection_name, ids, with_payload=True):
        for pid in ids:
            _uuid.UUID(str(pid))  # real server 400s on non-UUID point ids
        return [_FakePoint(pid, dict(self._points[pid])) for pid in ids if pid in self._points]

    async def scroll(self, collection_name, scroll_filter=None, limit=10,
                     offset=None, with_payload=True, with_vectors=False):
        out = []
        for pid, payload in self._points.items():
            if scroll_filter is not None:
                ok = True
                for cond in (scroll_filter.must or []):
                    if isinstance(cond, qm.FieldCondition):
                        if payload.get(cond.key) != cond.match.value:
                            ok = False
                if not ok:
                    continue
            out.append(_FakePoint(pid, dict(payload)))
        # Paginate like the real client: next_page_offset is None on the
        # last page.
        start = int(offset or 0)
        page = out[start:start + limit]
        next_offset = start + limit if start + limit < len(out) else None
        return page, next_offset

    async def set_payload(self, collection_name, payload, points):
        self.set_payload_calls.append((points[0], payload))
        for pid in points:
            _uuid.UUID(str(pid))
            self._points[pid].update(payload)


async def _seed(cache, qd, *, thread=THREAD, answer="the deployment is X",
                created_at=None, audit=None):
    pid = await cache.store(
        question="q", embedding=[0.1, 0.2, 0.3], answer=answer,
        tool_call_audit=audit if audit is not None else AUDIT_WITH_RUNBOOK,
        thread_id=thread, user_id="u1",
    )
    if created_at is not None:
        qd._points[pid]["created_at"] = created_at
    return pid


def _cache():
    qd = _FakeQdrant()
    return InvestigationCache(qd, threshold=0.0), qd


# ------------------------------------------------------------ resolution --

async def test_record_feedback_uuid_direct_still_works():
    cache, qd = _cache()
    pid = await _seed(cache, qd)

    res = await cache.record_feedback(pid, thumbs="up")

    assert bool(res) is True
    assert res.point_id == pid
    assert qd._points[pid]["feedback"]["up"] == 1


async def test_record_feedback_returns_resolved_query_and_answer():
    # The Postgres feedback audit row needs to show WHAT was rated. The
    # resolved investigation payload carries question+answer; record_feedback
    # surfaces them so the channel/route feedback path can persist snippets
    # (Slack feedback used to store thumbs only -> blank dashboard cards).
    cache, qd = _cache()
    pid = await cache.store(
        question="why is passport-be 400ing?",
        embedding=[0.1, 0.2, 0.3],
        answer="rs-backend rejects the org sync payload",
        tool_call_audit=[],
        thread_id=THREAD,
        user_id="u1",
    )

    res = await cache.record_feedback(pid, thumbs="down")

    assert res.query == "why is passport-be 400ing?"
    assert res.answer == "rs-backend rejects the org sync payload"


async def test_record_feedback_thread_id_resolves_newest():
    cache, qd = _cache()
    old = await _seed(cache, qd, answer="old turn", created_at=100.0)
    new = await _seed(cache, qd, answer="new turn", created_at=200.0)

    res = await cache.record_feedback(THREAD, thumbs="down")

    assert bool(res) is True
    assert res.point_id == new
    assert qd._points[new]["feedback"]["down"] == 1
    assert "feedback" not in qd._points[old]


async def test_record_feedback_snippet_picks_the_right_turn():
    cache, qd = _cache()
    old = await _seed(cache, qd, answer="consumer is acme-notes-appservice-consumers", created_at=100.0)
    await _seed(cache, qd, answer="something else entirely", created_at=200.0)

    res = await cache.record_feedback(
        THREAD, thumbs="up", answer_snippet="acme-notes-appservice-consumers",
    )

    assert res.point_id == old
    assert qd._points[old]["feedback"]["up"] == 1


async def test_record_feedback_unknown_thread_returns_falsy():
    cache, qd = _cache()
    await _seed(cache, qd)

    res = await cache.record_feedback("ffffffff-ffff-ffff-ffff-ffffffffffff_zz", thumbs="up")

    assert bool(res) is False
    assert qd.set_payload_calls == []


# ---------------------------------------------------------- runbook ids --

async def test_record_feedback_returns_loaded_runbook_ids():
    cache, qd = _cache()
    pid = await _seed(cache, qd)

    res = await cache.record_feedback(pid, thumbs="up")

    assert res.runbook_ids == [RB]


def test_extract_loaded_runbook_ids_edge_shapes():
    assert extract_loaded_runbook_ids(None) == []
    assert extract_loaded_runbook_ids([]) == []
    assert extract_loaded_runbook_ids([{"name": "runbook_load"}]) == []
    assert extract_loaded_runbook_ids([{"name": "runbook_load", "args": {}}]) == []


# ---------------------------------------------------------------- index --

async def test_ensure_collection_indexes_thread_id_even_when_collection_exists():
    cache, qd = _cache()
    await cache.store(question="q", embedding=[0.1], answer="a", thread_id="t", user_id="u")
    assert "thread_id" in qd.index_fields


async def test_snippet_matching_nothing_returns_falsy_not_wrong_turn():
    """qa-cache-hit answers never store a point under the user's thread —
    if the FE-sent snippet matches NO stored answer, guessing newest would
    misattribute thumbs+correction to a prior unrelated turn. Refuse
    instead (Postgres sink still records the feedback)."""
    cache, qd = _cache()
    await _seed(cache, qd, answer="turn one about postgres", created_at=100.0)

    res = await cache.record_feedback(
        THREAD, thumbs="down", answer_snippet="totally different cached answer",
    )

    assert bool(res) is False
    assert qd.set_payload_calls == []


async def test_thread_resolution_paginates_past_first_page(monkeypatch):
    from opsrag.agent.cache import investigation_cache as ic
    monkeypatch.setattr(ic, "_RESOLVE_PAGE_LIMIT", 2)
    cache, qd = _cache()
    newest = None
    for i in range(5):  # 5 points, page size 2 -> 3 pages
        newest = await _seed(cache, qd, answer=f"turn {i}", created_at=float(i))

    res = await cache.record_feedback(THREAD, thumbs="up")

    assert res.point_id == newest
