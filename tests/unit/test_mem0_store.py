"""Unit tests for the Mem0-backed operational memory store.

The mem0 `Memory` object is fully mocked -- these tests never require a live
Qdrant or LLM. They lock down the four design-critical behaviours:

(a) namespace tuple -> expected mem0 ``user_id`` string
(b) empty / None service is NOT written to a shared bucket
(c) PII (an email / token) is redacted before ``.add()``
(d) a raising mem0 client does NOT propagate (best-effort)
"""
from __future__ import annotations

from opsrag.interfaces.memory import Memory, MemoryStore
from opsrag.memory.mem0_store import (
    Mem0ServiceMemory,
    _namespace_to_user_id,
    _redact_pii,
)


class FakeMemory:
    """Records calls; returns mem0 v2-shaped ``{"results": [...]}``."""

    def __init__(self) -> None:
        self.add_calls: list[dict] = []
        self.search_calls: list[dict] = []
        self.get_all_calls: list[dict] = []
        self.delete_calls: list[dict] = []
        self._rows: list[dict] = []

    def seed(self, rows: list[dict]) -> None:
        self._rows = rows

    def add(self, messages, *, user_id=None, metadata=None, infer=True):
        self.add_calls.append(
            {
                "messages": messages,
                "user_id": user_id,
                "metadata": metadata,
                "infer": infer,
            }
        )
        return {"results": []}

    def search(self, query, *, filters=None, top_k=20, **kwargs):
        self.search_calls.append({"query": query, "filters": filters, "top_k": top_k})
        return {"results": self._rows}

    def get_all(self, *, filters=None, top_k=20, **kwargs):
        self.get_all_calls.append({"filters": filters, "top_k": top_k})
        return {"results": self._rows}

    def delete(self, *, memory_id=None):
        self.delete_calls.append({"memory_id": memory_id})


class RaisingMemory:
    """Every method raises -- proves best-effort swallowing."""

    def add(self, *a, **k):
        raise RuntimeError("boom-add")

    def search(self, *a, **k):
        raise RuntimeError("boom-search")

    def get_all(self, *a, **k):
        raise RuntimeError("boom-get_all")

    def delete(self, *a, **k):
        raise RuntimeError("boom-delete")


# --- (a) namespace -> user_id --------------------------------------------

def test_namespace_maps_to_colon_joined_user_id():
    assert _namespace_to_user_id(("ops", "acme-notes-be")) == "ops:acme-notes-be"
    assert _namespace_to_user_id(("user", "u1", "topics")) == "user:u1:topics"


async def test_put_uses_expected_user_id():
    fake = FakeMemory()
    store = Mem0ServiceMemory(fake, infer=True)
    await store.put(("ops", "acme-notes-be"), "k1", {"note": "scaled replicas"})
    assert len(fake.add_calls) == 1
    assert fake.add_calls[0]["user_id"] == "ops:acme-notes-be"


async def test_search_uses_expected_user_id_filter():
    fake = FakeMemory()
    store = Mem0ServiceMemory(fake, infer=True)
    await store.search(("ops", "acme-notes-be"), query="latency", limit=3)
    assert fake.search_calls[0]["filters"] == {"user_id": "ops:acme-notes-be"}
    assert fake.search_calls[0]["top_k"] == 3


# --- (b) empty / None service is not written to a shared bucket ----------

async def test_empty_namespace_is_not_written():
    fake = FakeMemory()
    store = Mem0ServiceMemory(fake, infer=True)
    await store.put((), "k1", {"note": "x"})
    assert fake.add_calls == []


async def test_blank_trailing_service_is_not_written():
    fake = FakeMemory()
    store = Mem0ServiceMemory(fake, infer=True)
    # trailing service segment is empty/whitespace -> treated as global, skip
    await store.put(("ops", ""), "k1", {"note": "x"})
    await store.put(("ops", "   "), "k2", {"note": "y"})
    assert fake.add_calls == []


async def test_serviceless_reads_return_empty_without_calling_mem0():
    fake = FakeMemory()
    store = Mem0ServiceMemory(fake, infer=True)
    assert await store.get((), "k") is None
    assert await store.search((), query="q") == []
    assert await store.delete((), "k") is False
    assert fake.get_all_calls == []
    assert fake.search_calls == []


# --- (c) PII redaction before .add() -------------------------------------

def test_redact_pii_strips_email_and_token():
    out = _redact_pii("contact alice@example.com with Bearer abc123secret")
    assert "alice@example.com" not in out
    assert "abc123secret" not in out
    assert "[redacted-email]" in out
    assert "[redacted-token]" in out


async def test_email_redacted_before_add():
    fake = FakeMemory()
    store = Mem0ServiceMemory(fake, infer=True)
    await store.put(
        ("ops", "acme-notes-be"),
        "incident",
        {"reporter": "alice@example.com", "note": "paged on-call"},
    )
    assert len(fake.add_calls) == 1
    blob = str(fake.add_calls[0]["messages"]) + str(fake.add_calls[0]["metadata"])
    assert "alice@example.com" not in blob
    assert "[redacted-email]" in blob


# --- (d) raising mem0 client must not propagate --------------------------

async def test_put_swallows_exceptions():
    store = Mem0ServiceMemory(RaisingMemory(), infer=True)
    # Must not raise.
    await store.put(("ops", "acme-notes-be"), "k", {"note": "x"})


async def test_get_search_delete_swallow_exceptions():
    store = Mem0ServiceMemory(RaisingMemory(), infer=True)
    assert await store.get(("ops", "acme-notes-be"), "k") is None
    assert await store.search(("ops", "acme-notes-be"), query="q") == []
    assert await store.delete(("ops", "acme-notes-be"), "k") is False


# --- adapter / Protocol sanity -------------------------------------------

async def test_search_adapts_results_to_memory_dataclass():
    fake = FakeMemory()
    fake.seed(
        [
            {
                "id": "row-1",
                "memory": "scaled acme-notes-be to 5 replicas",
                "metadata": {"_key": "scale-event", "service": "acme-notes-be"},
                "score": 0.91,
            }
        ]
    )
    store = Mem0ServiceMemory(fake, infer=True)
    out = await store.search(("ops", "acme-notes-be"), query="replicas", limit=5)
    assert len(out) == 1
    m = out[0]
    assert isinstance(m, Memory)
    assert m.key == "scale-event"
    assert m.namespace == ("ops", "acme-notes-be")
    assert m.value["memory"] == "scaled acme-notes-be to 5 replicas"
    # internal _key must not leak into surfaced metadata
    assert "_key" not in m.value["metadata"]


async def test_get_matches_by_logical_key():
    fake = FakeMemory()
    fake.seed(
        [
            {"id": "r1", "memory": "a", "metadata": {"_key": "other"}},
            {"id": "r2", "memory": "b", "metadata": {"_key": "wanted"}},
        ]
    )
    store = Mem0ServiceMemory(fake, infer=True)
    m = await store.get(("ops", "acme-notes-be"), "wanted")
    assert m is not None
    assert m.value["memory"] == "b"


def test_infer_flag_threaded_into_add(monkeypatch):
    fake = FakeMemory()
    store = Mem0ServiceMemory(fake, infer=False)
    import asyncio

    asyncio.run(store.put(("ops", "acme-notes-be"), "k", {"note": "x"}))
    assert fake.add_calls[0]["infer"] is False


def test_satisfies_memory_store_protocol():
    store = Mem0ServiceMemory(FakeMemory(), infer=True)
    assert isinstance(store, MemoryStore)


# --- server-side _key filter (fast path) + linear-scan fallback ----------
#
# Behaviour-equivalence guards for the get()/delete() optimisation: when the
# backend honours the {"user_id", "_key"} metadata filter we do strictly less
# work but return the SAME row; when it doesn't, the O(N) linear scan must
# still find the row. Both must yield the identical Memory the old code did.


class FilterAwareMemory(FakeMemory):
    """Mem0 fake that HONOURS the server-side ``_key`` metadata filter.

    Models a backend where ``get_all(filters={...,"_key":k})`` returns only the
    rows whose metadata ``_key`` equals ``k``. Records which calls were filtered
    so a test can assert the fast path (no full-table scan) was taken.
    """

    def __init__(self) -> None:
        super().__init__()
        self.filtered_get_all_keys: list[str] = []
        self.unfiltered_get_all_count: int = 0

    def get_all(self, *, filters=None, top_k=20, **kwargs):
        self.get_all_calls.append({"filters": filters, "top_k": top_k})
        filters = filters or {}
        if "_key" in filters:
            wanted = str(filters["_key"])
            self.filtered_get_all_keys.append(wanted)
            rows = [
                r
                for r in self._rows
                if str((r.get("metadata") or {}).get("_key")) == wanted
            ]
            return {"results": rows}
        self.unfiltered_get_all_count += 1
        return {"results": self._rows}


class FilterReorderingMemory(FakeMemory):
    """Mem0 fake that HONOURS the ``_key`` filter but REORDERS filtered hits.

    Models a vector store whose payload-filtered ``get_all`` returns matching
    rows in a DIFFERENT order than the unfiltered scan (e.g. by internal point
    id or score) -- exactly the condition under which a naive ``rows[0]`` fast
    path would diverge from the old first-match-in-iteration-order selection.

    * unfiltered ``get_all`` -> rows in seeded (==created) order (canonical)
    * filtered ``get_all`` (``_key`` present) -> matching rows REVERSED
    """

    def __init__(self) -> None:
        super().__init__()
        self.unfiltered_get_all_count: int = 0

    def get_all(self, *, filters=None, top_k=20, **kwargs):
        self.get_all_calls.append({"filters": filters, "top_k": top_k})
        filters = filters or {}
        if "_key" in filters:
            wanted = str(filters["_key"])
            matched = [
                r
                for r in self._rows
                if str((r.get("metadata") or {}).get("_key")) == wanted
            ]
            # Reverse to simulate a server-side order that differs from the
            # canonical unfiltered iteration order.
            return {"results": list(reversed(matched))}
        self.unfiltered_get_all_count += 1
        return {"results": self._rows}


class FilterUnawareMemory(FakeMemory):
    """Mem0 fake that IGNORES an unknown ``_key`` filter (returns everything).

    Models an older / stricter backend whose metadata-filter shape doesn't
    match -- it just returns all rows for the user_id regardless of ``_key``.
    Our Python re-verification + fallback must still pick the right row.
    """

    def __init__(self) -> None:
        super().__init__()
        self.unfiltered_get_all_count: int = 0

    def get_all(self, *, filters=None, top_k=20, **kwargs):
        # Ignores _key entirely -> always returns all rows.
        self.get_all_calls.append({"filters": filters, "top_k": top_k})
        self.unfiltered_get_all_count += 1
        return {"results": self._rows}


class FilterRaisingMemory(FakeMemory):
    """Mem0 fake where a ``_key`` filter RAISES (version-fragile shape).

    The unfiltered ``get_all`` (no ``_key``) still works -- proving the
    fallback engages on a raising filtered call and finds the row.
    """

    def get_all(self, *, filters=None, top_k=20, **kwargs):
        self.get_all_calls.append({"filters": filters, "top_k": top_k})
        if filters and "_key" in filters:
            raise RuntimeError("metadata filter shape not supported")
        return {"results": self._rows}


_TWO_ROWS = [
    {"id": "r1", "memory": "a", "metadata": {"_key": "other"}},
    {"id": "r2", "memory": "b", "metadata": {"_key": "wanted"}},
]


async def test_get_uses_server_side_filter_when_supported():
    fake = FilterAwareMemory()
    fake.seed(list(_TWO_ROWS))
    store = Mem0ServiceMemory(fake, infer=True)
    m = await store.get(("ops", "acme-notes-be"), "wanted")
    assert m is not None
    assert m.value["memory"] == "b"
    # Fast path: the server-side _key filter was used and NO unfiltered
    # full-table scan happened.
    assert fake.filtered_get_all_keys == ["wanted"]
    assert fake.unfiltered_get_all_count == 0


async def test_get_falls_back_when_filter_ignored():
    # Backend ignores the _key filter (returns all rows); Python re-verify must
    # still pick the right row. Same Memory the old linear scan produced.
    fake = FilterUnawareMemory()
    fake.seed(list(_TWO_ROWS))
    store = Mem0ServiceMemory(fake, infer=True)
    m = await store.get(("ops", "acme-notes-be"), "wanted")
    assert m is not None
    assert m.value["memory"] == "b"


async def test_get_falls_back_when_filter_raises():
    # Server-side filter raises -> linear-scan fallback finds the row.
    fake = FilterRaisingMemory()
    fake.seed(list(_TWO_ROWS))
    store = Mem0ServiceMemory(fake, infer=True)
    m = await store.get(("ops", "acme-notes-be"), "wanted")
    assert m is not None
    assert m.value["memory"] == "b"


async def test_get_missing_key_returns_none_filter_aware():
    fake = FilterAwareMemory()
    fake.seed(list(_TWO_ROWS))
    store = Mem0ServiceMemory(fake, infer=True)
    # No row carries this _key; fast path returns empty -> fallback scans ->
    # still empty -> None (same as old behaviour).
    assert await store.get(("ops", "acme-notes-be"), "nope") is None


async def test_get_equivalent_across_all_backend_shapes():
    """The Memory returned is identical regardless of filter support."""
    results = []
    for cls in (FakeMemory, FilterAwareMemory, FilterUnawareMemory, FilterRaisingMemory):
        fake = cls()
        fake.seed(list(_TWO_ROWS))
        store = Mem0ServiceMemory(fake, infer=True)
        m = await store.get(("ops", "acme-notes-be"), "wanted")
        results.append(None if m is None else (m.key, m.value["memory"]))
    # All four backend shapes yield the same logical result.
    assert results == [("wanted", "b")] * 4


# --- duplicate-logical-key ordering equivalence (regression) -------------
#
# Two rows share ONE logical _key. The pre-optimisation get() did a single
# unfiltered get_all() and returned the FIRST matching row in iteration order.
# The server-side _key fast path may return those rows in a DIFFERENT order
# (modelled by FilterReorderingMemory), so a naive rows[0] would pick the WRONG
# row. get() must reproduce the old first-match selection regardless.

_DUP_KEY_ROWS = [
    # First in unfiltered iteration / created order -> this is what the
    # pre-change linear scan returned for _key="dup".
    {"id": "first", "memory": "FIRST", "metadata": {"_key": "dup"}},
    {"id": "second", "memory": "SECOND", "metadata": {"_key": "dup"}},
    {"id": "x", "memory": "other", "metadata": {"_key": "elsewhere"}},
]


def _old_linear_scan_first_match(rows: list[dict], key: str) -> dict | None:
    """Reproduce the EXACT pre-change get() selection: first matching row in
    the unfiltered get_all() iteration order."""
    for item in rows:
        if str((item.get("metadata") or {}).get("_key")) == str(key):
            return item
    return None


async def test_get_dup_key_matches_old_linear_scan_when_filter_reorders():
    """Multi-row dup key + a backend that reorders filtered hits: get() must
    still return the SAME row the old unfiltered linear scan would have."""
    fake = FilterReorderingMemory()
    fake.seed([dict(r) for r in _DUP_KEY_ROWS])
    store = Mem0ServiceMemory(fake, infer=True)

    m = await store.get(("ops", "svc"), "dup")

    expected = _old_linear_scan_first_match(_DUP_KEY_ROWS, "dup")
    assert expected is not None
    assert m is not None
    # The pre-change code returned the FIRST row in iteration order ("first").
    assert m.key == "dup"
    assert m.value["memory"] == expected["memory"] == "FIRST"
    # And it must have DEFERRED to the unfiltered linear scan (multi-row +
    # order-sensitive) rather than trusting the reordered fast-path order.
    assert fake.unfiltered_get_all_count == 1


async def test_get_dup_key_identical_across_all_backend_shapes():
    """For a dup logical key, every backend shape -- including one that
    reorders filtered hits -- yields the identical first-match Memory the old
    linear scan produced."""
    expected = _old_linear_scan_first_match(_DUP_KEY_ROWS, "dup")
    assert expected is not None
    results = []
    for cls in (
        FakeMemory,
        FilterAwareMemory,
        FilterReorderingMemory,
        FilterUnawareMemory,
        FilterRaisingMemory,
    ):
        fake = cls()
        fake.seed([dict(r) for r in _DUP_KEY_ROWS])
        store = Mem0ServiceMemory(fake, infer=True)
        m = await store.get(("ops", "svc"), "dup")
        results.append(None if m is None else (m.key, m.value["memory"]))
    assert results == [("dup", expected["memory"])] * 5
    assert expected["memory"] == "FIRST"


async def test_get_single_row_dup_key_still_uses_fast_path():
    """Order-sensitivity only defers on MULTI-row hits: a single matching row
    is unambiguous, so the fast path is taken (no unfiltered full scan)."""
    fake = FilterReorderingMemory()
    fake.seed(
        [
            {"id": "only", "memory": "ONLY", "metadata": {"_key": "dup"}},
            {"id": "x", "memory": "other", "metadata": {"_key": "elsewhere"}},
        ]
    )
    store = Mem0ServiceMemory(fake, infer=True)
    m = await store.get(("ops", "svc"), "dup")
    assert m is not None
    assert m.value["memory"] == "ONLY"
    # Single-row fast path -> no unfiltered scan needed.
    assert fake.unfiltered_get_all_count == 0


async def test_delete_dup_key_sweeps_all_rows_under_reordering():
    """delete() is order-insensitive: under a reordering backend it still
    deletes the FULL SET of rows sharing the logical key."""
    fake = FilterReorderingMemory()
    fake.seed([dict(r) for r in _DUP_KEY_ROWS])
    store = Mem0ServiceMemory(fake, infer=True)
    deleted = await store.delete(("ops", "svc"), "dup")
    assert deleted is True
    got = sorted(c["memory_id"] for c in fake.delete_calls)
    assert got == ["first", "second"]


async def test_delete_uses_server_side_filter_when_supported():
    fake = FilterAwareMemory()
    fake.seed(list(_TWO_ROWS))
    store = Mem0ServiceMemory(fake, infer=True)
    deleted = await store.delete(("ops", "acme-notes-be"), "wanted")
    assert deleted is True
    # Only the matching row's id was deleted; no full scan.
    assert fake.delete_calls == [{"memory_id": "r2"}]
    assert fake.unfiltered_get_all_count == 0


async def test_delete_falls_back_when_filter_raises():
    fake = FilterRaisingMemory()
    fake.seed(list(_TWO_ROWS))
    store = Mem0ServiceMemory(fake, infer=True)
    deleted = await store.delete(("ops", "acme-notes-be"), "wanted")
    assert deleted is True
    assert fake.delete_calls == [{"memory_id": "r2"}]


async def test_delete_sweeps_all_rows_with_same_key():
    # Multiple rows share one logical key -> ALL are deleted (set, not order).
    rows = [
        {"id": "a", "memory": "x", "metadata": {"_key": "dup"}},
        {"id": "b", "memory": "y", "metadata": {"_key": "dup"}},
        {"id": "c", "memory": "z", "metadata": {"_key": "keep"}},
    ]
    for cls in (FilterAwareMemory, FilterUnawareMemory, FilterRaisingMemory):
        fake = cls()
        fake.seed([dict(r) for r in rows])
        store = Mem0ServiceMemory(fake, infer=True)
        deleted = await store.delete(("ops", "svc"), "dup")
        assert deleted is True
        got = sorted(c["memory_id"] for c in fake.delete_calls)
        assert got == ["a", "b"], f"{cls.__name__} deleted {got}"


async def test_delete_missing_key_returns_false():
    fake = FilterAwareMemory()
    fake.seed(list(_TWO_ROWS))
    store = Mem0ServiceMemory(fake, infer=True)
    assert await store.delete(("ops", "acme-notes-be"), "nope") is False
    assert fake.delete_calls == []
