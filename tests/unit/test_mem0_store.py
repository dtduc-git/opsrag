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
