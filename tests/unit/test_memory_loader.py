"""Unit tests for the memory_loader graph node.

These lock the latency optimisation that fans the three independent Mem0 reads
(preferences get; topics search; query-relevant search) out with
``asyncio.gather`` instead of running them serially. The assertions prove the
optimised node produces the SAME output state as a faithful serial reference
across every interesting case: empty store, partial population, full
population, and any single read raising / returning falsy.
"""
from __future__ import annotations

from datetime import UTC, datetime

from opsrag.agent.nodes.memory_loader import load_memory_node
from opsrag.interfaces.memory import Memory


def _mem(key: str, value: dict | None = None) -> Memory:
    now = datetime.now(UTC)
    return Memory(
        key=key,
        namespace=("user", "u1"),
        value=value if value is not None else {"memory": key},
        created_at=now,
        updated_at=now,
    )


class FakeStore:
    """Records reads; returns canned per-namespace results.

    ``get`` keys on the namespace tuple; ``search`` keys on namespace, with the
    semantic 'topics' vs 'query-relevant' calls distinguished by the presence
    of a ``query``. Any entry may be a callable that is invoked to allow raising
    (mimicking a backend method that does NOT swallow -- the node must cope).
    """

    def __init__(self, *, pref=None, topics=None, relevant=None) -> None:
        self.pref = pref
        self.topics = topics if topics is not None else []
        self.relevant = relevant if relevant is not None else []
        self.calls: list[tuple] = []

    @staticmethod
    def _resolve(v):
        return v() if callable(v) else v

    async def get(self, namespace, key):
        self.calls.append(("get", namespace, key))
        return self._resolve(self.pref)

    async def search(self, namespace, query=None, limit=10):
        self.calls.append(("search", namespace, query, limit))
        if query is None:
            return self._resolve(self.topics)
        return self._resolve(self.relevant)


async def _serial_reference(store, state: dict) -> dict:
    """A faithful re-implementation of the ORIGINAL serial node logic.

    Used as the oracle: the optimised node's output must equal this for every
    input we throw at it.
    """
    user_id = state.get("user_id", "anonymous")
    prefs: dict = {}
    context_parts: list[str] = []
    user_memories: list = []
    try:
        pref = await store.get(("user", user_id, "preferences"), "default")
        if pref:
            prefs = pref.value
        topics = await store.search(("user", user_id, "topics"), limit=5)
        if topics:
            topic_names = [t.key for t in topics]
            context_parts.append(f"Frequent topics: {', '.join(topic_names)}")
    except Exception:
        pass
    try:
        q = state.get("query") or ""
        mems = await store.search(("user", user_id), query=q, limit=6)
        user_memories = mems or []
    except Exception:
        user_memories = []
    return {
        "user_preferences": prefs,
        "session_context": " | ".join(context_parts) if context_parts else "",
        "user_memories": user_memories,
        "current_step": "memory_loaded",
    }


def _make_state():
    return {"user_id": "u1", "query": "why is checkout 500ing"}


# --- equivalence across the interesting cases ----------------------------

async def _assert_equiv(*, pref, topics, relevant, state=None):
    state = state if state is not None else _make_state()
    optimised = load_memory_node(FakeStore(pref=pref, topics=topics, relevant=relevant), None)
    out_opt = await optimised(dict(state))
    out_ref = await _serial_reference(
        FakeStore(pref=pref, topics=topics, relevant=relevant), dict(state)
    )
    assert out_opt == out_ref
    return out_opt


async def test_empty_store_matches_serial():
    out = await _assert_equiv(pref=None, topics=[], relevant=[])
    assert out == {
        "user_preferences": {},
        "session_context": "",
        "user_memories": [],
        "current_step": "memory_loaded",
    }


async def test_full_population_matches_serial():
    pref = _mem("default", {"tone": "concise", "lang": "en"})
    topics = [_mem("checkout"), _mem("payments")]
    relevant = [_mem("user prefers terse answers"), _mem("owns checkout svc")]
    out = await _assert_equiv(pref=pref, topics=topics, relevant=relevant)
    assert out["user_preferences"] == {"tone": "concise", "lang": "en"}
    assert out["session_context"] == "Frequent topics: checkout, payments"
    assert out["user_memories"] == relevant


async def test_only_prefs_matches_serial():
    pref = _mem("default", {"tone": "concise"})
    await _assert_equiv(pref=pref, topics=[], relevant=[])


async def test_only_topics_matches_serial():
    await _assert_equiv(pref=None, topics=[_mem("dns"), _mem("tls")], relevant=[])


async def test_only_relevant_matches_serial():
    await _assert_equiv(pref=None, topics=[], relevant=[_mem("uses bedrock")])


# --- a raising read must not sink the others (return_exceptions guard) ----
#
# NOTE on equivalence scope: the production MemoryStore methods are best-effort
# and NEVER raise (they swallow + return {} / [] internally), so for any
# contract-conforming store the gather path output is byte-identical to the
# serial path -- that is what test_*_matches_serial above proves. The
# return_exceptions=True guard is belt-and-braces for a MISBEHAVING store. We
# deliberately do NOT compare the raising case against the serial oracle,
# because the old serial code coupled prefs+topics in a SINGLE try block (a
# pref raise also nuked topics) -- an incidental artifact of sequencing, not a
# contract. The gather path correctly isolates each read; we assert that.

def _raise():
    raise RuntimeError("backend blew up")


async def test_pref_read_raising_isolated_from_others():
    # pref raises -> prefs={} -- but topics and relevant are independent reads
    # and still load. (Better isolation than the old serial code; never reached
    # in production since the store swallows internally.)
    relevant = [_mem("r")]
    node = load_memory_node(
        FakeStore(pref=_raise, topics=[_mem("x")], relevant=relevant), None
    )
    out = await node(_make_state())
    assert out["user_preferences"] == {}
    assert out["session_context"] == "Frequent topics: x"
    assert out["user_memories"] == relevant
    assert out["current_step"] == "memory_loaded"


async def test_relevant_read_raising_isolated_from_others():
    node = load_memory_node(
        FakeStore(
            pref=_mem("default", {"tone": "concise"}),
            topics=[_mem("x")],
            relevant=_raise,
        ),
        None,
    )
    out = await node(_make_state())
    assert out["user_memories"] == []
    assert out["user_preferences"] == {"tone": "concise"}
    assert out["session_context"] == "Frequent topics: x"


async def test_falsy_relevant_falls_back_to_empty_list():
    # `mems or []` -> None/empty becomes [].
    out = await _assert_equiv(pref=None, topics=[], relevant=None)
    assert out["user_memories"] == []


# --- the three reads actually go to the right namespaces ------------------

async def test_reads_target_expected_namespaces():
    store = FakeStore(pref=None, topics=[], relevant=[])
    node = load_memory_node(store, None)
    await node(_make_state())
    # All three reads issued, with the documented namespaces/args.
    assert ("get", ("user", "u1", "preferences"), "default") in store.calls
    assert ("search", ("user", "u1", "topics"), None, 5) in store.calls
    assert ("search", ("user", "u1"), "why is checkout 500ing", 6) in store.calls


async def test_missing_query_uses_empty_string():
    store = FakeStore(pref=None, topics=[], relevant=[])
    node = load_memory_node(store, None)
    await node({"user_id": "u1"})  # no "query"
    assert ("search", ("user", "u1"), "", 6) in store.calls


async def test_reads_run_concurrently_not_serially():
    """Proof of the optimisation: the three reads overlap in time.

    A serial implementation would take ~3x the per-read delay; the gather
    implementation overlaps them so total wall time is ~1x. We assert the three
    reads are all in-flight simultaneously (peak concurrency == 3).
    """
    import asyncio as _asyncio

    in_flight = 0
    peak = 0
    started = _asyncio.Event()

    class SlowStore:
        async def _work(self, ret):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            # Yield enough to let the sibling coroutines start before we finish.
            await _asyncio.sleep(0.02)
            in_flight -= 1
            return ret

        async def get(self, namespace, key):
            return await self._work(None)

        async def search(self, namespace, query=None, limit=10):
            return await self._work([])

    node = load_memory_node(SlowStore(), None)
    out = await node(_make_state())
    assert peak == 3, f"expected all 3 reads concurrent, peak was {peak}"
    # And output is still the canonical empty-state shape.
    assert out == {
        "user_preferences": {},
        "session_context": "",
        "user_memories": [],
        "current_step": "memory_loaded",
    }
