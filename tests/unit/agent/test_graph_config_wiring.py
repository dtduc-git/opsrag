"""Wiring regressions: knobs that existed but never reached their consumer.

Four interrelated dead-wire fixes, each tested at the seam where the value
was being dropped (NOT at the layer that already worked):

1. F13 carve-out: ``query_with_session`` / ``query_with_session_events`` must
   pass ``user_scope`` into ``investigation_cache.store`` on the tool-path
   branch -- the scope was only computed in the mutually-exclusive qa_cache
   branch, so the cross-user per-user-memory carve-out never activated.

2. recursion_limit: ``graph.py`` must forward ``MULTI_AGENT_RECURSION_LIMIT``
   into the runnable config (``ainvoke`` + ``astream_events``); LangGraph
   otherwise silently applies its default of 25.

3. verify_grounding: ``cfg.agent.verify_grounding_default`` must flow
   config -> ``build_multi_agent_graph(verify_grounding=...)``.

4. rerank content-dedup: ``cfg.agent.rerank_content_dedup`` /
   ``rerank_content_dedup_threshold`` must flow config -> builder ->
   ``rerank_node`` closure, so config-only settings (no state injection)
   actually take effect.
"""
from __future__ import annotations

import asyncio

from opsrag.agent.graph import (
    query_with_session,
    query_with_session_events,
)


class _FakeEmbedder:
    async def embed_query(self, text):  # noqa: ANN001
        return [0.0, 1.0, 0.0]


class _NoOpQaCache:
    """Never hits; records nothing -- keeps the qa_cache branch out of the way."""

    async def lookup(self, *a, **k):  # noqa: ANN001, ANN002
        return None

    async def store(self, *a, **k):  # noqa: ANN001, ANN002
        return "qa-id"


class _RecordingInvestigationCache:
    """Records the ``user_scope`` kwarg seen by every ``store`` call."""

    def __init__(self) -> None:
        self.scopes_seen: list[object] = []

    async def search(self, *a, **k):  # noqa: ANN001, ANN002
        return []

    async def store(self, *a, user_scope=None, **k):  # noqa: ANN001, ANN002
        self.scopes_seen.append(user_scope)
        return "inv-id"


def _tool_path_result(initial, *, user_memories):
    """A tool-path final-state dict, optionally carrying per-user memories."""
    out = {
        "query": initial.get("query"),
        "generation": "an answer from live tools",
        "generation_grounded": True,
        "grounding_checked": True,
        "tool_path_active": True,
        "tool_call_audit": [{"name": "k8s_list_pods"}],
        "model_route_decision": {},
        "final_chunks": [],
        "sources_searched": [],
    }
    if user_memories:
        out["user_memories"] = user_memories
    return out


class _ToolPathInvokeGraph:
    def __init__(self, *, user_memories):
        self._user_memories = user_memories

    async def ainvoke(self, initial, config=None) -> dict:  # noqa: ANN001
        return _tool_path_result(initial, user_memories=self._user_memories)


class _ToolPathStreamGraph:
    def __init__(self, *, user_memories):
        self._user_memories = user_memories

    async def astream_events(self, initial, config=None, version=None):  # noqa: ANN001
        yield {
            "event": "on_chain_end",
            "name": "LangGraph",
            "data": {"output": _tool_path_result(initial, user_memories=self._user_memories)},
        }


# --- Fix #1: investigation store receives user_scope (the WIRING) ----------


def test_investigation_store_scoped_when_answer_has_user_memories():
    """When the tool-path answer wove in per-user memories, the store call on
    the investigation cache must receive a NON-None user_scope == user_id."""
    inv = _RecordingInvestigationCache()
    asyncio.run(
        query_with_session(
            compiled_graph=_ToolPathInvokeGraph(user_memories=["alice likes prod"]),
            query="is my service healthy right now",
            user_id="alice",
            thread_id="t1",
            embedder=_FakeEmbedder(),
            qa_cache=_NoOpQaCache(),
            investigation_cache=inv,
        )
    )
    assert inv.scopes_seen == ["alice"], (
        "investigation store did not receive the per-user scope -- F13 carve-out is dead"
    )


def test_investigation_store_unscoped_when_no_user_memories():
    """A shared (no per-user-memory) tool-path answer must store with
    user_scope=None so it stays globally searchable."""
    inv = _RecordingInvestigationCache()
    asyncio.run(
        query_with_session(
            compiled_graph=_ToolPathInvokeGraph(user_memories=None),
            query="is the deploy pipeline green right now",
            user_id="alice",
            thread_id="t1",
            embedder=_FakeEmbedder(),
            qa_cache=_NoOpQaCache(),
            investigation_cache=inv,
        )
    )
    assert inv.scopes_seen == [None]


def test_investigation_store_scoped_events_path():
    """Same carve-out on the streaming path."""
    inv = _RecordingInvestigationCache()

    async def _go():
        async for _ in query_with_session_events(
            _ToolPathStreamGraph(user_memories=["alice likes prod"]),
            query="is my service healthy right now",
            user_id="alice",
            thread_id="t1",
            embedder=_FakeEmbedder(),
            qa_cache=_NoOpQaCache(),
            investigation_cache=inv,
        ):
            pass

    asyncio.run(_go())
    assert inv.scopes_seen == ["alice"]


# --- Fix #2: recursion_limit forwarded into the runnable config ------------


class _ConfigRecordingInvokeGraph:
    def __init__(self) -> None:
        self.config_seen: dict | None = None

    async def ainvoke(self, initial, config=None) -> dict:  # noqa: ANN001
        self.config_seen = config
        return {
            "query": initial.get("query"),
            "generation": "an answer",
            "generation_grounded": True,
            "grounding_checked": True,
            "final_chunks": [],
        }


class _ConfigRecordingStreamGraph:
    def __init__(self) -> None:
        self.config_seen: dict | None = None

    async def astream_events(self, initial, config=None, version=None):  # noqa: ANN001
        self.config_seen = config
        yield {
            "event": "on_chain_end",
            "name": "LangGraph",
            "data": {
                "output": {
                    "query": initial.get("query"),
                    "generation": "an answer",
                    "generation_grounded": True,
                    "grounding_checked": True,
                    "final_chunks": [],
                    "sources_searched": [],
                }
            },
        }


def test_recursion_limit_forwarded_to_ainvoke():
    from opsrag.agent.nodes.multi_agent import MULTI_AGENT_RECURSION_LIMIT

    g = _ConfigRecordingInvokeGraph()
    asyncio.run(
        query_with_session(
            compiled_graph=g,
            query="why did the deploy fail",
            user_id="u1",
            thread_id="t1",
            embedder=_FakeEmbedder(),
        )
    )
    assert g.config_seen is not None
    assert g.config_seen.get("recursion_limit") == MULTI_AGENT_RECURSION_LIMIT
    # configurable identity payload must still be present alongside it.
    assert g.config_seen["configurable"]["thread_id"] == "t1"


def test_recursion_limit_forwarded_to_astream_events():
    from opsrag.agent.nodes.multi_agent import MULTI_AGENT_RECURSION_LIMIT

    g = _ConfigRecordingStreamGraph()

    async def _go():
        async for _ in query_with_session_events(
            g,
            query="why did the deploy fail",
            user_id="u1",
            thread_id="t1",
            embedder=_FakeEmbedder(),
        ):
            pass

    asyncio.run(_go())
    assert g.config_seen is not None
    assert g.config_seen.get("recursion_limit") == MULTI_AGENT_RECURSION_LIMIT


# --- shared fakes for the config->builder spy tests ------------------------


class _Fake:
    """Stand-in provider whose every attribute access returns a no-op callable.
    Node factories only build closures at construction time, so they never
    actually exercise these."""

    def __getattr__(self, name):  # noqa: ANN001
        def _noop(*a, **k):  # noqa: ANN002, ANN003
            return None
        return _noop


class _FakeReranker:
    score_floor = 0.05
    trust_score = 0.65

    async def rerank(self, *a, **k):  # noqa: ANN001, ANN002
        return []


# --- Fix #3: verify_grounding flows config -> builder -> generator_node -----


def test_verify_grounding_flows_config_to_builder(monkeypatch):
    """``build_multi_agent_graph(verify_grounding=...)`` must forward the value
    into ``generator_node`` -- so cfg.agent.verify_grounding_default isn't a
    dead knob. Spy the generator factory the builder calls."""
    import opsrag.agent.graph as graph_mod

    captured: dict = {}

    def _spy_generator_node(*a, verify_grounding=True, **k):  # noqa: ANN002, ANN003
        captured["verify_grounding"] = verify_grounding
        async def _node(state):  # noqa: ANN001
            return {}
        return _node

    monkeypatch.setattr(graph_mod, "generator_node", _spy_generator_node)

    for flag in (False, True):
        captured.clear()
        graph_mod.build_multi_agent_graph(
            llm=_Fake(), vector_store=_Fake(), embedder=_Fake(),
            observability=_Fake(), reranker=_FakeReranker(),
            verify_grounding=flag,
        )
        assert captured["verify_grounding"] is flag


# --- Fix #4: rerank dedup flows config -> builder -> rerank_node closure ----


def test_rerank_content_dedup_flows_config_to_builder(monkeypatch):
    """All four builders must forward ``content_dedup`` /
    ``content_dedup_threshold`` into ``rerank_node`` -- otherwise the cfg
    values can never take effect (no state populates them, no closure fallback
    existed). Spy the rerank factory and exercise every builder."""
    import opsrag.agent.graph as graph_mod

    captured: list[dict] = []

    def _spy_rerank_node(*a, content_dedup=True, content_dedup_threshold=0.0, **k):  # noqa: ANN002, ANN003
        captured.append(
            {"content_dedup": content_dedup, "content_dedup_threshold": content_dedup_threshold}
        )
        async def _node(state):  # noqa: ANN001
            return {}
        return _node

    monkeypatch.setattr(graph_mod, "rerank_node", _spy_rerank_node)

    common = dict(
        llm=_Fake(), vector_store=_Fake(), embedder=_Fake(),
        observability=_Fake(), reranker=_FakeReranker(),
        rerank_content_dedup=False, rerank_content_dedup_threshold=0.7,
    )

    graph_mod.build_minimal_graph(**common)
    graph_mod.build_multi_agent_graph(**common)
    graph_mod.build_tool_calling_graph(**common)
    # build_full_graph requires reranker positionally (non-optional) -- same kwargs work.
    graph_mod.build_full_graph(**common)

    assert len(captured) == 4, "a builder did not construct the rerank node"
    for seen in captured:
        assert seen["content_dedup"] is False
        assert seen["content_dedup_threshold"] == 0.7
