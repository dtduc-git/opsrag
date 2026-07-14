"""R1 (zero-cost weak-retrieval gate) + R4 (artifact verifier) wiring on the
multi_agent / tool_calling graphs.

R1: on the RERANKER branch only, ``rerank`` must route through
``rerank_decision`` so a query that named anchors but matched none of the kept
chunks (with a sub-floor cross-encoder score) lands on ``insufficient_info``
instead of fabricating an answer from adjacent chunks. Mirrors build_full_graph.

R4: when ``verify_artifacts`` is on (cfg.agent.verify_artifacts_default,
default True), the precise artifact/citation ``verify_answer`` node must sit
after BOTH terminal generators (the tool-path ``generator`` and the
retrieval-path ``generate``). When off, both go straight to END.

These assert at the compiled-graph topology level (node set + edge set), which
is where the value was being dropped -- not at the node bodies, which are
tested elsewhere.
"""
from __future__ import annotations

import opsrag.agent.graph as graph_mod
from opsrag.agent.nodes import answer_verifier as av
from opsrag.agent.nodes.reranker import rerank_decision


class _Fake:
    """Every attribute access returns a no-op callable. Node factories only
    build closures at construction time, so they never exercise these."""

    def __getattr__(self, name):  # noqa: ANN001
        def _noop(*a, **k):  # noqa: ANN002, ANN003
            return None
        return _noop


class _FakeReranker:
    score_floor = 0.05
    trust_score = 0.65

    async def rerank(self, *a, **k):  # noqa: ANN001, ANN002
        return []


def _edges(compiled) -> set[tuple[str, str]]:
    g = compiled.get_graph()
    return {(e.source, e.target) for e in g.edges}


def _nodes(compiled) -> set[str]:
    return set(compiled.get_graph().nodes.keys())


# --- R1: rerank_decision returns weak_retrieval -----------------------------


def test_rerank_decision_weak_retrieval_when_anchors_unmatched_and_low_score():
    """Anchors present, none matched in kept results, best score below floor
    -> weak_retrieval."""
    state = {
        "anchors": ["acme-analytics-v3"],
        "anchors_matched_in_results": False,
        "best_rerank_score": 0.01,  # below the 0.05 default floor
        "min_rerank_score": 0.05,
        "merged_results": [object()],  # non-empty so the empty-set branch is skipped
    }
    assert rerank_decision(state) == "weak_retrieval"


def test_rerank_decision_ok_when_anchor_matched():
    """If an anchor matched a kept chunk, retrieval is on-topic -> ok, even at
    a low score."""
    state = {
        "anchors": ["acme-analytics-v3"],
        "anchors_matched_in_results": True,
        "best_rerank_score": 0.01,
        "min_rerank_score": 0.05,
        "merged_results": [object()],
    }
    assert rerank_decision(state) == "ok"


def test_rerank_decision_ok_when_score_above_floor():
    """No anchor match but a strong cross-encoder score -> ok (the doc is
    semantically relevant even without a path/repo token hit)."""
    state = {
        "anchors": ["acme-analytics-v3"],
        "anchors_matched_in_results": False,
        "best_rerank_score": 0.9,
        "min_rerank_score": 0.05,
        "merged_results": [object()],
    }
    assert rerank_decision(state) == "ok"


def test_rerank_decision_weak_retrieval_when_nothing_kept():
    state = {"merged_results": [], "anchors": [], "best_rerank_score": 0.0}
    assert rerank_decision(state) == "weak_retrieval"


# --- R1: multi_agent retrieval branch routes rerank -> insufficient_info ----


def test_multi_agent_rerank_gated_to_insufficient_info():
    compiled = graph_mod.build_multi_agent_graph(
        llm=_Fake(), vector_store=_Fake(), embedder=_Fake(),
        observability=_Fake(), reranker=_FakeReranker(),
    )
    edges = _edges(compiled)
    nodes = _nodes(compiled)
    assert "insufficient_info" in nodes
    # Conditional edge both legs present.
    assert ("rerank", "insufficient_info") in edges
    assert ("rerank", "generate") in edges
    # The weak-retrieval fallback is terminal.
    assert ("insufficient_info", "__end__") in edges


def test_multi_agent_no_reranker_has_no_weak_gate():
    """Without a reranker there's no best_rerank_score to gate on -- keep the
    direct vector_retrieve -> generate edge and don't wire insufficient_info."""
    compiled = graph_mod.build_multi_agent_graph(
        llm=_Fake(), vector_store=_Fake(), embedder=_Fake(),
        observability=_Fake(), reranker=None,
    )
    edges = _edges(compiled)
    assert "insufficient_info" not in _nodes(compiled)
    assert ("vector_retrieve", "generate") in edges


def test_tool_calling_rerank_gated_to_insufficient_info():
    compiled = graph_mod.build_tool_calling_graph(
        llm=_Fake(), vector_store=_Fake(), embedder=_Fake(),
        observability=_Fake(), reranker=_FakeReranker(),
    )
    edges = _edges(compiled)
    assert "insufficient_info" in _nodes(compiled)
    assert ("rerank", "insufficient_info") in edges
    assert ("rerank", "generate") in edges
    assert ("insufficient_info", "__end__") in edges


# --- R4: verify_answer wired after both generators when the flag is on -------


def test_multi_agent_verify_answer_wired_when_flag_on():
    compiled = graph_mod.build_multi_agent_graph(
        llm=_Fake(), vector_store=_Fake(), embedder=_Fake(),
        observability=_Fake(), reranker=_FakeReranker(),
        verify_artifacts=True,
    )
    edges = _edges(compiled)
    assert "verify_answer" in _nodes(compiled)
    # Tool path: generator -> verify_answer (not straight to END).
    assert ("generator", "verify_answer") in edges
    assert ("generator", "__end__") not in edges
    # Retrieval path: generate -> verify_answer.
    assert ("generate", "verify_answer") in edges
    assert ("generate", "__end__") not in edges
    # verify_answer is terminal.
    assert ("verify_answer", "__end__") in edges


def test_multi_agent_verify_answer_absent_when_flag_off():
    compiled = graph_mod.build_multi_agent_graph(
        llm=_Fake(), vector_store=_Fake(), embedder=_Fake(),
        observability=_Fake(), reranker=_FakeReranker(),
        verify_artifacts=False,
    )
    edges = _edges(compiled)
    assert "verify_answer" not in _nodes(compiled)
    # Both generators terminate directly at END.
    assert ("generator", "__end__") in edges
    assert ("generate", "__end__") in edges


def test_verify_artifacts_flows_config_to_builder(monkeypatch):
    """``build_multi_agent_graph(verify_artifacts=...)`` must decide whether to
    construct the verify_answer node -- so cfg.agent.verify_artifacts_default
    isn't a dead knob. Spy the factory and assert it's only built when on."""
    built: list[bool] = []

    def _spy_verify_answer_node(*a, **k):  # noqa: ANN002, ANN003
        built.append(True)
        async def _node(state):  # noqa: ANN001
            return {}
        return _node

    monkeypatch.setattr(graph_mod, "verify_answer_node", _spy_verify_answer_node)

    graph_mod.build_multi_agent_graph(
        llm=_Fake(), vector_store=_Fake(), embedder=_Fake(),
        observability=_Fake(), reranker=_FakeReranker(),
        verify_artifacts=False,
    )
    assert built == [], "verify_answer node built despite verify_artifacts=False"

    graph_mod.build_multi_agent_graph(
        llm=_Fake(), vector_store=_Fake(), embedder=_Fake(),
        observability=_Fake(), reranker=_FakeReranker(),
        verify_artifacts=True,
    )
    assert built == [True], "verify_answer node not built despite verify_artifacts=True"


# --- R4: live-tool evidence feeds the verifier (tool-only answers) -----------


class _CapturingVerifierLLM:
    """Captures the user message the verifier sends, returns an all-verified
    verdict so the node runs to completion."""

    model_name = "fake-model"

    def __init__(self) -> None:
        self.user_msg: str | None = None

    async def generate_structured(self, **kw):  # noqa: ANN003
        msgs = kw.get("messages") or []
        self.user_msg = msgs[0]["content"] if msgs else ""
        schema = kw["schema"]
        return schema(verified=["k8s_list_pods"], unverifiable=[])


async def test_verifier_feeds_live_tool_evidence():
    """A tool-only answer (no doc chunks) must not be flagged unverifiable:
    the verifier evidence block must include the tool audit + tool_result
    payloads so tool-grounded facts are matchable."""
    llm = _CapturingVerifierLLM()
    node = av.verify_answer_node(llm, None, None)
    state = {
        "generation": "The pod payments-7c9 is CrashLoopBackOff (via k8s_list_pods).",
        "final_chunks": [],  # tool-only turn: no doc evidence
        "tool_call_audit": [
            {"name": "k8s_list_pods", "args": {"namespace": "payments"}},
        ],
        "tool_message_history": [
            {"role": "tool_call", "name": "k8s_list_pods", "args": {"namespace": "payments"}},
            {
                "role": "tool_result",
                "name": "k8s_list_pods",
                "response": {"text": "payments-7c9  CrashLoopBackOff  restarts=14"},
            },
        ],
    }
    out = await node(state)
    assert llm.user_msg is not None
    # The audit (which tool fired) AND the tool_result payload must be in the
    # evidence the verifier saw.
    assert "k8s_list_pods" in llm.user_msg
    assert "CrashLoopBackOff" in llm.user_msg
    assert "Live tool calls" in llm.user_msg
    # All-verified verdict -> no hedge, so the node leaves `generation`
    # untouched (it only writes `generation` when it prepends a hedge).
    assert "generation" not in out
    assert out["verification_result"]["unverifiable"] == []


async def test_verifier_skips_errored_tool_results_as_evidence():
    """An errored tool result is not evidence -- it must not appear in the
    evidence block (otherwise an error string could spuriously 'verify' a
    claim)."""
    llm = _CapturingVerifierLLM()
    node = av.verify_answer_node(llm, None, None)
    state = {
        "generation": "some answer",
        "final_chunks": [],
        "tool_call_audit": [
            {"name": "k8s_list_pods", "error": "forbidden"},  # errored -> skipped
        ],
        "tool_message_history": [
            {
                "role": "tool_result",
                "name": "k8s_list_pods",
                "response": {"error": "forbidden"},
            },
        ],
    }
    await node(state)
    assert llm.user_msg is not None
    assert "forbidden" not in llm.user_msg
    # No successful tool evidence -> the live-tool section is omitted entirely.
    assert "Live tool calls" not in llm.user_msg
