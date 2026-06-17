"""L7 -- behavior-equivalence for PARALLELIZED tool dispatch.

`tool_caller_node`'s per-call loop used to `await` each `tool.call(...)`
serially: every network round-trip blocked the next call. The optimized
node fans the REAL MCP dispatches out concurrently (`asyncio.gather`) and
re-applies their results IN ORIGINAL pending ORDER, while keeping
`update_plan` (state mutation) and unknown-tool (F7 registry-miss) calls
on the serial path.

These tests prove the optimization is OBSERVABLY EQUIVALENT to the serial
version -- identical ordered `tool_message_history`, identical
`tool_call_audit` content + order, identical `tool_call_count` budget
accounting (F7), and identical de-duplicated retrieval chunks -- while the
dispatches actually run concurrently (network IO overlaps).

Quality invariant: ONLY *when* the IO runs changed. Not a single bit of
history / audit / budget / chunk content or order may differ.
"""
from __future__ import annotations

import asyncio

import pytest

import opsrag.agent.nodes.multi_agent as ma
from opsrag.agent.nodes.multi_agent import MAX_TOOL_CALLS, tool_caller_node
from opsrag.interfaces.chunker import Chunk
from opsrag.mcp import MCPTool
from opsrag.mcp.gitlab import GitLabMCPError
from opsrag.mcp.tool_cache import ToolOutputCache


@pytest.fixture(autouse=True)
def _isolate_cache(monkeypatch):
    """Fresh tool cache per test so cross-test (name,args) keys never leak,
    and the node never shares a cached result with another test run."""
    fresh = ToolOutputCache()
    monkeypatch.setattr(ma, "get_default_cache", lambda: fresh)
    yield


@pytest.fixture(autouse=True)
def _gitlab_token(monkeypatch):
    # The node always constructs a GitLabClient() (needs a token). No network
    # is hit: the fake tools below are not gitlab_* so they ignore the client.
    monkeypatch.setenv("GITLAB_TOKEN", "test-dummy-token")
    yield


def _mk_tool(name: str, handler):
    return MCPTool(
        name=name,
        description=f"fake {name}",
        input_schema={"type": "object", "properties": {}},
        handler=handler,
    )


def _install_registry(monkeypatch, tools: dict[str, MCPTool]):
    monkeypatch.setattr(ma, "_registry", lambda: dict(tools))


# --------------------------------------------------------------------------
# 1. Three independent real calls -> same ordered history/audit + budget,
#    and the dispatches actually overlap.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_three_independent_calls_equivalent_and_concurrent(monkeypatch):
    # A barrier-style probe: each handler records its dispatch order, waits on
    # a shared event, then returns. If dispatch is concurrent, all three are
    # "in flight" before any completes; a serial loop could never have >1
    # in flight at once.
    in_flight = 0
    max_in_flight = 0
    release = asyncio.Event()

    def make_handler(tag: str):
        async def _h(_client, args):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            # First two callers wait; the result return order is intentionally
            # decoupled from pending order to prove re-application is ordered.
            await release.wait()
            in_flight -= 1
            return {"tag": tag, "n": args.get("n")}

        return _h

    tools = {
        "tool_a": _mk_tool("tool_a", make_handler("A")),
        "tool_b": _mk_tool("tool_b", make_handler("B")),
        "tool_c": _mk_tool("tool_c", make_handler("C")),
    }
    _install_registry(monkeypatch, tools)

    caller = tool_caller_node(observability=None)
    state = {
        "tool_calls": [
            {"name": "tool_a", "args": {"n": 1}},
            {"name": "tool_b", "args": {"n": 2}},
            {"name": "tool_c", "args": {"n": 3}},
        ],
        "tool_message_history": [],
        "tool_call_audit": [],
        "tool_call_count": 0,
    }

    # Release everyone shortly after dispatch so gather can complete.
    async def _releaser():
        # Yield enough times for all three handlers to enter `release.wait()`.
        for _ in range(20):
            await asyncio.sleep(0)
        release.set()

    out, _ = await asyncio.gather(caller(state), _releaser())

    # --- concurrency proof: all three were in flight simultaneously ---
    assert max_in_flight == 3, (
        f"dispatch was not concurrent (max_in_flight={max_in_flight})"
    )

    # --- equivalence: history is ordered by PENDING order, not completion ---
    hist = out["tool_message_history"]
    assert [m["name"] for m in hist] == ["tool_a", "tool_b", "tool_c"]
    assert [m["role"] for m in hist] == ["tool_result"] * 3
    assert all("text" in m["response"] for m in hist)

    # --- audit rows in the same order, each with the success-row shape ---
    audit = out["tool_call_audit"]
    assert [a["name"] for a in audit] == ["tool_a", "tool_b", "tool_c"]
    for a in audit:
        assert "result_chars" in a and "chunks_lifted" in a
        assert "error" not in a

    # --- budget: 3 real calls each consume 1 (F7) ---
    assert out["tool_call_count"] == 3
    ev = out["agent_event"]["metadata"]
    assert ev["calls"] == 3
    assert ev["tools"] == ["tool_a", "tool_b", "tool_c"]
    assert ev["unknown_tool_rounds"] == 0


# --------------------------------------------------------------------------
# 2. An unknown tool mixed into a real-call hop does NOT consume budget (F7),
#    and history/audit order still follows pending order.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_tool_in_hop_does_not_consume_budget(monkeypatch):
    async def _ok(_client, args):
        return {"ok": True}

    tools = {"tool_a": _mk_tool("tool_a", _ok), "tool_b": _mk_tool("tool_b", _ok)}
    _install_registry(monkeypatch, tools)

    caller = tool_caller_node(observability=None)
    state = {
        "tool_calls": [
            {"name": "tool_a", "args": {}},
            {"name": "nope_not_a_tool", "args": {}},  # unknown -> no budget
            {"name": "tool_b", "args": {}},
        ],
        "tool_message_history": [],
        "tool_call_audit": [],
        "tool_call_count": 0,
    }

    out = await caller(state)

    # Real budget = 2 (the two real tools); the unknown tool charged 0 (F7).
    assert out["tool_call_count"] == 2

    # History/audit preserve pending order, including the unknown error row
    # sandwiched between the two real results.
    hist = out["tool_message_history"]
    assert [m["name"] for m in hist] == ["tool_a", "nope_not_a_tool", "tool_b"]
    assert "text" in hist[0]["response"]
    assert hist[1]["response"]["error"].startswith("TOOL DOES NOT EXIST:")
    assert "text" in hist[2]["response"]

    audit = out["tool_call_audit"]
    assert [a["name"] for a in audit] == ["tool_a", "nope_not_a_tool", "tool_b"]
    assert "error" in audit[1]  # unknown audit row carries the error

    ev = out["agent_event"]["metadata"]
    assert ev["unknown_tool_rounds"] == 1
    assert ev["unknown_tools_this_round"] == 1
    # `tools=` lists only the REAL calls actually dispatched, in order.
    assert ev["tools"] == ["tool_a", "tool_b"]
    # `calls` counts every processed call (real + unknown), as before.
    assert ev["calls"] == 3


# --------------------------------------------------------------------------
# 3. GitLabMCPError vs. generic exception payloads survive gather() unchanged.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exception_payloads_match_serial_branches(monkeypatch):
    async def _gitlab_err(_client, args):
        raise GitLabMCPError(404, "not found body", tool="tool_g")

    async def _generic_err(_client, args):
        raise ValueError("boom")

    async def _ok(_client, args):
        return {"ok": True}

    tools = {
        "tool_g": _mk_tool("tool_g", _gitlab_err),
        "tool_x": _mk_tool("tool_x", _generic_err),
        "tool_ok": _mk_tool("tool_ok", _ok),
    }
    _install_registry(monkeypatch, tools)

    caller = tool_caller_node(observability=None)
    state = {
        "tool_calls": [
            {"name": "tool_g", "args": {}},
            {"name": "tool_x", "args": {}},
            {"name": "tool_ok", "args": {}},
        ],
        "tool_message_history": [],
        "tool_call_audit": [],
        "tool_call_count": 0,
    }

    out = await caller(state)

    hist = out["tool_message_history"]
    # GitLabMCPError branch: {"error": str(exc), "status": exc.status}
    assert hist[0]["name"] == "tool_g"
    assert hist[0]["response"]["status"] == 404
    assert "not found body" in hist[0]["response"]["error"]
    # generic branch: {"error": f"unhandled: {exc}"}
    assert hist[1]["name"] == "tool_x"
    assert hist[1]["response"]["error"] == "unhandled: boom"
    # success
    assert hist[2]["name"] == "tool_ok"
    assert "text" in hist[2]["response"]

    audit = out["tool_call_audit"]
    assert audit[0]["error"] and "result_chars" not in audit[0]
    assert audit[1]["error"] == "unhandled: boom"
    assert "result_chars" in audit[2]

    # All three are real -> all three consume budget regardless of error.
    assert out["tool_call_count"] == 3
    assert out["agent_event"]["metadata"]["tools"] == ["tool_g", "tool_x", "tool_ok"]


# --------------------------------------------------------------------------
# 4. Retrieval-chunk dedupe: a single end-of-loop dedupe == per-call dedupe.
# --------------------------------------------------------------------------


def _serial_dedupe_reference(per_call_chunk_lists: list[list[Chunk]]) -> list[Chunk]:
    """Reproduce the OLD serial behaviour: dedupe after EACH call's extend."""
    from opsrag.agent.nodes.tool_caller import _dedupe_chunks

    acc: list[Chunk] = []
    for chunks in per_call_chunk_lists:
        if chunks:
            acc.extend(chunks)
            acc = _dedupe_chunks(acc)
    return acc


@pytest.mark.asyncio
async def test_chunk_dedupe_equivalent_to_serial(monkeypatch):
    # Two knowledge_search calls returning OVERLAPPING (repo, source_path)
    # so dedupe actually does work. The node must produce the same final
    # list as the per-call-dedupe reference.
    call1_results = {
        "results": [
            {"source": "a.md", "repo": "r1", "content": "A"},
            {"source": "b.md", "repo": "r1", "content": "B"},
        ]
    }
    call2_results = {
        "results": [
            {"source": "b.md", "repo": "r1", "content": "B-dup"},  # dup of call1
            {"source": "c.md", "repo": "r1", "content": "C"},
        ]
    }
    by_args = {1: call1_results, 2: call2_results}

    async def _ks(_client, args):
        return by_args[args["n"]]

    tools = {"knowledge_search": _mk_tool("knowledge_search", _ks)}
    _install_registry(monkeypatch, tools)

    caller = tool_caller_node(observability=None)
    state = {
        "tool_calls": [
            {"name": "knowledge_search", "args": {"n": 1}},
            {"name": "knowledge_search", "args": {"n": 2}},
        ],
        "tool_message_history": [],
        "tool_call_audit": [],
        "tool_call_count": 0,
        "tool_retrieved_chunks": [],
    }

    out = await caller(state)
    got = out["tool_retrieved_chunks"]

    # Reference: extract the same chunks and run per-call dedupe.
    from opsrag.agent.nodes.tool_caller import _extract_chunks_from_knowledge_search

    expected = _serial_dedupe_reference([
        _extract_chunks_from_knowledge_search(call1_results),
        _extract_chunks_from_knowledge_search(call2_results),
    ])

    assert [(c.repo, c.source_path) for c in got] == [
        (c.repo, c.source_path) for c in expected
    ]
    # First-seen wins: b.md keeps call1's content ("B"), not call2's "B-dup".
    keys = {(c.repo, c.source_path): c.content for c in got}
    assert keys[("r1", "b.md")] == "B"
    assert [(c.repo, c.source_path) for c in got] == [
        ("r1", "a.md"), ("r1", "b.md"), ("r1", "c.md")
    ]


# --------------------------------------------------------------------------
# 5. MAX_TOOL_CALLS cap is applied at the SAME slot as the serial `break`:
#    calls past the budget are dropped, unknown tools don't count toward it.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_cap_drops_calls_like_serial_break(monkeypatch):
    async def _ok(_client, args):
        return {"ok": args.get("n")}

    tools = {"tool_r": _mk_tool("tool_r", _ok)}
    _install_registry(monkeypatch, tools)

    caller = tool_caller_node(observability=None)
    # Start with budget already at MAX-1 so exactly ONE more real call fits;
    # an unknown tool before it must NOT consume the remaining slot, and the
    # second real call must be DROPPED (mirrors the serial `break`).
    state = {
        "tool_calls": [
            {"name": "nope", "args": {}},          # unknown: no budget
            {"name": "tool_r", "args": {"n": 1}},  # fits (last slot)
            {"name": "tool_r", "args": {"n": 2}},  # dropped (cap reached)
        ],
        "tool_message_history": [],
        "tool_call_audit": [],
        "tool_call_count": MAX_TOOL_CALLS - 1,
    }

    out = await caller(state)

    assert out["tool_call_count"] == MAX_TOOL_CALLS
    hist = out["tool_message_history"]
    # The unknown error + the single real result; the 2nd real call dropped.
    assert [m["name"] for m in hist] == ["nope", "tool_r"]
    assert out["agent_event"]["metadata"]["tools"] == ["tool_r"]
    # The dropped real call left no audit/history trace, exactly like break.
    assert sum(1 for m in hist if m["name"] == "tool_r") == 1


# --------------------------------------------------------------------------
# 6. Cancellation semantics: a tool coroutine raising asyncio.CancelledError
#    must PROPAGATE out of tool_caller_node -- NOT be swallowed into a
#    tool_result error row.
#
#    `asyncio.gather(..., return_exceptions=True)` captures BaseException
#    subclasses (CancelledError/KeyboardInterrupt) as ordinary results; the
#    ORIGINAL serial loop used `except Exception`, which does NOT catch those,
#    so they propagated -- preserving correct cancellation/interrupt behaviour.
#    The node must re-raise such results after the gather rather than turn a
#    cancelled hop into a fabricated tool_result.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancellederror_propagates_not_swallowed(monkeypatch):
    async def _cancel(_client, args):
        raise asyncio.CancelledError()

    async def _ok(_client, args):
        return {"ok": True}

    tools = {
        "tool_cancel": _mk_tool("tool_cancel", _cancel),
        "tool_ok": _mk_tool("tool_ok", _ok),
    }
    _install_registry(monkeypatch, tools)

    caller = tool_caller_node(observability=None)
    state = {
        "tool_calls": [
            {"name": "tool_ok", "args": {}},
            {"name": "tool_cancel", "args": {}},
        ],
        "tool_message_history": [],
        "tool_call_audit": [],
        "tool_call_count": 0,
    }

    # The CancelledError must propagate OUT of the node (exactly as the serial
    # loop's `except Exception` let it through) -- not become a tool_result.
    with pytest.raises(asyncio.CancelledError):
        await caller(state)

    # And it must NOT have been recorded as a fabricated tool_result/audit row
    # (the swallow-bug would have appended an "unhandled: ..." entry).
    assert state["tool_message_history"] == []
    assert state["tool_call_audit"] == []
