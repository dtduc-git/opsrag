"""Gemini 3.x rejects replayed `functionCall` parts that lack a
`thought_signature` (400 "Function call is missing a thought_signature").
Observed in prod 2026-07-12: every pro-tier multi-hop turn died at the
first reasoner hop, masked by google-api-core's streaming-error formatter
as "'list' object has no attribute 'get'"; the generator then answered
from parametric memory and the citation guard refused the turn.

Fix under test:
- `ToolCall` carries the base64 signature captured from response parts
  (`_part_thought_signature_b64`).
- `_messages_to_gemini_contents` replays it; when absent on a Gemini 3+
  model it injects Google's documented dummy
  (base64 of b"skip_thought_signature_validator") so synthetic /
  guard-injected calls survive strict validation. Pre-3 models get no
  injected field (unchanged wire shape).
- triage/reasoner persist the signature into `tool_message_history`.
- The LiteLLM lane round-trips it via `provider_specific_fields`.
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("vertexai")  # optional [vertex] extra; skip when absent

from opsrag.agent.nodes.multi_agent import reasoner_node, triage_node
from opsrag.llms.litellm_provider import LiteLLMLLM
from opsrag.llms.vertex import (
    ToolCall,
    ToolCallingResponse,
    _messages_to_gemini_contents,
    _part_thought_signature_b64,
)

_DUMMY_B64 = "c2tpcF90aG91Z2h0X3NpZ25hdHVyZV92YWxpZGF0b3I="  # b"skip_thought_signature_validator"


# --- ToolCall dataclass -------------------------------------------------


def test_toolcall_thought_signature_optional_default_none():
    tc = ToolCall(name="code_grep", args={"pattern": "x"})
    assert tc.thought_signature is None
    tc2 = ToolCall(name="code_grep", args={}, thought_signature="QUJD")
    assert tc2.thought_signature == "QUJD"


# --- capture: response part -> base64 -----------------------------------


def test_part_signature_extracted_as_base64():
    part = SimpleNamespace(
        _raw_part=SimpleNamespace(thought_signature=b"\x01\x02\x03")
    )
    assert _part_thought_signature_b64(part) == "AQID"


def test_part_signature_none_when_empty_or_missing():
    empty = SimpleNamespace(_raw_part=SimpleNamespace(thought_signature=b""))
    assert _part_thought_signature_b64(empty) is None
    assert _part_thought_signature_b64(SimpleNamespace()) is None


# --- replay: _messages_to_gemini_contents --------------------------------


def _replayed_sig(contents) -> str | None:
    """b64 thought_signature of the first part, None when absent."""
    return contents[0].to_dict()["parts"][0].get("thought_signature")


def test_replay_uses_captured_signature_on_gemini3():
    msgs = [{
        "role": "tool_call", "name": "code_list_repos", "args": {},
        "thought_signature": "QUJD",
    }]
    contents = _messages_to_gemini_contents(msgs, model="gemini-3.1-pro-preview")
    assert _replayed_sig(contents) == "QUJD"


def test_replay_injects_documented_dummy_when_missing_on_gemini3():
    # Synthetic calls (code-intent guard, keyword-guard) have no captured
    # signature -- Google's documented dummy keeps strict validation happy.
    msgs = [{"role": "tool_call", "name": "code_list_repos", "args": {}}]
    contents = _messages_to_gemini_contents(msgs, model="gemini-3-flash-preview")
    assert _replayed_sig(contents) == _DUMMY_B64


def test_replay_no_injection_on_pre_gemini3():
    msgs = [{"role": "tool_call", "name": "code_list_repos", "args": {}}]
    contents = _messages_to_gemini_contents(msgs, model="gemini-2.5-pro")
    assert _replayed_sig(contents) is None


def test_replay_real_signature_still_replayed_on_pre_gemini3():
    # A captured signature is never dropped -- older models ignore it.
    msgs = [{
        "role": "tool_call", "name": "code_grep", "args": {"pattern": "x"},
        "thought_signature": "QUJD",
    }]
    contents = _messages_to_gemini_contents(msgs, model="gemini-2.5-pro")
    assert _replayed_sig(contents) == "QUJD"


def test_replay_default_model_arg_stays_backward_compatible():
    # No model passed -> no dummy injection (wire shape unchanged for
    # callers that haven't opted in).
    msgs = [{"role": "tool_call", "name": "code_list_repos", "args": {}}]
    contents = _messages_to_gemini_contents(msgs)
    assert _replayed_sig(contents) is None


# --- persistence: triage + reasoner history rows -------------------------


class _SignedToolCallLLM:
    """Stub LLM that emits one tool call carrying a thought_signature,
    then (2nd invocation) declines further tools."""

    def __init__(self) -> None:
        self.calls = 0
        self.model_name = "stub"

    async def generate_with_tools(self, **_kwargs) -> ToolCallingResponse:
        self.calls += 1
        if self.calls > 1:
            return ToolCallingResponse(tool_calls=[], text="done", model="stub")
        return ToolCallingResponse(
            tool_calls=[ToolCall(
                name="code_grep",
                args={"repo": "saas/sight-be", "pattern": "barcode"},
                thought_signature="U0lHTkFUVVJF",
            )],
            text="",
            model="stub",
        )

    # No `generate_with_tools_stream` -> nodes take the non-stream path.


@pytest.mark.asyncio
async def test_reasoner_persists_thought_signature_into_history():
    reason = reasoner_node(_SignedToolCallLLM(), observability=None, model_router=None)
    state = {
        "tool_message_history": [{"role": "user", "content": "trace the endpoint"}],
        "tool_call_count": 1,
        "turn_started_at": time.monotonic(),
    }
    out = await reason(state)
    rows = [m for m in out["tool_message_history"] if m.get("role") == "tool_call"]
    assert rows, "reasoner must append the emitted tool_call to history"
    assert rows[-1]["thought_signature"] == "U0lHTkFUVVJF"


@pytest.mark.asyncio
async def test_triage_persists_thought_signature_into_history():
    triage = triage_node(_SignedToolCallLLM(), observability=None, model_router=None)
    out = await triage({"query": "list pods please", "conversation_history": []})
    rows = [m for m in out["tool_message_history"] if m.get("role") == "tool_call"]
    assert rows, "triage must append the emitted tool_call to history"
    assert rows[0]["thought_signature"] == "U0lHTkFUVVJF"


@pytest.mark.asyncio
async def test_tool_decide_persists_thought_signature_into_history():
    # Legacy `agent.mode: tool_calling` lane -- same ToolCall dataclass,
    # same replay path, must not silently downgrade a captured signature
    # to the dummy on every hop.
    from opsrag.agent.nodes.tool_caller import tool_decide_node

    decide = tool_decide_node(_SignedToolCallLLM(), observability=None)
    out = await decide({"query": "trace the endpoint", "tool_call_count": 0})
    rows = [m for m in out["tool_message_history"] if m.get("role") == "tool_call"]
    assert rows, "tool_decide must append the emitted tool_call to history"
    assert rows[0]["thought_signature"] == "U0lHTkFUVVJF"


# --- LiteLLM lane: replay + capture --------------------------------------


def test_litellm_replay_carries_signature_in_provider_specific_fields():
    internal = [{
        "role": "tool_call", "name": "code_grep", "args": {"pattern": "a"},
        "thought_signature": "QUJD",
    }]
    out = LiteLLMLLM._to_openai_messages(internal, None)
    fn = out[0]["tool_calls"][0]["function"]
    assert fn["provider_specific_fields"]["thought_signature"] == "QUJD"


def test_litellm_replay_omits_provider_fields_when_no_signature():
    internal = [{"role": "tool_call", "name": "code_grep", "args": {"pattern": "a"}}]
    out = LiteLLMLLM._to_openai_messages(internal, None)
    assert "provider_specific_fields" not in out[0]["tool_calls"][0]["function"]


@pytest.mark.asyncio
async def test_litellm_captures_signature_from_response(monkeypatch):
    import litellm as _litellm

    fake_resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content="",
            tool_calls=[SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(
                    name="code_grep",
                    arguments=json.dumps({"pattern": "x"}),
                    provider_specific_fields={"thought_signature": "U0lH"},
                ),
            )],
        ))],
        model="vertex_ai/gemini-3-flash-preview",
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )

    async def _fake_acompletion(**_kwargs):
        return fake_resp

    monkeypatch.setattr(_litellm, "acompletion", _fake_acompletion)

    llm = LiteLLMLLM(model="vertex_ai/gemini-3-flash-preview")
    resp = await llm.generate_with_tools(
        messages=[{"role": "user", "content": "q"}], tools=[],
    )
    assert resp.tool_calls[0].thought_signature == "U0lH"
