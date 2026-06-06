"""Tests for LiteLLMLLM tool-calling: the internal->OpenAI message translation
(the part that pairs tool_results to tool_calls via synthesized ids)."""
import json

from opsrag.llms.litellm_provider import LiteLLMLLM


def test_translate_plain_and_tool_loop():
    internal = [
        {"role": "user", "content": "what version of flask?"},
        {"role": "tool_call", "name": "code_dependency_lookup", "args": {"repo": "saas/sight-be", "package": "flask"}},
        {"role": "tool_result", "name": "code_dependency_lookup", "response": {"resolved_version": "2.3.3"}},
        {"role": "assistant", "content": "Flask 2.3.3"},
    ]
    out = LiteLLMLLM._to_openai_messages(internal, system_prompt="sys")
    assert out[0] == {"role": "system", "content": "sys"}
    assert out[1] == {"role": "user", "content": "what version of flask?"}
    # tool_call -> assistant message with a tool_calls block carrying an id
    asst = out[2]
    assert asst["role"] == "assistant" and asst["content"] is None
    cid = asst["tool_calls"][0]["id"]
    assert asst["tool_calls"][0]["function"]["name"] == "code_dependency_lookup"
    assert json.loads(asst["tool_calls"][0]["function"]["arguments"])["package"] == "flask"
    # tool_result -> role:tool with MATCHING tool_call_id
    tr = out[3]
    assert tr["role"] == "tool" and tr["tool_call_id"] == cid
    assert json.loads(tr["content"])["resolved_version"] == "2.3.3"
    assert out[4] == {"role": "assistant", "content": "Flask 2.3.3"}


def test_multiple_calls_pair_fifo_by_name():
    internal = [
        {"role": "tool_call", "name": "code_grep", "args": {"pattern": "a"}},
        {"role": "tool_call", "name": "code_grep", "args": {"pattern": "b"}},
        {"role": "tool_result", "name": "code_grep", "response": {"hit": "a"}},
        {"role": "tool_result", "name": "code_grep", "response": {"hit": "b"}},
    ]
    out = LiteLLMLLM._to_openai_messages(internal, None)
    ids = [c["id"] for c in out[0]["tool_calls"]]
    assert len(ids) == 2 and ids[0] != ids[1]
    # results pair to calls in order
    assert out[1]["tool_call_id"] == ids[0]
    assert out[2]["tool_call_id"] == ids[1]


def test_has_generate_with_tools_method():
    # the whole point: the wrapper now exposes tool-calling
    assert hasattr(LiteLLMLLM, "generate_with_tools")
