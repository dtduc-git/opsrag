"""Tolerant first-object JSON extraction for LLM structured outputs.

Gemini-3-flash in json mode occasionally emits a valid object followed by
junk -- captured live 2026-07-13: '{\\n  "grounded": false\\n}\\n}' (stray
trailing brace) -> strict json.loads dies with "Extra data", the
groundedness gate fails CLOSED, and every answer gets a spurious
"could not be verified" note (plus wasted regenerate loops on the
retrieval lane). All five providers shared the same strict parse block.

`extract_first_json_object` takes the FIRST valid JSON object and ignores
trailing junk; clean output parses identically to strict json.loads, so
providers that already emit clean JSON (Claude/Bedrock/OpenAI) see zero
behavior change.
"""
from __future__ import annotations

import json

import pytest

from opsrag.llms.json_extract import extract_first_json_object

# Captured live from gemini-3-flash-preview (repro 2026-07-13).
STRAY_BRACE = '{\n  "grounded": false\n}\n}'


def test_stray_trailing_brace_fixture_parses():
    assert extract_first_json_object(STRAY_BRACE) == {"grounded": False}


def test_clean_json_identical_to_strict_loads():
    raw = '{"verified": ["a/b.py"], "unverifiable": []}'
    assert extract_first_json_object(raw) == json.loads(raw)


def test_double_object_takes_first():
    raw = '{"grounded": true}\n{"grounded": false}'
    assert extract_first_json_object(raw) == {"grounded": True}


def test_markdown_fenced_json():
    raw = '```json\n{"grounded": true}\n```'
    assert extract_first_json_object(raw) == {"grounded": True}


def test_prose_wrapped_object():
    raw = 'Here is the verdict:\n{"grounded": false}\nHope that helps!'
    assert extract_first_json_object(raw) == {"grounded": False}


def test_nested_braces_survive():
    raw = '{"a": {"b": [1, 2]}, "c": "x}y"} trailing'
    assert extract_first_json_object(raw) == {"a": {"b": [1, 2]}, "c": "x}y"}


@pytest.mark.parametrize("bad", ["", "   ", "no braces here", "{truncated"])
def test_unparseable_raises_valueerror(bad):
    with pytest.raises(ValueError):
        extract_first_json_object(bad)


# -- provider-level regression: the exact prod failure through LiteLLM ------


@pytest.mark.asyncio
async def test_litellm_generate_structured_tolerates_stray_brace(monkeypatch):
    from types import SimpleNamespace

    pytest.importorskip("litellm")  # optional [litellm] extra; skip when absent
    import litellm as _litellm
    from pydantic import BaseModel

    from opsrag.llms.litellm_provider import LiteLLMLLM

    class G(BaseModel):
        grounded: bool

    fake_resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=STRAY_BRACE))],
        model="vertex_ai/gemini-3-flash-preview",
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )

    async def _fake_acompletion(**_kwargs):
        return fake_resp

    monkeypatch.setattr(_litellm, "acompletion", _fake_acompletion)

    llm = LiteLLMLLM(model="vertex_ai/gemini-3-flash-preview")
    out = await llm.generate_structured(
        messages=[{"role": "user", "content": "q"}], schema=G,
    )
    assert out.grounded is False
