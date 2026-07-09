"""`_image_context` makes tool SELECTION image-aware (not just the generator).

In multi_agent mode triage/reasoner pick tools from the text query alone, so an
attached chart never steered *what got investigated*. `_image_context` runs one
vision call up front and returns a factual description the caller splices into
the query. These tests pin the model-selection + best-effort contract without a
real LLM or image decode.
"""
from __future__ import annotations

import pytest

from opsrag.agent.graph import _image_context
from opsrag.llms.content import ImagePart

_IMG = [ImagePart(data=b"\x89PNG\r\n", mime_type="image/png", name="chart.png")]
_DESC = "Grafana panel for web-svc: liveness-probe failures spiking to 12/min."


class _FakeLLM:
    """Minimal LLM stub: records the call and returns a canned description."""

    def __init__(self, model_name: str):
        self._model = model_name
        self.calls: list[dict] = []

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        from types import SimpleNamespace
        return SimpleNamespace(content=_DESC)


class _Boom(_FakeLLM):
    async def generate(self, **kwargs):
        raise RuntimeError("vision backend down")


@pytest.mark.asyncio
async def test_describes_with_vision_fallback():
    v = _FakeLLM("gemini-2.5-flash")
    out = await _image_context(v, None, _IMG)
    assert out == _DESC
    assert v.calls and v.calls[0]["purpose"] == "vision"
    assert v.calls[0]["temperature"] == 0.0


@pytest.mark.asyncio
async def test_falls_back_to_main_llm_when_it_can_see():
    # The common single-model deployment: no dedicated vision_llm, but the main
    # generation model is itself vision-capable -> describe with it anyway.
    main = _FakeLLM("claude-sonnet-4-5")
    out = await _image_context(None, main, _IMG)
    assert out == _DESC
    assert main.calls


@pytest.mark.asyncio
async def test_prefers_vision_llm_over_main():
    v = _FakeLLM("gemini-2.5-flash")
    main = _FakeLLM("claude-sonnet-4-5")
    await _image_context(v, main, _IMG)
    assert v.calls and not main.calls


@pytest.mark.asyncio
async def test_blind_models_return_empty():
    blind = _FakeLLM("text-embedding-3-large")
    out = await _image_context(blind, blind, _IMG)
    assert out == "" and not blind.calls


@pytest.mark.asyncio
async def test_no_images_returns_empty():
    v = _FakeLLM("gemini-2.5-flash")
    assert await _image_context(v, None, []) == ""
    assert not v.calls


@pytest.mark.asyncio
async def test_generate_error_is_swallowed():
    out = await _image_context(_Boom("gemini-2.5-flash"), None, _IMG)
    assert out == ""  # best-effort: raw bytes still reach the generator
