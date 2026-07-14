"""answer_verifier must use `generate_structured` (json mode + provider
token floor), not plain `generate(max_tokens=1024)`.

Live repro 2026-07-13 on gemini-3-flash: thinking tokens count against
max_output_tokens on the plain-generate path -> finish_reason=length with
content_len=0 -> verdict None -> the fail-closed CAUTION landed on
essentially EVERY answer. `generate_structured` sets json response mode
and floors max_tokens at the provider default, so the verdict can't be
starved by thinking. The fail-closed CAUTION stays as the backstop when
the structured call itself errors.
"""
from __future__ import annotations

import pytest

from opsrag.agent.nodes import answer_verifier as av


class _Chunk:
    source_path = "docs/a.md"
    repo = "r"
    content = "apps/foo/values.yaml is the deploy file"


class _StructuredVerdictLLM:
    """generate_structured returns a clean verdict; plain generate must NOT
    be called anymore (guard against regressing to the old path)."""

    model_name = "fake-model"

    def __init__(self, verified=None, unverifiable=None):
        self._verified = verified or []
        self._unverifiable = unverifiable or []
        self.structured_calls = 0

    async def generate(self, **_kw):
        raise AssertionError("verifier must use generate_structured, not generate")

    async def generate_structured(self, *, schema, **kw):
        self.structured_calls += 1
        self.kwargs = kw
        return schema(verified=self._verified, unverifiable=self._unverifiable)


class _BoomStructuredLLM:
    model_name = "fake-model"

    async def generate_structured(self, **_kw):
        raise RuntimeError("verifier llm down")


@pytest.mark.asyncio
async def test_clean_verdict_no_caution_no_hedge():
    llm = _StructuredVerdictLLM(verified=["apps/foo/values.yaml"])
    node = av.verify_answer_node(llm, None, None)
    answer = "We deploy via `apps/foo/values.yaml`."
    out = await node({"generation": answer, "final_chunks": [_Chunk()]})
    assert llm.structured_calls == 1
    # Starvation guard: gemini-3 thinking scales with prompt size; a big
    # verifier prompt ate the whole 4096 default (finish_reason=length,
    # empty content -- observed live). The node must ask for a generous cap.
    assert llm.kwargs.get("max_tokens", 0) >= 8192
    assert av._CAUTION not in out.get("generation", answer)
    assert out["verification_result"]["skipped"] is False
    assert out["verification_result"]["unverifiable"] == []
    # No hedge prefix, no rewritten generation needed.
    assert "Treat with caution" not in out.get("generation", answer)


@pytest.mark.asyncio
async def test_unverifiable_claims_still_hedged():
    llm = _StructuredVerdictLLM(unverifiable=["ghost/file.py"])
    node = av.verify_answer_node(llm, None, None)
    answer = "See `ghost/file.py`."
    out = await node({"generation": answer, "final_chunks": [_Chunk()]})
    assert out["generation"].startswith("Warning: Some claims could not be verified")
    assert "ghost/file.py" in out["generation"]


@pytest.mark.asyncio
async def test_structured_error_still_fails_closed_with_caution():
    node = av.verify_answer_node(_BoomStructuredLLM(), None, None)
    answer = "We deploy via `apps/foo/values.yaml`."
    out = await node({"generation": answer, "final_chunks": [_Chunk()]})
    assert out["generation"].startswith(answer)
    assert av._CAUTION in out["generation"]
    assert out["verification_result"]["fail_closed"] is True


class _SlowStructuredLLM:
    model_name = "fake-model"

    async def generate_structured(self, **_kw):
        import asyncio
        await asyncio.sleep(0.2)
        raise AssertionError("should have timed out before completing")


@pytest.mark.asyncio
async def test_slow_verifier_times_out_and_fails_closed(monkeypatch):
    """SSE regression guard: verify_answer emits NOTHING while the LLM
    thinks; a 59.9s verify starved the stream past the ~60s proxy idle
    timeout and the live answer bubble arrived empty (observed in prod
    2026-07-13 10:58 -- the stored turn was fine on reload). The node
    must bound the silent window and fail closed on timeout."""
    monkeypatch.setattr(av, "_VERIFY_TIMEOUT_SEC", 0.05)
    node = av.verify_answer_node(_SlowStructuredLLM(), None, None)
    answer = "We deploy via `apps/foo/values.yaml`."
    out = await node({"generation": answer, "final_chunks": [_Chunk()]})
    assert out["generation"].startswith(answer)
    assert av._CAUTION in out["generation"]
    assert out["verification_result"]["fail_closed"] is True
