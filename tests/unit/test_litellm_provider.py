"""Unit tests for the scoped LiteLLM adapter.

The providers import ``litellm`` lazily inside their methods, so these tests
inject a fake ``litellm`` module into ``sys.modules`` -- no real litellm
install required. We assert the providers return vectors / text AND record
usage with the right purpose tags.
"""
from __future__ import annotations

import sys
import types

import pytest
from pydantic import BaseModel

from opsrag.embedders.litellm_provider import LiteLLMEmbeddings
from opsrag.llms.litellm_provider import LiteLLMLLM
from opsrag.usage import tracker


def _install_fake_litellm(monkeypatch, *, aembedding=None, acompletion=None):
    fake = types.ModuleType("litellm")
    if aembedding is not None:
        fake.aembedding = aembedding
    if acompletion is not None:
        fake.acompletion = acompletion
    monkeypatch.setitem(sys.modules, "litellm", fake)
    return fake


def _embed_response(vectors):
    """Mimic LiteLLM's OpenAI-shaped embedding response: r.data[i]['embedding']."""
    resp = types.SimpleNamespace()
    resp.data = [{"embedding": v} for v in vectors]
    return resp


def _completion_response(content, *, model="gemini/gemini-2.5-flash",
                         prompt_tokens=11, completion_tokens=7):
    msg = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=msg)
    usage = types.SimpleNamespace(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    return types.SimpleNamespace(choices=[choice], usage=usage, model=model)


# --------------------------- embeddings ---------------------------

async def test_embed_texts_returns_vectors_and_records_usage(monkeypatch):
    calls: list[dict] = []

    async def fake_aembedding(**kwargs):
        calls.append(kwargs)
        return _embed_response([[0.1, 0.2, 0.3] for _ in kwargs["input"]])

    _install_fake_litellm(monkeypatch, aembedding=fake_aembedding)

    emb = LiteLLMEmbeddings(model="voyage/voyage-code-3", dimension=1024)

    vecs = await emb.embed_texts(["alpha", "beta"])

    assert vecs == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert emb.dimension == 1024
    assert emb.model_name == "voyage/voyage-code-3"
    # model string forwarded verbatim to litellm
    assert calls[0]["model"] == "voyage/voyage-code-3"
    assert calls[0]["input"] == ["alpha", "beta"]

    # usage recorded under embed-index for this model
    summary = tracker.get_summary()
    model_row = summary["models"]["voyage/voyage-code-3"]
    assert "embed-index" in model_row["by_purpose"]
    assert model_row["by_purpose"]["embed-index"]["input_tokens"] > 0


async def test_embed_query_records_embed_query_purpose(monkeypatch):
    async def fake_aembedding(**kwargs):
        return _embed_response([[0.5] * 4])

    _install_fake_litellm(monkeypatch, aembedding=fake_aembedding)

    emb = LiteLLMEmbeddings(model="cohere/embed-english-v3.0", dimension=4)
    vec = await emb.embed_query("how do i restart the pod")

    assert vec == [0.5, 0.5, 0.5, 0.5]
    summary = tracker.get_summary()
    model_row = summary["models"]["cohere/embed-english-v3.0"]
    assert "embed-query" in model_row["by_purpose"]


async def test_embed_texts_empty_returns_empty(monkeypatch):
    async def fake_aembedding(**kwargs):  # pragma: no cover - should not be called
        raise AssertionError("aembedding should not be called for empty input")

    _install_fake_litellm(monkeypatch, aembedding=fake_aembedding)
    emb = LiteLLMEmbeddings(model="voyage/voyage-3", dimension=8)
    assert await emb.embed_texts([]) == []


async def test_embed_passes_api_base_for_self_hosted(monkeypatch):
    calls: list[dict] = []

    async def fake_aembedding(**kwargs):
        calls.append(kwargs)
        return _embed_response([[1.0, 2.0]])

    _install_fake_litellm(monkeypatch, aembedding=fake_aembedding)
    monkeypatch.setenv("QWEN_API_KEY", "secret-token")

    emb = LiteLLMEmbeddings(
        model="openai/Qwen3-Embedding-8B",
        dimension=2,
        api_base="http://qwen-tei.internal:8080/v1",
        api_key_env="QWEN_API_KEY",
    )
    await emb.embed_query("ping")

    assert calls[0]["api_base"] == "http://qwen-tei.internal:8080/v1"
    assert calls[0]["api_key"] == "secret-token"


# ------------------------------ LLM ------------------------------

async def test_generate_returns_text_and_records_usage(monkeypatch):
    calls: list[dict] = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return _completion_response("hello world")

    _install_fake_litellm(monkeypatch, acompletion=fake_acompletion)

    llm = LiteLLMLLM(model="gemini/gemini-2.5-flash")
    resp = await llm.generate(
        messages=[{"role": "user", "content": "hi"}],
        system_prompt="you are ops",
        purpose="generation",
    )

    assert resp.content == "hello world"
    assert resp.usage == {"input_tokens": 11, "output_tokens": 7}
    # system prompt is sent as a leading system message
    sent = calls[0]["messages"]
    assert sent[0] == {"role": "system", "content": "you are ops"}
    assert sent[1] == {"role": "user", "content": "hi"}
    assert calls[0]["model"] == "gemini/gemini-2.5-flash"

    summary = tracker.get_summary()
    model_row = summary["models"]["gemini/gemini-2.5-flash"]
    assert "generation" in model_row["by_purpose"]


async def test_generate_estimates_tokens_when_usage_absent(monkeypatch):
    async def fake_acompletion(**kwargs):
        resp = _completion_response("some output")
        resp.usage = None  # backend didn't report usage
        return resp

    _install_fake_litellm(monkeypatch, acompletion=fake_acompletion)

    llm = LiteLLMLLM(model="openai/local-model", api_base="http://vllm:8000/v1")
    resp = await llm.generate(messages=[{"role": "user", "content": "estimate me"}])

    # fell back to estimated tokens -> both > 0
    assert resp.usage["input_tokens"] > 0
    assert resp.usage["output_tokens"] > 0


async def test_generate_structured_parses_json(monkeypatch):
    class Out(BaseModel):
        answer: str
        score: int

    async def fake_acompletion(**kwargs):
        return _completion_response('{"answer": "ok", "score": 5}')

    _install_fake_litellm(monkeypatch, acompletion=fake_acompletion)

    llm = LiteLLMLLM(model="anthropic/claude-sonnet-4-20250514")
    out = await llm.generate_structured(
        messages=[{"role": "user", "content": "grade this"}],
        schema=Out,
    )

    assert isinstance(out, Out)
    assert out.answer == "ok"
    assert out.score == 5


async def test_generate_structured_strips_code_fences(monkeypatch):
    class Out(BaseModel):
        ok: bool

    async def fake_acompletion(**kwargs):
        return _completion_response('```json\n{"ok": true}\n```')

    _install_fake_litellm(monkeypatch, acompletion=fake_acompletion)

    llm = LiteLLMLLM(model="gemini/gemini-2.5-flash")
    out = await llm.generate_structured(
        messages=[{"role": "user", "content": "x"}], schema=Out
    )
    assert out.ok is True


async def test_generate_structured_rejects_non_basemodel(monkeypatch):
    _install_fake_litellm(monkeypatch, acompletion=None)
    llm = LiteLLMLLM(model="gemini/gemini-2.5-flash")
    with pytest.raises(TypeError):
        await llm.generate_structured(messages=[], schema=dict)
