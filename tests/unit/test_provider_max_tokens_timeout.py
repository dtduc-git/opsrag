"""Behavior-equivalence tests for the providers latency/robustness knobs.

Two PURE additions are exercised here, both proven to be quality-neutral:

L1 -- ``generate_structured(..., max_tokens=...)`` (Protocol + all 5 providers).
  The boolean/verdict gates (F6 groundedness, F7 budget verdicts) can now cap
  the structured output. We prove:
    * a valid boolean/verdict schema STILL parses to the SAME object with a
      small cap (64 / 128) -- truncation is impossible for the ~10-20-token
      payload, so the answer is byte-identical to the uncapped path;
    * the cap is threaded into the inner ``generate()`` ONLY when set; left
      None, ``generate`` is NOT given ``max_tokens`` so its default-4096
      behavior (and therefore every existing caller) is unchanged.

L2 -- optional client-construction timeout/retry knobs (Anthropic + Bedrock).
  We prove construction succeeds with the new args and that the underlying
  SDK / botocore ``Config`` receives them ONLY when provided (SDK-native
  defaults preserved when unset). These are connection-layer settings: they
  cannot change a retrieval result, routing, grounding decision, or answer
  text -- they only govern how a transport failure is timed out / retried.
"""
from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from pydantic import BaseModel

from opsrag.interfaces.llm import LLMResponse


# --- schemas mirroring the real boolean / verdict gate payloads -------------
class _Verdict(BaseModel):
    """Shape of the F6/F7 boolean gate payload (tiny -- can't truncate)."""
    grounded: bool


class _BudgetVerdict(BaseModel):
    grounded: bool
    reason: str


def _captured_generate(captured: dict, *, content: str):
    """Build a fake ``generate`` coroutine that records its kwargs and returns
    a canned LLMResponse, so we can assert exactly what generate_structured
    threads into it without any network."""

    async def _fake_generate(**kwargs: Any) -> LLMResponse:
        captured.clear()
        captured.update(kwargs)
        return LLMResponse(content=content, model="fake-model", usage={})

    return _fake_generate


# ---------------------------------------------------------------------------
# L1 -- max_tokens threading + parse equivalence, per provider
# ---------------------------------------------------------------------------
def _provider_factories():
    """(label, constructed-instance) for every provider. Heavy SDK init is
    avoided: each provider's generate() is monkeypatched, so we only need a
    constructed object whose generate_structured plumbing we can drive."""
    from opsrag.llms.anthropic import AnthropicLLM
    from opsrag.llms.litellm_provider import LiteLLMLLM
    from opsrag.llms.openai import OpenAILLM
    from opsrag.llms.vertex import VertexAILLM

    return [
        ("anthropic", AnthropicLLM(api_key="test")),
        ("openai", OpenAILLM(api_key="test")),
        ("litellm", LiteLLMLLM(model="gemini/gemini-2.5-flash")),
        ("vertex", VertexAILLM(model="gemini-2.0-flash")),
    ]


# Providers whose default structured path triggers Gemini-style thinking, where a
# small max_output_tokens would be eaten by thought tokens (no response_schema is
# set on the in-prompt path). They must IGNORE a small gate cap (floor it at the
# default) rather than thread it through -- see opsrag/llms/vertex.py and
# litellm_provider.py generate_structured.
_THINKING_PRONE = {"vertex", "litellm"}


@pytest.mark.parametrize("cap", [64, 128])
async def test_generate_structured_caps_threaded_when_set(monkeypatch, cap):
    """A small cap is forwarded into generate() AND the boolean payload still
    parses to the identical object -- proving the cap is quality-neutral.

    On non-thinking providers the cap is HONORED verbatim. On thinking-prone
    providers (Vertex/LiteLLM-Gemini) a small cap is IGNORED (floored to the
    provider default) so it can never truncate Gemini thinking tokens -- the
    parsed verdict is still identical either way.
    """
    for label, llm in _provider_factories():
        captured: dict = {}
        monkeypatch.setattr(
            llm, "generate", _captured_generate(captured, content='{"grounded": true}')
        )

        out = await llm.generate_structured(
            messages=[{"role": "user", "content": "is this grounded?"}],
            schema=_Verdict,
            max_tokens=cap,
        )

        assert isinstance(out, _Verdict), label
        assert out.grounded is True, label
        if label in _THINKING_PRONE:
            # A tiny gate cap must NOT reach generate() as-is -- it is floored
            # to the safe default so Gemini thinking can never be truncated.
            threaded = captured.get("max_tokens")
            assert threaded is not None, f"{label}: floor not applied"
            assert threaded >= llm._default_max_tokens, (
                f"{label}: small cap leaked below the safe default -> "
                "would truncate Gemini thinking tokens"
            )
            assert threaded > cap, f"{label}: tiny cap was NOT floored"
        else:
            assert captured.get("max_tokens") == cap, f"{label}: cap not threaded"


async def test_generate_structured_no_cap_preserves_default(monkeypatch):
    """Left None (the existing-caller path), generate() is NOT given
    max_tokens -> its default-4096 behavior is untouched. This is the
    back-compat guarantee for every current caller."""
    for label, llm in _provider_factories():
        captured: dict = {}
        monkeypatch.setattr(
            llm, "generate", _captured_generate(captured, content='{"grounded": false}')
        )

        out = await llm.generate_structured(
            messages=[{"role": "user", "content": "verdict?"}],
            schema=_Verdict,
        )

        assert isinstance(out, _Verdict), label
        assert out.grounded is False, label
        assert "max_tokens" not in captured, (
            f"{label}: max_tokens leaked into generate() when unset -- "
            "would override the default-4096 path"
        )


async def test_capped_and_uncapped_yield_identical_verdict(monkeypatch):
    """The same model output parses to the SAME object whether or not a cap is
    set -- the cap changes ONLY the transport ceiling, never the parsed answer.
    """
    from opsrag.llms.anthropic import AnthropicLLM

    llm = AnthropicLLM(api_key="test")
    payload = '{"grounded": true, "reason": "covered by source [1]"}'

    async def _fake_generate(**kwargs: Any) -> LLMResponse:
        return LLMResponse(content=payload, model="fake", usage={})

    monkeypatch.setattr(llm, "generate", _fake_generate)

    uncapped = await llm.generate_structured(
        messages=[{"role": "user", "content": "x"}], schema=_BudgetVerdict
    )
    capped = await llm.generate_structured(
        messages=[{"role": "user", "content": "x"}], schema=_BudgetVerdict,
        max_tokens=128,
    )

    assert uncapped == capped
    assert capped.grounded is True
    assert capped.reason == "covered by source [1]"


# ---------------------------------------------------------------------------
# TRACK gemini -- a tiny structured-gate cap can NEVER truncate a thinking model
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "factory",
    [
        lambda: __import__(
            "opsrag.llms.vertex", fromlist=["VertexAILLM"]
        ).VertexAILLM(model="gemini-2.5-flash"),
        lambda: __import__(
            "opsrag.llms.litellm_provider", fromlist=["LiteLLMLLM"]
        ).LiteLLMLLM(model="gemini/gemini-2.5-flash"),
    ],
    ids=["vertex", "litellm"],
)
async def test_gemini_structured_gate_cap_cannot_truncate_thinking(monkeypatch, factory):
    """REGRESSION (track=gemini): on the thinking-prone providers, the tiny gate
    cap (128) must NOT be threaded into generate() as-is. The in-prompt structured
    path sets no response_schema, so Gemini thinking tokens count against
    max_output_tokens -- a 128 ceiling could be wholly consumed by thinking,
    yielding empty output, a json parse failure, and (per the gates' fail-closed /
    fallback logic) a CHANGED verdict. So the cap must be floored to >= the safe
    default. Default left None must still mean "no max_tokens" so the existing
    default-4096 path is untouched."""
    llm = factory()
    default = llm._default_max_tokens

    # 1) A tiny cap is floored, never passed through verbatim.
    captured: dict = {}
    monkeypatch.setattr(
        llm, "generate", _captured_generate(captured, content='{"grounded": true}')
    )
    out = await llm.generate_structured(
        messages=[{"role": "user", "content": "grounded?"}],
        schema=_Verdict,
        max_tokens=128,
    )
    assert out.grounded is True
    threaded = captured.get("max_tokens")
    assert threaded is not None
    assert threaded != 128, "tiny gate cap leaked through -- Gemini thinking could truncate"
    assert threaded >= default, "cap floored below the safe default"

    # 2) A cap ABOVE the default is honored (floor is a no-op there).
    big = default + 1000
    captured.clear()
    out = await llm.generate_structured(
        messages=[{"role": "user", "content": "grounded?"}],
        schema=_Verdict,
        max_tokens=big,
    )
    assert out.grounded is True
    assert captured.get("max_tokens") == big

    # 3) Unset -> generate() gets NO max_tokens (default-4096 path untouched).
    captured.clear()
    monkeypatch.setattr(
        llm, "generate", _captured_generate(captured, content='{"grounded": false}')
    )
    out = await llm.generate_structured(
        messages=[{"role": "user", "content": "grounded?"}],
        schema=_Verdict,
    )
    assert out.grounded is False
    assert "max_tokens" not in captured


async def test_generate_structured_max_tokens_keyword_only():
    """The contract requires max_tokens be keyword-only (positional callers
    stay back-compatible). Passing it positionally must fail."""
    from opsrag.llms.anthropic import AnthropicLLM

    llm = AnthropicLLM(api_key="test")
    with pytest.raises(TypeError):
        # (messages, schema, system_prompt, purpose, <max_tokens positional>)
        await llm.generate_structured([], _Verdict, None, None, 128)


def test_protocol_generate_structured_accepts_max_tokens():
    """The Protocol signature itself must carry the keyword-only max_tokens so
    other tracks/callers can depend on it."""
    import inspect

    from opsrag.interfaces.llm import LLMProvider

    sig = inspect.signature(LLMProvider.generate_structured)
    p = sig.parameters["max_tokens"]
    assert p.kind == inspect.Parameter.KEYWORD_ONLY
    assert p.default is None


def test_all_providers_expose_keyword_only_max_tokens():
    """Every provider's generate_structured must match the contract: a
    keyword-only ``max_tokens: ... = None`` parameter."""
    import inspect

    from opsrag.llms.anthropic import AnthropicLLM
    from opsrag.llms.bedrock import BedrockLLM
    from opsrag.llms.litellm_provider import LiteLLMLLM
    from opsrag.llms.openai import OpenAILLM
    from opsrag.llms.vertex import VertexAILLM

    for cls in (AnthropicLLM, OpenAILLM, BedrockLLM, LiteLLMLLM, VertexAILLM):
        sig = inspect.signature(cls.generate_structured)
        assert "max_tokens" in sig.parameters, cls.__name__
        p = sig.parameters["max_tokens"]
        assert p.kind == inspect.Parameter.KEYWORD_ONLY, cls.__name__
        assert p.default is None, cls.__name__


# ---------------------------------------------------------------------------
# L2 -- client construction with timeout/retry args
# ---------------------------------------------------------------------------
def _install_fake_anthropic(monkeypatch):
    """Replace AsyncAnthropic in the provider module with a recorder so we can
    assert which kwargs reach the SDK constructor."""
    import opsrag.llms.anthropic as mod

    recorded: dict = {}

    class _FakeAsyncAnthropic:
        def __init__(self, **kwargs):
            recorded.clear()
            recorded.update(kwargs)

    monkeypatch.setattr(mod, "AsyncAnthropic", _FakeAsyncAnthropic)
    return recorded


def test_anthropic_construct_default_omits_timeout_retry(monkeypatch):
    """Unset -> the SDK constructor is given ONLY api_key (SDK-native timeout
    + retry defaults preserved). This is the existing-caller path."""
    from opsrag.llms.anthropic import AnthropicLLM

    recorded = _install_fake_anthropic(monkeypatch)
    llm = AnthropicLLM(api_key="k")
    llm._get_client()  # client is lazy

    assert recorded == {"api_key": "k"}
    assert "timeout" not in recorded
    assert "max_retries" not in recorded


def test_anthropic_construct_threads_timeout_retry(monkeypatch):
    from opsrag.llms.anthropic import AnthropicLLM

    recorded = _install_fake_anthropic(monkeypatch)
    llm = AnthropicLLM(api_key="k", timeout=30.0, max_retries=4)
    llm._get_client()

    assert recorded["api_key"] == "k"
    assert recorded["timeout"] == 30.0
    assert recorded["max_retries"] == 4


def test_anthropic_construct_partial_timeout_only(monkeypatch):
    from opsrag.llms.anthropic import AnthropicLLM

    recorded = _install_fake_anthropic(monkeypatch)
    llm = AnthropicLLM(api_key="k", timeout=12.5)
    llm._get_client()

    assert recorded["timeout"] == 12.5
    assert "max_retries" not in recorded


def _install_fake_boto(monkeypatch):
    """Stub boto3 + botocore.config so BedrockLLM construction is offline and we
    can capture the client() call's config kwarg."""
    captured: dict = {"client_kwargs": None, "config_kwargs": None}

    class _FakeConfig:
        def __init__(self, **kwargs):
            captured["config_kwargs"] = kwargs

    class _FakeSession:
        def __init__(self, **kwargs):
            pass

        def client(self, name, **kwargs):
            captured["client_kwargs"] = kwargs
            return object()

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.Session = _FakeSession

    fake_botocore = types.ModuleType("botocore")
    fake_botocore_config = types.ModuleType("botocore.config")
    fake_botocore_config.Config = _FakeConfig
    fake_botocore.config = fake_botocore_config

    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(sys.modules, "botocore", fake_botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", fake_botocore_config)
    return captured


def test_bedrock_construct_default_omits_config(monkeypatch):
    """Unset -> NO botocore Config is built; boto3 keeps its native timeout +
    retry defaults. Existing-caller path is byte-identical."""
    captured = _install_fake_boto(monkeypatch)
    from opsrag.llms.bedrock import BedrockLLM

    BedrockLLM(model="anthropic.claude-x:0")

    assert captured["client_kwargs"] == {}
    assert captured["config_kwargs"] is None


def test_bedrock_construct_builds_config_when_set(monkeypatch):
    """Mirrors embedders/bedrock.py: read_timeout/connect_timeout/adaptive
    retries land on the botocore Config when provided."""
    captured = _install_fake_boto(monkeypatch)
    from opsrag.llms.bedrock import BedrockLLM

    BedrockLLM(
        model="anthropic.claude-x:0",
        request_timeout=45.0,
        connect_timeout=5.0,
        max_retries=6,
    )

    assert "config" in captured["client_kwargs"]
    cfg = captured["config_kwargs"]
    assert cfg["read_timeout"] == 45.0
    assert cfg["connect_timeout"] == 5.0
    assert cfg["retries"] == {"max_attempts": 6, "mode": "adaptive"}


def test_bedrock_construct_partial_only_set_fields(monkeypatch):
    """Only the provided knobs appear on the Config -- a single field still
    builds a Config without inventing the others."""
    captured = _install_fake_boto(monkeypatch)
    from opsrag.llms.bedrock import BedrockLLM

    BedrockLLM(model="anthropic.claude-x:0", max_retries=3)

    assert "config" in captured["client_kwargs"]
    cfg = captured["config_kwargs"]
    assert cfg == {"retries": {"max_attempts": 3, "mode": "adaptive"}}
    assert "read_timeout" not in cfg
    assert "connect_timeout" not in cfg
