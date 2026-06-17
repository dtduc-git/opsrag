"""L2 wiring: provider timeout/retry knobs thread into EVERY Anthropic/Bedrock
client construction (factory llm + vision; model_router cheap lanes).

These are PURE latency/robustness knobs (the provider track defaults them to
None == SDK-native behaviour). The wiring track's job is to forward
``cfg.llm.request_timeout`` / ``connect_timeout`` / ``max_retries`` so no path
keeps a bare client. The tests below spy the LLM constructors and assert the
exact kwargs forwarded -- equivalence-by-construction: with the default
LLMConfig the values are the documented defaults (120.0 / 10.0 / 2), and they
flow unchanged into the client.
"""
from __future__ import annotations

from opsrag.config import OpsRAGConfig


class _RecordingLLM:
    """Captures the kwargs each LLM client was constructed with."""

    calls: list[dict] = []

    def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        # store the (provider-tagged) construction kwargs
        type(self).calls.append(dict(kwargs, _model=kwargs.get("model")))

    # PurposeRouter dedupes by (provider, model) and seeds with the default
    # llm; nothing here is actually invoked.
    @property
    def model_name(self):  # noqa: ANN201
        return self.calls[-1].get("model") if self.calls else "stub"


def _make_recording_cls():
    """Fresh recording class so call lists don't leak between tests."""

    class _Rec(_RecordingLLM):
        calls: list[dict] = []

    return _Rec


def _offline_config(provider: str) -> OpsRAGConfig:
    """A fully-offline config: in-memory qdrant, openai embedder (no network at
    construct time), no graph/memory/reranker/sources. Only the llm provider
    varies so we exercise the anthropic vs bedrock construction branches."""
    return OpsRAGConfig.model_validate(
        {
            "scm": {"provider": "github", "base_url": "https://api.github.com"},
            "embedding": {"provider": "openai", "model": "text-embedding-3-small",
                          "dimension": 8, "api_key_env": "UNUSED_KEY_ENV"},
            "vector_store": {"provider": "qdrant", "url": ":memory:",
                             "collection": "t", "api_key_env": None},
            "llm": {"provider": provider,
                    "model": ("claude-x" if provider == "anthropic"
                              else "anthropic.claude-x:0")},
            "reranker": {"provider": "noop"},
            "session": {"provider": "memory"},
            "observability": {"provider": "console"},
            "knowledge_graph": {"provider": "none"},
            "entity_extraction": {"method": "none"},
            "memory": {"provider": "none"},
            "vision": {"enabled": False},
        }
    )


def test_factory_anthropic_llm_gets_configured_timeouts(monkeypatch):
    import opsrag.factory as factory

    rec = _make_recording_cls()
    monkeypatch.setattr(factory, "AnthropicLLM", rec)

    cfg = _offline_config("anthropic")
    factory.build_providers(cfg)

    # The classic llm client (and PurposeRouter's seeded default, same target)
    # must carry the configured robustness knobs.
    assert rec.calls, "AnthropicLLM was never constructed"
    main = rec.calls[0]
    assert main["timeout"] == cfg.llm.request_timeout == 120.0
    assert main["max_retries"] == cfg.llm.max_retries == 2


def test_factory_bedrock_llm_gets_configured_timeouts(monkeypatch):
    import opsrag.factory as factory

    # Patch the lazily-imported BedrockLLM symbol at its source module so the
    # factory's `from opsrag.llms.bedrock import BedrockLLM` picks up the spy.
    import opsrag.llms.bedrock as bedrock_mod
    rec = _make_recording_cls()
    monkeypatch.setattr(bedrock_mod, "BedrockLLM", rec)

    cfg = _offline_config("bedrock")
    factory.build_providers(cfg)

    assert rec.calls, "BedrockLLM was never constructed"
    main = rec.calls[0]
    assert main["request_timeout"] == cfg.llm.request_timeout == 120.0
    assert main["connect_timeout"] == cfg.llm.connect_timeout == 10.0
    assert main["max_retries"] == cfg.llm.max_retries == 2


def test_factory_threads_custom_timeout_values(monkeypatch):
    """Non-default config values must flow through verbatim (not hard-coded)."""
    import opsrag.factory as factory

    rec = _make_recording_cls()
    monkeypatch.setattr(factory, "AnthropicLLM", rec)

    cfg = _offline_config("anthropic")
    cfg.llm.request_timeout = 12.5
    cfg.llm.max_retries = 7
    factory.build_providers(cfg)

    main = rec.calls[0]
    assert main["timeout"] == 12.5
    assert main["max_retries"] == 7


def test_factory_vision_anthropic_gets_timeouts(monkeypatch):
    """The vision fallback client is a path that must NOT keep a bare client."""
    import opsrag.factory as factory

    rec = _make_recording_cls()
    monkeypatch.setattr(factory, "AnthropicLLM", rec)

    cfg = _offline_config("anthropic")
    # Force a separate vision client: enable vision + a text-only main model so
    # resolve_vision_model returns an explicit anthropic target.
    cfg.vision.enabled = True
    cfg.vision.provider = "anthropic"
    cfg.vision.model = "claude-vision-x"
    factory.build_providers(cfg)

    vision_calls = [c for c in rec.calls if c.get("model") == "claude-vision-x"]
    assert vision_calls, "vision AnthropicLLM was never constructed"
    v = vision_calls[0]
    assert v["timeout"] == cfg.llm.request_timeout
    assert v["max_retries"] == cfg.llm.max_retries


def test_model_router_build_llm_anthropic_threads_timeouts():
    """model_router._build_llm reads the same timeouts from settings.llm."""
    from opsrag import model_router as mr

    captured: dict = {}

    class _Spy:
        def __init__(self, *a, **k):  # noqa: ANN002, ANN003
            captured.update(k)

    import opsrag.llms.anthropic as anth_mod
    orig = anth_mod.AnthropicLLM
    anth_mod.AnthropicLLM = _Spy
    try:
        cfg = _offline_config("anthropic")
        cfg.llm.request_timeout = 33.0
        cfg.llm.max_retries = 5
        mr._build_llm("anthropic", "claude-cheap", cfg)
    finally:
        anth_mod.AnthropicLLM = orig

    assert captured["timeout"] == 33.0
    assert captured["max_retries"] == 5
    assert captured["model"] == "claude-cheap"


def test_model_router_build_llm_bedrock_threads_timeouts():
    from opsrag import model_router as mr

    captured: dict = {}

    class _Spy:
        def __init__(self, *a, **k):  # noqa: ANN002, ANN003
            captured.update(k)

    import opsrag.llms.bedrock as bedrock_mod
    orig = bedrock_mod.BedrockLLM
    bedrock_mod.BedrockLLM = _Spy
    try:
        cfg = _offline_config("bedrock")
        cfg.llm.request_timeout = 21.0
        cfg.llm.connect_timeout = 4.0
        cfg.llm.max_retries = 9
        mr._build_llm("bedrock", "anthropic.cheap:0", cfg)
    finally:
        bedrock_mod.BedrockLLM = orig

    assert captured["request_timeout"] == 21.0
    assert captured["connect_timeout"] == 4.0
    assert captured["max_retries"] == 9
    assert captured["model"] == "anthropic.cheap:0"
