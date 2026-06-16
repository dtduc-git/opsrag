from opsrag.config import LLMConfig, VisionConfig


def _resolve(vision: VisionConfig, llm: LLMConfig):
    # Mirror the factory's resolution rule (pure function under test).
    from opsrag.factory import resolve_vision_model
    return resolve_vision_model(vision, llm)


def test_resolve_uses_explicit_override():
    v = VisionConfig(model="claude-sonnet-4-6", provider="bedrock")
    out = _resolve(v, LLMConfig(provider="bedrock", model="some-text-only"))
    assert out == ("bedrock", "claude-sonnet-4-6")


def test_resolve_reuses_active_model_when_vision_capable():
    v = VisionConfig()
    llm = LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514")
    # active model already sees -> no separate vision model needed
    assert _resolve(v, llm) is None


def test_resolve_falls_back_to_provider_default():
    v = VisionConfig()
    llm = LLMConfig(provider="vertex", model="text-bison")   # not vision-capable
    assert _resolve(v, llm) == ("vertex", "gemini-3-flash-preview")


def test_resolve_disabled_returns_none():
    assert _resolve(VisionConfig(enabled=False), LLMConfig()) is None
