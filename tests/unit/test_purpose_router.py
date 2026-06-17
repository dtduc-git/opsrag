"""Unit tests for PurposeRouter (models feature, F)."""
from __future__ import annotations

import pytest

from opsrag.config import Settings
from opsrag.model_bundles import resolve_cloud_bundle
from opsrag.model_router import PurposeRouter

# Bedrock LLM construction opens a boto3 session/client; stub it so unit
# tests need no live AWS. We patch the constructor used by the router.


class _StubLLM:
    def __init__(self, model: str, **kwargs):
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model


@pytest.fixture
def stub_bedrock(monkeypatch):
    """Replace BedrockLLM with a no-network stub for the router build path."""
    import opsrag.llms.bedrock as bedrock_mod

    def _factory(model="m", region=None, profile=None, default_max_tokens=4096, **kwargs):
        # **kwargs tolerates the timeout/retry robustness knobs the router now
        # threads in (request_timeout / connect_timeout / max_retries); the
        # real BedrockLLM accepts them, so the stub must too.
        return _StubLLM(model)

    monkeypatch.setattr(bedrock_mod, "BedrockLLM", _factory)
    return _factory


def test_reason_vs_tool_call_differ_for_aws(stub_bedrock):
    cfg = Settings(cloud_provider="aws")
    resolve_cloud_bundle(cfg)
    router = PurposeRouter(cfg)

    reason = router.reason
    tool_call = router.tool_call

    assert reason.model_name == "us.anthropic.claude-sonnet-4-6"
    assert tool_call.model_name == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert reason.model_name != tool_call.model_name


def test_dedup_returns_same_instance_for_same_target(stub_bedrock):
    cfg = Settings(cloud_provider="aws")
    resolve_cloud_bundle(cfg)
    router = PurposeRouter(cfg)

    # tool_call, summarize, extract all map to the same Haiku (provider,model)
    # -> the client must be deduped to a single instance (prompt caching).
    tc = router.pick("tool_call")
    summ = router.pick("summarize")
    extract = router.pick("extract")
    assert tc is summ is extract

    # Repeated picks of reason also return the same memoized instance.
    assert router.pick("reason") is router.pick("reason")


def test_no_bundle_falls_back_to_default_llm():
    # No cloud_provider, no models -> every purpose returns the factory's
    # default client (passed in as the already-built providers.llm).
    cfg = Settings()  # classic anthropic default
    default = _StubLLM("claude-sonnet-4-20250514")
    router = PurposeRouter(cfg, default_llm=default)

    assert router.reason is default
    assert router.tool_call is default
    assert router.pick("summarize") is default
    assert router.pick("extract") is default
    # Unknown purpose also falls back to the default client.
    assert router.pick("totally-unknown") is default
    assert router.default_llm is default


def test_reason_target_equal_to_llm_slot_reuses_default(stub_bedrock):
    # When the bundle's reason equals the (already bundle-filled) llm slot,
    # the router seeds that target with the passed default client and reuses
    # it (no second Bedrock client built for reason).
    cfg = Settings(cloud_provider="aws")
    resolve_cloud_bundle(cfg)
    default = _StubLLM(cfg.llm.model)  # (bedrock, opus) -- same as reason
    router = PurposeRouter(cfg, default_llm=default)
    assert router.reason is default
