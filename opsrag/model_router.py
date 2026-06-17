"""PurposeRouter -- per-PURPOSE LLM client selection (models feature, F).

Given a resolved ``Settings`` (after ``resolve_cloud_bundle``), returns an
``LLMProvider`` for a logical purpose:

  - ``reason``    -> the strong model (e.g. aws Bedrock Opus)
  - ``tool_call`` -> the cheap model (e.g. aws Bedrock Haiku)
  - ``summarize`` / ``extract`` -> the cheap model lane by default

Clients are **deduped by (provider, model)** so one client instance is
reused across calls with the same target. This is load-bearing for prompt
caching: the high-volume ``tool_call`` loop reuses one Haiku client, so a
stable cached prefix (system + tool defs) survives across calls. The
router must never rebuild that prefix per call -- it only selects a client.

Degrades to ``settings.llm`` (the classic slot) when no bundle / no
``models`` override resolves a purpose, so a no-bundle deployment is a
tested no-op: every purpose returns the same object as ``settings.llm``.
``embed`` / ``rerank`` are NOT LLMProviders and are not governed here --
they stay on the embedder / reranker chains.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from opsrag.interfaces.llm import LLMProvider

if TYPE_CHECKING:  # pragma: no cover
    from opsrag.config import ModelSpec, Settings

_log = logging.getLogger("opsrag.model_router")

# Logical LLM purposes this router governs. embed/rerank are excluded.
_LLM_PURPOSES = ("reason", "tool_call", "summarize", "extract")

# Purposes that map to the cheap lane when their own spec is unset.
_CHEAP_PURPOSES = ("tool_call", "summarize", "extract")


def _build_llm(
    provider: str,
    model: str,
    settings: Settings,
) -> LLMProvider:
    """Construct an LLMProvider for (provider, model), reusing the same
    constructor wiring as the factory llm chain. Region / project / key
    env come from the classic ``settings.llm`` slot (bundles carry only
    model families)."""
    slot = settings.llm
    import os

    def _env(name: str) -> str | None:
        v = os.environ.get(name)
        return v if v else None

    if provider == "anthropic":
        from opsrag.llms.anthropic import AnthropicLLM
        return AnthropicLLM(
            api_key=_env(slot.api_key_env),
            model=model,
            default_max_tokens=slot.max_tokens,
            # Bound provider tail latency on the cheap lanes too -- read the
            # same timeout/retry knobs as the factory's classic llm slot.
            timeout=slot.request_timeout,
            max_retries=slot.max_retries,
        )
    if provider == "openai":
        from opsrag.llms.openai import OpenAILLM
        return OpenAILLM(
            api_key=_env(slot.api_key_env),
            model=model,
            default_max_tokens=slot.max_tokens,
        )
    if provider == "vertex":
        from opsrag.llms.vertex import VertexAILLM
        return VertexAILLM(
            model=model,
            project=slot.project,
            location=slot.location or "us-central1",
            default_max_tokens=slot.max_tokens,
        )
    if provider == "bedrock":
        from opsrag.llms.bedrock import BedrockLLM
        return BedrockLLM(
            model=model,
            region=slot.aws_region,
            profile=slot.aws_profile,
            default_max_tokens=slot.max_tokens,
            request_timeout=slot.request_timeout,
            connect_timeout=slot.connect_timeout,
            max_retries=slot.max_retries,
        )
    raise NotImplementedError(f"PurposeRouter: LLM provider {provider!r} not available")


class PurposeRouter:
    """Selects + memoizes an LLMProvider per logical purpose.

    Construct once from a resolved ``Settings``. ``pick(purpose)`` returns a
    cached client; same ``(provider, model)`` target -> same instance.
    """

    def __init__(
        self,
        settings: Settings,
        default_llm: LLMProvider | None = None,
    ) -> None:
        """``default_llm`` is the already-built classic client (the factory's
        ``providers.llm``). When omitted, the router builds one lazily from
        ``settings.llm`` so the class is usable in isolation. Passing the
        factory's client preserves the no-bundle invariant: every purpose
        returns the *same object* the rest of the app already holds."""
        self._settings = settings
        # (provider, model) -> client. Seeded with the classic llm client so
        # the no-bundle / fallback path returns the SAME object as
        # providers.llm (preserves prompt-cache + back-compat invariants).
        self._clients: dict[tuple[str, str], LLMProvider] = {}
        if default_llm is None:
            default_llm = _build_llm(
                settings.llm.provider, settings.llm.model, settings,
            )
        self._default_llm: LLMProvider = default_llm
        self._clients[(settings.llm.provider, settings.llm.model)] = self._default_llm
        # purpose -> resolved (provider, model) or None (fall back to default).
        self._resolved: dict[str, tuple[str, str] | None] = {}
        for purpose in _LLM_PURPOSES:
            self._resolved[purpose] = self._resolve_target(purpose)

    def _resolve_target(self, purpose: str) -> tuple[str, str] | None:
        """Resolve a purpose to a concrete (provider, model), or None to
        fall back to the classic ``settings.llm`` slot."""
        models = self._settings.models
        spec: ModelSpec | None = None
        if models is not None:
            spec = getattr(models, purpose, None)
            # Cheap purposes inherit tool_call when their own spec is unset.
            if spec is None and purpose in _CHEAP_PURPOSES:
                spec = getattr(models, "tool_call", None)
        if spec is None or spec.provider is None or spec.model is None:
            return None
        return (spec.provider, spec.model)

    def pick(self, purpose: str) -> LLMProvider:
        """Return the LLMProvider for ``purpose`` (memoized).

        Unknown purposes and unresolved purposes both fall back to the
        classic ``settings.llm`` client."""
        target = self._resolved.get(purpose)
        if target is None:
            return self._default_llm
        provider, model = target
        existing = self._clients.get(target)
        if existing is not None:
            return existing
        client = _build_llm(provider, model, self._settings)
        self._clients[target] = client
        _log.info(
            "purpose_router built client purpose=%s provider=%s model=%s",
            purpose, provider, model,
        )
        return client

    @property
    def reason(self) -> LLMProvider:
        """The strong model lane."""
        return self.pick("reason")

    @property
    def tool_call(self) -> LLMProvider:
        """The cheap model lane."""
        return self.pick("tool_call")

    @property
    def default_llm(self) -> LLMProvider:
        """The classic ``settings.llm`` client (no-bundle default)."""
        return self._default_llm
