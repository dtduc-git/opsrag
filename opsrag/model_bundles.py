"""Cloud model bundles + fill-unset resolver (models feature, DESIGN 4).

A ``cloud_provider`` flag ("aws" | "gcp" | None) selects a curated
``CLOUD_BUNDLES`` table keyed by ``provider -> purpose -> ModelSpec``.
``resolve_cloud_bundle`` is a pure, deterministic defaulting layer that
runs in ``Settings.load`` *after* ``_apply_env_overrides`` so explicit
slot config + ``models`` overrides + env always win. It only fills
slots/purposes the operator left UNSET; it never overrides an
explicitly-set or env-provided value.

Bundles carry **model families only** -- per-deployment region / project /
account stay in the existing slot blocks (``aws_region`` / ``project`` /
``location``), satisfying Constitution Principle VI.

aws (research-best): reason=Bedrock Opus, tool_call/summarize/extract=
Bedrock Haiku, embed=Bedrock Titan v2 (1024-dim), rerank=Cohere
`Cohere Rerank 3.5 hosted on Bedrock (cohere.rerank-v3-5:0).

gcp (upstream mirror): reason/tool_call/summarize/extract=Vertex
``gemini-2.5-flash`` (Pro escalation handled by the router/agent), embed=
Vertex ``text-embedding-005`` (768-dim), rerank=Vertex
``semantic-ranker-default-004``.
"""
from __future__ import annotations

from opsrag.config import ModelsConfig, ModelSpec, Settings

# Default slot values that count as "unset" for fill purposes. The
# classic provider blocks ship with eager defaults (e.g. llm.provider=
# "anthropic", llm.model="claude-sonnet-4-20250514"); when the operator
# left those at the shipped default we still want the bundle to fill them.
# We treat a slot value as overridable iff it equals the shipped default.
_DEFAULT_LLM_PROVIDER = "anthropic"
_DEFAULT_LLM_MODEL = "claude-sonnet-4-20250514"
_DEFAULT_EMBED_PROVIDER = "openai"
_DEFAULT_EMBED_MODEL = "text-embedding-3-large"
# Keep in sync with RerankerConfig defaults: the bundle fills the reranker slot
# only when it's still at this default pair (i.e. the operator didn't pick one).
_DEFAULT_RERANK_PROVIDER = "fastembed"
_DEFAULT_RERANK_MODEL = "rerank-v3.5"


# provider -> purpose -> ModelSpec. embed/rerank ModelSpecs carry a
# `dimension`-bearing model family; the embed dimension is filled
# separately via _BUNDLE_EMBED_DIMENSION below.
CLOUD_BUNDLES: dict[str, dict[str, ModelSpec]] = {
    "aws": {
        "reason": ModelSpec(
            provider="bedrock",
            # Sonnet 4.6 (not Opus 4.8): with strong context/chunking the
            # reasoning lane doesn't need the largest model -- Sonnet 4.6 is
            # ~5x cheaper and fast enough for synthesis. See pro_model below.
            model="us.anthropic.claude-sonnet-4-6",
            effort="medium",
        ),
        "tool_call": ModelSpec(
            provider="bedrock",
            model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        ),
        "summarize": ModelSpec(
            provider="bedrock",
            model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        ),
        "extract": ModelSpec(
            provider="bedrock",
            model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        ),
        "embed": ModelSpec(
            provider="bedrock",
            model="us.cohere.embed-v4:0",       # Cohere Embed v4 (1536, 128K ctx) > Titan
        ),
        "rerank": ModelSpec(
            provider="bedrock",                 # Cohere Rerank 3.5 ON Bedrock
            model="cohere.rerank-v3-5:0",       # no COHERE_API_KEY needed
        ),
    },
    "gcp": {
        "reason": ModelSpec(
            provider="vertex",
            model="gemini-2.5-flash",
            effort="medium",
        ),
        "tool_call": ModelSpec(
            provider="vertex",
            model="gemini-2.5-flash",
        ),
        "summarize": ModelSpec(
            provider="vertex",
            model="gemini-2.5-flash",
        ),
        "extract": ModelSpec(
            provider="vertex",
            model="gemini-2.5-flash",
        ),
        "embed": ModelSpec(
            provider="vertex",
            model="gemini-embedding-001",   # stronger than text-embedding-005; code-capable, MRL dims
        ),
        "rerank": ModelSpec(
            provider="vertex",
            model="semantic-ranker-default-004",
        ),
    },
}

# Embedding dimension carried by each bundle's embed model. Filled onto
# embedding.dimension only when unset (None).
_BUNDLE_EMBED_DIMENSION: dict[str, int] = {
    "aws": 1536,   # us.cohere.embed-v4:0 (Matryoshka; 1536 = full)
    "gcp": 3072,   # gemini-embedding-001 (native; MRL can reduce to 1536/768)
}

# gcp Pro-escalation model folded into the bundle so escalation is
# provider-agnostic (server.py hardcodes this today). Filled onto
# agent.pro_model only when unset.
_BUNDLE_PRO_MODEL: dict[str, str | None] = {
    "aws": "us.anthropic.claude-sonnet-4-6",   # was Opus 4.8; Sonnet 4.6 escalation
    "gcp": "gemini-2.5-pro",
}

# The six purpose keys carried by a bundle / ModelsConfig.
_PURPOSES = ("reason", "tool_call", "embed", "rerank", "summarize", "extract")
# The LLM-bearing purposes (everything except embed/rerank). Used to keep the
# bundle from injecting a provider into these that conflicts with an
# explicitly-pinned classic `llm` slot (split-brain guard in resolve_cloud_bundle).
_LLM_PURPOSES = frozenset({"reason", "tool_call", "summarize", "extract"})


def _llm_slot_is_default(settings: Settings) -> bool:
    """True iff llm.provider/model are still at the shipped defaults."""
    return (
        settings.llm.provider == _DEFAULT_LLM_PROVIDER
        and settings.llm.model == _DEFAULT_LLM_MODEL
    )


def _embed_slot_is_default(settings: Settings) -> bool:
    return (
        settings.embedding.provider == _DEFAULT_EMBED_PROVIDER
        and settings.embedding.model == _DEFAULT_EMBED_MODEL
    )


def _rerank_slot_is_default(settings: Settings) -> bool:
    return (
        settings.reranker.provider == _DEFAULT_RERANK_PROVIDER
        and settings.reranker.model == _DEFAULT_RERANK_MODEL
    )


def _fill_models_purpose(
    models: ModelsConfig,
    purpose: str,
    bundle_spec: ModelSpec,
) -> None:
    """Fill an UNSET ``models.<purpose>`` ModelSpec field-by-field from the
    bundle spec. An explicitly-set field on the existing spec always wins."""
    existing: ModelSpec | None = getattr(models, purpose, None)
    if existing is None:
        # Whole purpose unset -> copy the bundle spec (a fresh instance so
        # later mutation of one settings object never aliases the table).
        setattr(models, purpose, bundle_spec.model_copy(deep=True))
        return
    # Purpose present but some fields may be unset -> fill those only.
    if existing.provider is None and bundle_spec.provider is not None:
        existing.provider = bundle_spec.provider
    if existing.model is None and bundle_spec.model is not None:
        existing.model = bundle_spec.model
    if existing.effort is None and bundle_spec.effort is not None:
        existing.effort = bundle_spec.effort


def resolve_cloud_bundle(settings: Settings) -> None:
    """Mutate ``settings`` in place: fill UNSET model slots + per-purpose
    ``models`` entries from the selected ``cloud_provider`` bundle.

    No-op when ``cloud_provider`` is None (today's behavior). Pure +
    deterministic: explicit slot config + env-provided values + explicit
    ``models`` fields are never overridden.
    """
    provider = settings.cloud_provider
    if provider is None:
        return
    bundle = CLOUD_BUNDLES.get(provider)
    if bundle is None:
        return

    # 1. Per-purpose `models` block. Create it if absent so the router can
    #    consult resolved purposes regardless of operator config.
    if settings.models is None:
        settings.models = ModelsConfig()
    llm_overridden = not _llm_slot_is_default(settings)
    for purpose in _PURPOSES:
        spec = bundle.get(purpose)
        if spec is None:
            continue
        # Split-brain guard: when the operator explicitly pinned the classic
        # `llm` slot to a DIFFERENT provider than this bundle, don't inject the
        # bundle's provider into the LLM purposes -- otherwise e.g. models.reason
        # becomes vertex while llm.provider stays bedrock, and the router builds
        # a Vertex client using bedrock's slot config. The router falls back to
        # the explicit classic llm client for skipped purposes. Matching
        # providers fill as normal (preserves the cheap tool/summarize lane).
        if (
            purpose in _LLM_PURPOSES
            and llm_overridden
            and spec.provider is not None
            and spec.provider != settings.llm.provider
        ):
            continue
        _fill_models_purpose(settings.models, purpose, spec)

    # 2. Classic `llm` slot <- bundle `reason` (the strong/default model).
    reason = bundle.get("reason")
    if reason is not None and _llm_slot_is_default(settings):
        if reason.provider is not None:
            settings.llm.provider = reason.provider  # type: ignore[assignment]
        if reason.model is not None:
            settings.llm.model = reason.model

    # 3. Classic `embedding` slot <- bundle `embed`.
    embed = bundle.get("embed")
    if embed is not None and _embed_slot_is_default(settings):
        if embed.provider is not None:
            settings.embedding.provider = embed.provider  # type: ignore[assignment]
        if embed.model is not None:
            settings.embedding.model = embed.model
    # Embedding dimension: fill only when unset (None). This is the
    # load-bearing vector-store seam -- never override an explicit value.
    if settings.embedding.dimension is None:
        dim = _BUNDLE_EMBED_DIMENSION.get(provider)
        if dim is not None:
            settings.embedding.dimension = dim

    # 4. Classic `reranker` slot <- bundle `rerank`.
    rerank = bundle.get("rerank")
    if rerank is not None and _rerank_slot_is_default(settings):
        if rerank.provider is not None:
            settings.reranker.provider = rerank.provider  # type: ignore[assignment]
        if rerank.model is not None:
            settings.reranker.model = rerank.model

    # 5. Pro-escalation model folded into the bundle so escalation is
    #    provider-agnostic. Fill only when unset.
    if settings.agent.pro_model is None:
        pro = _BUNDLE_PRO_MODEL.get(provider)
        if pro is not None:
            settings.agent.pro_model = pro
