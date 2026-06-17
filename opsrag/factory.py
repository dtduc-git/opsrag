"""Dependency wiring - builds concrete providers from OpsRAGConfig.

Keeps the knobs in one place so new phases can swap implementations
without touching call sites.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from opsrag.chunkers.fixed_size import FixedSizeChunker
from opsrag.chunkers.parent_child import ParentChildChunker
from opsrag.config import OpsRAGConfig
from opsrag.embedders.cached import CachedEmbedder
from opsrag.embedders.openai import OpenAIEmbeddings
from opsrag.extractors.hybrid import HybridExtractor
from opsrag.extractors.llm_extractor import LLMEntityExtractor
from opsrag.extractors.rule_based import RuleBasedExtractor
from opsrag.graphstores.null import NullGraphStore
from opsrag.indexed_files.noop import NoopIndexedFilesTracker
from opsrag.indexed_files.postgres import PostgresIndexedFilesTracker
from opsrag.interfaces.chunker import ChunkingStrategy
from opsrag.interfaces.embedder import EmbeddingProvider
from opsrag.interfaces.entity_extractor import EntityExtractor
from opsrag.interfaces.graphstore import KnowledgeGraphStore
from opsrag.interfaces.indexed_files import IndexedFilesTracker
from opsrag.interfaces.llm import LLMProvider
from opsrag.interfaces.memory import MemoryStore
from opsrag.interfaces.observability import ObservabilityProvider
from opsrag.interfaces.parser import DocumentParser
from opsrag.interfaces.reranker import Reranker
from opsrag.interfaces.scm import SCMProvider
from opsrag.interfaces.session import SessionStore
from opsrag.interfaces.vectorstore import VectorStore
from opsrag.llms.anthropic import AnthropicLLM
from opsrag.memory.memory import InMemoryMemoryStore
from opsrag.memory.postgres import PostgresMemoryStore
from opsrag.model_router import PurposeRouter
from opsrag.observability.console import ConsoleObservability
from opsrag.observability.phoenix import PhoenixObservability
from opsrag.parsers.alert import AlertParser
from opsrag.parsers.generic import GenericConfigParser
from opsrag.parsers.helm import HelmParser
from opsrag.parsers.k8s import K8sManifestParser
from opsrag.parsers.markdown import GenericMarkdownParser
from opsrag.parsers.postmortem import PostmortemParser
from opsrag.parsers.runbook import RunbookParser
from opsrag.parsers.terraform import TerraformParser
from opsrag.rerankers.cohere import CohereReranker
from opsrag.rerankers.noop import NoOpReranker
from opsrag.scm.github import GitHubSCM
from opsrag.scm.gitlab import GitLabSCM
from opsrag.sessions.memory import InMemorySessionStore
from opsrag.sessions.postgres import PostgresSessionStore
from opsrag.vectorstores.qdrant import QdrantVectorStore


@dataclass
class Providers:
    scm: SCMProvider
    parsers: list[DocumentParser]
    chunker: ChunkingStrategy
    embedder: EmbeddingProvider
    vector_store: VectorStore
    llm: LLMProvider
    reranker: Reranker
    session_store: SessionStore
    observability: ObservabilityProvider
    graph_store: KnowledgeGraphStore | None = None
    entity_extractor: EntityExtractor | None = None
    memory_store: MemoryStore | None = None
    # Per-purpose model router (reason / tool_call / summarize / extract).
    # Falls back to `llm` when no cloud bundle / models overrides are set.
    purpose_router: PurposeRouter | None = None
    indexed_files: IndexedFilesTracker | None = None
    # Phase 2 - non-git sources, keyed by `source_type`. Empty dict by
    # default; populated when `cfg.confluence.enabled` (and similarly
    # rootly, slack, etc.) is true.
    sources: dict = field(default_factory=dict)
    # Optional code-specific embedder + collection. Both are None when
    # the code lane is disabled (default). When set, the ingestion
    # pipeline dual-writes code DocType chunks to `code_vector_store`
    # using `code_embedder` (with a code-specific query task type); the
    # retrieval pipeline adds a 4th RRF lane for identifier-heavy queries.
    code_embedder: EmbeddingProvider | None = None
    code_vector_store: VectorStore | None = None
    # Lightweight entity-graph (Postgres edges) for the entity-expansion
    # retrieval lane. None when light_graph.enabled is false.
    light_graph: Any | None = None
    # Durable indexing job-state (Postgres). Writers flush the in-memory
    # tracker here; backend pods read it so /indexing/status is consistent
    # across replicas. None when sessions aren't on Postgres (dev no-op ->
    # routes fall back to the in-memory tracker).
    index_store: Any | None = None
    # Live, operator-editable agent settings (Postgres) -- backs the admin
    # "Agent Guidance" page (custom instructions). None when sessions aren't on
    # Postgres (prompt injection then uses the config seed only).
    agent_settings: Any | None = None
    # Provider-aware vision fallback LLM. Built once at startup only when the
    # active `llm` model can't see (else None; the generator reuses `llm`).
    # Defaults to None so existing constructions don't break.
    vision_llm: LLMProvider | None = None


def _env(name: str) -> str | None:
    v = os.environ.get(name)
    return v if v else None


def resolve_vision_model(vision, llm_cfg) -> tuple[str, str] | None:
    """Return (provider, model) for the vision fallback LLM, or None when no
    separate vision client is needed (disabled, or the active model already
    sees). Explicit vision.model/provider always wins (spec FR-011)."""
    from opsrag.llms.content import default_vision_model, is_vision_capable

    if not getattr(vision, "enabled", True):
        return None
    if vision.model:
        return (vision.provider or llm_cfg.provider, vision.model)
    if is_vision_capable(llm_cfg.provider, llm_cfg.model):
        return None  # active model can already see; generator reuses it
    provider = vision.provider or llm_cfg.provider
    model = default_vision_model(provider)
    return (provider, model) if model else None


def build_providers(config: OpsRAGConfig) -> Providers:
    token = _env(config.scm.token_env) or ""

    if config.scm.clone_mode and config.scm.provider in ("gitlab", "github"):
        from opsrag.scm.git_clone import GitCloneSCM
        scm: SCMProvider = GitCloneSCM(
            base_url=config.scm.base_url,
            token=token,
            provider=config.scm.provider,
            use_ssh=config.scm.use_ssh,
            ssh_host=config.scm.ssh_host,
            ssh_user=config.scm.ssh_user,
        )
    elif config.scm.provider == "gitlab":
        scm = GitLabSCM(token=token, base_url=config.scm.base_url)
    elif config.scm.provider == "github":
        scm = GitHubSCM(
            token=token,
            base_url=config.scm.base_url
            if config.scm.base_url != "https://gitlab.com"
            else "https://api.github.com",
        )
    else:
        raise NotImplementedError(f"SCM provider {config.scm.provider!r} not available")

    # Order matters: most specific first. GenericMarkdown handles .md only,
    # GenericConfig is the last-resort catch-all for text/config files
    # nothing else claimed (CI YAML, scripts, Dockerfiles, etc.).
    # See tests/unit/test_parser_priority.py for conflict resolution tests.
    parsers: list[DocumentParser] = [
        RunbookParser(),
        PostmortemParser(),
        AlertParser(),
        K8sManifestParser(),
        HelmParser(),
        TerraformParser(),
        GenericMarkdownParser(),
        GenericConfigParser(),
    ]

    if config.chunking.strategy == "parent_child":
        chunker: ChunkingStrategy = ParentChildChunker(
            child_size=config.chunking.child_size,
            child_overlap=config.chunking.child_overlap,
            parent_max_tokens=config.chunking.parent_max_tokens,
            code_parent_max_tokens=config.chunking.code_parent_max_tokens,
        )
    elif config.chunking.strategy == "fixed_size":
        chunker = FixedSizeChunker(
            chunk_size=config.chunking.chunk_size,
            overlap=config.chunking.overlap,
        )
    else:
        raise NotImplementedError(f"Chunker {config.chunking.strategy!r} not available")

    if config.embedding.provider == "openai":
        embedder: EmbeddingProvider = OpenAIEmbeddings(
            api_key=_env(config.embedding.api_key_env),
            model=config.embedding.model,
            dimension=config.embedding.dimension,
        )
    elif config.embedding.provider == "fastembed":
        from opsrag.embedders.fastembed import FastEmbedEmbeddings
        embedder = FastEmbedEmbeddings(model=config.embedding.model)
    elif config.embedding.provider == "vertex":
        from opsrag.embedders.vertex import VertexAIEmbeddings
        embedder = VertexAIEmbeddings(
            model=config.embedding.model,
            project=config.embedding.project,
            location=config.embedding.location or "us-central1",
            # Pass the configured dimension through as Matryoshka
            # output_dimensionality, so `.dimension` (collection size) matches
            # the vectors actually produced. Without this, models like
            # gemini-embedding-001 report a default dim but return native-size
            # vectors -> every upsert is rejected by the vector store.
            output_dimensionality=config.embedding.dimension,
        )
    elif config.embedding.provider == "bedrock":
        from opsrag.embedders.bedrock import BedrockEmbeddings
        embedder = BedrockEmbeddings(
            model=config.embedding.model or "amazon.titan-embed-text-v2:0",
            region=config.embedding.aws_region,
            profile=config.embedding.aws_profile,
            dimension=config.embedding.dimension,
        )
    elif config.embedding.provider == "litellm":
        from opsrag.embedders.litellm_provider import LiteLLMEmbeddings
        embedder = LiteLLMEmbeddings(
            model=config.embedding.model,
            dimension=config.embedding.dimension,
            api_base=config.embedding.api_base,
            api_key_env=config.embedding.api_key_env,
        )
    else:
        raise NotImplementedError(f"Embedder {config.embedding.provider!r} not available")

    # T2.3 - wrap with a small in-memory LRU+TTL. Same query gets
    # re-embedded 3-5x per chat turn (cache lookup, classifier,
    # semantic router, retrieval, qa_cache). 60s TTL + 1k entries is
    # tuned for chat-burst patterns and keeps memory well under 100MB.
    embedder = CachedEmbedder(
        embedder,
        max_size=config.embedding.cache_max_size,
        ttl_seconds=config.embedding.cache_ttl_seconds,
    )

    if config.vector_store.provider == "qdrant":
        vector_store: VectorStore = QdrantVectorStore(
            url=config.vector_store.url,
            api_key=_env(config.vector_store.api_key_env) if config.vector_store.api_key_env else None,
            collection_name=config.vector_store.collection,
            dimension=embedder.dimension,
            allow_dimension_change=config.vector_store.allow_dimension_change,
        )
    elif config.vector_store.provider == "pgvector":
        from opsrag.vectorstores.pgvector import PgVectorStore
        dsn = config.vector_store.dsn or _env(config.vector_store.dsn_env)
        if not dsn:
            raise ValueError("pgvector vector store requires a DSN")
        vector_store = PgVectorStore(
            dsn=dsn,
            dimension=embedder.dimension,
            allow_dimension_change=config.vector_store.allow_dimension_change,
        )
    else:
        raise NotImplementedError(f"Vector store {config.vector_store.provider!r} not available")

    if config.llm.provider == "anthropic":
        llm: LLMProvider = AnthropicLLM(
            api_key=_env(config.llm.api_key_env),
            model=config.llm.model,
            default_max_tokens=config.llm.max_tokens,
            # Bound provider tail latency: thread the configured timeout/retry
            # so no path keeps a bare client (Anthropic SDK takes timeout +
            # max_retries directly).
            timeout=config.llm.request_timeout,
            max_retries=config.llm.max_retries,
        )
    elif config.llm.provider == "openai":
        from opsrag.llms.openai import OpenAILLM
        llm = OpenAILLM(
            api_key=_env(config.llm.api_key_env),
            model=config.llm.model,
            default_max_tokens=config.llm.max_tokens,
        )
    elif config.llm.provider == "vertex":
        from opsrag.llms.vertex import VertexAILLM
        # OPSRAG_LLM_MODEL env var lets operators A/B-swap models without
        # editing the config file. Region resolution is config-driven:
        # set `llm.location` to a region that supports the selected model
        # family. When left unset we fall back to a sensible default
        # ("us-central1"); deployments whose model family is only served
        # in another region MUST set `llm.location` explicitly.
        model = _env("OPSRAG_LLM_MODEL") or config.llm.model
        location = config.llm.location or "us-central1"
        llm = VertexAILLM(
            model=model,
            project=config.llm.project,
            location=location,
            default_max_tokens=config.llm.max_tokens,
        )
    elif config.llm.provider == "bedrock":
        from opsrag.llms.bedrock import BedrockLLM
        llm = BedrockLLM(
            model=config.llm.model or "anthropic.claude-sonnet-4-20250514-v1:0",
            region=config.llm.aws_region,
            profile=config.llm.aws_profile,
            default_max_tokens=config.llm.max_tokens,
            # Bound provider tail latency: Bedrock applies these via a botocore
            # Config (read/connect timeouts + adaptive retries).
            request_timeout=config.llm.request_timeout,
            connect_timeout=config.llm.connect_timeout,
            max_retries=config.llm.max_retries,
        )
    elif config.llm.provider == "litellm":
        from opsrag.llms.litellm_provider import LiteLLMLLM
        llm = LiteLLMLLM(
            model=config.llm.model,
            default_max_tokens=config.llm.max_tokens,
            api_base=config.llm.api_base,
            api_key_env=config.llm.api_key_env,
        )
    else:
        raise NotImplementedError(f"LLM {config.llm.provider!r} not available")

    # Per-purpose model router. default_llm=llm so the no-bundle path reuses the
    # already-built client (prompt-cache + back-compat invariant). Built here
    # (before the entity extractor) so the cheap `extract` lane can use it.
    purpose_router = PurposeRouter(config, default_llm=llm)

    # Vision fallback LLM (only when the active model can't see). Built once at
    # startup so per-turn vision routing costs no client setup.
    vision_llm: LLMProvider | None = None
    _vision_target = resolve_vision_model(config.vision, config.llm)
    if _vision_target is not None:
        v_provider, v_model = _vision_target
        if v_provider == "anthropic":
            vision_llm = AnthropicLLM(
                api_key=_env(config.llm.api_key_env),
                model=v_model,
                default_max_tokens=config.llm.max_tokens,
                timeout=config.llm.request_timeout,
                max_retries=config.llm.max_retries,
            )
        elif v_provider == "bedrock":
            from opsrag.llms.bedrock import BedrockLLM
            vision_llm = BedrockLLM(
                model=v_model,
                region=config.llm.aws_region,
                profile=config.llm.aws_profile,
                default_max_tokens=config.llm.max_tokens,
                request_timeout=config.llm.request_timeout,
                connect_timeout=config.llm.connect_timeout,
                max_retries=config.llm.max_retries,
            )
        elif v_provider == "vertex":
            from opsrag.llms.vertex import VertexAILLM
            vision_llm = VertexAILLM(
                model=v_model,
                project=config.llm.project,
                location=config.llm.location or "us-central1",
                default_max_tokens=config.llm.max_tokens,
            )
        elif v_provider == "openai":
            from opsrag.llms.openai import OpenAILLM
            vision_llm = OpenAILLM(
                api_key=_env(config.llm.api_key_env),
                model=v_model,
                default_max_tokens=config.llm.max_tokens,
            )
        elif v_provider == "litellm":
            from opsrag.llms.litellm_provider import LiteLLMLLM
            vision_llm = LiteLLMLLM(
                model=v_model,
                default_max_tokens=config.llm.max_tokens,
                api_base=config.llm.api_base,
                api_key_env=config.llm.api_key_env,
            )

    # Knowledge-graph store. Provider-selected via
    # `config.knowledge_graph.provider` (FR-019). The null backend is the
    # default so a minimal deployment ships with no graph database running.
    graph_store: KnowledgeGraphStore
    if config.knowledge_graph.provider == "none":
        graph_store = NullGraphStore()
    elif config.knowledge_graph.provider == "neo4j":
        try:
            from opsrag.graphstores.neo4j import Neo4jGraphStore
        except ImportError as exc:
            raise NotImplementedError(
                "neo4j graph store selected but its dependencies are not "
                "installed; install the neo4j driver or set "
                "knowledge_graph.provider to 'none'"
            ) from exc
        graph_store = Neo4jGraphStore(
            url=config.knowledge_graph.url,
            username=config.knowledge_graph.username,
            password=_env(config.knowledge_graph.password_env) or "",
            database=config.knowledge_graph.database,
        )
    else:
        raise NotImplementedError(
            f"Graph store {config.knowledge_graph.provider!r} not available"
        )

    # Entity extractor for the graph lane. Built whenever extraction is
    # enabled (method != none, default "hybrid"); the ingestion pipeline only
    # upserts to the graph when graph_store is non-null (NullGraphStore makes
    # the lane a no-op). The LLM prose lane runs on the CHEAP `extract` purpose
    # (falls back to the default client when no cloud bundle sets it), not the
    # expensive reasoner -- extraction is a high-volume, low-reasoning task, so
    # this was thousands of full-model calls per large repo before.
    entity_extractor: EntityExtractor | None = None
    _ = LLMEntityExtractor, RuleBasedExtractor  # available for explicit opt-in
    if config.entity_extraction.method != "none":
        _extract_llm = (
            purpose_router.pick("extract")
            if config.entity_extraction.method in ("hybrid", "llm")
            else None
        )
        entity_extractor = HybridExtractor(
            llm=_extract_llm,
            method=config.entity_extraction.method,
        )

    if config.reranker.provider == "fastembed":
        # The default. If the `fastembed` extra isn't installed, warn loudly and
        # fall back to no-op rather than crash boot -- the operator didn't
        # explicitly pick a cloud reranker, so a hard failure here would be
        # surprising for a minimal install.
        try:
            from opsrag.rerankers.fastembed_reranker import FastEmbedReranker
            reranker: Reranker = FastEmbedReranker(model=config.reranker.model)
        except Exception as exc:
            import logging
            logging.getLogger("opsrag.factory").warning(
                "fastembed reranker unavailable (%s) -- falling back to no-op. "
                "Install the `fastembed` extra to enable local reranking.", exc,
            )
            reranker = NoOpReranker()
    elif config.reranker.provider == "cohere":
        api_key = _env(config.reranker.api_key_env)
        if not api_key:
            # Fail fast: an EXPLICIT cohere selection with no key is a
            # misconfiguration, not a reason to silently degrade to no-op (the
            # exact silent-failure class that burned us on Neo4j/APOC).
            raise ValueError(
                f"reranker.provider=cohere but {config.reranker.api_key_env} is "
                "unset. Set the API key or choose a different reranker.provider."
            )
        reranker = CohereReranker(api_key=api_key, model=config.reranker.model)
    elif config.reranker.provider == "bedrock":
        from opsrag.rerankers.bedrock import BedrockReranker
        # Cohere Rerank 3.5 (or amazon.rerank) hosted ON Bedrock -- same AWS
        # creds as the LLM/embedder, no separate COHERE_API_KEY.
        reranker = BedrockReranker(
            model=config.reranker.model or "cohere.rerank-v3-5:0",
            region=config.reranker.aws_region,
            profile=config.reranker.aws_profile,
        )
    elif config.reranker.provider == "vertex":
        from opsrag.rerankers.vertex import VertexReranker
        project = config.reranker.project or _env("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise ValueError(
                "Vertex reranker requires reranker.project or GOOGLE_CLOUD_PROJECT env"
            )
        reranker = VertexReranker(
            project=project,
            model=config.reranker.model or "semantic-ranker-default-004",
            location=config.reranker.location,
        )
    else:
        reranker = NoOpReranker()

    if config.session.provider == "postgres":
        dsn = config.session.dsn or _env(config.session.dsn_env)
        if not dsn:
            raise ValueError("Postgres session store requires a DSN")
        session_store: SessionStore = PostgresSessionStore(dsn=dsn)
    else:
        session_store = InMemorySessionStore()

    # Step 7: indexed_files dedup tracker. Reuses POSTGRES_DSN when sessions
    # are on postgres; falls back to no-op so on-disk state doesn't drift
    # from in-memory sessions when both are misconfigured together.
    indexed_files: IndexedFilesTracker
    if config.session.provider == "postgres":
        dedup_dsn = config.session.dsn or _env(config.session.dsn_env)
        indexed_files = (
            PostgresIndexedFilesTracker(dsn=dedup_dsn)
            if dedup_dsn else NoopIndexedFilesTracker()
        )
    else:
        indexed_files = NoopIndexedFilesTracker()

    # Memory store (optional)
    mem_store: MemoryStore | None = None
    if config.memory.provider == "postgres":
        dsn = config.memory.dsn or _env(config.memory.dsn_env)
        if dsn:
            mem_store = PostgresMemoryStore(dsn=dsn)
    elif config.memory.provider == "memory":
        mem_store = InMemoryMemoryStore()
    elif config.memory.provider == "mem0":
        # Mem0 operational memory: reuses the main Qdrant (distinct
        # collection) + the project LLM/embedder; no separate API key,
        # graph layer off. Per-service, best-effort (never raises into
        # the agent). The QA cache is unaffected.
        from opsrag.memory.mem0_store import build_mem0_store
        mem_store = build_mem0_store(
            config,
            config.vector_store.url,
            config.llm,
            config.embedding,
        )

    if config.observability.provider == "phoenix":
        observability: ObservabilityProvider = PhoenixObservability(
            endpoint=config.observability.endpoint,
            api_key=_env("PHOENIX_API_KEY"),
        )
    elif config.observability.provider == "console":
        observability = ConsoleObservability()
    else:
        raise NotImplementedError(
            f"Observability {config.observability.provider!r} not available"
        )
    observability.setup(config.observability.project_name)

    # Phase 2 - non-git sources. Each is opt-in via its config block.
    sources: dict = {}
    if config.confluence.enabled:
        from opsrag.sources.confluence import ConfluenceClient, ConfluenceSource
        try:
            confluence_client = ConfluenceClient.from_config(config.confluence)
            sources["confluence"] = ConfluenceSource(
                client=confluence_client,
                spaces_allowlist=config.confluence.spaces_allowlist,
                spaces_denylist=config.confluence.spaces_denylist,
                label_denylist=config.confluence.label_denylist,
            )
        except RuntimeError as exc:
            # Config enabled but creds missing - log and skip rather
            # than crash the whole app on startup.
            import logging
            logging.getLogger("opsrag.factory").warning(
                "confluence enabled in config but disabled at runtime: %s", exc,
            )

    if config.rootly.enabled:
        from opsrag.sources.rootly.client import RootlyClient
        from opsrag.sources.rootly.source import RootlySource
        token = config.rootly.api_token or _env(config.rootly.api_token_env)
        if not token:
            import logging
            logging.getLogger("opsrag.factory").warning(
                "rootly enabled in config but %s is empty - skipping",
                config.rootly.api_token_env,
            )
        else:
            try:
                rootly_client = RootlyClient(
                    api_token=token,
                    base_url=config.rootly.base_url,
                    max_retries=config.rootly.max_retries,
                    retry_base_seconds=config.rootly.retry_base_seconds,
                )
                sources["rootly"] = RootlySource(
                    client=rootly_client,
                    statuses=tuple(config.rootly.statuses),
                    post_mortem_statuses=tuple(config.rootly.post_mortem_statuses),
                    skip_private=config.rootly.skip_private,
                )
            except (ValueError, RuntimeError) as exc:
                import logging
                logging.getLogger("opsrag.factory").warning(
                    "rootly enabled in config but disabled at runtime: %s", exc,
                )

    if config.slack.enabled:
        from opsrag.sources.slack.client import SlackClient
        from opsrag.sources.slack.source import SlackSource
        token = config.slack.bot_token or _env(config.slack.bot_token_env)
        if not token:
            import logging
            logging.getLogger("opsrag.factory").warning(
                "slack enabled in config but %s is empty - skipping",
                config.slack.bot_token_env,
            )
        elif not config.slack.channels_allowlist:
            import logging
            logging.getLogger("opsrag.factory").warning(
                "slack enabled but channels_allowlist is empty - skipping",
            )
        else:
            try:
                slack_client = SlackClient(
                    bot_token=token,
                    max_retries=config.slack.max_retries,
                    retry_base_seconds=config.slack.retry_base_seconds,
                )
                sources["slack"] = SlackSource(
                    client=slack_client,
                    channels_allowlist=config.slack.channels_allowlist,
                    backfill_days=config.slack.backfill_days,
                    min_replies_per_thread=config.slack.min_replies_per_thread,
                    skip_bot_messages=config.slack.skip_bot_messages,
                )
            except (ValueError, RuntimeError) as exc:
                import logging
                logging.getLogger("opsrag.factory").warning(
                    "slack enabled in config but disabled at runtime: %s", exc,
                )

    # Optional code-specific embedder + collection. Built only when BOTH
    # `code_embedding` and `code_vector_store` config blocks are set. Only
    # the Vertex/Qdrant pair is wired here; extending to other providers
    # is mechanical.
    code_embedder_provider: EmbeddingProvider | None = None
    code_vector_store_provider: VectorStore | None = None
    if config.code_embedding is not None and config.code_vector_store is not None:
        ce_cfg = config.code_embedding
        cv_cfg = config.code_vector_store
        if ce_cfg.provider != "vertex":
            raise NotImplementedError(
                f"code embedder currently only supports 'vertex' provider, "
                f"got {ce_cfg.provider!r}"
            )
        from opsrag.embedders.vertex import VertexAIEmbeddings
        code_embedder_provider = VertexAIEmbeddings(
            model=ce_cfg.model,
            project=ce_cfg.project,
            location=ce_cfg.location or "us-central1",
            # Code retrieval task types: documents use RETRIEVAL_DOCUMENT
            # (same as prose), queries use the code-specific task type.
            # Overridable for non-default Vertex embed models.
            document_task_type=ce_cfg.code_document_task_type,
            query_task_type=ce_cfg.code_query_task_type,
            # Pass the configured dimension THROUGH (no `or 768` fallback): a
            # None here lets VertexAIEmbeddings' unknown-model guard raise
            # instead of silently baking a wrong-dim code collection. A known
            # model resolves its own dimension; an unknown model + unset
            # code_embedding.dimension is a hard error, by design.
            output_dimensionality=ce_cfg.dimension,
        )
        code_embedder_provider = CachedEmbedder(
            code_embedder_provider,
            max_size=ce_cfg.cache_max_size,
            ttl_seconds=ce_cfg.cache_ttl_seconds,
        )
        if cv_cfg.provider != "qdrant":
            raise NotImplementedError(
                f"code vector store currently only supports 'qdrant', got {cv_cfg.provider!r}"
            )
        code_vector_store_provider = QdrantVectorStore(
            url=cv_cfg.url,
            api_key=_env(cv_cfg.api_key_env) if cv_cfg.api_key_env else None,
            collection_name=cv_cfg.collection,
            dimension=code_embedder_provider.dimension,
            allow_dimension_change=cv_cfg.allow_dimension_change,
        )

    # (PurposeRouter is built earlier, before the entity extractor.)

    # Lightweight entity-graph (Postgres edges) for the entity-expansion
    # retrieval lane. Constructed (not opened -- factory is sync; the server
    # lifespan opens it). Independent of the Neo4j graph_store above.
    light_graph = None
    if config.light_graph.enabled:
        from opsrag.light_graph import LightGraphStore
        lg_dsn = config.light_graph.dsn or _env(config.light_graph.dsn_env)
        if lg_dsn:
            light_graph = LightGraphStore(dsn=lg_dsn)

    # Durable indexing job-state. Reuses the session Postgres DSN (same as the
    # indexed_files dedup tracker). The lifespan opens it + drives the flush
    # loop on writer roles; backend pods read it. proc_token makes this
    # writer's run rows unique so concurrent Jobs never collide.
    index_store = None
    agent_settings = None
    if config.session.provider == "postgres":
        idx_dsn = config.session.dsn or _env(config.session.dsn_env)
        if idx_dsn:
            import uuid

            from opsrag.indexing import PostgresIndexStore
            index_store = PostgresIndexStore(dsn=idx_dsn, proc_token=uuid.uuid4().hex[:12])
            from opsrag.agent_settings import AgentSettingsStore
            agent_settings = AgentSettingsStore(dsn=idx_dsn)

    return Providers(
        scm=scm,
        parsers=parsers,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
        reranker=reranker,
        session_store=session_store,
        observability=observability,
        graph_store=graph_store,
        entity_extractor=entity_extractor,
        memory_store=mem_store,
        purpose_router=purpose_router,
        indexed_files=indexed_files,
        sources=sources,
        code_embedder=code_embedder_provider,
        code_vector_store=code_vector_store_provider,
        light_graph=light_graph,
        index_store=index_store,
        agent_settings=agent_settings,
        vision_llm=vision_llm,
    )
