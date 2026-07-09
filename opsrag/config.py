"""Pydantic v2 configuration models loaded from YAML / env vars.

The root model is ``Settings``. ``OpsRAGConfig`` is preserved as a
backward-compatible alias for existing imports; new code should import
``Settings`` directly.

The schema mirrors the contract at
``specs/001-port-opsrag-opensource/contracts/config-schema.md``; the
``deployment`` field is the operator-supplied ``DeploymentContext`` per
``specs/001-port-opsrag-opensource/contracts/deployment-context.md``.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from opsrag.channels.config import ChannelsConfig, SlackChannelConfig
from opsrag.config_mcp import (
    KNOWN_MCP_NAMES,
    MCP_CONFIG_TYPES,
    MCPConfigBlock,
)
from opsrag.config_mcp import (
    default_mcp_map as _default_mcp_map,
)
from opsrag.context import DeploymentContext
from opsrag.slack_bot.config import SlackBotConfig


class RepoEntry(BaseModel):
    """Per-repo override. Either a bare string or an object with a branch."""
    name: str
    branch: str | None = None


class SCMConfig(BaseModel):
    provider: Literal["gitlab", "github", "gitea", "local"] = "gitlab"
    base_url: str = "https://gitlab.com"
    token_env: str = "GITLAB_TOKEN"
    # Each entry is either "owner/repo" (uses default_branch) or
    # {name: "owner/repo", branch: "main"} for per-repo branch overrides.
    repos: list[str | RepoEntry] = Field(default_factory=list)
    default_branch: str = "main"
    auto_index: bool = True
    clone_mode: bool = True           # True = git clone (fast), False = API per file (slow)
    use_ssh: bool = False             # True = clone via git@host:repo.git (bypasses HTTPS proxies)
    ssh_host: str | None = None       # Override host for SSH URL; falls back to base_url host
    ssh_user: str = "git"

    def repos_with_branch(self) -> list[tuple[str, str]]:
        """Return [(repo_path, branch)] tuples, applying per-entry overrides."""
        out: list[tuple[str, str]] = []
        for r in self.repos:
            if isinstance(r, str):
                out.append((r, self.default_branch))
            else:
                out.append((r.name, r.branch or self.default_branch))
        return out

    def repo_names(self) -> list[str]:
        return [name for name, _ in self.repos_with_branch()]


class ChunkingConfig(BaseModel):
    # Only the strategies the chunker factory can actually build (see
    # opsrag/factory.py). "semantic" was accepted here but unimplemented,
    # so a bad config deferred to a runtime NotImplementedError -- now it
    # fails fast at config-load with a clear pydantic ValidationError.
    strategy: Literal["fixed_size", "parent_child"] = "parent_child"
    chunk_size: int = 512
    overlap: int = 64
    child_size: int = 256
    child_overlap: int = 32
    parent_max_tokens: int = 1024
    # Code parents get a larger budget so whole functions/classes fit in one
    # parent (matches ParentChildChunker's default). Tunable without a rebuild.
    code_parent_max_tokens: int = 2048


class EmbeddingConfig(BaseModel):
    # Only the providers the embedder factory can actually build (see
    # opsrag/factory.py). "cohere"/"ollama" were accepted here but
    # unimplemented, so a bad config deferred to a runtime
    # NotImplementedError -- now it fails fast at config-load with a clear
    # pydantic ValidationError (matches ChunkingConfig.strategy above).
    provider: Literal["openai", "vertex", "bedrock", "fastembed", "litellm"] = "openai"
    model: str = "text-embedding-3-large"
    dimension: int | None = None
    api_key_env: str = "OPENAI_API_KEY"
    aws_region: str | None = None
    aws_profile: str | None = None
    project: str | None = None        # Vertex AI: GCP project id
    location: str | None = None       # Vertex AI: region (default us-central1)
    # LiteLLM: optional base URL for self-hosted / OpenAI-compatible
    # endpoints (e.g. a Qwen TEI server). Ignored by the other providers.
    api_base: str | None = None
    # Read-through embedding cache (CachedEmbedder wraps the provider): the
    # same query is re-embedded 3-5x per turn. Defaults match the previously
    # hardcoded values; raise cache_max_size for high query cardinality.
    cache_max_size: int = 1000
    cache_ttl_seconds: float = 60.0
    # Vertex code-retrieval task types -- only consulted when this block is the
    # `code_embedding` lane (ignored by the main embedder / non-Vertex providers).
    code_document_task_type: str = "RETRIEVAL_DOCUMENT"
    code_query_task_type: str = "CODE_RETRIEVAL_QUERY"


class VectorStoreConfig(BaseModel):
    # Only the providers the factory can build. "weaviate"/"chroma" were
    # accepted here but unimplemented, deferring to a runtime
    # NotImplementedError -- now they fail fast at config-load (matches
    # EmbeddingConfig.provider above).
    provider: Literal["qdrant", "pgvector"] = "qdrant"
    url: str = "http://localhost:6333"
    collection: str = "opsrag"
    api_key_env: str | None = None
    dsn: str | None = None
    dsn_env: str = "PGVECTOR_DSN"
    # Vector distance metric. cosine default keeps all current deployments
    # byte-identical; wired into Qdrant + pgvector.
    distance: Literal["cosine", "dot", "euclid"] = "cosine"
    # Fail-closed embedding-dimension guard (shared seam across the main
    # index, the QA cache, and investigations). When False (default), the
    # factory refuses to start if the embedder's dimension differs from an
    # existing collection's dimension (a silent mismatch corrupts retrieval).
    # Set True only for an intentional reindex after an embed-model switch.
    allow_dimension_change: bool = False


class LLMConfig(BaseModel):
    # Only the providers the LLM factory can build. "ollama" was accepted
    # here but unimplemented natively (run an Ollama model through
    # provider="litellm" instead) -- it deferred to a runtime
    # NotImplementedError, so it now fails fast at config-load.
    provider: Literal["anthropic", "openai", "vertex", "bedrock", "litellm"] = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key_env: str = "ANTHROPIC_API_KEY"
    aws_region: str | None = None
    aws_profile: str | None = None
    max_tokens: int = 4096
    project: str | None = None        # Vertex AI: GCP project id
    location: str | None = None       # Vertex AI: region (default us-central1)
    # LiteLLM: optional base URL for self-hosted / OpenAI-compatible
    # endpoints (e.g. a vLLM/TGI Qwen server). Ignored by other providers.
    api_base: str | None = None
    # Bound provider client tail latency so a hung/slow upstream can't stall
    # a turn indefinitely. Anthropic uses max_retries directly; Bedrock applies
    # these via a botocore Config in adaptive mode.
    # 120s is generous enough never to cut a valid generation -- the generator
    # already self-budgets ~45s -- while still bounding upstream hangs/retry-storms
    # vs the SDK 600s default.
    request_timeout: float = 120.0
    connect_timeout: float = 10.0
    max_retries: int = 2
    # Per-model cost-telemetry price overrides (USD). Key = model id, matched
    # exact-then-suffix (e.g. "gemini-3.1-pro-preview" also matches the litellm
    # "vertex_ai/gemini-3.1-pro-preview" spelling). Value is either token rates
    # {"input_per_1m": 0.5, "output_per_1m": 3.0} or a per-call rate
    # {"per_call": 0.001} for rerankers. Lets you price preview / MaaS / custom
    # models without code changes; overrides win over the built-in pricing table.
    model_prices: dict[str, dict[str, float]] = Field(default_factory=dict)


class VisionConfig(BaseModel):
    """Image/vision behaviour. All overridable via env/YAML (no rebuild).

    `model`/`provider` are the auto-route fallback used ONLY when an image
    arrives and the active model isn't vision-capable; left None, a
    provider-aware default is resolved at factory time
    (opsrag.llms.content.default_vision_model). Bytes are ephemeral -- never
    persisted (spec FR-003).
    """

    enabled: bool = True
    model: str | None = None
    provider: Literal["anthropic", "openai", "vertex", "bedrock", "litellm"] | None = None
    max_images: int = 4
    max_bytes: int = 5 * 1024 * 1024
    allowed_mime: list[str] = Field(
        default_factory=lambda: ["image/png", "image/jpeg", "image/gif", "image/webp"]
    )


class ObservabilityConfig(BaseModel):
    # Only the providers the factory can build. "datadog" was accepted here
    # but unimplemented, deferring to a runtime NotImplementedError -- now it
    # fails fast at config-load.
    provider: Literal["console", "phoenix"] = "console"
    project_name: str = "opsrag"
    endpoint: str | None = None


class GraphStoreConfig(BaseModel):
    provider: Literal["neo4j", "networkx", "none"] = "none"
    url: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password_env: str = "NEO4J_PASSWORD"
    database: str = "neo4j"
    # When True (and provider != none), the agent augments vector retrieval
    # with a graph-traversal lane. Default False: the graph is populated at
    # ingestion but treated as low-trust soft-boost, opt-in for retrieval.
    use_in_retrieval: bool = False


class LightGraphConfig(BaseModel):
    """Lightweight entity-graph for the entity-expansion retrieval lane
    (Postgres edges table, NOT Neo4j). Industry-recommended way to get
    multi-hop WITHOUT a graph engine: write deterministic entity ids onto each
    chunk's Qdrant payload at index time + keep a tiny adjacency table; the
    retriever does a 1-hop lookup AFTER vector search and pulls related chunks
    via a metadata filter. NEVER the main line -- fail-safe (empty graph -> no
    extra chunks). Works with knowledge_graph.provider=none."""
    enabled: bool = False
    dsn_env: str = "POSTGRES_DSN"     # reuse the main Postgres
    dsn: str | None = None
    seed_chunks: int = 6              # top vector chunks to seed entity ids from
    max_neighbors: int = 40           # cap on 1-hop neighbor entities
    expand_top_k: int = 4             # extra chunks pulled in per query


class EntityExtractionConfig(BaseModel):
    method: Literal["llm", "rule_based", "hybrid", "none"] = "hybrid"


class ModelSpec(BaseModel):
    """A per-purpose model override. Any unset field falls back to the
    cloud-bundle default, then to the classic provider block."""
    provider: str | None = None
    model: str | None = None
    effort: str | None = None        # e.g. reasoning effort / thinking budget


class ModelsConfig(BaseModel):
    """Per-PURPOSE model routing (reason / tool_call / embed / rerank /
    summarize). Resolved against the `cloud_provider` bundle in
    Settings.load (explicit values here always win over the bundle).
    All-None = behavior identical to today (classic provider blocks)."""
    reason: ModelSpec | None = None
    tool_call: ModelSpec | None = None
    embed: ModelSpec | None = None
    rerank: ModelSpec | None = None
    summarize: ModelSpec | None = None
    extract: ModelSpec | None = None


class APIConfig(BaseModel):
    api_keys: list[str] = Field(default_factory=list)
    api_keys_env: str = "OPSRAG_API_KEYS"
    rate_limit_rpm: int = 60
    rate_limit_enabled: bool = True
    # Where rate-limit state lives. "memory" (default) keeps the current
    # in-process behavior -- correct for a single replica but loses state
    # across replicas. "redis" shares state across replicas via a Redis
    # backend (the `redis` extra), in which case Redis is REQUIRED: the API
    # fails fast at startup if it can't reach the server named by
    # `redis_url_env`.
    rate_limit_backend: Literal["memory", "redis"] = "memory"
    redis_url_env: str = "OPSRAG_REDIS_URL"


class AgentConfig(BaseModel):
    # Default to "full" -- the value the bundled config.yaml ships -- so the
    # schema default matches the shipped deploy and an env/YAML-less load picks
    # the same path. "hybrid" stays in the Literal for back-compat (server.py
    # warns and maps hybrid->full).
    mode: Literal["minimal", "full", "hybrid", "tool_calling", "multi_agent"] = "full"
    top_k: int = 10
    rerank_top_k: int = 5
    max_retries: int = 3
    # Post-rerank MMR diversity penalty, the (1 - lambda) weight in [0, 1].
    # enabled by default (MMR diversity); set 0.0 to disable. >0 re-orders the
    # kept top_k via Maximal Marginal Relevance so near-duplicate config
    # variants don't crowd out distinct docs. ~0.3 is a gentle nudge; 1.0 is
    # pure diversity. See opsrag/rerankers/mmr.py.
    rerank_diversity: float = 0.3
    # Content dedup BEFORE the rerank call -- a cheap pre-filter distinct from
    # the post-rerank MMR diversity above.
    rerank_content_dedup: bool = True   # drop exact-duplicate-content chunks BEFORE the rerank call (cost win, safe)
    rerank_content_dedup_threshold: float = 0.0   # near-duplicate Jaccard merge before rerank; 0.0 = off (can drop distinct env-specific configs)
    # Run the shared, fail-closed groundedness gate on the default multi_agent
    # path (the hybrid/minimal/full paths already gate; this extends it to the
    # default).
    verify_grounding_default: bool = True   # run the shared, fail-closed groundedness gate on the default multi_agent path (adds ~1 LLM call/turn; set false to trade safety for latency/cost)
    # R4 -- run the precise artifact/citation verify_answer check on the default
    # multi_agent path (distinct from the groundedness gate above: this asserts
    # cited artifacts/citations actually back the answer). Adds +1 LLM
    # verification call/turn; set false to disable.
    verify_artifacts_default: bool = True
    # Phase 03 Pillar 3 -- Flash/Pro hybrid routing.
    # When set (e.g. "gemini-2.5-pro"), complex queries (root-cause /
    # cross-source / multi-step) escalate to Pro for synthesis;
    # everything else stays on the configured `llm` (Flash). When
    # None, all queries use Flash and escalation patterns are still
    # logged for telemetry.
    pro_model: str | None = None


class RerankerConfig(BaseModel):
    # Default to the local cross-encoder (FastEmbed): reranking is the single
    # highest-ROI retrieval lever, it runs offline with no API key, and shipping
    # `noop` meant the default deployment fed raw bi-encoder order to the LLM.
    provider: Literal["fastembed", "cohere", "bedrock", "vertex", "noop"] = "fastembed"
    model: str = "rerank-v3.5"
    api_key_env: str = "COHERE_API_KEY"
    # Bedrock rerank (Cohere Rerank 3.5 hosted on Bedrock) -- no COHERE_API_KEY.
    aws_region: str | None = None
    aws_profile: str | None = None
    project: str | None = None        # Vertex Discovery Engine: GCP project id
    location: str = "global"          # Vertex ranking config location (typically "global")


class SessionConfig(BaseModel):
    provider: Literal["postgres", "memory"] = "memory"
    dsn_env: str = "POSTGRES_DSN"
    dsn: str | None = None


class MemoryConfig(BaseModel):
    provider: Literal["postgres", "memory", "none", "mem0"] = "memory"
    dsn_env: str = "POSTGRES_DSN"
    dsn: str | None = None
    # Mem0 operational memory (per-service). Only consulted when
    # provider == "mem0". Reuses the main Qdrant client + the project's
    # configured LLM/embedder -- no separate API key, no second client path.
    # `infer` routes fact-distillation through the cheap summarize model;
    # set False to store raw turns. Graph layer stays OFF.
    mem0_collection: str = "opsrag_mem0_ops"
    mem0_infer: bool = True
    # Mem0 embedder override. The main `embedding` model is tuned for code
    # retrieval (e.g. Cohere Embed v4 on Bedrock) -- but mem0's built-in
    # provider embedders don't all speak every model (mem0's aws_bedrock
    # embedder only sends Titan-style payloads, so Cohere v4 -> "Malformed
    # request"). Memory facts are short natural language, so a simpler embedder
    # is fine. When set, mem0 uses these instead of the main `embedding`.
    # Leave unset to reuse the main embedder (works for OpenAI / Vertex).
    mem0_embed_provider: str | None = None      # e.g. "bedrock"
    mem0_embed_model: str | None = None          # e.g. "amazon.titan-embed-text-v2:0"
    mem0_embed_dimension: int | None = None      # must match the model (Titan v2 = 1024)


class ConfluenceConfig(BaseModel):
    """Phase 2 -- Confluence connector.

    Disabled by default; flip `enabled` to true after creating an Atlassian
    service-account API token and exporting it as `CONFLUENCE_API_TOKEN`.
    """

    enabled: bool = False
    # Per Constitution Principle VI, the engine carries no example URLs.
    # Operators set this to their own Atlassian site (e.g. "https://
    # myorg.atlassian.net") or, preferably, leave it unset and surface
    # the URL via ``DeploymentContext.source_urls.confluence``.
    base_url: str | None = None
    email_env: str = "CONFLUENCE_EMAIL"
    email: str | None = None
    api_token_env: str = "CONFLUENCE_API_TOKEN"
    api_token: str | None = None
    # Hard allowlist -- never crawl personal "~username" spaces or unlisted
    # spaces. The Atlassian API can return more than what we want.
    spaces_allowlist: list[str] = Field(default_factory=list)
    spaces_denylist: list[str] = Field(default_factory=list)
    # Pages carrying any of these labels are skipped (HR / personal /
    # salary / etc.). Use lowercase; matched case-insensitively.
    label_denylist: list[str] = Field(default_factory=list)
    # Concurrent page fetches. Atlassian Cloud rate-limits at ~5000
    # req/hr/IP; 5 is comfortably under the ceiling for a 500-page
    # space without contention.
    fetch_concurrency: int = 5
    # Backoff policy for 429s.
    max_retries: int = 3
    retry_base_seconds: float = 2.0


class SlackConfig(BaseModel):
    """Phase 2 -- Slack channel archive connector.

    Disabled by default; flip `enabled` to true after creating a Slack
    bot user, granting it `channels:history` + `channels:read` +
    `users:read` scopes, and exporting the bot token as
    `SLACK_BOT_TOKEN`. Bot must be invited (`/invite @<bot>`) into
    each channel listed in `channels_allowlist` -- Slack returns
    `not_in_channel` otherwise for `conversations.history`.

    Indexing model: each thread becomes one Markdown document. Single-
    message threads with no replies can be filtered via
    `min_replies_per_thread` to drop noise.
    """

    enabled: bool = False
    bot_token_env: str = "SLACK_BOT_TOKEN"
    bot_token: str | None = None
    # Workspace web URL -- used to render channel/thread links in
    # answers. Format: `https://<workspace>.slack.com`. Per Constitution
    # Principle VI, no default; operators set this or leave it None.
    workspace_url: str | None = None
    # Hard allowlist of channel IDs (e.g. CC448TKTQ for #devops).
    # IDs not names -- channel names can be renamed; IDs are stable.
    channels_allowlist: list[str] = Field(default_factory=list)
    # Backfill window for the initial run. Daily delta scheduler picks
    # up new messages from the last successful run timestamp afterwards.
    backfill_days: int = 30
    # Threads with fewer replies are usually noise (single notifications,
    # one-off posts). Set to 0 to index every message.
    min_replies_per_thread: int = 1
    # Skip messages from bots (Datadog alerts, GitLab notifications,
    # etc.) -- they're config noise, not knowledge.
    skip_bot_messages: bool = True
    # Slack rate limits are tiered by method: conversations.history is
    # tier 3 (~50 req/min), so a low concurrency keeps us comfortably
    # under the ceiling.
    fetch_concurrency: int = 3
    max_retries: int = 3
    retry_base_seconds: float = 2.0


class RootlyConfig(BaseModel):
    """Phase 2 -- Rootly incidents + post-mortems connector.

    Disabled by default; flip `enabled` to true after creating a Rootly
    API token with `incidents:read` + `post_mortems:read` scopes plus
    the reference-data scopes (severities, causes, environments,
    incident_types, incident_roles) and exporting it as
    `ROOTLY_API_TOKEN`.

    Indexing model: one incident = one Markdown document. The matching
    post-mortem is appended at fetch time so retrieval surfaces both
    summary and root-cause analysis as a coherent unit.

    Alerts are NOT indexed -- config noise, not knowledge. They were
    explicitly dropped from RAG scope on 2026-05-06; live alert state
    is a Phase 4 MCP concern.
    """

    enabled: bool = False
    api_token_env: str = "ROOTLY_API_TOKEN"
    api_token: str | None = None
    base_url: str = "https://api.rootly.com/v1"
    # Web UI base -- distinct from API base. Used to render
    # incident links in answers; not used for fetching. Per Constitution
    # Principle VI, no default; operators set this or leave it None.
    web_base_url: str | None = None
    # Single-tenant per Rootly account, but the scheduler API still
    # wants a `scope` value -- use the workspace name as the canonical
    # identifier (free-form, only matters for tracker key).
    scope: str = "acme"
    # Statuses to index. `resolved` + `mitigated` carry post-incident
    # knowledge; `cancelled` / `scheduled` / `planning` don't.
    statuses: list[str] = Field(default_factory=lambda: ["resolved", "mitigated"])
    # Post-mortem statuses. `published` only by default -- drafts are
    # work-in-progress and would noise up retrieval.
    post_mortem_statuses: list[str] = Field(default_factory=lambda: ["published"])
    # Skip incidents marked `private: true`. Rare but possible --
    # excluded by default since they're sensitive by intent.
    skip_private: bool = True
    max_retries: int = 3
    retry_base_seconds: float = 2.0


class SchedulerConfig(BaseModel):
    """Step 6 -- daily indexing scheduler (APScheduler).

    In-memory jobstore (single-replica). The cron schedule rebuilds from
    config on every container start, so persistence is unnecessary for
    this use case.
    """

    enabled: bool = False
    # Cron trigger: fires once per day at this local hour.
    timezone: str = "Asia/Ho_Chi_Minh"
    cron_hour: int = 2
    cron_minute: int = 0
    # +/-jitter in seconds applied by APScheduler to spread the start time
    # so we don't slam Vertex at exactly :00 every day. 900s = +/-15min.
    jitter_seconds: int = 900
    # How many repos to index in parallel during the daily run. Same cap
    # as the startup auto-index. Vertex per-minute token quotas are the
    # binding constraint.
    parallel_limit: int = 3


class InvestigationHistoryConfig(BaseModel):
    """Sub-sprint 5 phase-2 -- promote settled past investigations into the
    main RAG corpus as historical-reference docs. See
    `opsrag.sources.investigations` for the source-of-truth/snapshot
    distinction and the freshness-warning header format."""

    enabled: bool = False
    # Investigations younger than this are still settling -- exclude.
    min_age_days: int = 7
    # Older than this is a stale snapshot (kafka 3->5 brokers scenario)
    # -- exclude AND prune from corpus on each daily run.
    max_age_days: int = 90
    # Safety cap per run. With ~10 useful investigations/day, 500 covers
    # roughly 50 days of accumulation.
    max_docs_per_run: int = 500
    # Drop entries flagged thumbs-down entirely.
    skip_thumbs_down: bool = True


class BrandConfig(BaseModel):
    """White-label / multi-tenant branding. All fields can be overridden
    at runtime via env vars (see `OpsRAGConfig.load`) so the same image
    deploys for any tenant without rebuilding the UI."""

    name: str = "OpsRAG"
    subtitle: str = "DevOps Intelligence"
    assistant_name: str = "OpsRAG"
    # Absolute URL or path served by the UI host. Empty = no favicon link
    # injected (browser shows a generic page icon).
    favicon_url: str = ""
    # Optional accent color override for the UI primary action / brand
    # mark. Hex string (e.g. "#6384ff"). Empty = use UI default.
    accent_color: str = ""


class K8sClusterCoords(BaseModel):
    """GCP coordinates for a single GKE cluster. Used by the K8s MCP to
    authenticate via Workload Identity -> GCP Container API -> cluster
    endpoint + CA cert. Mirrors the airflow-infra `GKEStartPodOperator`
    pattern; no kubeconfig file required.
    """
    project: str
    location: str
    name: str


class K8sConfig(BaseModel):
    """DEPRECATED: prefer environments.targets.

    K8s MCP cluster registry (OPTIONAL GKE Workload-Identity provider).

    When non-empty, the K8s MCP uses ADC (Workload Identity in-pod, gcloud
    locally) + the GCP Container API to fetch cluster credentials for each
    registered cluster name. Empty (the default) selects the vendor-neutral
    path: the in-cluster ServiceAccount when running in a pod, otherwise a
    standard kubeconfig (``KUBECONFIG`` / ``~/.kube/config``) honouring its
    own auth (client certs, tokens, or exec plugins like ``aws eks
    get-token``). Multi-cluster in that mode is by kubeconfig context name.
    """
    default_cluster: str | None = None
    clusters: dict[str, K8sClusterCoords] = Field(default_factory=dict)


class ElasticsearchConfig(BaseModel):
    """DEPRECATED: prefer environments.targets.

    Elasticsearch / OpenSearch MCP -- direct read-only search.

    Reaches a single Elasticsearch (or OpenSearch) endpoint directly over
    HTTPS with an API key or basic auth. No K8s / port-forward coupling.
    The URL + credentials are resolved from env vars (so secrets never live
    in config); ``url`` may also be set inline for convenience.

    Read-only tools: list_indices, get_mappings, search, esql_query
    (Elasticsearch only), cluster_health.
    """
    enabled: bool = False
    # Endpoint base URL, e.g. ``https://es.example.com:9200``. If blank,
    # resolved from the ``url_env`` environment variable.
    url: str = ""
    url_env: str = "ES_URL"
    # Auth: prefer an API key (base64 ``id:api_key``). Falls back to basic
    # auth when the API-key env is empty. Credentials come from env only.
    api_key_env: str = "ES_API_KEY"
    username_env: str = "ES_USERNAME"
    password_env: str = "ES_PASSWORD"
    # ``elasticsearch`` or ``opensearch`` -- gates ES|QL (``_query``), which
    # OpenSearch does not implement.
    backend: str = "elasticsearch"
    default_index: str = "*"
    verify_ssl: bool = True


# --- Unified multi-environment registry (Approach A) -----------------
# One OpsRAG instance can target N environments; each environment bundles
# how to reach its kubernetes + prometheus + elasticsearch. Names in
# `environments.targets` are the canonical env list (the engine derives
# DeploymentContext.environments from them -- no hardcoded env enum).
# See docs/superpowers/specs/2026-06-09-multi-env-environments-registry-design.md
# Fields are intentionally permissive (most Optional); invalid combinations
# surface as STRUCTURED errors at resolve/use time, never silent defaults.


class K8sTarget(BaseModel):
    """How to reach one environment's Kubernetes API."""

    model_config = ConfigDict(extra="forbid")

    # gke       -> Workload Identity (ADC) + GCP Container API (project/location/name)
    # kubeconfig-> vendor-neutral: a KUBECONFIG context (EKS via aws-eks-get-token,
    #              client certs, in-cluster SA when context is None).
    mode: Literal["gke", "kubeconfig"] = "kubeconfig"
    # gke mode:
    project: str | None = None
    location: str | None = None
    name: str | None = None
    # kubeconfig mode (None -> current-context / in-cluster SA):
    context: str | None = None
    # shared:
    default_namespace: str | None = None
    pod_label_selector: str | None = None


class PrometheusTarget(BaseModel):
    """How to reach one environment's Prometheus."""

    model_config = ConfigDict(extra="forbid")

    # k8s_proxy -> through the cluster's API server service-proxy.
    # direct    -> a reachable Prometheus base URL.
    reach: Literal["k8s_proxy", "direct"] = "k8s_proxy"
    # k8s_proxy mode (generic defaults -- NOT monitoring-main-prometheus):
    namespace: str = "monitoring"
    service: str = "kube-prometheus-stack-prometheus"
    port: int = 9090
    # extra named services on the same cluster, e.g. {"istio": "..."}.
    extra_services: dict[str, str] = Field(default_factory=dict)
    # direct mode:
    url: str | None = None
    bearer_token_env: str | None = None


class EsTarget(BaseModel):
    """How to reach one environment's Elasticsearch / OpenSearch."""

    model_config = ConfigDict(extra="forbid")

    # direct       -> a reachable ES base URL (HTTPS).
    # port_forward -> tunnel a pod port over the k8s API (ECK-style; the API
    #                 proxy strips Authorization, so port-forward preserves
    #                 API-key auth). proxy -> k8s service-proxy.
    reach: Literal["direct", "port_forward", "proxy"] = "direct"
    # direct mode:
    url: str | None = None
    # in-cluster (port_forward / proxy):
    service: str | None = None
    namespace: str | None = None
    port: int = 9200
    pod_label_selector: str | None = None
    # auth (env-only; prefer API key):
    api_key_env: str | None = None
    username_env: str | None = None
    password_env: str | None = None
    # query shaping:
    index_pattern: str = "*"
    backend: Literal["elasticsearch", "opensearch"] = "elasticsearch"
    verify_ssl: bool = True
    # logical -> physical ES field mapping (de-hardcodes one org's schema,
    # e.g. {"timestamp": "@timestamp", "service": "kubernetes.labels.app_name"}).
    fields: dict[str, str] = Field(default_factory=dict)


class EnvironmentTarget(BaseModel):
    """One environment: how to reach its k8s + prometheus + elasticsearch.
    Any integration may be None (that integration is unavailable for the env)."""

    model_config = ConfigDict(extra="forbid")

    kubernetes: K8sTarget | None = None
    prometheus: PrometheusTarget | None = None
    elasticsearch: EsTarget | None = None


class EnvironmentsConfig(BaseModel):
    """Unified multi-environment registry. Empty `targets` -> the engine
    synthesizes a registry from the legacy `k8s` / `elasticsearch` /
    `deployment` blocks (backward compatible)."""

    model_config = ConfigDict(extra="forbid")

    default: str | None = None
    targets: dict[str, EnvironmentTarget] = Field(default_factory=dict)


class CodeCacheConfig(BaseModel):
    """Pre-warm settings for the MCP ``code_*`` tool clone cache.

    The backend pod's ``/tmp/opsrag-repos/`` is emptyDir -> wipes on
    every pod restart. Lazy-clone re-pays the 2-30s cost per repo on
    the FIRST code_* query. Pre-warming during lifespan startup
    eliminates that latency for users (background task -- doesn't block
    /health -> Ready).
    """

    prewarm_on_start: bool = True
    concurrency: int = 6           # max parallel clones (GitLab tolerates this)


class QACacheConfig(BaseModel):
    """Q&A semantic cache (``opsrag.qa_cache``) -- the separate Qdrant
    collection that serves cached corpus answers for near-duplicate queries.

    The cache is gated by an LLM "same question?" judge over a borderline
    cosine band ``[floor, qa_judge_upper]``: a score ``>= qa_judge_upper``
    auto-accepts (skip the judge, high confidence); below the band is a plain
    miss; in-band calls the cheap judge. The borderline floor stays a judge-
    module concern (``opsrag.qa_cache_judge``); this block surfaces the upper
    bound + the NER discriminator toggle so ``server.py`` can thread them.

    ``qa_ner_guard`` enables the spaCy free-form NER entity-swap discriminator
    by default. It graceful-degrades: when the spaCy model is absent the
    extractor logs once and returns no tokens (regex discriminators still run),
    so turning it on never hard-fails a deployment without the model installed.
    """

    qa_judge_upper: float = 0.99
    qa_ner_guard: bool = True
    # R14 -- behaviour when the borderline-band judge LLM itself errors. Default
    # fail-CLOSED (False): a judge error in-band re-queries the corpus rather
    # than serving a wrong-but-close cached answer (QUALITY > latency). Set True
    # to fail-open and prefer availability (serve the cached candidate when the
    # judge can't decide).
    qa_judge_fail_open: bool = False


class TrackingUserConfig(BaseModel):
    """M1 -- Pomerium identity + M2 -- per-user token attribution.

    When `enabled` is False the entire identity path is short-circuited:
    `extract_current_user` returns `CurrentUser.anonymous()` without ever
    touching headers / JWKS, and the `/api/me` endpoint reports
    `tracking_enabled: false` so the UI knows not to render a user chip.

    When enabled, Pomerium forwards the user's identity as a signed JWT
    in the `X-Pomerium-Jwt-Assertion` request header. We verify the
    signature against the JWKS served at `pomerium_jwks_url` (ES256 by
    default -- Pomerium's standard signing alg). `pomerium_audience` is
    the route's expected `aud` claim; leave None to skip audience check.

    `admin_group_oid` is the Azure AD group OID (UUID) that grants
    access to the admin-gated `/api/usage/by_user` endpoint. Membership
    is asserted by the JWT's `groups` claim.
    """

    enabled: bool = False
    require_auth: bool = False
    pomerium_jwks_url: str | None = None
    pomerium_audience: str | None = None
    admin_group_oid: str | None = None


class SessionConfigAuth(BaseModel):
    """Cookie-session signing + lifetimes for auth.mode='login'. The signing
    key is sourced from a path or env ONLY -- inline key material is refused
    at load time (opsrag.auth.sessions.load_signing_key)."""
    model_config = ConfigDict(extra="forbid")
    # Login methods (easy switching): password / SSO / both. Set
    # password_enabled=false for SSO-only; enable SSO via the `sso` block.
    # The /auth/providers endpoint reports both to the UI so the login page
    # shows exactly the available methods.
    password_enabled: bool = True
    signing_key_path: str | None = None
    signing_key_env: str | None = "OPSRAG_SESSION_SIGNING_KEY"
    # External base URL where the API is reachable by the BROWSER, including
    # any reverse-proxy prefix (e.g. "https://opsrag.example.com/api" or, for
    # the local demo behind the UI proxy, "http://localhost:5173/api"). Used
    # to build the SSO redirect_uri so it matches what you register with each
    # IdP. When None, falls back to request.url_for (only correct when the API
    # is hit directly, not behind a path-stripping proxy).
    sso_callback_base: str | None = None
    session_ttl_seconds: int = 900
    refresh_ttl_seconds: int = 14 * 24 * 3600
    cookie_secure: bool = True
    cookie_samesite: Literal["lax", "strict", "none"] = "lax"
    cookie_domain: str | None = None
    login_max_attempts: int = 5
    login_window_seconds: int = 300
    login_lockout_seconds: int = 900


class SSOProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    client_id: str | None = None
    client_secret_env: str | None = None   # secret via env, never inline
    scopes: list[str] = Field(default_factory=list)
    server_metadata_url: str | None = None  # single-tenant Entra override


class SSOConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    google: SSOProviderConfig = Field(default_factory=SSOProviderConfig)
    microsoft: SSOProviderConfig = Field(default_factory=SSOProviderConfig)
    github: SSOProviderConfig = Field(default_factory=SSOProviderConfig)


class AuthConfig(BaseModel):
    """OIDC Bearer-token verification + (mode='login') first-party login.

    Replaces the upstream Pomerium-specific path with a generic OIDC
    issuer / audience model. The Settings loader probes
    ``<issuer>/.well-known/openid-configuration`` at startup; failure
    refuses to start with ``AUTH_MISCONFIGURED:<reason>``.
    """

    model_config = ConfigDict(extra="forbid")

    # Auth mode. Authentication is ALWAYS enforced -- there is no anonymous /
    # "open" mode. `login` (default) = first-party login: a built-in admin
    # account plus optional SSO providers, with signed cookie sessions. `oidc`
    # = verify incoming Bearer JWTs against an external IdP's `issuer`/
    # `audience` (requires both). Non-allowlisted requests without a valid
    # identity are rejected (401) in either mode.
    mode: Literal["oidc", "login"] = "login"
    issuer: str | None = Field(
        default=None,
        description="OIDC discovery base URL; no trailing slash. Required when mode='oidc'.",
    )
    audience: str | None = Field(
        default=None,
        description="Expected aud claim on incoming JWTs. Required when mode='oidc'.",
    )
    jwks_cache_seconds: int = 300
    # RBAC: map an IdP group/claim value -> opsrag role names
    # (e.g. {"sre-admins": ["admin"], "oncall": ["member_investigate"]}).
    # Empty = everyone authenticated gets the default member role.
    role_mappings: dict[str, list[str]] = Field(default_factory=dict)
    # RBAC (per-connector): map an opsrag role -> the MCP connectors that role
    # grants (e.g. {"finance": ["gcp"], "sre": ["datadog", "kubernetes"]}). A
    # value of ["*"] grants every enabled connector. Only MATTERS for connectors
    # flagged `restricted: true` (non-restricted connectors are usable by all);
    # the `admin` role implicitly grants everything. Empty = no role grants, so
    # restricted connectors are reachable only via per-user allow overrides.
    # See opsrag.auth.connector_perms.
    role_connectors: dict[str, list[str]] = Field(default_factory=dict)
    # mode='login' first-party login: cookie sessions + SSO providers.
    login: SessionConfigAuth = Field(default_factory=SessionConfigAuth)
    sso: SSOConfig = Field(default_factory=SSOConfig)

    @model_validator(mode="after")
    def _require_oidc_fields(self) -> AuthConfig:
        if self.mode == "oidc" and (not self.issuer or not self.audience):
            raise ValueError(
                "auth.mode='oidc' requires both 'issuer' and 'audience'"
            )
        return self


# ---------------------------------------------------------------------------
# MCP configuration block re-exports.
# ---------------------------------------------------------------------------
# The canonical ``MCPConfigBlock`` base, the per-integration subclass map,
# the canonical name set, and the default-map factory now live in
# ``opsrag.config_mcp``. They are re-imported here (above) and re-exported
# below so existing callers can keep importing them from ``opsrag.config``.
__all__ = (
    "AuthConfig",
    "KNOWN_MCP_NAMES",
    "MCPConfigBlock",
    "MCP_CONFIG_TYPES",
    "OpsRAGConfig",
    "Settings",
)


class Settings(BaseModel):
    """Root configuration model. Single source of truth for runtime
    settings; mirrors ``contracts/config-schema.md``."""

    model_config = ConfigDict(extra="forbid")

    # ------------------------------------------------------------------
    # New v1 fields (T029): OIDC auth, MCP map, operator-supplied
    # deployment context. Every value that varies between deployments
    # belongs under ``deployment`` per Constitution Principle VI.
    # ------------------------------------------------------------------
    # Auth is ALWAYS present -- the product never serves anonymous traffic.
    # When no `auth` block is supplied (e.g. the compose demo), it defaults
    # to first-party `login` mode (built-in admin + optional SSO).
    auth: AuthConfig = Field(default_factory=lambda: AuthConfig(mode="login"))
    mcp: dict[str, MCPConfigBlock] = Field(default_factory=_default_mcp_map)
    deployment: DeploymentContext = Field(default_factory=DeploymentContext)

    # Cloud model bundle. null = use the classic provider blocks exactly as
    # today; "aws"/"gcp" fill UNSET model slots from a per-purpose bundle
    # (explicit provider blocks + `models` overrides + env always win). The
    # resolver runs in Settings.load (models feature track).
    cloud_provider: Literal["aws", "gcp"] | None = None
    models: ModelsConfig | None = None

    @field_validator("mcp", mode="before")
    @classmethod
    def _dispatch_mcp_subclasses(
        cls,
        v: object,
    ) -> dict[str, MCPConfigBlock]:
        """Dispatch each ``mcp.<name>`` entry through the per-integration
        ``MCPConfigBlock`` subclass keyed by ``name`` (see
        ``opsrag.config_mcp``). Unknown names raise ``ValueError`` so
        ``Settings.model_validate(...)`` surfaces them with the standard
        Pydantic error envelope -- satisfies the contract test
        ``test_config_unknown_mcp_rejected``.
        """
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("mcp must be a mapping of integration name -> block")
        out: dict[str, MCPConfigBlock] = {}
        for name, block in v.items():
            if name not in MCP_CONFIG_TYPES:
                raise ValueError(
                    f"Unknown MCP integration {name!r}; "
                    f"valid names: {sorted(MCP_CONFIG_TYPES)}"
                )
            target = MCP_CONFIG_TYPES[name]
            if isinstance(block, target):
                out[name] = block
            elif isinstance(block, MCPConfigBlock):
                # Wrong subclass: re-dispatch through the right one.
                out[name] = target.model_validate(block.model_dump())
            else:
                # dict-like; let the subclass validate.
                payload = dict(block) if block is not None else {}
                payload.setdefault("name", name)
                out[name] = target.model_validate(payload)
        return out

    # ------------------------------------------------------------------
    # Provider blocks. These predate T029 and remain as classic Pydantic
    # models with ``Literal`` discriminators on ``.provider``; T031+
    # introduces full discriminated unions once the registry lands.
    # ------------------------------------------------------------------
    scm: SCMConfig = Field(default_factory=SCMConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    knowledge_graph: GraphStoreConfig = Field(default_factory=GraphStoreConfig)
    light_graph: LightGraphConfig = Field(default_factory=LightGraphConfig)
    entity_extraction: EntityExtractionConfig = Field(default_factory=EntityExtractionConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    confluence: ConfluenceConfig = Field(default_factory=ConfluenceConfig)
    slack: SlackConfig = Field(default_factory=SlackConfig)
    # Legacy single-channel Slack bot config. KEPT for back-compat: an existing
    # `slack_bot:` block still boots the Slack adapter (mirrored into
    # `channels.slack` at load -- see `_mirror_legacy_slack_bot`).
    slack_bot: SlackBotConfig = Field(default_factory=SlackBotConfig)
    # Unified multi-channel bot config (Slack / Telegram / Discord / Teams).
    # New deployments configure bots here; `slack_bot:` is the deprecated alias.
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    code_cache: CodeCacheConfig = Field(default_factory=CodeCacheConfig)
    # Q&A semantic cache judge/NER knobs (the cache enable toggle stays the
    # OPSRAG_QA_CACHE env in opsrag.qa_cache). server.py threads these into the
    # judge upper bound + the spaCy NER entity-swap discriminator.
    qa_cache: QACacheConfig = Field(default_factory=QACacheConfig)
    rootly: RootlyConfig = Field(default_factory=RootlyConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    brand: BrandConfig = Field(default_factory=BrandConfig)
    # --- Three overlapping config layers (F10(c) clarity) -------------------
    # These blocks look similar but serve DIFFERENT roles -- do not conflate:
    #   * slack / rootly (above) = INGESTION sources for the RAG corpus
    #     (crawl content, chunk + embed it into the vector store).
    #   * k8s / elasticsearch (below) = DEPRECATED legacy live-tool connection
    #     config, now synthesized into environments.targets (see
    #     opsrag/environments.py). Prefer environments.targets for new configs.
    #   * mcp.<name> (the `mcp` map above) = the per-connector enable/secret
    #     toggle for each in-process MCP integration.
    k8s: K8sConfig = Field(default_factory=K8sConfig)
    elasticsearch: ElasticsearchConfig = Field(default_factory=ElasticsearchConfig)
    # Unified multi-environment registry (Approach A). When `targets` is
    # empty the engine synthesizes it from the legacy k8s/elasticsearch/
    # deployment blocks above, so existing deployments keep working.
    environments: EnvironmentsConfig = Field(default_factory=EnvironmentsConfig)
    investigation_history: InvestigationHistoryConfig = Field(
        default_factory=InvestigationHistoryConfig,
    )
    # M1/M2 -- legacy Pomerium identity. Scheduled for removal in T032
    # (replaced by ``auth`` above + ``opsrag/auth/oidc.py``).
    tracking_user: TrackingUserConfig = Field(default_factory=TrackingUserConfig)

    # P3 (2026-05-18) -- code-specific embedding lane. When BOTH
    # `code_embedding` and `code_vector_store` are non-None, the
    # ingestion pipeline dual-writes code DocType chunks to the code
    # collection (in addition to the main collection), and the
    # hybrid_search 4th lane is activated for identifier-heavy queries.
    # When either is None, behavior is identical to pre-P3.
    code_embedding: EmbeddingConfig | None = None
    code_vector_store: VectorStoreConfig | None = None

    @model_validator(mode="after")
    def _validate_code_lane(self) -> Settings:
        # The optional code-embedding lane is only wired for Vertex embeddings
        # written to a Qdrant collection (see opsrag/factory.py). Any other
        # provider deferred to a runtime NotImplementedError at provider-build
        # time; reject it at config-load with a clear, field-named error.
        if self.code_embedding is not None and self.code_embedding.provider != "vertex":
            raise ValueError(
                "code_embedding.provider must be 'vertex' "
                f"(the code lane only supports Vertex), got {self.code_embedding.provider!r}"
            )
        if (
            self.code_vector_store is not None
            and self.code_vector_store.provider != "qdrant"
        ):
            raise ValueError(
                "code_vector_store.provider must be 'qdrant' "
                f"(the code lane only supports Qdrant), got {self.code_vector_store.provider!r}"
            )
        return self

    @classmethod
    def load(cls, path: str | Path | None = None) -> Settings:
        if path is None:
            path = os.environ.get("OPSRAG_CONFIG", "config.yaml")
        p = Path(path)
        if p.exists():
            data = yaml.safe_load(p.read_text()) or {}
            cfg = cls.model_validate(data)
        else:
            cfg = cls()
        _apply_env_overrides(cfg)
        # Cloud model bundle: fill UNSET model slots from the per-purpose
        # bundle AFTER env overrides so explicit slot/env values always win.
        # No-op when cloud_provider is None. Local import avoids a circular
        # import (model_bundles imports Settings).
        from opsrag.model_bundles import resolve_cloud_bundle
        resolve_cloud_bundle(cfg)
        _mirror_legacy_slack_bot(cfg)
        return cfg


# Backward-compat alias for existing imports. New code should import
# ``Settings`` directly.
OpsRAGConfig = Settings


def _mirror_legacy_slack_bot(cfg: Settings) -> None:
    """Mirror a legacy ``slack_bot:`` block into ``channels.slack``.

    Back-compat (design section 5): when an operator still configures the deprecated
    top-level ``slack_bot:`` block (``enabled=True``) AND has not configured the
    new ``channels.slack`` block (it is still at defaults, i.e. disabled), copy
    the legacy settings into ``channels.slack`` so the unified boot path picks
    them up. If ``channels.slack`` is already enabled, the new block wins and
    the legacy block is ignored (with a warning).

    Both paths boot the same Slack adapter, so this keeps existing
    ``config.yaml`` / Helm ``slackBot:`` deployments working unchanged.
    """
    legacy = cfg.slack_bot
    if not getattr(legacy, "enabled", False):
        return
    slack_ch = cfg.channels.slack
    if slack_ch.enabled:
        _log.warning(
            "config: both slack_bot.enabled and channels.slack.enabled are set; "
            "channels.slack wins (slack_bot is deprecated -- migrate to channels.slack)."
        )
        return
    _log.warning(
        "config: slack_bot.* is DEPRECATED -- mirroring it into channels.slack. "
        "Migrate to the unified `channels.slack` block."
    )
    cfg.channels.slack = SlackChannelConfig(
        enabled=True,
        bot_token_env=legacy.bot_token_env,
        app_token_env=legacy.app_token_env,
        allowlist=list(legacy.channels_allowlist),
        per_user_daily_quota=legacy.per_user_daily_quota,
        streaming_enabled=legacy.streaming_enabled,
        workspace_url=legacy.workspace_url,
        web_ui_base_url=legacy.web_ui_base_url,
        thread_context_message_cap=getattr(
            legacy, "thread_context_message_cap", 20
        ),
    )


# Env var -> (sub-model attr, field) mapping. Values set here override
# YAML so the same image can be deployed for multiple tenants by just
# changing env vars at the cluster level. Keep the surface narrow:
# only fields that legitimately differ per deployment (URLs, branding).
# Tokens already live behind their own `*_token_env` indirection -- don't
# duplicate that here.
_ENV_OVERRIDES: list[tuple[str, str, str]] = [
    # 4 base URLs used to render clickable source links in answers.
    ("OPSRAG_CONFLUENCE_BASE_URL", "confluence", "base_url"),
    ("OPSRAG_SLACK_WORKSPACE_URL", "slack", "workspace_url"),
    ("OPSRAG_ROOTLY_WEB_URL", "rootly", "web_base_url"),
    ("OPSRAG_SCM_BASE_URL", "scm", "base_url"),
    # White-label / favicon -- surfaced via /ui-config to the React UI.
    ("OPSRAG_BRAND_NAME", "brand", "name"),
    ("OPSRAG_BRAND_SUBTITLE", "brand", "subtitle"),
    ("OPSRAG_BRAND_ASSISTANT_NAME", "brand", "assistant_name"),
    ("OPSRAG_BRAND_FAVICON_URL", "brand", "favicon_url"),
    ("OPSRAG_BRAND_ACCENT_COLOR", "brand", "accent_color"),
]

_log = logging.getLogger("opsrag.config")


def _as_bool(raw: str) -> bool:
    """Parse a boolean env value. Raises ValueError on anything ambiguous so the
    override is skipped with a warning (rather than silently truthy)."""
    v = raw.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(raw)


# Model / provider selection via env. Lets the SAME image switch cloud provider
# (cost vs quality) and bump model ids -- e.g. when a version is deprecated --
# WITHOUT editing YAML or rebuilding. Applied BEFORE resolve_cloud_bundle so an
# env-set value counts as "explicitly set" and the per-provider bundle won't
# overwrite it. Precedence: env > YAML slot > cloud-bundle default.
#   (env_var, section | None for a top-level Settings field, field, caster)
_MODEL_ENV_OVERRIDES: list[tuple[str, str | None, str, Any]] = [
    ("OPSRAG_CLOUD_PROVIDER",     None,        "cloud_provider", str),
    ("OPSRAG_LLM_PROVIDER",       "llm",       "provider",       str),
    ("OPSRAG_LLM_MODEL",          "llm",       "model",          str),
    ("OPSRAG_PRO_MODEL",          "agent",     "pro_model",      str),
    ("OPSRAG_EMBEDDING_PROVIDER", "embedding", "provider",       str),
    ("OPSRAG_EMBEDDING_MODEL",    "embedding", "model",          str),
    ("OPSRAG_EMBEDDING_DIMENSION","embedding", "dimension",      int),
    ("OPSRAG_RERANKER_PROVIDER",  "reranker",  "provider",       str),
    ("OPSRAG_RERANKER_MODEL",     "reranker",  "model",          str),
    # Vision auto-route fallback (only used when the active model can't see).
    ("OPSRAG_VISION_ENABLED",     "vision",    "enabled",        _as_bool),
    ("OPSRAG_VISION_PROVIDER",    "vision",    "provider",       str),
    ("OPSRAG_VISION_MODEL",       "vision",    "model",          str),
]

# cloud_provider is a closed set -> guard so a typo doesn't silently select an
# empty bundle. Other provider fields are Pydantic Literals validated at use.
_CLOUD_PROVIDERS = {"aws", "gcp"}


def _apply_env_overrides(cfg: Settings) -> None:
    for env_var, section, field in _ENV_OVERRIDES:
        val = os.environ.get(env_var)
        if val is None:
            continue
        sub = getattr(cfg, section, None)
        if sub is None or not hasattr(sub, field):
            continue
        setattr(sub, field, val)

    # Model / provider knobs (typed; top-level or sub-model).
    for env_var, section, field, caster in _MODEL_ENV_OVERRIDES:
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        if field == "cloud_provider" and raw not in _CLOUD_PROVIDERS:
            _log.warning(
                "ignoring %s=%r: must be one of %s", env_var, raw, sorted(_CLOUD_PROVIDERS)
            )
            continue
        try:
            value = caster(raw)
        except (TypeError, ValueError):
            _log.warning("ignoring %s=%r: not a valid %s", env_var, raw, caster.__name__)
            continue
        target = cfg if section is None else getattr(cfg, section, None)
        if target is None or not hasattr(target, field):
            continue
        setattr(target, field, value)
        _log.info("config override from %s: %s.%s",
                  env_var, section or "(root)", field)
