"""FastAPI application factory."""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from opsrag import __version__

_log = logging.getLogger("opsrag.server")
from opsrag.agent.graph import (
    build_full_graph,
    build_minimal_graph,
    build_multi_agent_graph,
    build_tool_calling_graph,
)
from opsrag.api.errors import register_error_handlers
from opsrag.api.middleware import RateLimitMiddleware
from opsrag.api.oidc_enforcement import OIDCAuthMiddleware
from opsrag.api.rate_limit_backend import (
    MemoryRateLimitBackend,
    RedisRateLimitBackend,
    make_redis_client,
)
from opsrag.api.routes import router
from opsrag.api.routes_health import health_router
from opsrag.auth import (
    PomeriumVerifier,
    UserStore,
    build_verifier_from_settings,
    current_user_oid_var,
)
from opsrag.config import OpsRAGConfig
from opsrag.correction_store import CorrectionStore
from opsrag.factory import build_providers
from opsrag.feedback_store import FeedbackStore
from opsrag.graphstores.neo4j import Neo4jGraphStore
from opsrag.indexed_files.postgres import PostgresIndexedFilesTracker
from opsrag.ingestion.pipeline import IngestionPipeline
from opsrag.llms.bedrock import BedrockLLM
from opsrag.llms.vertex import VertexAILLM, VertexResult
from opsrag.mcp_server import AuditLogger, MCPServer, MCPTokenStore, TokenRateLimiter
from opsrag.qa_cache import QAVectorCache
from opsrag.qa_cache import is_enabled as qa_cache_enabled
from opsrag.sessions.postgres import PostgresSessionStore
from opsrag.usage import tracker as usage_tracker
from opsrag.usage_persistence import UsagePersistence


def _build_rate_limit_backend(cfg: OpsRAGConfig):
    """Construct the rate-limit backend from ``cfg.api``.

    ``rate_limit_backend == "memory"`` (default) returns an in-process
    backend and touches neither the ``redis`` extra nor the network.

    ``rate_limit_backend == "redis"`` is the require-redis path: it builds a
    ``redis.asyncio`` client from the env var named by ``cfg.api.redis_url_env``
    and PINGs it, FAILING FAST with a clear error if the var is unset or the
    server is unreachable. The shared backend is returned for both the request
    limiter and the login lockout so replicas agree.
    """
    if cfg.api.rate_limit_backend != "redis":
        # Per-pod, in-process state: with N replicas the effective limit is
        # N x rpm and the login lockout is not shared. Fine for single-replica
        # / local dev; set api.rate_limit_backend=redis for a shared limit.
        _log.warning(
            "rate-limit: using the in-memory backend (per-pod). Limits are "
            "NOT shared across replicas -- with N replicas the effective rate "
            "is N x %s rpm. Set api.rate_limit_backend=redis to share state.",
            getattr(cfg.api, "rate_limit_rpm", "configured"),
        )
        return MemoryRateLimitBackend()

    url = os.environ.get(cfg.api.redis_url_env, "").strip()
    if not url:
        raise RuntimeError(
            "rate_limit_backend=redis requires a Redis URL; set the "
            f"{cfg.api.redis_url_env} environment variable"
        )
    client = make_redis_client(url)
    try:
        asyncio.run(client.ping())
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "rate_limit_backend=redis: cannot reach Redis at "
            f"${cfg.api.redis_url_env} ({exc}). Redis is required when the "
            "redis rate-limit backend is selected."
        ) from exc
    _log.info("rate-limit: redis backend connected (%s)", cfg.api.redis_url_env)
    return RedisRateLimitBackend(client=client)


def _build_agent_graph(cfg, providers, checkpointer, known_repos, model_router):
    """Select + build the agent graph for ``cfg.agent.mode``.

    Pure dispatch (no I/O of its own) extracted from the lifespan so the
    mode->builder mapping is unit-testable. Behaviour is byte-for-byte the
    same as the inline chain it replaces:

      - multi_agent / tool_calling / minimal -> their dedicated builders
      - full / hybrid                         -> build_full_graph (hybrid is a
        REMOVED legacy alias: it still builds the full graph but emits a
        one-time warning so operators migrate the knob)
      - anything else                         -> ValueError (fail fast; a
        typo'd mode is a startup error, not a silent default)

    The pydantic ``Literal`` on ``agent.mode`` already rejects true typos at
    config-load, so the final ``raise`` is a defensive backstop for graphs
    built from programmatically-constructed configs / future mode additions.
    """
    if cfg.agent.mode == "multi_agent":
        # When the NLI groundedness gate is OFF but the artifact verifier is
        # ON, warn once so operators know the default path still has a
        # fail-closed safety net (just not the entailment check). Both off is a
        # deliberate latency/cost trade and not warned here.
        if not cfg.agent.verify_grounding_default and cfg.agent.verify_artifacts_default:
            _log.warning(
                "groundedness check disabled on multi_agent "
                "(agent.verify_grounding_default=false); artifact verifier "
                "still runs (agent.verify_artifacts_default=true)"
            )
        return build_multi_agent_graph(
            llm=providers.llm,
            vector_store=providers.vector_store,
            embedder=providers.embedder,
            observability=providers.observability,
            reranker=providers.reranker,
            checkpointer=checkpointer,
            top_k=cfg.agent.top_k,
            rerank_top_k=cfg.agent.rerank_top_k,
            rerank_diversity=cfg.agent.rerank_diversity,
            rerank_content_dedup=cfg.agent.rerank_content_dedup,
            rerank_content_dedup_threshold=cfg.agent.rerank_content_dedup_threshold,
            known_repos=known_repos,
            model_router=model_router,
            code_embedder=providers.code_embedder,
            code_store=providers.code_vector_store,
            light_graph=providers.light_graph,
            verify_grounding=cfg.agent.verify_grounding_default,
            verify_artifacts=cfg.agent.verify_artifacts_default,
        )
    elif cfg.agent.mode == "tool_calling":
        return build_tool_calling_graph(
            llm=providers.llm,
            vector_store=providers.vector_store,
            embedder=providers.embedder,
            observability=providers.observability,
            reranker=providers.reranker,
            checkpointer=checkpointer,
            top_k=cfg.agent.top_k,
            rerank_top_k=cfg.agent.rerank_top_k,
            rerank_diversity=cfg.agent.rerank_diversity,
            rerank_content_dedup=cfg.agent.rerank_content_dedup,
            rerank_content_dedup_threshold=cfg.agent.rerank_content_dedup_threshold,
            known_repos=known_repos,
            model_router=model_router,
            code_embedder=providers.code_embedder,
            code_store=providers.code_vector_store,
            light_graph=providers.light_graph,
        )
    elif cfg.agent.mode == "minimal":
        return build_minimal_graph(
            llm=providers.llm,
            vector_store=providers.vector_store,
            embedder=providers.embedder,
            observability=providers.observability,
            reranker=providers.reranker,
            checkpointer=checkpointer,
            top_k=cfg.agent.top_k,
            rerank_top_k=cfg.agent.rerank_top_k,
            rerank_diversity=cfg.agent.rerank_diversity,
            rerank_content_dedup=cfg.agent.rerank_content_dedup,
            rerank_content_dedup_threshold=cfg.agent.rerank_content_dedup_threshold,
            known_repos=known_repos,
            code_embedder=providers.code_embedder,
            code_store=providers.code_vector_store,
        )
    elif cfg.agent.mode in ("full", "hybrid"):
        # NB: the legacy `hybrid` mode (graph-anchored retrieval) was removed
        # with the Neo4j lane (its build_hybrid_graph stub is now deleted too);
        # `agent.mode: hybrid` degrades to the full graph rather than failing.
        # Make the mismatch EXPLICIT (don't silently map unknown modes here):
        # `full` is the real target; `hybrid` is a removed legacy alias kept
        # building the full graph, but loudly, so operators migrate the knob.
        if cfg.agent.mode == "hybrid":
            _log.warning(
                "agent.mode='hybrid' is a removed legacy alias; running "
                "build_full_graph -- set mode to "
                "full/minimal/tool_calling/multi_agent"
            )
        return build_full_graph(
            llm=providers.llm,
            vector_store=providers.vector_store,
            embedder=providers.embedder,
            reranker=providers.reranker,
            observability=providers.observability,
            memory_store=providers.memory_store,
            checkpointer=checkpointer,
            top_k=cfg.agent.top_k,
            rerank_top_k=cfg.agent.rerank_top_k,
            rerank_diversity=cfg.agent.rerank_diversity,
            rerank_content_dedup=cfg.agent.rerank_content_dedup,
            rerank_content_dedup_threshold=cfg.agent.rerank_content_dedup_threshold,
            known_repos=known_repos,
            light_graph=providers.light_graph,
            # Pro/answer model (Sonnet 4.6) for the final generation; cheap
            # nodes stay on the base llm. Without this the full graph
            # generated with the base llm (Haiku) and pro_model was unused.
            model_router=model_router,
            code_embedder=providers.code_embedder,
            code_store=providers.code_vector_store,
        )
    else:
        # Fail fast on an unknown/typo'd mode rather than silently building
        # the full graph for everything -- a misconfigured knob is a startup
        # error, not a quiet default.
        raise ValueError(f"unknown agent.mode={cfg.agent.mode!r}")


def create_app(config: OpsRAGConfig | None = None) -> FastAPI:
    cfg = config or OpsRAGConfig.load()

    # Cost-telemetry pricing: install operator overrides (USD -> micro-cents)
    # then warn loudly about any CONFIGURED model with no price -- so an
    # unpriced model surfaces at deploy time, not later as a "$0" row in /usage.
    from opsrag.llms import pricing as _pricing
    _tok_over: dict[str, tuple[int, int]] = {}
    _pc_over: dict[str, int] = {}
    for _mid, _spec in (cfg.llm.model_prices or {}).items():
        if "per_call" in _spec:
            _pc_over[_mid] = int(round(float(_spec["per_call"]) * 1e8))
        if "input_per_1m" in _spec or "output_per_1m" in _spec:
            _tok_over[_mid] = (
                int(round(float(_spec.get("input_per_1m", 0)) * 1e8)),
                int(round(float(_spec.get("output_per_1m", 0)) * 1e8)),
            )
    _pricing.set_overrides(_tok_over, _pc_over)
    _configured_models = [
        cfg.llm.model,
        getattr(cfg.agent, "pro_model", None),
        getattr(cfg.embedding, "model", None),
        getattr(cfg.reranker, "model", None),
        getattr(cfg.memory, "mem0_embed_model", None),
    ]
    _unpriced = [
        m for m in dict.fromkeys(filter(None, _configured_models))
        if not _pricing.has_price(m)
    ]
    if _unpriced:
        _log.warning(
            "no pricing for configured model(s) %s -> their usage cost will be "
            "recorded as $0. Set llm.model_prices['<model>'] = "
            "{input_per_1m: .., output_per_1m: ..} (or {per_call: ..}) to fix.",
            _unpriced,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Expand the asyncio default ThreadPoolExecutor before any work
        # starts. Default sizing is `min(32, cpu_count + 4)` which on a
        # 4-core container is 8 workers. Indexing throws sync work at this
        # pool from many directions: Vertex SDK embed calls, parser.parse,
        # chunker.chunk, regex-based entity extraction. With 12 concurrent
        # file-process pipelines, 8 workers saturate, and to_thread calls
        # queue -> request handlers (e.g. /health) wait for a thread that
        # never frees -> loop appears wedged.
        import concurrent.futures
        loop = asyncio.get_running_loop()
        loop.set_default_executor(
            concurrent.futures.ThreadPoolExecutor(
                max_workers=32,
                thread_name_prefix="opsrag-pool",
            )
        )

        # Run DB migrations early -- before any module opens its own pool /
        # tries to CREATE TABLE IF NOT EXISTS. The runner is idempotent
        # and only acts on missing migrations.
        #
        # Opt out by setting OPSRAG_AUTO_MIGRATE=false (or 0/no/off) -- in
        # production you may prefer to run `python -m opsrag.db.migrate up`
        # manually as a separate k8s Job before rolling the backend, so
        # a buggy migration can't crash the pod into CrashLoopBackoff.
        _auto_migrate_env = os.environ.get("OPSRAG_AUTO_MIGRATE", "true").lower()
        _auto_migrate = _auto_migrate_env not in ("false", "0", "no", "off")
        if cfg.session.provider == "postgres" and _auto_migrate:
            mig_dsn = cfg.session.dsn or os.environ.get(cfg.session.dsn_env, "")
            if mig_dsn:
                try:
                    from psycopg_pool import AsyncConnectionPool

                    from opsrag.db.migrate import apply_all
                    _mig_pool = AsyncConnectionPool(
                        conninfo=mig_dsn, min_size=1, max_size=1, open=False,
                        kwargs={"autocommit": False},
                    )
                    await _mig_pool.open()
                    try:
                        await apply_all(_mig_pool)
                    finally:
                        await _mig_pool.close()
                except Exception as exc:
                    _log.warning(
                        "DB migrations failed (%s); proceeding anyway -- "
                        "modules that own their own CREATE TABLE IF NOT EXISTS "
                        "will still work, but new identity/MCP tables may be missing",
                        exc,
                    )
        elif not _auto_migrate:
            _log.info(
                "OPSRAG_AUTO_MIGRATE=false -- skipping auto-apply, "
                "run `python -m opsrag.db.migrate up` manually before rollouts"
            )

        providers = build_providers(cfg)

        # Fail-closed embedding-dimension guard (shared Qdrant seam: the main
        # index, the QA cache, and investigations share the client). Refuse to
        # start if the embedder's dimension differs from an existing
        # collection's dimension, unless allow_dimension_change is set.
        # Runs here (async lifespan) -- build_providers is sync, so it can't
        # await this itself. No-op on a missing collection (first boot).
        if cfg.vector_store.provider == "qdrant":
            try:
                from opsrag.vectorstore_guard import assert_dimension_compatible
                await assert_dimension_compatible(
                    providers.vector_store._client,
                    cfg.vector_store.collection,
                    providers.embedder.dimension,
                    cfg.vector_store.allow_dimension_change,
                )
            except AttributeError:
                # vector store exposes no async _client -- skip the guard.
                pass

        # Same fail-closed guard for the SEPARATE code collection (dual-write
        # lane). It has its own embedder + dimension; a code-embedder swap would
        # otherwise silently create a wrong-dim code collection on first boot.
        if (
            cfg.code_vector_store is not None
            and cfg.code_vector_store.provider == "qdrant"
            and providers.code_vector_store is not None
            and providers.code_embedder is not None
        ):
            try:
                from opsrag.vectorstore_guard import assert_dimension_compatible
                await assert_dimension_compatible(
                    providers.code_vector_store._client,
                    cfg.code_vector_store.collection,
                    providers.code_embedder.dimension,
                    cfg.code_vector_store.allow_dimension_change,
                )
            except AttributeError:
                pass

        # Fail-fast APOC guard: if the neo4j graph lane is explicitly
        # selected, refuse to start when APOC is missing (a silent-empty
        # graph is worse than a clear startup error). No-op for provider=none.
        if cfg.knowledge_graph.provider == "neo4j" and providers.graph_store is not None:
            check_apoc = getattr(providers.graph_store, "check_apoc", None)
            if check_apoc is not None:
                await check_apoc()

        # mode='login': build first-party login runtime state. Lazy imports
        # keep the `login` extra out of oidc deployments. Best-effort:
        # a setup failure logs + disables login rather than crashing boot.
        if cfg.auth.mode == "login":
            try:
                import os as _os

                from opsrag.auth.login import LoginRateLimiter
                from opsrag.auth.sessions import SessionManager, load_signing_key
                from opsrag.auth.sso import ProviderConfig, build_oauth_registry
                from opsrag.auth.user_store import (
                    InMemoryAuthUserStore,
                    PostgresAuthUserStore,
                )
                lc = cfg.auth.login
                key = load_signing_key(key_path=lc.signing_key_path, key_env=lc.signing_key_env)
                app.state.session_manager = SessionManager(
                    key,
                    session_ttl_seconds=lc.session_ttl_seconds,
                    refresh_ttl_seconds=lc.refresh_ttl_seconds,
                    cookie_secure=lc.cookie_secure,
                    cookie_samesite=lc.cookie_samesite,
                    cookie_domain=lc.cookie_domain,
                )
                dsn = cfg.session.dsn or _os.environ.get(cfg.session.dsn_env, "")
                if cfg.session.provider == "postgres" and dsn:
                    aus = PostgresAuthUserStore(dsn=dsn)
                    await aus.open()
                else:
                    aus = InMemoryAuthUserStore()
                app.state.auth_user_store = aus
                # Seed an initial admin from a SECRET (env var), never inline:
                # OPSRAG_ADMIN_EMAIL + OPSRAG_ADMIN_PASSWORD (the password
                # comes from a mounted secret in production). Idempotent --
                # only creates the user if it doesn't already exist.
                admin_email = _os.environ.get("OPSRAG_ADMIN_EMAIL")
                admin_pw = _os.environ.get("OPSRAG_ADMIN_PASSWORD")
                if admin_email and admin_pw:
                    existing = await aus.get_user_by_email(admin_email)
                    if existing is None:
                        from opsrag.auth.password import hash_password
                        await aus.create_user(
                            email=admin_email,
                            password_hash=hash_password(admin_pw),
                            email_verified=True,
                            roles=("admin",),
                        )
                        _log.info("auth: seeded admin user %s", admin_email)
                # Share the rate-limit backend with login ONLY when it's the
                # distributed (redis) one, so the lockout is enforced across
                # replicas. On the default memory backend, leave backend=None
                # so login keeps its own byte-identical in-process state.
                _rl_backend = getattr(app.state, "rate_limit_backend", None)
                _login_backend = (
                    _rl_backend
                    if isinstance(_rl_backend, RedisRateLimitBackend)
                    else None
                )
                app.state.login_rate_limiter = LoginRateLimiter(
                    max_attempts=lc.login_max_attempts,
                    window_seconds=lc.login_window_seconds,
                    lockout_seconds=lc.login_lockout_seconds,
                    backend=_login_backend,
                )
                _sso_blocks = (
                    ("google", cfg.auth.sso.google),
                    ("microsoft", cfg.auth.sso.microsoft),
                    ("github", cfg.auth.sso.github),
                )
                app.state.sso_oauth = build_oauth_registry({
                    name: ProviderConfig(
                        enabled=p.enabled,
                        client_id=p.client_id,
                        client_secret=_os.environ.get(p.client_secret_env or "", "") or None,
                        scopes=tuple(p.scopes),
                        server_metadata_url=p.server_metadata_url,
                    )
                    for name, p in _sso_blocks
                })
                # Advertise available login methods to the UI (easy switching:
                # password / SSO / both). password_enabled + enabled providers.
                app.state.login_password_enabled = cfg.auth.login.password_enabled
                app.state.sso_providers = [n for n, p in _sso_blocks if p.enabled]
                # External base for the SSO redirect_uri (matches what you
                # register with each IdP). None -> derive from the request.
                app.state.sso_callback_base = cfg.auth.login.sso_callback_base
                _log.info(
                    "auth: login-mode ready (password=%s, sso=%s)",
                    cfg.auth.login.password_enabled,
                    app.state.sso_providers,
                )
            except Exception as exc:  # noqa: BLE001
                _log.error("auth: login-mode setup failed (%s); login disabled", exc)

        # Unified multi-environment registry (Approach A): bind it FIRST so
        # the k8s / prometheus / elasticsearch MCPs resolve per-env targets
        # (cluster coords, prometheus service/ns/port, ES endpoint+fields).
        # When the `environments:` block is empty this synthesizes a registry
        # from the legacy k8s/elasticsearch/deployment config -- so existing
        # deployments + the demo keep working (the register_clusters /
        # es_mcp.bind calls below are now back-compat shims).
        try:
            from opsrag.environments import bind_environments
            bind_environments(cfg)
        except Exception as exc:  # noqa: BLE001
            _log.warning("environments registry bind failed: %s", exc)

        # Register K8s cluster coordinates with the K8s MCP for the optional
        # GKE Workload-Identity provider (ADC + GCP Container API). Empty
        # config (the default) falls through to the vendor-neutral path:
        # in-cluster ServiceAccount, or a standard kubeconfig.
        if cfg.k8s.clusters:
            try:
                from opsrag.mcp.kubernetes import register_clusters
                register_clusters({
                    name: coords.model_dump()
                    for name, coords in cfg.k8s.clusters.items()
                })
            except Exception as exc:
                _log.warning("k8s mcp cluster registration failed: %s", exc)

        # Elasticsearch / OpenSearch MCP -- direct read-only client against
        # ES_URL with API-key / basic auth (creds from env). Disabled by
        # default.
        if cfg.elasticsearch.enabled:
            try:
                from opsrag.mcp import elasticsearch as es_mcp
                es_mcp.bind(cfg.elasticsearch)
                _log.info(
                    "elasticsearch MCP bound (backend=%s)",
                    getattr(cfg.elasticsearch, "backend", "elasticsearch"),
                )
            except Exception as exc:  # noqa: BLE001
                _log.warning("elasticsearch MCP bind failed: %s", exc)

        # -- Cloudflare MCP (LIVE API surface) ----------------------
        # Token source priority:
        #   1. CLOUDFLARE_API_KEY env (prod path -- ExternalSecret from
        #      GSM `opsrag-cloudflare-api-key`).
        #   2. CLOUDFLARE_API_KEY_FILE env (local dev path -- file
        #      `.cloudflare-api-key` mounted via docker-compose volume).
        # Empty token -> bind is no-op; every cloudflare_* tool surfaces
        # reason="not_bound".
        try:
            from opsrag.mcp import cloudflare as cloudflare_mcp
            cf_token = os.environ.get("CLOUDFLARE_API_KEY", "").strip()
            if not cf_token:
                cf_token_file = os.environ.get("CLOUDFLARE_API_KEY_FILE", "").strip()
                if cf_token_file and os.path.isfile(cf_token_file):
                    try:
                        with open(cf_token_file) as f:
                            cf_token = f.read().strip()
                    except Exception as exc:  # noqa: BLE001
                        _log.warning(
                            "cloudflare: failed to read token from %s: %s",
                            cf_token_file, exc,
                        )
            cloudflare_mcp.bind(token=cf_token)
            if cf_token:
                _log.info("cloudflare MCP bound (live API)")
            else:
                _log.info("cloudflare MCP not bound -- no CLOUDFLARE_API_KEY env/file")
        except Exception as exc:  # noqa: BLE001
            _log.warning("cloudflare MCP bind failed: %s", exc)

        # Instantiate Pro LLM if configured. Used for
        # complex-query escalation in tool_synthesize / generate. Falls
        # back to Flash silently when None.
        pro_llm = None
        if cfg.agent.pro_model:
            try:
                # Build the Pro (escalation/synthesis) model on the SAME
                # provider as the main llm so a Bedrock deployment escalates to
                # Bedrock (e.g. Opus), not Vertex. (VertexAILLM is imported at
                # module top; the redundant local import here previously
                # shadowed it -> UnboundLocalError in the on_usage hook.)
                if cfg.llm.provider == "bedrock":
                    pro_llm = BedrockLLM(
                        model=cfg.agent.pro_model,
                        region=cfg.llm.aws_region,
                        profile=cfg.llm.aws_profile,
                        # Bound the escalation client's tail latency with the
                        # same timeout/retry knobs as the main llm slot.
                        request_timeout=cfg.llm.request_timeout,
                        connect_timeout=cfg.llm.connect_timeout,
                        max_retries=cfg.llm.max_retries,
                    )
                else:
                    pro_llm = VertexAILLM(model=cfg.agent.pro_model)
                _log.info("Pro LLM enabled: %s (%s)", cfg.agent.pro_model, cfg.llm.provider)
            except Exception as exc:
                _log.warning(
                    "Pro LLM init failed (%s); routing degrades to Flash-only",
                    exc,
                )
        from opsrag.agent.model_router import ModelRouter
        model_router = ModelRouter(flash_llm=providers.llm, pro_llm=pro_llm)
        app.state.model_router = model_router

        # Investigation cache. Stores tool-path
        # investigation outcomes so the reasoner can reference past
        # similar investigations on follow-up queries.
        investigation_cache = None
        try:
            from qdrant_client import AsyncQdrantClient

            from opsrag.agent.cache import InvestigationCache
            qdrant_url = (
                cfg.vector_store.url
                if hasattr(cfg.vector_store, "url") and cfg.vector_store.url
                else os.environ.get("QDRANT_URL", "http://qdrant:6333")
            )
            inv_qdrant = AsyncQdrantClient(url=qdrant_url)
            # Size the cache collection to the ACTIVE embedder dimension, not the
            # legacy 768 default (text-embedding-005). With Cohere Embed v4 the
            # embedder produces 1536; a 768 collection made every investigation
            # lookup 400 ("Vector dimension error: expected dim: 768, got 1536")
            # -- fail-safe, but it silently disabled investigation caching.
            _inv_dim = getattr(cfg.embedding, "dimension", None) or 768
            investigation_cache = InvestigationCache(qdrant=inv_qdrant, vector_size=_inv_dim)
            _log.info(
                "investigation cache wired (collection=opsrag_investigations, dim=%d)",
                _inv_dim,
            )
        except Exception as exc:
            _log.warning("investigation cache init failed: %s", exc)
        app.state.investigation_cache = investigation_cache

        # InvestigationsSource: promote settled
        # past investigations into the corpus as historical-reference
        # docs. Registered on `providers.sources` so the existing
        # ingestion pipeline + daily scheduler can pull them like any
        # other source. Requires `investigation_cache` (above).
        if cfg.investigation_history.enabled and investigation_cache is not None:
            try:
                from opsrag.sources.investigations import InvestigationsSource
                providers.sources["investigation-history"] = InvestigationsSource(
                    investigation_cache=investigation_cache,
                    min_age_days=cfg.investigation_history.min_age_days,
                    max_age_days=cfg.investigation_history.max_age_days,
                    max_docs=cfg.investigation_history.max_docs_per_run,
                    skip_thumbs_down=cfg.investigation_history.skip_thumbs_down,
                )
                _log.info(
                    "investigation-history source registered (age %d-%dd, max_docs=%d)",
                    cfg.investigation_history.min_age_days,
                    cfg.investigation_history.max_age_days,
                    cfg.investigation_history.max_docs_per_run,
                )
            except Exception as exc:
                _log.warning("investigation-history source init failed: %s", exc)

        if isinstance(providers.session_store, PostgresSessionStore):
            await providers.session_store.open()

        # Lightweight entity-graph (Postgres edges) for the entity-expansion
        # retrieval lane. Open the pool; non-fatal (the lane is augment-only).
        if providers.light_graph is not None:
            try:
                await providers.light_graph.open()
                _log.info("light-graph (entity-expansion) store opened")
            except Exception as exc:
                _log.warning("light-graph open failed (%s); entity-expansion disabled", exc)
                providers.light_graph = None

        # Durable indexing job-state (Postgres). Backend pods read it so
        # /indexing/status is consistent across replicas; writer roles flush
        # the in-memory tracker into it (started below). Non-fatal: on failure
        # the status routes fall back to the in-memory tracker.
        app.state.index_store = None
        if providers.index_store is not None:
            try:
                await providers.index_store.open()
                app.state.index_store = providers.index_store
                _log.info("indexing job-state store opened (durable /indexing/status)")
            except Exception as exc:
                _log.warning("index-state store open failed (%s); using in-memory tracker", exc)
                providers.index_store = None

        # Live, operator-editable agent guidance (custom instructions). Open the
        # store, seed it from the config default on first boot, install the live
        # value into the prompt layer, and refresh it on a short interval so
        # UI edits take effect on the next query across replicas (no restart).
        app.state.agent_settings = None
        app.state._agent_settings_stop = None
        app.state._agent_settings_task = None
        if providers.agent_settings is not None:
            try:
                from opsrag.agent.prompt_render import set_custom_instructions_live
                from opsrag.agent_settings import CUSTOM_INSTRUCTIONS_KEY
                await providers.agent_settings.open()
                app.state.agent_settings = providers.agent_settings
                meta = await providers.agent_settings.get_meta(CUSTOM_INSTRUCTIONS_KEY)
                if meta is None:
                    # First boot: seed the editable value from the config default
                    # (deployment.custom_instructions) so the UI shows it.
                    seed = (cfg.deployment.custom_instructions or "").strip()
                    if seed:
                        try:
                            await providers.agent_settings.set(
                                CUSTOM_INSTRUCTIONS_KEY, seed, updated_by="config-seed",
                            )
                        except Exception:
                            pass
                        set_custom_instructions_live(seed)
                    else:
                        set_custom_instructions_live(None)
                else:
                    set_custom_instructions_live(meta.get("value") or "")
                _log.info("agent-guidance store opened (live custom instructions)")

                stop_evt = asyncio.Event()
                app.state._agent_settings_stop = stop_evt

                async def _refresh_agent_guidance() -> None:
                    while not stop_evt.is_set():
                        try:
                            await asyncio.wait_for(stop_evt.wait(), timeout=15)
                        except TimeoutError:
                            pass
                        if stop_evt.is_set():
                            break
                        try:
                            m = await providers.agent_settings.get_meta(CUSTOM_INSTRUCTIONS_KEY)
                            set_custom_instructions_live(m.get("value") or "" if m else None)
                        except Exception:
                            pass

                app.state._agent_settings_task = asyncio.create_task(_refresh_agent_guidance())
            except Exception as exc:
                _log.warning("agent-guidance store open failed (%s); using config seed only", exc)
                providers.agent_settings = None

        # Ensure the indexed_files schema is in place before the
        # pipeline can call should_skip / record. Safe even on warm DB --
        # the DDL is IF NOT EXISTS.
        if isinstance(providers.indexed_files, PostgresIndexedFilesTracker):
            await providers.indexed_files.open()

        # Usage telemetry persistence. Reuses POSTGRES_DSN. Without this
        # the in-memory UsageTracker resets on every restart and the
        # Usage & Cost dashboard shows zero. Order matters: open the
        # pool, seed historical totals into the live tracker, THEN start
        # the flush loop and wire the persistence hook so new records
        # land in DB without double-counting the seeded ones.
        usage_persist: UsagePersistence | None = None
        if cfg.session.provider == "postgres":
            usage_dsn = cfg.session.dsn or os.environ.get(cfg.session.dsn_env, "")
            if usage_dsn:
                usage_persist = UsagePersistence(dsn=usage_dsn)
                try:
                    await usage_persist.open()
                    await usage_persist.seed_tracker(usage_tracker)
                    usage_persist.start()
                    usage_tracker.set_persistence_hook(usage_persist.enqueue)
                    app.state.usage_persistence = usage_persist
                except Exception as exc:
                    _log.warning("usage persistence init failed: %s", exc)
                    usage_persist = None

        # Thumbs-up/down feedback persistence. Reuses
        # the same Postgres DSN. Same idempotent-schema pattern as
        # UsagePersistence. Failures are non-fatal: the route handler
        # checks `app.state.feedback_store` and gracefully no-ops if the
        # store isn't available.
        feedback_store: FeedbackStore | None = None
        app.state.feedback_store = None
        if cfg.session.provider == "postgres":
            fb_dsn = cfg.session.dsn or os.environ.get(cfg.session.dsn_env, "")
            if fb_dsn:
                feedback_store = FeedbackStore(dsn=fb_dsn)
                try:
                    await feedback_store.open()
                    app.state.feedback_store = feedback_store
                except Exception as exc:
                    _log.warning("feedback_store init failed: %s", exc)
                    feedback_store = None

        # RunbookStore (hand-authored runbooks).
        # Reuses the same Postgres DSN as feedback_store, with its own
        # tiny pool so a busy feedback path doesn't starve runbook
        # CRUD. Embedding goes into a dedicated Qdrant collection
        # (`opsrag_runbooks_vec`) using the same embedder the agent
        # uses for query embedding.
        app.state.runbook_store = None
        app.state.runbook_pool = None
        if cfg.session.provider == "postgres":
            rb_dsn = cfg.session.dsn or os.environ.get(cfg.session.dsn_env, "")
            if rb_dsn:
                try:
                    from psycopg_pool import AsyncConnectionPool

                    from opsrag.runbooks import RunbookStore
                    runbook_pool = AsyncConnectionPool(
                        conninfo=rb_dsn,
                        min_size=1, max_size=4, open=False,
                        kwargs={"autocommit": False},
                    )
                    await runbook_pool.open()
                    # Reuse the same Qdrant client investigation_cache
                    # uses (already constructed above). Falls back to a
                    # fresh client if that didn't init.
                    rb_qdrant = None
                    inv_cache = app.state.investigation_cache
                    if inv_cache is not None:
                        rb_qdrant = getattr(inv_cache, "_qdrant", None)
                    if rb_qdrant is None:
                        from qdrant_client import AsyncQdrantClient
                        rb_qurl = (
                            cfg.vector_store.url
                            if hasattr(cfg.vector_store, "url") and cfg.vector_store.url
                            else os.environ.get("QDRANT_URL", "http://qdrant:6333")
                        )
                        rb_qdrant = AsyncQdrantClient(url=rb_qurl)
                    app.state.runbook_pool = runbook_pool
                    app.state.runbook_store = RunbookStore(
                        pg_pool=runbook_pool,
                        qdrant=rb_qdrant,
                        embedder=providers.embedder,
                    )
                    _log.info(
                        "runbook_store wired (Postgres pool=1-4, Qdrant collection=opsrag_runbooks_vec)"
                    )
                except Exception as exc:
                    _log.warning("runbook_store init failed: %s", exc)
                    app.state.runbook_store = None

        # Investigation event ledger + runner.
        # Reuses the same Postgres pool as the runbook store (small,
        # only used by the SSE tail-cursor reads which are cheap).
        app.state.investigation_event_store = None
        app.state.investigation_runner = None
        if cfg.session.provider == "postgres" and app.state.runbook_pool is not None:
            try:
                from opsrag.investigations import InvestigationEventStore
                from opsrag.investigations.runner import (
                    InvestigationDeps,
                    InvestigationRunner,
                )
                from opsrag.mcp import ALL_MCP_TOOLS

                inv_store = InvestigationEventStore(pg_pool=app.state.runbook_pool)
                # Decide Pro vs Flash. `model_router` is wired later in
                # this function -- for now read from providers.llm and
                # fall back to llm for both Flash + Pro if no Pro is
                # configured (degenerate but functional).
                flash_llm = providers.llm
                pro_llm = getattr(providers, "pro_llm", None) or providers.llm
                tool_registry = {t.name: t for t in ALL_MCP_TOOLS}
                gitlab_client = getattr(providers, "gitlab_client", None)
                deps = InvestigationDeps(
                    event_store=inv_store,
                    flash_llm=flash_llm,
                    pro_llm=pro_llm,
                    embedder=providers.embedder,
                    runbook_store=app.state.runbook_store,
                    investigation_cache=app.state.investigation_cache,
                    tool_registry=tool_registry,
                    gitlab_client=gitlab_client,
                )
                app.state.investigation_event_store = inv_store
                app.state.investigation_runner = InvestigationRunner(deps)
                _log.info(
                    "investigation_event_store + runner wired (tools=%d)",
                    len(tool_registry),
                )
            except Exception as exc:  # noqa: BLE001
                _log.warning("investigation runner init failed: %s", exc)

        # Surface the investigate feature-gate state at boot. The Investigate
        # tab only appears in the UI when >=1 live-telemetry MCP (datadog /
        # prometheus / kubernetes / ...) is enabled, so a config flip that
        # silently hides the tab is otherwise hard to diagnose.
        try:
            from opsrag.investigations.feature_gate import (
                investigation_live_telemetry_enabled,
            )

            _gate = investigation_live_telemetry_enabled(cfg)
            _log.info(
                "investigate feature gate: live telemetry %s -> Investigate tab %s",
                "enabled" if _gate else "disabled",
                "VISIBLE" if _gate else "HIDDEN (enable a datadog/prometheus/k8s MCP)",
            )
        except Exception as exc:  # noqa: BLE001
            _log.debug("feature-gate boot log skipped: %s", exc)

        # Pomerium identity verifier + user store. The verifier is
        # only constructed when tracking is enabled AND a JWKS URL is
        # configured. When verifier/user_store are None, the auth
        # dependency falls through to CurrentUser.anonymous().
        app.state.pomerium_verifier = None
        app.state.tracking_user_config = cfg.tracking_user
        user_store: UserStore | None = None
        app.state.user_store = None
        if cfg.tracking_user.enabled and cfg.tracking_user.pomerium_jwks_url:
            try:
                app.state.pomerium_verifier = PomeriumVerifier(
                    jwks_url=cfg.tracking_user.pomerium_jwks_url,
                    expected_audience=cfg.tracking_user.pomerium_audience,
                )
                _log.info(
                    "pomerium identity enabled (jwks=%s, require_auth=%s)",
                    cfg.tracking_user.pomerium_jwks_url,
                    cfg.tracking_user.require_auth,
                )
            except Exception as exc:
                _log.warning("pomerium verifier init failed: %s", exc)
            if cfg.session.provider == "postgres":
                us_dsn = cfg.session.dsn or os.environ.get(cfg.session.dsn_env, "")
                if us_dsn:
                    try:
                        user_store = UserStore(dsn=us_dsn)
                        await user_store.open()
                        await user_store.init_schema()
                        app.state.user_store = user_store
                    except Exception as exc:
                        _log.warning("user_store init failed: %s", exc)
                        user_store = None
        else:
            _log.info("tracking_user disabled -- running in anonymous mode")

        # Wire Vertex on_usage hook so per-call token usage is
        # persisted with the current user_oid (read from ContextVar set
        # by the query handler). Falls back gracefully when usage
        # persistence isn't available.
        if usage_persist is not None:
            def _on_vertex_usage(result: VertexResult) -> None:
                try:
                    usage_persist.enqueue(  # type: ignore[union-attr]
                        model=result.model,
                        input_tokens=result.prompt_tokens,
                        output_tokens=result.completion_tokens,
                        latency_ms=getattr(result, "latency_ms", 0.0),
                        session_id=None,
                        purpose="generation",
                        user_oid=current_user_oid_var.get(),
                    )
                except Exception as exc:
                    _log.debug("on_vertex_usage hook failed: %s", exc)
            try:
                if isinstance(providers.llm, VertexAILLM):
                    providers.llm.set_on_usage(_on_vertex_usage)
                if pro_llm is not None and isinstance(pro_llm, VertexAILLM):
                    pro_llm.set_on_usage(_on_vertex_usage)
            except Exception as exc:
                _log.warning("Vertex on_usage hook wiring failed: %s", exc)

        # MCP server-as-proxy. Bearer-token + Pomerium-mediated
        # external tool surface. Token store + audit share the Postgres
        # DSN; rate limiter is in-process only. Graceful degradation:
        # any init failure leaves the routes to 503 rather than breaking
        # startup.
        app.state.mcp_token_store = None
        app.state.mcp_audit = None
        app.state.mcp_rate_limiter = TokenRateLimiter()
        app.state.mcp_server = None
        mcp_token_store: MCPTokenStore | None = None
        mcp_audit: AuditLogger | None = None
        if cfg.session.provider == "postgres":
            mcp_dsn = cfg.session.dsn or os.environ.get(cfg.session.dsn_env, "")
            if mcp_dsn:
                try:
                    mcp_token_store = MCPTokenStore(dsn=mcp_dsn)
                    await mcp_token_store.open()
                    app.state.mcp_token_store = mcp_token_store

                    mcp_audit = AuditLogger(dsn=mcp_dsn)
                    await mcp_audit.open()
                    mcp_audit.start()
                    app.state.mcp_audit = mcp_audit

                    app.state.mcp_server = MCPServer(
                        rate_limiter=app.state.mcp_rate_limiter,
                        audit=mcp_audit,
                    )
                    _log.info(
                        "mcp-server-as-proxy wired (tools=%d)",
                        len(app.state.mcp_server._tools),
                    )
                except Exception as exc:
                    _log.warning("mcp_server init failed: %s", exc)
                    mcp_token_store = None
                    mcp_audit = None

        if isinstance(providers.graph_store, Neo4jGraphStore):
            await providers.graph_store.ensure_indexes()

        checkpointer = providers.session_store.get_checkpointer()

        # Pass the configured repo list down so the retriever can detect when
        # a query names a specific repo and scope the search.
        known_repos = cfg.scm.repo_names()

        # Loud warning if the light-graph lane is enabled but the active mode
        # can't run it -- this is the "silently-dead graph for months" failure
        # the Neo4j fail-fast guard was written to prevent, one layer up.
        if providers.light_graph is not None and cfg.agent.mode == "minimal":
            _log.warning(
                "light_graph is ENABLED but agent.mode=minimal has no "
                "entity_expand node -- the graph lane is DEAD (edges + entity_ids "
                "are still computed at index time but nothing reads them). Use "
                "mode=full/multi_agent/tool_calling, or disable light_graph."
            )
        agent_graph = _build_agent_graph(
            cfg, providers, checkpointer, known_repos, model_router,
        )

        app.state.config = cfg
        app.state.providers = providers
        app.state.ingestion_pipeline = IngestionPipeline(
            scm=providers.scm,
            parsers=providers.parsers,
            chunker=providers.chunker,
            embedder=providers.embedder,
            vector_store=providers.vector_store,
            graph_store=providers.graph_store,
            entity_extractor=providers.entity_extractor,
            llm=providers.llm,  # for contextual chunking when enabled
            indexed_files=providers.indexed_files,
            sources=providers.sources,
            # Optional code lane; both None when feature disabled.
            code_embedder=providers.code_embedder,
            code_vector_store=providers.code_vector_store,
            light_graph=providers.light_graph,
        )
        app.state.agent_graph = agent_graph
        app.state.session_store = providers.session_store

        # Optional Q&A semantic cache. Reuses the Qdrant client +
        # embedding dimension from the vector store. Toggle with
        # OPSRAG_QA_CACHE env var (default on).
        app.state.qa_cache = None
        if qa_cache_enabled() and hasattr(providers.vector_store, "_client"):
            # Thread the QA-cache precision knobs (M4): widen the LLM-judge band
            # and enable the spaCy NER entity-swap guard by default. configure()
            # sets the module defaults that the env vars still override.
            from opsrag import qa_cache_judge as _qa_judge
            from opsrag import qa_cache_ner as _qa_ner
            _qa_judge.configure(
                qa_judge_upper=cfg.qa_cache.qa_judge_upper,
                qa_judge_fail_open=cfg.qa_cache.qa_judge_fail_open,
            )
            _qa_ner.configure(qa_ner_guard=cfg.qa_cache.qa_ner_guard)
            app.state.qa_cache = QAVectorCache(
                client=providers.vector_store._client,
                dimension=providers.embedder.dimension,
            )
            _log.info("qa_cache enabled (collection=%s)", app.state.qa_cache._collection)

        # Feedback-as-correction store. Writes synthetic
        # Q+A chunks into the main retrieval collection with
        # priority="user-correction" (2.5x boost). Shares the Qdrant
        # client + embedder with the regular vector store. Wired only
        # when both are available; otherwise the /correction endpoint
        # returns 503 gracefully.
        app.state.correction_store = None
        if hasattr(providers.vector_store, "_client") and providers.embedder is not None:
            try:
                app.state.correction_store = CorrectionStore(
                    qdrant=providers.vector_store._client,
                    embedder=providers.embedder,
                    collection_name=cfg.vector_store.collection,
                )
                _log.info(
                    "correction_store enabled (collection=%s, boost=1.8x, moderated)",
                    cfg.vector_store.collection,
                )
            except Exception as exc:
                _log.warning("correction_store init failed: %s", exc)
                app.state.correction_store = None

        # Moderation queue for corrections. POST /correction enqueues here
        # (pending, invisible to retrieval); an operator approves before the
        # boosted chunk is injected via correction_store. Reuses POSTGRES_DSN.
        app.state.pending_correction_store = None
        try:
            from opsrag.pending_corrections import PendingCorrectionStore
            pc_dsn = cfg.session.dsn or os.environ.get(cfg.session.dsn_env, "")
            if pc_dsn:
                pc_store = PendingCorrectionStore(dsn=pc_dsn)
                await pc_store.open()
                app.state.pending_correction_store = pc_store
                _log.info("pending_correction_store enabled (moderation queue)")
        except Exception as exc:
            _log.warning("pending_correction_store init failed: %s", exc)
            app.state.pending_correction_store = None

        # Bind the corpus to the `knowledge_search` MCP tool
        # so the multi-agent tool-loop can pull runbook/policy chunks
        # mid-flight (e.g. after a Slack URL fetch reveals an SRE access
        # request, chain into the SRE-KB docs for next-step guidance).
        #
        # The entity-anchored retrieval lane (Neo4j as a 3rd RRF lane) is
        # optional: when graph_store is None the tool transparently falls
        # back to 2-lane hybrid; when llm is None the entity extractor uses
        # rule-based-only (no LLM fallback for fuzzy mentions).
        try:
            from opsrag.mcp import bind_knowledge
            # The `graph_store=` and `llm=` kwargs were dropped after the
            # cartography refactor stripped them from `knowledge.bind()`.
            # Passing the old call shape would raise
            # `TypeError: bind() got an unexpected keyword argument 'graph_store'`,
            # which the except below would swallow, leaving _embedder None so
            # every knowledge_search call returns "not configured".
            bind_knowledge(
                providers.embedder,
                providers.vector_store,
                # Optional code-specific lane. Both None unless config
                # specified code_embedding + code_vector_store blocks.
                code_embedder=providers.code_embedder,
                code_vector_store=providers.code_vector_store,
                # Enables LLM-driven multi-query decomposition WHEN the operator
                # sets OPSRAG_DECOMPOSE_QUERIES=1 (no-op otherwise).
                llm=providers.llm,
                # Rerank the tool path too (was raw bi-encoder order before).
                reranker=providers.reranker,
                # Forward the SAME rerank-enrichment config the LangGraph
                # rerank_node gets (path-anchor boost / MMR / content-dedup),
                # so the tool path can't re-diverge from the graph path (M1).
                rerank_diversity=cfg.agent.rerank_diversity,
                rerank_content_dedup=cfg.agent.rerank_content_dedup,
                rerank_content_dedup_threshold=cfg.agent.rerank_content_dedup_threshold,
            )
            _log.info("knowledge_search MCP tool bound to corpus")
        except Exception as exc:  # noqa: BLE001
            _log.warning("knowledge_search bind failed: %s", exc)

        # Lazy-clone for code_* MCP tools.
        # Backend pods don't share the indexer's clone cache (no RWX
        # PVC), so without this the agent's code-exploration tools
        # return "repo not in cache" for every query on a fresh pod.
        # GitCloneSCM is the same provider the indexer uses; binding
        # it here lets the tools shallow-clone configured repos on
        # first miss into the pod's emptyDir. Allowlist is the same
        # SCMConfig.repos_with_branch() list -- agents can't ask us to
        # clone arbitrary external repos.
        try:
            from opsrag.mcp import bind_code_scm
            if hasattr(providers.scm, "_ensure_cloned"):
                bind_code_scm(
                    providers.scm,
                    cfg.scm.repos_with_branch(),
                )
                _log.info("code_* MCP tools bound for lazy-clone")
            else:
                _log.info(
                    "code_* MCP tools NOT bound for lazy-clone "
                    "(SCM provider %s lacks _ensure_cloned -- expected "
                    "with non-git providers)",
                    type(providers.scm).__name__,
                )
        except Exception as exc:  # noqa: BLE001
            _log.warning("code_scm bind failed: %s", exc)

        # Auto-warm code clone cache as a background task. Backend pods
        # use emptyDir for /tmp/opsrag-repos/, so the cache is empty
        # on every pod start. Without this, the FIRST code_* query
        # against each repo pays a 2-30s clone cost. We fire-and-forget
        # so /health -> Ready is immediate; in-flight queries still hit
        # the lazy-clone fallback in opsrag.mcp.code while warming.
        app.state.code_cache_task = None
        if (
            cfg.code_cache.prewarm_on_start
            and hasattr(providers.scm, "warm_repo_cache")
        ):
            async def _warm_code_cache() -> None:
                try:
                    repo_pairs_local = cfg.scm.repos_with_branch()
                    _log.info(
                        "code-cache prewarm starting (repos=%d, concurrency=%d)",
                        len(repo_pairs_local), cfg.code_cache.concurrency,
                    )
                    await providers.scm.warm_repo_cache(
                        repo_pairs_local,
                        concurrency=cfg.code_cache.concurrency,
                    )
                except Exception as exc:  # noqa: BLE001
                    _log.warning("code-cache prewarm failed: %s", exc, exc_info=True)
            app.state.code_cache_task = asyncio.create_task(
                _warm_code_cache(), name="opsrag-code-cache-prewarm",
            )
        elif cfg.code_cache.prewarm_on_start:
            _log.info(
                "code-cache prewarm SKIPPED -- SCM provider %s lacks warm_repo_cache",
                type(providers.scm).__name__,
            )

        # Semantic-router classifier (forensic / live /
        # procedural / mixed). Embed reference anchors once at startup;
        # at request time, the router reuses the query embedding the
        # retrieval path already computes -- runtime cost is one in-process
        # cosine sweep over ~30 anchors. Failures are non-fatal -- the
        # classifier silently degrades to legacy regex-only behaviour.
        app.state.semantic_router = None
        if os.environ.get("OPSRAG_CLASSIFIER", "1").lower() in ("1", "true", "yes", "on"):
            try:
                from opsrag.agent.classifier import SemanticRouter
                router = SemanticRouter(embedder=providers.embedder)
                await router.fit()
                app.state.semantic_router = router
                _log.info("semantic router fitted (forensic/live/procedural)")
            except Exception as exc:
                _log.warning("semantic router fit failed: %s -- classifier degraded", exc)

        # Auto-index configured repos in the background
        from opsrag.indexing_tracker import indexing_tracker
        index_task = None
        repo_pairs = cfg.scm.repos_with_branch()

        # Role gate -- production splits backend (query/serve, N replicas) and
        # indexer (auto-index + scheduler, 1 replica) into two deployments off
        # the same image. Set OPSRAG_ROLE=backend on serving pods to suppress
        # the auto-index loop and APScheduler; OPSRAG_ROLE=indexer (or unset
        # for local dev) keeps the legacy behavior of running both.
        _role = (os.environ.get("OPSRAG_ROLE") or "").strip().lower()
        _is_indexer = _role in ("", "indexer")
        if _role == "backend":
            _log.info("OPSRAG_ROLE=backend -- skipping auto-index loop + daily scheduler")
        elif _role == "indexer":
            _log.info("OPSRAG_ROLE=indexer -- running auto-index loop + daily scheduler")
        else:
            _log.info("OPSRAG_ROLE unset -- running auto-index loop + daily scheduler (legacy / dev mode)")

        # Backfill the in-memory tracker from what's already in Qdrant
        # so the dashboard reflects existing index state after every
        # restart. Without this, Confluence disappears entirely (no
        # config-side auto-register) and git repos show 0 chunks
        # because the tracker only counts chunks from the current
        # process run. Runs in the background so we don't block startup.
        repo_to_branch = {r: b for r, b in repo_pairs}

        async def _backfill_indexing_state() -> None:
            try:
                if not hasattr(providers.vector_store, "_client"):
                    return
                from collections import Counter
                client = providers.vector_store._client
                chunks_per_repo: Counter[str] = Counter()
                files_per_repo: dict[str, set[str]] = {}
                cursor = None
                # Walk until exhausted -- collection can hit ~1M points
                # once Slack archive grows. Safety cap at 1500 pages
                # (3M chunks) so a runaway never blocks startup.
                # Use the configured collection name -- was hardcoded "opsrag"
                # which 404s once the config moves to a non-default collection
                # (e.g. "opsrag_v2"). The bug silently shows "0 chunks in
                # vector store" in the UI even when Qdrant has data.
                collection_name = cfg.vector_store.collection
                for _ in range(1500):
                    points, cursor = await client.scroll(
                        collection_name=collection_name,
                        limit=2000,
                        offset=cursor,
                        with_payload=["repo", "source_path"],
                        with_vectors=False,
                    )
                    for p in points:
                        pl = p.payload or {}
                        r = pl.get("repo")
                        sp = pl.get("source_path")
                        if not r:
                            continue
                        chunks_per_repo[r] += 1
                        if sp:
                            files_per_repo.setdefault(r, set()).add(sp)
                    if not cursor:
                        break

                # Real branch per repo from the indexed_files table (Qdrant
                # payloads don't carry branch). Prevents mislabeling repos
                # indexed on `master` (e.g. genapp) as `main` -> no duplicate
                # rows in /indexing/status. Best-effort; empty on no-Postgres.
                idx_branches: dict[str, str] = {}
                _idx = getattr(providers, "indexed_files", None)
                if _idx is not None and hasattr(_idx, "repo_branches"):
                    try:
                        idx_branches = await _idx.repo_branches()
                    except Exception:
                        idx_branches = {}

                for repo, chunks in chunks_per_repo.items():
                    if repo.startswith(("confluence:", "rootly:", "slack:")):
                        source_type = repo.split(":", 1)[0]
                        branch = source_type
                    else:
                        source_type = "git"
                        # Config wins; else the actually-indexed branch; else main.
                        branch = (repo_to_branch.get(repo)
                                  or idx_branches.get(repo) or "main")
                    indexing_tracker.backfill_done(
                        repo=repo,
                        branch=branch,
                        source_type=source_type,
                        total_chunks=chunks,
                        total_files=len(files_per_repo.get(repo, [])),
                    )

                # Resolve display names for Slack channels via the Slack
                # API. Cheap (1 conversations.info per channel) and
                # produces a readable label without depending on what
                # the chunker happens to leave in the Qdrant payload.
                slack_source = (providers.sources or {}).get("slack")
                if slack_source is not None:
                    for repo in list(chunks_per_repo):
                        if not repo.startswith("slack:"):
                            continue
                        channel_id = repo.split(":", 1)[1]
                        try:
                            info = await slack_source._client.get_channel(channel_id)
                            indexing_tracker.set_display_name(
                                repo, "slack", f"slack:#{info.name}",
                            )
                        except Exception as exc:
                            _log.debug(
                                "slack name lookup failed for %s: %s",
                                channel_id, exc,
                            )
                # Persist the restored state to Postgres (guarded -- never
                # stomps a live Job's row) so backend pods reflect pre-existing
                # index content even before any new Job runs.
                if app.state.index_store is not None:
                    try:
                        await app.state.index_store.backfill_upsert(
                            indexing_tracker.get_summary().get("repos", []),
                            indexing_tracker.get_jobs().get("jobs", []),
                        )
                    except Exception as exc:
                        _log.warning("index-state backfill persist failed: %s", exc)
                _log.info(
                    "indexing tracker backfilled: repos=%d total_chunks=%d",
                    len(chunks_per_repo), sum(chunks_per_repo.values()),
                )
            except Exception as exc:
                _log.warning("indexing tracker backfill failed: %s", exc)

        backfill_task = asyncio.create_task(_backfill_indexing_state())

        # Writer roles flush the in-memory tracker into Postgres on a throttle
        # so backend pods see near-real-time progress without per-file DB
        # writes. A "writer" is any role that can run indexing in-process: the
        # `api`/`indexer`/dev roles (POST /index/repo runs in-process) and the
        # job-indexer Job. Only the pure-serving `backend` role is read-only.
        # NOTE: the entrypoint defaults OPSRAG_ROLE to "api", so gating on
        # `_is_indexer` (== role in {"", "indexer"}) wrongly excluded the
        # common compose/dev case -- gate on "not backend" instead.
        _is_index_writer = _role != "backend"

        # Indexing trigger: in production POST /index/repo creates an ephemeral
        # k8s Job (the API stays pure-serving). In dev / no-cluster the launcher
        # is None and the routes run indexing in-process (legacy behaviour).
        # Initialised BEFORE the flush loop so we know whether a separate Job
        # writer can exist -- that decides whether the serving pod's flush must
        # be guarded (see below).
        try:
            from opsrag.job.launcher import JobLauncher
            app.state.job_launcher = JobLauncher.from_env()
            if app.state.job_launcher is not None:
                _log.info("indexing trigger: k8s Job launcher active")
            else:
                _log.info("indexing trigger: in-process (no Job launcher)")
        except Exception as exc:  # noqa: BLE001
            _log.warning("job launcher init failed (%s); using in-process indexing", exc)
            app.state.job_launcher = None

        app.state._index_flush_task = None
        app.state._index_flush_stop = None
        if _is_index_writer and app.state.index_store is not None:
            from opsrag.indexing.pg_store import flush_loop
            # When a Job launcher is active, indexing runs in a separate
            # ephemeral Job that owns the live `listing`/`indexing` PROGRESS
            # rows. The serving pod must NOT flush those rows back to `done`
            # with its own stale (restored-from-Qdrant) counts -> guard it.
            # Dev/in-process (no launcher) runs indexing here, so it stays
            # unguarded and can advance its own `indexing` -> `done`.
            _flush_guarded = app.state.job_launcher is not None
            stop_event = asyncio.Event()
            app.state._index_flush_stop = stop_event
            app.state._index_flush_task = asyncio.create_task(
                flush_loop(app.state.index_store, indexing_tracker,
                           stop_event=stop_event, guarded=_flush_guarded)
            )
            _log.info(
                "indexing job-state flush loop started (writer role, guarded=%s)",
                _flush_guarded,
            )

        if _is_indexer and cfg.scm.auto_index and repo_pairs:
            pipeline = app.state.ingestion_pipeline

            for repo, branch in repo_pairs:
                indexing_tracker.queue_repo(repo, branch)

            # Index multiple repos in parallel. Vertex embeddings + Discovery
            # Engine ranker are I/O bound, so concurrency is a big win. Cap at
            # 3 to stay under Vertex per-minute token quotas. The embedder
            # itself retries 429s with backoff, but lower concurrency reduces
            # contention. Override via OPSRAG_INDEX_PARALLEL env var.
            parallel_limit = max(
                1,
                min(int(os.environ.get("OPSRAG_INDEX_PARALLEL", "3")), len(repo_pairs)),
            )
            sem = asyncio.Semaphore(parallel_limit)

            async def _index_one(repo: str, branch: str) -> None:
                async with sem:
                    try:
                        _log.info("auto-index starting repo=%s branch=%s", repo, branch)
                        count = await pipeline.index_repo(repo, branch=branch)
                        _log.info("auto-index done repo=%s chunks=%d", repo, count)
                    except Exception as exc:
                        _log.warning("auto-index failed repo=%s: %s", repo, exc)
                        indexing_tracker.repo_failed(repo, branch, str(exc))

            async def _auto_index():
                _log.info(
                    "auto-index batch start: %d repos, parallel_limit=%d",
                    len(repo_pairs), parallel_limit,
                )
                await asyncio.gather(
                    *(_index_one(repo, branch) for repo, branch in repo_pairs),
                    return_exceptions=False,
                )
                _log.info("auto-index batch done")

            index_task = asyncio.create_task(_auto_index())

        # Daily indexing scheduler (APScheduler).
        # Independent of the startup auto-index above; that runs once per
        # process lifecycle, this fires once per day on cron.
        scheduler = None
        # Confluence spaces are added to the daily run if the
        # connector is enabled AND a non-empty allowlist is configured.
        # Personal `~space` keys are filtered defensively even if they
        # somehow ended up in the allowlist (the connector's own check
        # also blocks them).
        confluence_scopes: list[tuple[str, str]] = []
        if cfg.confluence.enabled and "confluence" in (providers.sources or {}):
            confluence_scopes = [
                ("confluence", k)
                for k in cfg.confluence.spaces_allowlist
                if k and not k.startswith("~")
            ]
        slack_scopes: list[tuple[str, str]] = []
        if cfg.slack.enabled and "slack" in (providers.sources or {}):
            slack_scopes = [
                ("slack", channel_id)
                for channel_id in cfg.slack.channels_allowlist
                if channel_id
            ]
        rootly_scopes: list[tuple[str, str]] = []
        if cfg.rootly.enabled and "rootly" in (providers.sources or {}):
            # Rootly is single-tenant per token -- one synthetic scope.
            rootly_scopes = [("rootly", cfg.rootly.scope)]
        # Investigation-history source: synthetic
        # single scope ("opsrag") since investigations live in one global
        # collection.
        investigation_scopes: list[tuple[str, str]] = []
        if (
            cfg.investigation_history.enabled
            and "investigation-history" in (providers.sources or {})
        ):
            investigation_scopes = [("investigation-history", "opsrag")]
        source_scopes = (
            confluence_scopes + slack_scopes + rootly_scopes + investigation_scopes
        )

        if _is_indexer and cfg.scheduler.enabled and (repo_pairs or source_scopes):
            from functools import partial

            from opsrag.scheduler import build_scheduler, daily_index_job

            pipeline = app.state.ingestion_pipeline
            job_callable = partial(
                daily_index_job,
                repo_pairs,
                pipeline,
                cfg.scheduler.parallel_limit,
                source_scopes,
            )
            scheduler = build_scheduler(cfg.scheduler, job_callable)
            scheduler.start()
            _log.info(
                "scheduler started: daily index at %02d:%02d %s +/-%ds jitter "
                "(repos=%d, source_scopes=%d)",
                cfg.scheduler.cron_hour, cfg.scheduler.cron_minute,
                cfg.scheduler.timezone, cfg.scheduler.jitter_seconds,
                len(repo_pairs), len(source_scopes),
            )
            app.state.scheduler = scheduler
        else:
            app.state.scheduler = None

        # -- Multi-channel chat bots (Slack / Telegram / Discord) --
        # The role-gating fix (design D6): a channel worker boots iff
        # OPSRAG_ROLE maps to a channel AND that channel is enabled.
        # ``build_and_start`` returns the connected adapter for shutdown, or
        # None on the api/backend/indexer roles (no outbound worker). This
        # replaces the old `enabled`-alone Slack boot, which on N api replicas
        # opened N Socket Mode connections = duplicate answers.
        #
        # Bound to the same agent_graph + providers as the HTTP /query
        # endpoint, so a channel worker pod is one process per pod.
        app.state.channel = None
        from types import SimpleNamespace
        caches = SimpleNamespace(
            qa_cache=app.state.qa_cache,
            investigation_cache=app.state.investigation_cache,
            semantic_router=app.state.semantic_router,
            feedback_store=app.state.feedback_store,
        )
        try:
            from opsrag.channels.boot import build_and_start
            app.state.channel = await build_and_start(
                _role, cfg, agent_graph, providers, caches,
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("channel worker startup failed: %s", exc, exc_info=True)

        # Teams is inbound-only: its Bot Framework webhook is mounted on the
        # `api` role (Bot Framework pushes activities to the public ingress the
        # API already has). Only mount it when Teams is enabled.
        if _role in ("", "api") and getattr(cfg.channels.teams, "enabled", False):
            try:
                from opsrag.channels.adapters.teams.router import build_teams_router
                app.include_router(
                    build_teams_router(
                        agent_graph, providers, caches, cfg.channels.teams,
                        getattr(cfg, "vision", None),
                    )
                )
                _log.info("teams webhook router mounted on api role (/api/channels/teams)")
            except Exception as exc:  # noqa: BLE001
                _log.warning("teams webhook router mount failed: %s", exc, exc_info=True)

        try:
            yield
        finally:
            if app.state.channel is not None:
                try:
                    await app.state.channel.close()
                except Exception as exc:  # noqa: BLE001
                    _log.warning("channel worker close raised: %s", exc)
            if scheduler is not None:
                # wait=False so shutdown doesn't block on a job in flight;
                # the next process picks up via the persisted jobstore.
                scheduler.shutdown(wait=False)
            if index_task and not index_task.done():
                index_task.cancel()
                try:
                    await index_task
                except asyncio.CancelledError:
                    pass
            # Stop the index-state flush loop (does a final flush in its
            # finally), then close the store.
            flush_stop = getattr(app.state, "_index_flush_stop", None)
            flush_task = getattr(app.state, "_index_flush_task", None)
            if flush_stop is not None:
                flush_stop.set()
            if flush_task is not None and not flush_task.done():
                try:
                    await asyncio.wait_for(flush_task, timeout=5)
                except (TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            if app.state.index_store is not None:
                try:
                    await app.state.index_store.close()
                except Exception as exc:  # noqa: BLE001
                    _log.warning("index-state store close failed: %s", exc)
            # Stop the agent-guidance refresher + close its store.
            ag_stop = getattr(app.state, "_agent_settings_stop", None)
            ag_task = getattr(app.state, "_agent_settings_task", None)
            if ag_stop is not None:
                ag_stop.set()
            if ag_task is not None and not ag_task.done():
                try:
                    await asyncio.wait_for(ag_task, timeout=5)
                except (TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            if getattr(app.state, "agent_settings", None) is not None:
                try:
                    await app.state.agent_settings.close()
                except Exception as exc:  # noqa: BLE001
                    _log.warning("agent-guidance store close failed: %s", exc)
            cache_task = getattr(app.state, "code_cache_task", None)
            if cache_task is not None and not cache_task.done():
                cache_task.cancel()
                try:
                    await cache_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            if hasattr(providers.scm, "close"):
                await providers.scm.close()
            if isinstance(providers.graph_store, Neo4jGraphStore):
                await providers.graph_store.close()
            if isinstance(providers.session_store, PostgresSessionStore):
                await providers.session_store.close()
            if isinstance(providers.indexed_files, PostgresIndexedFilesTracker):
                await providers.indexed_files.close()
            if usage_persist is not None:
                # Detach the live hook BEFORE close so any in-flight
                # record() calls during shutdown go straight to /dev/null
                # instead of refilling a buffer we're about to drop.
                usage_tracker.set_persistence_hook(None)
                await usage_persist.close()
            if feedback_store is not None:
                try:
                    await feedback_store.close()
                except Exception as exc:
                    _log.warning("feedback_store close failed: %s", exc)
            if getattr(app.state, "pending_correction_store", None) is not None:
                try:
                    await app.state.pending_correction_store.close()
                except Exception as exc:
                    _log.warning("pending_correction_store close failed: %s", exc)
            if user_store is not None:
                try:
                    await user_store.close()
                except Exception as exc:
                    _log.warning("user_store close failed: %s", exc)
            if mcp_audit is not None:
                try:
                    await mcp_audit.close()
                except Exception as exc:
                    _log.warning("mcp_audit close failed: %s", exc)
            if mcp_token_store is not None:
                try:
                    await mcp_token_store.close()
                except Exception as exc:
                    _log.warning("mcp_token_store close failed: %s", exc)

    app = FastAPI(
        title="OpsRAG",
        version=__version__,
        description="Agentic GraphRAG for DevOps/SRE",
        lifespan=lifespan,
    )
    # Make the config readable off app.state before lifespan runs (handlers
    # and contract tests read app.state.config).
    app.state.config = cfg

    # Install the deployment context so prompt templates render against the
    # operator's facts (Principle VI). Empty by default -> org-free prompts.
    from opsrag.agent.prompt_render import set_active_deployment
    set_active_deployment(cfg.deployment)

    # Fail fast (FR-004) if any enabled MCP is missing its required env /
    # config. Raises MCP_MISCONFIGURED:<name>:<missing> before serving.
    from opsrag.mcp.registry import validate_enabled_mcps
    validate_enabled_mcps(cfg)

    # Gate the agent's MCP tools to the operator-enabled integrations (T091).
    # Default config has all 14 disabled -> the agent sees zero MCP tools and
    # answers from the corpus only (US1), until an operator enables some (US2).
    from opsrag.mcp_server.registry_loader import (
        enabled_integration_names,
        set_active_enabled,
    )
    set_active_enabled(enabled_integration_names(cfg))

    # Stable error envelope on every non-2xx response (contracts/http-api.md).
    register_error_handlers(app)

    # Auth (FR-016): authentication is ALWAYS enforced -- there is no
    # anonymous / "open" mode. The global middleware rejects unauthenticated
    # requests on every route except the public allowlist (health/metadata,
    # /auth/*, SCM webhooks, MCP wire endpoints). `oidc` builds a Bearer
    # verifier from the `auth` block (construction is cheap -- JWKS discovery
    # is lazy on first verify); `login` (default) enforces the first-party
    # session cookie. The legacy X-API-Key gate is removed.
    _auth_mode = cfg.auth.mode
    if _auth_mode == "oidc":
        app.state.oidc_verifier = build_verifier_from_settings(cfg.auth)
        _log.info("auth: OIDC enforcement enabled (issuer=%s)", cfg.auth.issuer)
    else:
        app.state.oidc_verifier = None
        _log.info("auth: login mode (password + SSO); first-party sessions")
    # RBAC: expose auth mode + role->scope mappings so opsrag.auth.scopes can
    # resolve each request's roles/scopes.
    app.state.auth_config = cfg.auth
    app.state.role_mappings = cfg.auth.role_mappings
    # mode='login': register the first-party login/SSO router. Imported
    # lazily so oidc deployments don't need the `login` extra (authlib,
    # pwdlib, ...). The login-mode runtime state (SessionManager, user store,
    # SSO registry, rate limiter) is built in the lifespan (needs async open).
    if _auth_mode == "login":
        from opsrag.auth.login import router as login_router
        app.include_router(login_router)
        # Authlib's OAuth dance stashes the per-flow state/nonce in
        # request.session, so SessionMiddleware must be installed. We sign it
        # with the same session signing key; SameSite=lax so the cookie
        # survives the top-level GET redirect back from the IdP, short max_age
        # since it only needs to live for the round-trip. This is separate
        # from our first-party opsrag_session login cookie.
        try:
            from starlette.middleware.sessions import SessionMiddleware

            from opsrag.auth.sessions import load_signing_key as _load_sk
            _lc = cfg.auth.login
            _oauth_key = _load_sk(
                key_path=_lc.signing_key_path, key_env=_lc.signing_key_env
            )
            app.add_middleware(
                SessionMiddleware,
                secret_key=_oauth_key,
                session_cookie="opsrag_oauth",
                same_site="lax",
                https_only=bool(_lc.cookie_secure),
                max_age=600,
            )
        except Exception as exc:  # noqa: BLE001
            _log.error("auth: SSO SessionMiddleware setup failed (%s)", exc)
    app.add_middleware(OIDCAuthMiddleware)
    # Rate-limit backend (FLAG-GATED): "memory" (default) keeps in-process
    # state; "redis" shares it across replicas and is REQUIRED -- this fails
    # fast here if Redis is unreachable. The same backend is reused for the
    # login lockout (built in the lifespan) so both honor one shared store.
    rate_limit_backend = _build_rate_limit_backend(cfg)
    app.state.rate_limit_backend = rate_limit_backend
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=cfg.api.rate_limit_rpm,
        enabled=cfg.api.rate_limit_enabled,
        backend=rate_limit_backend,
    )

    # Health/readiness first so the allowlist paths are always present.
    app.include_router(health_router)
    # SCM push webhooks (HMAC/secret-authed; bypass OIDC).
    from opsrag.api.routes_webhooks import webhooks_router
    app.include_router(webhooks_router)
    app.include_router(router)

    # (The legacy hypothesis-tree investigation engine was retired; the
    # event-driven InvestigationRunner -- mounted below as
    # investigations_router -- owns all /investigations/* routes.)

    # MCP-server-as-proxy. Token management (Pomerium-authed) +
    # MCP transport (bearer-token-authed) endpoints under /api/mcp/*.
    from opsrag.api.mcp_routes import router as mcp_router
    app.include_router(mcp_router)

    # Hand-authored runbook CRUD + promote-from-investigation.
    from opsrag.api.routes_runbooks import runbooks_router
    app.include_router(runbooks_router)

    # Event-driven Investigate-mode pipeline.
    from opsrag.api.routes_investigations import investigations_router
    app.include_router(investigations_router)

    # Read-only browse of shared-channel (Slack/Discord/Teams/Telegram-group)
    # conversations -- scope-gated, never exposes private DMs or web threads.
    from opsrag.api.routes_channels import channels_router
    app.include_router(channels_router)

    # Admin RBAC: list users + assign roles (login-mode user management).
    from opsrag.api.routes_admin_users import admin_users_router
    app.include_router(admin_users_router)
    return app


app = create_app()
