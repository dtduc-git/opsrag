# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_Nothing yet._

## [0.2.0] - 2026-06-15

### Security

- **Chat-bot DMs are now deny-by-default.** Previously any user who found a bot
  could DM it and query internal data (DMs bypassed the allowlist; only the
  per-user quota applied). Each channel now has a `dm_allowlist` (platform user
  ids): an empty list denies every DM, listed ids are allowed, and `["*"]`
  re-opens DMs to anyone. Unauthorized DMs are denied **silently** (logged, no
  reply, so the bot's existence isn't confirmed). Applies uniformly to
  Slack/Telegram/Discord/Teams via `ChannelPermission`.

### Added

- **Browse shared-channel conversations in the web UI** — a read-only
  **Channels** page lets any authenticated user (scope `chat`) list and read
  conversations that happened in shared chat channels (Slack/Discord/Teams
  channels, Telegram groups) via `GET /channels/conversations` and
  `GET /channels/conversations/{thread_id}/messages`. Privacy is enforced by the
  `thread_id` prefix: only `<platform>-thread:` sessions are exposed; private 1:1
  DMs (`<platform>-dm:`) and web threads are excluded server-side (the messages
  endpoint 404s on any non-`-thread:` id). Channel users stay anonymous — only
  the platform is shown — and the surface is read-only (no reply/continue/delete).
  See [`docs/channels.md`](docs/channels.md).

## [0.1.1] - 2026-06-15

First **complete** public release. (The `v0.1.0` tag's release build failed on
the UI image — a stale `ui/package-lock.json` plus an arm64/QEMU crash — so
`v0.1.1` is the first fully-published tag.) Published by the `release` workflow:
the multi-arch (amd64+arm64) backend image `ghcr.io/dtduc-git/opsrag`, the
amd64 UI image `ghcr.io/dtduc-git/opsrag-ui`, and the Helm chart
`oci://ghcr.io/dtduc-git/charts/opsrag`.

### Added

- Initial public open-source fork of the opsrag agentic GraphRAG project,
  released under the Apache License 2.0. The codebase previously lived as a
  vendor-internal toolchain at its founding organization; that organization is
  intentionally not named in shipped artefacts. The public fork has:
  - Removed every organization-specific identifier, hostname, account ID,
    Slack channel ID, and runbook URL from shipped code and configuration.
  - Translated every non-English prompt, log message, and comment to English.
  - Placed each of the **twenty** read-only MCP integrations (`aws`, `azure`,
    `cloudflare`, `code`, `datadog`, `elasticsearch`, `gcp`, `github`,
    `gitlab`, `grafana`, `knowledge`, `kubernetes`, `loki`, `prometheus`,
    `rootly`, `runbooks`, `sentry`, `slack`, `splunk`, `tool_cache`) behind an
    explicit `mcp.<name>.enabled` flag whose default is `false`. Missing
    credentials on an enabled integration cause a named, fail-fast startup
    error (`MCP_MISCONFIGURED:<name>:<env>`).
  - Replaced the upstream Pomerium-specific JWT path with a generic OIDC
    Bearer-token middleware; a bundled local Dex issuer keeps the
    fifteen-minute new-evaluator bring-up.
  - Introduced a built-in null knowledge-graph backend so a minimal deployment
    needs only an LLM key, a vector store, and the OIDC issuer — no Neo4j.
  - Reworked the Helm chart (`deploy/helm/opsrag/`) to standard conventions
    (`values.schema.json`, NetworkPolicy, PDB, `helm test`, NOTES.txt) and
    exposed every MCP flag through `values.yaml`.
  - Added an automated vendor-neutrality audit (`scripts/audit-vendor-neutrality.sh`)
    that scans for proprietary names, non-English text, and hardcoded hosts; CI
    fails on any violation.
  - Introduced a `DeploymentContext` model: the engine carries no organization
    knowledge; operator-supplied facts render into prompts at runtime.
  - Shipped a synthetic sample corpus (the fictional "Acme Notes" product) plus
    a local-filesystem indexer and `scripts/seed-sample-corpus.sh`.
  - Added a fake backend + offline test for every MCP integration, and an
    end-to-end investigation test driven by a scripted LLM.
- **Authentication & RBAC.** Three auth modes — `open` / `oidc` / `login` —
  with first-party email+password login, SSO (Google / GitHub / Microsoft
  Entra), cookie sessions + rotating refresh tokens, and a four-scope role
  model (`chat` / `investigate` / `mcp` / `admin`) surfaced in a Users & Roles
  admin view. The bootstrap admin is seeded from `OPSRAG_ADMIN_EMAIL` /
  `OPSRAG_ADMIN_PASSWORD`.
- **Centralized read-only MCP server** (`/api/mcp/sse`, `/api/mcp/messages`,
  `/api/mcp/tokens`): one governed MCP endpoint for external clients (Claude
  Code, Cursor, ...) with per-client tokens, per-tool + global rate limits, and
  a Postgres audit log of every tool call (who/token, tool, args **hash**,
  latency, status) surfaced in an admin **MCP Audit** page.
- **Channel bots for Slack, Telegram, Discord, and Microsoft Teams** — full
  parity with the web UI (streaming progress, thread context, cited answers,
  thumbs up/down feedback, per-channel allowlist + per-user quota). A shared
  transport-agnostic core (`opsrag/channels/`) calls the agent pipeline
  in-process; each platform is a thin adapter over one `ChannelAdapter` port.
  Slack/Telegram/Discord run as role-gated outbound workers
  (`OPSRAG_ROLE=slackbot|telegrambot|discordbot`); Teams is a Bot Framework
  webhook on the `api` role. Configured under a unified `channels:` block (the
  legacy `slack_bot:` block is mirrored into `channels.slack` for back-compat).
  Answers render as platform-native rich text (Telegram HTML, Discord embeds);
  machine-only `diagram-json` blocks are replaced by a callout, and long answers
  are paginated across multiple messages/embeds instead of being truncated. A
  `channels`-profile compose target runs the worker bots locally.
  See [`docs/channels.md`](docs/channels.md).
- **Multi-environment `environments:` registry**: one instance targeting N
  environments' Kubernetes / Prometheus / Elasticsearch, with a per-target
  field-mapping layer (`opsrag/environments.py`).
- **Conversational / operational memory** via Mem0 — per-user cross-session
  recall reusing the existing Qdrant + LLM/embedder, with recursive PII
  redaction and the user's message only (never generated text).
- **Cloud model bundles**: `cloud_provider: aws | gcp` presets for the
  LLM / embedding / reranker / pro-escalation slots; every model id is
  overridable via env or YAML with no rebuild.
- **pgvector** vector-store backend (FTS + a pg_trgm identifier lane) alongside
  Qdrant, and an optional Vertex **code-embedding lane** (dual code collection).
- **Optional bundled Phoenix** LLM observability behind a single Helm toggle
  (`config.observability.enabled`) — flips the config provider and brings a
  Phoenix Deployment up/down with no image rebuild.
- **Per-user usage & cost** tracking — every user sees their own spend (the
  "Mine" view); admins additionally see the org-wide aggregate and per-user
  breakdown.
- Comprehensive user/ops documentation under `docs/` (getting-started,
  configuration, architecture, RAG pipeline, investigations, MCP integrations,
  multi-environment, auth, memory, evaluation, operations, API reference, Helm
  chart) plus ready-made Helm scenario values (GCP / AWS / MCP / multi-env /
  minimal).

### Changed

- Ingestion now runs as ephemeral Kubernetes Jobs cloned from a CronJob, with
  durable Postgres job-state — replacing the always-on indexer pod, so serving
  pods stay pure-serving and `/indexing/status` is consistent across replicas.
- Retrieval pipeline hardened: per-doc-type chars/token ratios (config ~2.5,
  code ~3.5, prose ~4.0) with a reindex-safety warning on change; hybrid
  dense + BM25 + code-lane fusion (RRF); MMR diversity; calibrated per-provider
  reranker score floors; CRAG / Self-RAG with hard budgets; a grounding-gated
  semantic answer cache with a judged similarity band.
- Default code embedder is Cohere Embed v4 on Bedrock; a fail-closed
  embedding-dimension guard refuses to start on a silent dimension mismatch.
- Investigations converged onto a single event-ledger engine
  (`opsrag/investigations/`) with hard budgets (wall-clock / tool-call /
  per-tool timeout), an "absence is not confirmation" evaluator rule,
  per-hypothesis citations, and resumable SSE.
- Conversation listing and usage are scoped per user (admins see all); the
  Investigate tab is feature-gated on having a live-telemetry MCP enabled.
- Configuration fails fast at load: provider values with no factory
  implementation (`ollama`, `weaviate`, `chroma`, `datadog` observability) are
  rejected by the schema instead of crashing at runtime.

### Fixed

- Chunk IDs now hash full content (were a 64-char prefix), eliminating a silent
  stale-vector / collision risk on edits.
- The `cloud_provider` bundle no longer creates a split-brain client set when
  the classic `llm` slot is explicitly pinned to a different provider.
- Many retrieval/eval correctness fixes (CRAG no-op, ungrounded-cache writes,
  BM25 identifier sub-tokenization, classifier edge cases, MMR, NaN guards,
  deleted-file index purge, dimension guards) plus CI lint/unit greening and a
  security CVE bump (aiohttp).
- Per-turn agent state no longer leaks across turns on a `thread_id`: the
  streaming entrypoint (`query_with_session_events`, used by every chat channel
  **and** the web stream) didn't reset `tool_call_count` / tool scratch, so the
  LangGraph checkpointer carried them forward — a busy chat thread (e.g. a
  long-lived DM) hit the tool-loop cap (10) on its first tool call and answered
  with no tools (ungrounded/hallucinated). It now resets them like the
  non-streaming path; a drift-guard test keeps the two entrypoints in sync.
- `knowledge_search` no longer returns zero hits when the reranker yields an
  empty list (e.g. a transient Vertex `:rank` response) — it falls back to the
  pre-rerank candidate pool.
- The triage prompt no longer advertises the removed `cartography_*` tool
  family, and unknown/unbound tool calls no longer consume the agent's
  tool-call budget — eliminating a wasted call + a fabricated-citation risk on
  infra questions.
- Teams adapter supports **single-tenant** Azure Bots (`MicrosoftAppType` +
  tenant id, via `OPSRAG_TEAMS_APP_TYPE` / `OPSRAG_TEAMS_APP_TENANT_ID`).
  Microsoft deprecated multi-tenant bot creation (2025); the adapter previously
  hardcoded `MultiTenant`. Defaults remain `MultiTenant` for existing bots.

### Removed

- The disabled hypothesis-tree investigation engine (`opsrag/agents/investigation/`,
  "Engine A") — superseded by the event-ledger `opsrag/investigations/` engine.

### Security

- Per-session conversation ownership: bound to the authenticated principal,
  enforced with 404-not-403 (no existence oracle); three authorization/IDOR
  gaps (resuming another user's `/query` thread, reading any session's usage,
  spoofable investigation-feedback identity) closed.
- Org-wide usage/cost and the MCP audit log are admin-scoped; the entire MCP
  tool surface is read-only by construction (verb allow-lists + output clamps).
- The vendor-neutrality audit gates CI (proprietary names / non-English text /
  hardcoded hosts), with Trivy, gitleaks, CodeQL, and pip-audit scanning.

[Unreleased]: https://github.com/dtduc-git/opsrag/commits/master
