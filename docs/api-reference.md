# API Reference

A grouped reference for the OpsRAG HTTP API — every endpoint with its method, path, auth scope, and a one-line purpose. The routers live in `opsrag/api/` (`routes.py`, `routes_investigations.py`, `routes_health.py`, `routes_webhooks.py`, `mcp_routes.py`, `routes_runbooks.py`, `routes_admin_users.py`, and `opsrag/auth/login.py`) and are mounted in `opsrag/api/server.py`.

## Path prefix and auth model

- **`/api` prefix.** The UI's reverse proxy strips a leading `/api/` before forwarding to FastAPI, so the FastAPI routers register bare paths (`/query`, `/investigations`, ...) while clients call `/api/query`, `/api/investigations`, and so on. This reference lists the **public** `/api/...` path. Health, webhooks, and SSO callbacks are hit directly (no `/api` prefix) — those are called out per group.
- **Scopes.** Authorization is scope-based (`opsrag/auth/scopes.py`): `chat`, `investigate`, `mcp`, `admin`. Handlers gate on scopes via `require_scope(...)`; the UI reads the same model from `/api/me` so nav never drifts from server enforcement.
- **Auth modes.** In **open** mode (default, zero-config) every request is anonymous and carries *all* scopes, so scope gates are transparent. In **oidc** / **login** mode, `require_scope` returns **403** (`{error: forbidden, reason: missing_scope}`) for an authenticated-but-unscoped caller, distinct from the **401** the auth middleware raises for an unauthenticated request. See [./auth.md](./auth.md).
- **Per-session ownership.** Routes under `/api/sessions/*` enforce per-session ownership: the persisted owner is bound to the authenticated `current_user.oid` (never the spoofable request body). A non-owner read/delete returns **404** (not 403, to avoid an existence oracle). Open / anonymous mode does not enforce, and legacy anonymous-owned threads are grandfathered. `GET /api/sessions/{user_id}` additionally overrides the path id with the caller's verified oid (IDOR fix).
- **"Auth: none (gated by middleware)"** below means the handler carries no per-route scope dependency; it relies on the global session/OIDC enforcement middleware. **"Auth: secret"** means the route authenticates with its own shared secret (webhooks, MCP bearer tokens), bypassing OIDC.

## Query / SSE

| Method | Path | Scope | Purpose |
| --- | --- | --- | --- |
| POST | `/api/query` | `chat` | Ask the agent. `stream:true` returns an SSE stream (`status`/`node_start`/`node_end`/`reasoner_token`/`cache_hit`/`chunk`/`render_component`/`done`/`error`/`close`); otherwise a single `QueryResponse`. Binds the session owner to the authenticated identity. |

## Sessions

Per-session ownership enforced (see above).

| Method | Path | Scope | Purpose |
| --- | --- | --- | --- |
| GET | `/api/sessions/{user_id}` | authenticated | List the caller's own sessions (path id is overridden with the verified oid). |
| DELETE | `/api/sessions/{thread_id}` | `chat` | Delete a session you own (404 for non-owners). |
| GET | `/api/sessions/{thread_id}/messages` | authenticated | Replay a thread's message history (404 for non-owners). |

## Index

| Method | Path | Scope | Purpose |
| --- | --- | --- | --- |
| POST | `/api/index/repo` | `admin` | Queue a git repo for indexing (fire-and-forget; spawns a k8s Job in prod, in-process in dev). |
| POST | `/api/index/source` | none (gated by middleware) | Trigger ingestion of a non-git source (e.g. a Confluence space). |
| GET | `/api/indexing/status` | none (gated by middleware) | Current per-source catalog: file/chunk counts and status (durable Postgres job-state). |
| GET | `/api/indexing/jobs` | none (gated by middleware) | Indexing run history, newest-first (start time, duration, status, error). |
| POST | `/api/admin/index/investigation-history` | none (gated by middleware) | On-demand index pass over the investigation-history source. |
| POST | `/api/admin/reaugment/confluence` | none (gated by middleware) | Re-run contextual chunking on Confluence children missing the `[Context: ...]` prefix (`dry_run`, `scope`, `max_docs`). |
| POST | `/api/admin/light-graph/backfill` | `admin` | Activate the entity-expansion lane on existing chunks without re-embedding (optional `repo` scope). |

## Corrections / Feedback

| Method | Path | Scope | Purpose |
| --- | --- | --- | --- |
| POST | `/api/correction` | `chat` | Submit a user-authored correction into the moderation queue (pending; not live until approved). Submitter is the authenticated principal. |
| GET | `/api/corrections/pending` | `admin` | List corrections awaiting operator review. |
| POST | `/api/corrections/{pending_id}/approve` | `admin` | Approve a pending correction — inject the boosted retrieval chunk and mark approved (idempotent). |
| POST | `/api/corrections/{pending_id}/reject` | `admin` | Reject a pending correction (never reaches retrieval). |
| GET | `/api/corrections` | `admin` | List recent *approved* (live) corrections. |
| DELETE | `/api/corrections/{chunk_id}` | `admin` | Remove a single live correction by chunk id. |
| POST | `/api/investigation/{investigation_id}/feedback` | none (gated by middleware) | Attach up/down feedback (+ optional correction) to a cached answer; dual-writes Qdrant cache + Postgres `opsrag_feedback`. |
| GET | `/api/feedback` | none (gated by middleware) | List recent feedback rows for SRE triage (`?direction=-1&limit=50`). |
| POST | `/api/slack/interactivity` | secret (Slack HMAC) | Slack button (up/down) interactivity callback; Slack-signature verified, then dual-writes feedback. |
| GET | `/api/cache/summary` | none (gated by middleware) | Unified summary across the Q&A, investigation, and tool-output caches. |
| POST | `/api/cache/purge` | `admin` | Multi-strategy purge across caches (target = `qa`/`investigation`/`tool`/`all`). |
| GET | `/api/investigation/cache/summary` | none (gated by middleware) | Audit summary of the investigation answer-cache (total, stale, low-quality counts). |

## Investigations

Event-driven `InvestigationRunner` engine (`opsrag/investigations/`); see [./investigations.md](./investigations.md). The **entire** router is gated on the `investigate` scope (router-level dependency). EventSource clients send the session cookie, so the SSE stream resolves the user the same way as the JSON endpoints.

| Method | Path | Scope | Purpose |
| --- | --- | --- | --- |
| POST | `/api/investigations` | `investigate` | Kick off an investigation from `alert_text`; runs in the background, returns `{investigation_id}` immediately. |
| GET | `/api/investigations` | `investigate` | Sidebar listing — most-recent-first lifecycle rows (`?limit=N`). |
| GET | `/api/investigations/{id}` | `investigate` | Full snapshot — lifecycle row + every event to date (for mount/refresh). |
| GET | `/api/investigations/{id}/events` | `investigate` | SSE tail-cursor stream from the Postgres event ledger; reconnect with `?since=<lastSeq>` (resumable, ~30s window). |

## Runbooks

Hand-authored runbook CRUD + promote-from-investigation (`opsrag/api/routes_runbooks.py`, prefix `/runbooks`).

| Method | Path | Scope | Purpose |
| --- | --- | --- | --- |
| GET | `/api/runbooks` | authenticated | List runbooks. |
| GET | `/api/runbooks/{runbook_id}` | authenticated | Fetch one runbook. |
| POST | `/api/runbooks` | authenticated | Create a runbook. |
| PUT | `/api/runbooks/{runbook_id}` | authenticated | Update a runbook. |
| DELETE | `/api/runbooks/{runbook_id}` | authenticated | Delete a runbook. |
| GET | `/api/runbooks/{runbook_id}/versions` | authenticated | List a runbook's version history. |
| POST | `/api/runbooks/from-investigation/{investigation_id}` | authenticated | Pro LLM converts a closed investigation into a runbook draft for review. |

## MCP server proxy

The MCP-server-as-proxy surface (`opsrag/api/mcp_routes.py`, prefix `/mcp`). Token management is scope-gated; the wire-protocol endpoints are authed by the minted bearer token. See [./mcp-integrations.md](./mcp-integrations.md).

| Method | Path | Scope | Purpose |
| --- | --- | --- | --- |
| POST | `/api/mcp/tokens` | `mcp` | Mint a new MCP bearer token for the calling user. |
| GET | `/api/mcp/tokens` | `mcp` | List the caller's MCP tokens (no plaintext). |
| DELETE | `/api/mcp/tokens/{token_id}` | `mcp` | Revoke one of the caller's tokens. |
| GET | `/api/mcp/sse` | secret (bearer token) | Open the MCP server→client SSE stream (emits the `endpoint` event + keep-alives). |
| POST | `/api/mcp/messages` | secret (bearer token) | MCP client→server JSON-RPC inbox (`tools/call`, etc.). |

## Admin

| Method | Path | Scope | Purpose |
| --- | --- | --- | --- |
| GET | `/api/admin/agent-guidance` | `admin` | Current deployment-wide custom instructions (live DB value, else config seed). |
| PUT | `/api/admin/agent-guidance` | `admin` | Save deployment-wide custom instructions; effective on the next query (no restart). |
| GET | `/api/admin/roles` | `admin` | Role catalog (the bundles in `opsrag/auth/scopes.py`). |
| GET | `/api/admin/users` | `admin` | List users (login-mode user management). |
| PUT | `/api/admin/users/{user_id}/roles` | `admin` | Assign roles to a user. |
| GET | `/api/admin/usage` | `admin` | Per-user usage leaderboard, ordered by cost (requires usage persistence). |
| GET | `/api/integrations` | none (gated by middleware) | Enumerate every MCP integration with enabled state, tool count, and health-probe flag. |
| GET | `/api/graph/stats` | none (gated by middleware) | Knowledge-graph backend status + schema (falls back to the lightweight entity-graph). |
| GET | `/api/graph/view` | none (gated by middleware) | Filtered subgraph for the Knowledge Graph UI (`view = business`/`public`/`private`). |

## Auth

First-party login/SSO router (`opsrag/auth/login.py`), registered only in `login` mode. All paths are on the no-auth `/auth/*` prefix so the OIDC enforcement middleware lets them through; they are hit directly (no `/api` prefix). Each handler degrades to **503** when its login-mode state isn't wired.

| Method | Path | Scope | Purpose |
| --- | --- | --- | --- |
| GET | `/auth/providers` | none (public) | Which login methods are enabled (password and/or SSO providers). |
| POST | `/auth/login` | none (public) | Password login (rate-limited + lockout); sets session/refresh/CSRF cookies. |
| POST | `/auth/logout` | none (public) | Clear cookies + revoke the presented refresh session. |
| POST | `/auth/refresh` | none (public) | Rotate the refresh token and re-mint the session cookie. |
| GET | `/auth/sso/{provider}/login` | none (public) | Begin the OAuth redirect (providers: `google`, `github`, `microsoft`). |
| GET | `/auth/sso/{provider}/callback` | none (public) | Complete OAuth, link/resolve the user, mint a session, redirect to `/`. |

## Health

Liveness/readiness (`opsrag/api/routes_health.py`); both bypass auth and are hit directly (no `/api` prefix).

| Method | Path | Scope | Purpose |
| --- | --- | --- | --- |
| GET | `/healthz` | none (public) | Liveness — 200 while the process serves. |
| GET | `/readyz` | none (public) | Readiness — 200 only once providers + backing stores are reachable; probes each enabled MCP's health URL; 503 with a per-component breakdown otherwise. |
| GET | `/api/health` | none (gated by middleware) | App-level health/version (the `routes.py` variant, behind the `/api` proxy). |

## Webhooks

SCM push webhooks (`opsrag/api/routes_webhooks.py`); authenticated by a per-provider shared secret (not OIDC), hit directly (no `/api` prefix). A missing/disabled secret returns 503.

| Method | Path | Scope | Purpose |
| --- | --- | --- | --- |
| POST | `/webhook/gitlab` | secret (GitLab token) | Trigger a reindex on a GitLab push. |
| POST | `/webhook/github` | secret (GitHub HMAC) | Trigger a reindex on a GitHub push. |

## UI-config

| Method | Path | Scope | Purpose |
| --- | --- | --- | --- |
| GET | `/api/ui-config` | none (gated by middleware) | White-label / runtime UI config (brand, source-link bases, model name, `investigation_enabled` gate). Fetched by the SPA on boot. |
| GET | `/api/me` | authenticated | Identity as the backend sees it: tracking flag, oid/sub/email/name, groups, roles, scopes, `is_admin`. |

## Usage

| Method | Path | Scope | Purpose |
| --- | --- | --- | --- |
| GET | `/api/usage` | none (gated by middleware) | Token-usage summary with per-model cost (pod-agnostic; reads shared Postgres). |
| GET | `/api/usage/weekly` | none (gated by middleware) | Per-week token + cost buckets for the Home dashboard chart. |
| GET | `/api/usage/{session_id}` | none (gated by middleware) | Token usage for one session. |
| GET | `/api/me/usage` | authenticated | The caller's own usage roll-up (always 200; anonymous → empty). |

## See also

- [Authentication & RBAC](./auth.md) — auth modes, scopes, sessions, SSO.
- [Investigations](./investigations.md) — the event-driven investigation engine behind `/api/investigations/*`.
- [MCP integrations](./mcp-integrations.md) — the read-only tool surface and the MCP-server proxy.
