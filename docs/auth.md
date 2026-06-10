# Authentication

opsrag has three authentication modes (`auth.mode`): `open` (no
enforcement), `oidc` (verify incoming Bearer JWTs — the default and the
focus of most of this document), and `login` (first-party cookie sessions
with password and/or SSO). Beyond authentication, opsrag enforces
**per-session ownership** (a thread can only be read or deleted by its
owner) and **rate limiting** (a per-request throttle plus a login brute-
force lockout). This document explains JWT verification, per-provider IdP
setup, the ownership model, and rate limiting.

## How auth works

Every endpoint except `/healthz` and `/readyz` requires an
`Authorization: Bearer <token>` header. (The schema/docs routes
`/openapi.json`, `/docs`, `/redoc`, and the legacy `/health` path are also
left open so tooling and the UI can read the spec.)

On startup the app builds a single OIDC verifier from the `auth` block in
your config and attaches it to the running app. For each protected request
the verifier:

1. Fetches `<issuer>/.well-known/openid-configuration` once (lazily, on
   first use) and reads `jwks_uri` from it. The `issuer` returned by
   discovery must match the configured `issuer`, or startup/verification
   fails.
2. Fetches the JWKS document and caches the signing keys by `kid` for
   `jwksCacheSeconds`. Keys are refreshed on cache miss or TTL expiry, so
   IdP key rotation is picked up automatically.
3. Verifies the incoming JWT:
   - signature against the JWK whose `kid` matches the token header
   - `iss` claim equals the configured `issuer`
   - `aud` claim equals the configured `audience`
   - `exp` claim against the current wall clock (token not expired)
   - the token must also carry a `sub` claim
4. Accepted RSA and ECDSA algorithms by default: RS256/384/512,
   ES256/384/512, PS256/384/512.

On success the token's `sub` claim is propagated into request-scoped
context for usage attribution. The `sub` is NEVER logged in cleartext and
is never returned in responses. The Bearer token itself is never logged.

If no `auth` block is configured, the verifier is absent and the API runs
in local-dev open mode (all requests pass through). Configure `auth` for
any shared or production deployment.

### Rejection envelope

A rejected request returns HTTP 401 with a stable JSON envelope:

```json
{"error": "unauthenticated", "reason": "<reason>", "request_id": "<uuid>"}
```

`reason` is one of a closed set, mapped from the verification failure:

| reason             | meaning                                              |
|--------------------|------------------------------------------------------|
| `missing_bearer`   | no `Authorization: Bearer ...` header on the request |
| `invalid_signature`| signature/kid/malformed token (catch-all reject)     |
| `issuer_mismatch`  | token `iss` does not match `auth.issuer`             |
| `audience_mismatch`| token `aud` does not match `auth.audience`           |
| `expired`          | token `exp` is in the past                            |

The `request_id` is also stamped onto logs for the same request so a
rejection can be correlated without logging the token.

## Config block

```yaml
auth:
  issuer: https://your-idp.example.com   # OIDC discovery base URL (required)
  audience: opsrag                       # expected token "aud" (required)
  jwks_cache_seconds: 300                # signing-key cache TTL (default 300)
```

- `issuer` - the OIDC issuer base URL. opsrag appends
  `/.well-known/openid-configuration` to discover the JWKS. Must exactly
  match the `iss` claim your IdP puts in tokens (no trailing slash needed;
  it is normalized).
- `audience` - the value opsrag requires in the token `aud` claim. Set this
  to the client/application/API identifier your IdP issues tokens for.
- `jwks_cache_seconds` - how long signing keys are cached before refetch.

The compose quickstart sets these via environment, e.g.
`OPSRAG_OIDC_ISSUER`.

## First-party login mode (`auth.mode: login`) + the admin user

`oidc` mode has **no user database** — identity comes entirely from the IdP's
token, so there is no "admin account" to sign in with. To run opsrag's own
login (email + password, SSO, cookie sessions) with a real **admin user**,
switch to `login` mode:

```yaml
auth:
  mode: login
  session:
    signing_key_env: OPSRAG_SESSION_SIGNING_KEY   # signs session cookies
    password_enabled: true                        # enable email + password login
```

Then supply the secrets via environment — never inline in committed config:

```sh
# A random key (>= 32 bytes) signs session cookies. Generate one, e.g.:
#   python3 -c "import secrets; print(secrets.token_urlsafe(48))"
OPSRAG_SESSION_SIGNING_KEY=<your-32+-byte-random-key>

# The bootstrap admin -- choose your OWN email + password.
OPSRAG_ADMIN_EMAIL=admin@opsrag.local
OPSRAG_ADMIN_PASSWORD=<choose-a-strong-password>
```

There is nothing to "retrieve": on startup opsrag **seeds** this admin user
(role `admin`) from those two env vars if it does not already exist —
idempotent, and the password is read only from the env var (or a mounted
secret in production), never from config (`api/server.py`). Log in to get a
session cookie, then call the API with it:

```sh
curl -sf -X POST http://localhost:8080/auth/login \
  -d "email=$OPSRAG_ADMIN_EMAIL" -d "password=$OPSRAG_ADMIN_PASSWORD" \
  -c cookies.txt

curl -sf -X POST http://localhost:8080/query -b cookies.txt \
  -H 'Content-Type: application/json' \
  -d '{"query":"How do I roll back an Acme Notes deployment?"}' | jq
```

The web UI presents a login screen in this mode. To add SSO (Google / GitHub /
Microsoft-Entra) on top of password login, configure the `sso` block alongside
the provider setup below. The admin user can list and manage **every**
conversation; other users are scoped to their own (see
[Per-session ownership](#per-session-ownership)).

## Per-provider setup

For all providers below, replace placeholder hosts and IDs with your own.
The key task is making the configured `issuer` match the token `iss`, and
the configured `audience` match the token `aud`.

### Dex (bundled local issuer)

The compose quickstart ships a Dex instance as a local OIDC issuer with a
static user (`evaluator@example.com` / `evaluator`) and a static client
`opsrag-local`. Dex supports the resource-owner password grant, so you can
mint a token without a browser:

```sh
curl -s http://localhost:5556/dex/token \
  -d grant_type=password \
  -d client_id=opsrag-local \
  -d client_secret=local-secret \
  -d username=evaluator@example.com \
  -d password=evaluator \
  -d scope="openid email profile" \
  | python3 -c "import sys, json; print(json.load(sys.stdin)['id_token'])"
```

Use the returned `id_token` as the Bearer token:

```sh
TOKEN=$(curl -s http://localhost:5556/dex/token \
  -d grant_type=password -d client_id=opsrag-local \
  -d client_secret=local-secret -d username=evaluator@example.com \
  -d password=evaluator -d scope="openid email profile" \
  | python3 -c "import sys, json; print(json.load(sys.stdin)['id_token'])")

curl -s http://localhost:8080/usage -H "Authorization: Bearer $TOKEN"
```

- issuer: the Dex issuer URL (see the gotcha below).
- audience: the Dex client id, `opsrag-local` in the bundled config. Dex
  puts the client id into the token `aud`, so `auth.audience` must equal it.

This Dex config is for local eval only. Do not use the placeholder client
secret or static password outside local development.

#### Local-Dex issuer-vs-cluster-host gotcha

Dex advertises a single canonical issuer in its config and in every token's
`iss` claim. The bundled Dex sets:

```yaml
issuer: http://localhost:5556/dex
```

Inside the compose network, however, the API container reaches Dex by its
service name at `http://dex:5556/dex`, not `localhost`. This creates a
mismatch: tokens carry `iss = http://localhost:5556/dex`, but the API may
be configured with `issuer: http://dex:5556/dex` for in-cluster discovery.
Discovery and `iss` verification will then fail (`issuer_mismatch`), because
the verifier requires the discovered/configured issuer to match the token
`iss` exactly.

Align them on ONE value used everywhere - in the Dex `issuer` field, in the
token `iss`, and in `auth.issuer`:

- Simplest for the quickstart: set Dex `issuer` to the same host the API
  uses to reach it (for example `http://dex:5556/dex`), and set
  `auth.issuer` to that same value. The browser/UI must then also be able
  to resolve `dex` (for example via a hosts entry or by running the token
  exchange from inside the network).
- Alternatively, keep `issuer: http://localhost:5556/dex` and make the API
  resolve `localhost:5556` to the Dex container (shared network namespace or
  a host alias), then set `auth.issuer: http://localhost:5556/dex`.

The rule: Dex `issuer`, the token `iss`, and `auth.issuer` must be byte-for-
byte the same string, and every party must be able to reach that URL.

### Keycloak

- issuer: `https://your-keycloak.example.com/realms/<realm>`. Keycloak's
  per-realm discovery lives at
  `https://your-keycloak.example.com/realms/<realm>/.well-known/openid-configuration`.
- audience: by default Keycloak does not put your client id in `aud` unless
  you add an audience mapper. Either add a "Audience" protocol mapper that
  emits the value you set in `auth.audience`, or set `auth.audience` to a
  value Keycloak does include. Verify the actual `aud` of a sample token.

### Okta

- issuer: your authorization server issuer, for example
  `https://your-org.okta.com/oauth2/<authServerId>` (or
  `https://your-org.okta.com/oauth2/default` for the default server).
- audience: the audience configured on that Okta authorization server (the
  "Audience" field, often an API URI like `api://opsrag`). Set
  `auth.audience` to that exact value. Tokens minted by the org
  authorization server (no custom server) may not carry a usable `aud`; use
  a custom authorization server so you control the `aud`.

### Auth0

- issuer: `https://your-tenant.us.auth0.com/` (Auth0 issuers include the
  trailing slash in the `iss` claim; the verifier normalizes trailing
  slashes, but keep issuer and `auth.issuer` consistent).
- audience: the API Identifier of the Auth0 API you created, for example
  `https://api.opsrag.example.com`. The client must request that audience
  when fetching the token, and `auth.audience` must equal it. Without a
  requested audience Auth0 returns an opaque token, not a verifiable JWT.

### Azure AD (Microsoft Entra ID)

- issuer: `https://login.microsoftonline.com/<tenant-id>/v2.0`. Discovery
  is at
  `https://login.microsoftonline.com/<tenant-id>/v2.0/.well-known/openid-configuration`.
  Use the v2.0 endpoint so the `iss` claim is predictable.
- audience: the Application (client) ID of the API app registration, or the
  Application ID URI (for example `api://<client-id>`). Entra access tokens
  set `aud` to that value; set `auth.audience` to match.
- gotcha: Entra tokens also carry an `azp` (authorized party) claim for the
  calling client; opsrag verifies `aud`, not `azp`. Make sure the token's
  `aud` (not just `azp`) equals `auth.audience`. Confirm with a decoded
  sample token, since v1.0 and v2.0 endpoints emit different `aud`/`iss`
  shapes.

## Per-session ownership

Authentication answers *who is calling*; ownership answers *what they may
touch*. A conversation thread (session) is owned by the verified identity
that created it, and only that owner can read or delete it. The owner is the
caller's stable subject id (`current_user.oid`, derived from the JWT `sub`
in `oidc` mode or the SSO/`sub` identity in `login` mode), recorded into the
checkpoint metadata as `user_id` when the thread is first written.

### Enforcement

On every single-thread read or delete (for example `GET /sessions/{thread_id}`,
`GET /sessions/{thread_id}/messages`, `DELETE /sessions/{thread_id}`), opsrag
looks up the thread's recorded owner via `store.get_session_owner(thread_id)`
and compares it to the caller. The check lives in
`opsrag/api/routes.py:_deny_if_not_owner`:

```python
if _is_real_owner(owner) and owner != current_user.oid:
    raise HTTPException(status_code=404, detail="session not found")
```

Three deliberate properties:

- **Cross-user access returns 404, not 403.** A non-owner gets "session not
  found" — the same response as a thread that does not exist. Returning 403
  would be an existence oracle, leaking that the thread is real but
  forbidden. 404 reveals nothing.
- **Open / anonymous mode is unenforced.** When the caller is anonymous
  (`current_user.is_anonymous` or no `oid` — i.e. `auth.mode: open`, no
  `auth` block, or a request with no usable identity), the ownership check
  is a no-op. This preserves zero-config local-dev behavior where every
  caller shares one anonymous identity.
- **Legacy anonymous-owned threads are grandfathered.** Threads created
  before owner binding existed have an empty or `"anonymous"` owner.
  `_is_real_owner` treats only a non-empty, non-`"anonymous"` owner as a
  lockable identity, so these pre-auth threads stay readable and deletable
  by anyone. opsrag cannot retroactively assign them an owner, so locking
  them down would orphan them; only threads with a real authenticated owner
  are guarded.

Ownership binds to the *verified* id. A client may still pass a `user_id` in
the request body for memory/personalization, but it never overrides the
owner binding — that always uses `current_user.oid`.

## Rate limiting

opsrag applies two independent throttles, both backed by a pluggable
storage seam (`opsrag/api/rate_limit_backend.py`):

1. **Per-request rate limit** — `RateLimitMiddleware` caps requests per key
   (per client) to `api.rate_limit_rpm` requests per minute. Over the limit
   returns HTTP 429 with a `Retry-After`.
2. **Login lockout** — `LoginRateLimiter` (used only in `auth.mode: login`)
   throttles failed `POST /auth/login` attempts. After
   `login_max_attempts` failures within `login_window_seconds`, the
   key (email/IP) is locked for `login_lockout_seconds`; a successful login
   clears the counter.

### Backend selection: memory vs redis

`api.rate_limit_backend` selects where the throttle state lives:

```yaml
api:
  rate_limit_enabled: true
  rate_limit_rpm: 60
  rate_limit_backend: memory        # "memory" (default) or "redis"
  redis_url_env: OPSRAG_REDIS_URL   # env var holding the Redis URL
```

- `memory` (default) — in-process counters. Correct and dependency-free for
  a **single replica**, but state is per-process: with multiple replicas
  each enforces its own limit, so the effective aggregate limit is
  `rpm x replicas` and a login lockout on one pod is not seen by the others.
- `redis` — shared state across replicas. The request limiter uses an atomic
  fixed-window counter (`INCR` + `EXPIRE`); login uses a failure counter plus
  a `SETEX` lockout key whose TTL is the authoritative retry-after. Use this
  for any horizontally-scaled deployment.

**Redis is REQUIRED when selected.** `redis` is an optional extra (the
`redis` import is lazy, so the API stays importable without it). When
`rate_limit_backend: redis`, opsrag reads the URL from `redis_url_env`,
`PING`s the server at startup, and **fails fast** if it is unreachable —
there is no silent fallback to in-memory, because a half-enforced limiter is
worse than a loud boot failure. The session config also exposes
`login_max_attempts` / `login_window_seconds` / `login_lockout_seconds`
(under `auth.login`) so both throttles ride the same backend.

See [`operations.md`](./operations.md#rate-limiting-across-replicas) for the
multi-replica deployment checklist.

## See also

- [`configuration.md`](./configuration.md) — the full `auth` and `api`
  config blocks and env precedence.
- [`operations.md`](./operations.md) — day-2 ops, including rate limiting
  across replicas and the security hardening checklist.
- [`architecture.md`](./architecture.md) — where the verifier and middleware
  sit in the request flow.
