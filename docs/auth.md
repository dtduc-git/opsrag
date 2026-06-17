# Authentication

This is the complete guide to authentication and authorization in OpsRAG:
how to choose an auth mode, how to wire each one up against real identity
providers, how scopes and roles work, how to mint MCP tokens, and how to
harden a production deployment.

Authentication is ALWAYS enforced -- there is no anonymous / "open"
mode. Every non-allowlisted request must carry a valid identity or it is
rejected with HTTP 401. OpsRAG has two authentication modes, set by
`auth.mode`:

- `login` (the default) -- OpsRAG runs its own first-party login: email +
  password and/or SSO (Google / Microsoft / GitHub), backed by signed
  cookie sessions and a local user store, with a seeded `admin` account.
- `oidc` -- OpsRAG verifies an incoming `Authorization: Bearer <JWT>`
  against an external identity provider (IdP). There are no local credentials:
  identity is asserted by the token, and the `admin` role is granted by mapping
  an IdP `groups` claim via `role_mappings`. Authenticated identities are still
  recorded in the `opsrag_user` table (for audit) in both modes.

On top of authentication ("who is calling"), OpsRAG enforces:

- **RBAC scopes** (`chat` / `investigate` / `mcp` / `admin`) -- what a
  caller may do (see [Scopes and roles](#scopes-and-roles)).
- **Per-session ownership** -- a conversation thread can only be read or
  deleted by the verified identity that created it (see
  [Per-session ownership](#per-session-ownership)).
- **Rate limiting** -- a per-request throttle plus a login brute-force
  lockout (see [Rate limiting](#rate-limiting)).

> Conventions in this doc: all examples are copy-pasteable. Replace
> placeholder hosts, client IDs, and secrets with your own. Secrets are
> ALWAYS supplied via environment variables (or mounted secret files),
> never inline in `config.yaml`.

---

## Which mode do I pick?

```
                       Do you have an existing IdP
                    (Okta / Entra / Google / Keycloak)
                      that already issues JWTs to your
                            users or services, or a
                      gateway that injects a Bearer JWT
                            per request?
                                   |
                  +----------------+----------------+
                 yes                                no
                  |                                  |
            ->  mode: oidc                    ->  mode: login
                                              (the default; OpsRAG runs
                                               its own login screen +
                                               seeded admin account)
```

Quick rules of thumb:

- You are **evaluating OpsRAG locally** or running the compose demo
  -> `login` (the default). You will hit a sign-in screen and log in as
  the seeded `admin` account (see [The admin user](#the-admin-user)).
- You want OpsRAG to **be the login system** -- a hosted product with its
  own login page, a real `admin` account, password and/or "Sign in with
  Google" buttons -> `login`.
- You already run an **IdP and want OpsRAG to trust its tokens**, or
  OpsRAG sits **behind a gateway** (Pomerium, oauth2-proxy, an API
  gateway) that mints/forwards a JWT -> `oidc`.

### Mode comparison

| Property | `login` | `oidc` |
|---|---|---|
| Identity source | OpsRAG's own users | external IdP JWT |
| Credential storage | yes (password / SSO links) | no (token-asserted) |
| Identity records (`opsrag_user`) | yes | yes (audit; no credentials) |
| First-party admin account | yes (seeded from env) | no |
| Credential the client sends | signed session cookie | `Authorization: Bearer <JWT>` |
| Web UI shows a login screen | yes | no (UI assumes a token/gateway) |
| SSO (Google / MS / GitHub) | built in (`sso` block) | via your IdP |
| Password login | yes (`password_enabled`) | -- |
| RBAC scopes enforced | yes (from stored roles) | yes (from token `groups`) |
| Per-session ownership | yes | yes |
| Required config | `session.signing_key` + admin env | `issuer` + `audience` |
| Typical use | hosted product, local dev / demo | API/CI, behind a gateway |

The default when **no `auth` block** is present is `auth.mode: login`.
Authentication is always enforced: a non-allowlisted request without a
valid identity is rejected with 401 in either mode -- there is no
anonymous / all-scopes path.

---

## How verification works (oidc mode)

In `oidc` mode every endpoint except the public allowlist below requires
an `Authorization: Bearer <token>` header. The always-allowed paths are:
`/healthz`, `/readyz`, the legacy `/health`, the schema/docs routes
(`/openapi.json`, `/docs`, `/docs/oauth2-redirect`, `/redoc`), the
pre-auth branding endpoint `/ui-config`, the SCM webhooks
(`/webhook/gitlab`, `/webhook/github`, which authenticate with their own
HMAC secret), the first-party login surface (`/auth/*`), and the MCP wire
protocol (`/mcp/sse`, `/mcp/messages`, which carry their own `opsrag_`
bearer token).

On startup the app builds a single OIDC verifier from the `auth` block
and attaches it to the running app. For each protected request the
verifier (`opsrag/auth/oidc.py`):

1. Fetches `<issuer>/.well-known/openid-configuration` once (lazily, on
   first use) and reads `jwks_uri`. The `issuer` returned by discovery
   must match the configured `issuer`, or verification fails.
2. Fetches the JWKS document and caches the signing keys by `kid` for
   `jwks_cache_seconds`. Keys are refreshed on cache miss or TTL expiry,
   so IdP key rotation is picked up automatically.
3. Verifies the incoming JWT:
   - signature against the JWK whose `kid` matches the token header
   - `iss` claim equals the configured `issuer`
   - `aud` claim equals the configured `audience`
   - `exp` claim against the current wall clock (token not expired)
   - the token must also carry a `sub` claim
4. Accepted algorithms by default: RS256/384/512, ES256/384/512,
   PS256/384/512 (RSA and ECDSA).

On success the standard OIDC claims (`sub`, `email`, `name`, `picture`,
and `groups` or `roles`) are read into a `CurrentUser`. The `sub` is
propagated into request-scoped context for usage attribution; it is NEVER
logged in cleartext and is never returned in responses. The Bearer token
itself is never logged.

### Rejection envelope

A rejected request returns HTTP 401 with a stable JSON envelope:

```json
{"error": "unauthenticated", "reason": "<reason>", "request_id": "<uuid>"}
```

`reason` is one of a closed set:

| reason               | mode  | meaning                                              |
|----------------------|-------|------------------------------------------------------|
| `missing_bearer`     | oidc  | no `Authorization: Bearer ...` header                |
| `invalid_signature`  | oidc  | signature/kid/malformed token (catch-all reject)     |
| `issuer_mismatch`    | oidc  | token `iss` does not match `auth.issuer`             |
| `audience_mismatch`  | oidc  | token `aud` does not match `auth.audience`           |
| `expired`            | oidc  | token `exp` is in the past                           |
| `missing_session`    | login | no / invalid / expired signed session cookie         |
| `login_unavailable`  | login | login mode configured but the session manager isn't wired (fails closed) |
| `auth_misconfigured` | oidc  | oidc mode but no verifier is wired (fails closed; no anonymous fallback) |

The `request_id` is also stamped onto logs for the same request, so a
rejection can be correlated without logging the token.

A 401 means "re-authenticate". A **403** with `{"error": "forbidden",
"reason": "missing_scope", "scope": "<scope>"}` means "authenticated, but
you lack the required scope" -- see [Scopes and roles](#scopes-and-roles).

---

## oidc mode setup

### Config block

```yaml
auth:
  mode: oidc
  issuer: https://your-idp.example.com   # OIDC discovery base URL (required)
  audience: opsrag                       # expected token "aud" (required)
  jwks_cache_seconds: 300                # signing-key cache TTL (default 300)
  # Optional RBAC: map an IdP group/role claim value -> opsrag roles.
  role_mappings:
    sre-admins: [admin]
    oncall:     [member_investigate]
```

- `issuer` -- the OIDC issuer base URL. OpsRAG appends
  `/.well-known/openid-configuration` to discover the JWKS. Must exactly
  match the `iss` claim your IdP puts in tokens (trailing slash is
  normalized).
- `audience` -- the value OpsRAG requires in the token `aud` claim. Set
  this to the client / application / API identifier your IdP issues
  tokens for.
- `jwks_cache_seconds` -- how long signing keys are cached before refetch.
- `role_mappings` -- optional `{group: [roles]}`. Empty means every
  authenticated user gets the default `member_investigate` role (chat +
  investigate). See [Scopes and roles](#scopes-and-roles).

`auth.mode: oidc` requires BOTH `issuer` and `audience` -- config load
fails fast otherwise.

The compose quickstart sets these via environment (e.g.
`OPSRAG_OIDC_ISSUER`).

### Bundled Dex demo issuer

The compose quickstart ships a Dex instance as a local OIDC issuer with a
static user (`evaluator@example.com` / `evaluator`) and a static client
`opsrag-local`. Dex supports the resource-owner password grant, so you
can mint a token without a browser:

```sh
TOKEN=$(curl -s http://localhost:5556/dex/token \
  -d grant_type=password \
  -d client_id=opsrag-local \
  -d client_secret=local-secret \
  -d username=evaluator@example.com \
  -d password=evaluator \
  -d scope="openid email profile" \
  | python3 -c "import sys, json; print(json.load(sys.stdin)['id_token'])")

curl -s http://localhost:8080/usage -H "Authorization: Bearer $TOKEN"
```

- issuer: the Dex issuer URL (see the gotcha below).
- audience: the Dex client id, `opsrag-local` in the bundled config. Dex
  puts the client id into the token `aud`, so `auth.audience` must equal
  it.

This Dex config is for local eval only. Do not use the placeholder client
secret or static password outside local development. The bundled Dex
exists to demonstrate the `oidc` path: point `auth.mode: oidc` at it (set
`OPSRAG_OIDC_ISSUER` / `OPSRAG_OIDC_AUDIENCE`) and the minted token above
becomes the required Bearer credential. The compose demo defaults to
`login` mode, where you sign in as the seeded admin instead (see
[login mode setup](#login-mode-setup-first-party-accounts)).

#### The Dex issuer-vs-cluster-host gotcha

Dex advertises a single canonical issuer in its config and in every
token's `iss` claim. The bundled Dex sets:

```yaml
issuer: http://localhost:5556/dex
```

Inside the compose network, however, the API container reaches Dex by its
service name at `http://dex:5556/dex`, not `localhost`. This creates a
mismatch: tokens carry `iss = http://localhost:5556/dex`, but the API may
be configured with `issuer: http://dex:5556/dex` for in-cluster
discovery. Discovery and `iss` verification then fail
(`issuer_mismatch`), because the verifier requires the
discovered/configured issuer to match the token `iss` exactly.

Align them on ONE value used everywhere -- in the Dex `issuer` field, in
the token `iss`, and in `auth.issuer`:

- Simplest: set Dex `issuer` to the host the API uses to reach it (e.g.
  `http://dex:5556/dex`), and set `auth.issuer` to that same value. The
  browser/UI must then also be able to resolve `dex` (e.g. via a hosts
  entry, or run the token exchange from inside the network).
- Alternatively, keep `issuer: http://localhost:5556/dex` and make the
  API resolve `localhost:5556` to the Dex container (shared network
  namespace or a host alias), then set
  `auth.issuer: http://localhost:5556/dex`.

The rule: Dex `issuer`, the token `iss`, and `auth.issuer` must be
byte-for-byte the same string, and every party must be able to reach that
URL.

### Production IdPs

For every provider the two tasks are the same: make `auth.issuer` match
the token `iss`, and make `auth.audience` match the token `aud`. Always
confirm against a DECODED sample token (paste it into a JWT decoder, or
`python3 -c "import jwt,sys;print(jwt.decode(sys.argv[1],options={'verify_signature':False}))" <token>`).

#### Okta

```yaml
auth:
  mode: oidc
  issuer: https://your-org.okta.com/oauth2/<authServerId>
  audience: api://opsrag
```

- issuer: your authorization server issuer, e.g.
  `https://your-org.okta.com/oauth2/<authServerId>` (or
  `https://your-org.okta.com/oauth2/default` for the default server).
- audience: the "Audience" configured on that Okta authorization server,
  often an API URI like `api://opsrag`. Set `auth.audience` to that exact
  value.
- Use a CUSTOM authorization server so you control the `aud`. Tokens
  minted by the bare org authorization server may not carry a usable
  `aud`.
- Getting a token: register an OIDC app (or API service app for
  machine-to-machine), then use the standard Okta `/v1/token` flow (e.g.
  client-credentials for services, authorization-code for browsers) and
  send the resulting access token as the Bearer.

#### Microsoft Entra ID (Azure AD)

```yaml
auth:
  mode: oidc
  issuer: https://login.microsoftonline.com/<tenant-id>/v2.0
  audience: api://<client-id>   # or the bare <client-id>
```

- issuer: `https://login.microsoftonline.com/<tenant-id>/v2.0`. Discovery
  is at `.../v2.0/.well-known/openid-configuration`. Use the v2.0
  endpoint so the `iss` claim is predictable.
- audience: the Application (client) ID of the API app registration, or
  its Application ID URI (e.g. `api://<client-id>`). Entra access tokens
  set `aud` to that value.
- Gotcha: Entra tokens also carry an `azp` (authorized party) claim for
  the calling client; OpsRAG verifies `aud`, NOT `azp`. Make sure the
  token's `aud` equals `auth.audience`. v1.0 vs v2.0 endpoints emit
  different `aud`/`iss` shapes -- confirm with a decoded sample.
- Getting a token: register an app, expose an API (App ID URI), grant the
  caller the API permission, then run the OAuth2 flow that fits the caller
  (client-credentials for services, auth-code for browsers).

#### Google

```yaml
auth:
  mode: oidc
  issuer: https://accounts.google.com
  audience: <your-google-oauth-client-id>.apps.googleusercontent.com
```

- issuer: `https://accounts.google.com`. Discovery is at
  `https://accounts.google.com/.well-known/openid-configuration`.
- audience: the OAuth 2.0 Client ID that requested the ID token; Google
  puts it in the `aud` claim. Set `auth.audience` to that client ID.
- Note: OpsRAG verifies the Google **ID token** (a JWT). Google OAuth
  *access* tokens are opaque and not verifiable here -- send the
  `id_token`. For human users who should log in *with* Google rather than
  present a pre-minted token, prefer `login` mode with the Google SSO
  block below.

#### Keycloak

```yaml
auth:
  mode: oidc
  issuer: https://your-keycloak.example.com/realms/<realm>
  audience: opsrag
```

- issuer: `https://your-keycloak.example.com/realms/<realm>`. Per-realm
  discovery lives at `.../realms/<realm>/.well-known/openid-configuration`.
- audience: by default Keycloak does not put your client id in `aud`
  unless you add an audience mapper. Either add an "Audience" protocol
  mapper that emits the value you set in `auth.audience`, or set
  `auth.audience` to a value Keycloak does include. Verify a sample token.

#### Auth0

```yaml
auth:
  mode: oidc
  issuer: https://your-tenant.us.auth0.com/
  audience: https://api.opsrag.example.com
```

- issuer: `https://your-tenant.us.auth0.com/` (Auth0 issuers include the
  trailing slash in `iss`; the verifier normalizes it, but keep both
  sides consistent).
- audience: the API Identifier of the Auth0 API you created. The client
  must REQUEST that audience when fetching the token; without a requested
  audience Auth0 returns an opaque token, not a verifiable JWT.

---

## login mode setup (first-party accounts)

`login` mode makes OpsRAG its own identity system: it runs a login
screen, stores users (email + password and/or federated SSO links),
issues signed cookie sessions, and seeds a real `admin` account. There is
no OIDC verifier in this mode -- the global middleware enforces a valid
session cookie on every protected route instead.

### Minimal config

```yaml
auth:
  mode: login
  login:
    password_enabled: true                        # email + password login
    signing_key_env: OPSRAG_SESSION_SIGNING_KEY   # signs session cookies
    # External URL the BROWSER uses to reach the API (for SSO redirects):
    sso_callback_base: https://opsrag.example.com/api
    cookie_secure: true                           # require HTTPS (default true)
    cookie_samesite: lax                          # lax | strict | none
    session_ttl_seconds: 900                      # session cookie lifetime
    refresh_ttl_seconds: 1209600                  # refresh token lifetime (14d)
```

The user store backs onto Postgres when `session.provider: postgres` and
a DSN is available; otherwise it falls back to an in-memory store (fine
for a single replica / dev, but users and sessions are lost on restart
and not shared across replicas). For a real deployment use Postgres:

```yaml
session:
  provider: postgres
  dsn_env: POSTGRES_DSN
```

### Secrets via environment

Never put key material or passwords in committed config. Supply them as
env vars (or mounted secret files):

```sh
# A random key (>= 32 bytes) signs the session cookies. Generate one:
#   python3 -c "import secrets; print(secrets.token_urlsafe(48))"
OPSRAG_SESSION_SIGNING_KEY=<your-32+-byte-random-key>

# The bootstrap admin -- choose your OWN email + password.
OPSRAG_ADMIN_EMAIL=admin@opsrag.local
OPSRAG_ADMIN_PASSWORD=<choose-a-strong-password>

# Postgres DSN (when session.provider: postgres).
POSTGRES_DSN=postgresql://opsrag:...@postgres:5432/opsrag
```

The signing key is loaded ONLY from `signing_key_path` (a file) or
`signing_key_env` (an env var); inline key material in config is refused
at load time (`opsrag/auth/sessions.py: load_signing_key`). For
production, prefer a mounted secret file:

```yaml
auth:
  login:
    signing_key_path: /run/secrets/opsrag-session-key
```

### The admin user

There is nothing to "retrieve": on startup, when `auth.mode: login`,
OpsRAG **seeds** an admin user (role `admin`) from `OPSRAG_ADMIN_EMAIL` +
`OPSRAG_ADMIN_PASSWORD` if that email does not already exist. It is
idempotent, and the password is read only from the env var (or a mounted
secret), never from config (`opsrag/api/server.py`).

Log in to get a session cookie, then call the API with it:

```sh
curl -sf -X POST http://localhost:8080/auth/login \
  -d "email=$OPSRAG_ADMIN_EMAIL" -d "password=$OPSRAG_ADMIN_PASSWORD" \
  -c cookies.txt

curl -sf -X POST http://localhost:8080/query -b cookies.txt \
  -H 'Content-Type: application/json' \
  -d '{"query":"How do I roll back an Acme Notes deployment?"}' | jq
```

The web UI presents a login screen in this mode. It calls
`GET /auth/providers` first to learn which methods are available
(`password_enabled` plus the list of enabled SSO providers), so the login
page renders exactly the buttons you configured.

### Session, refresh, and CSRF cookies

On a successful login (password or SSO) the server sets three cookies via
`SessionManager` (`opsrag/auth/sessions.py`):

| Cookie | Purpose | Notes |
|---|---|---|
| `opsrag_session` | short-lived signed session (`session_ttl_seconds`, default 15m) | carries the user id, email, and baked-in roles |
| `opsrag_refresh` | rotating refresh token (`refresh_ttl_seconds`, default 14d) | stored server-side as a SHA-256 hash only; rotated on each `POST /auth/refresh` |
| `opsrag_csrf` | double-submit CSRF token | |

`POST /auth/refresh` rotates the refresh token (revokes the presented
one, issues a fresh session). `POST /auth/logout` clears the cookies and
revokes the presented refresh session.

### SSO providers (Google / Microsoft / GitHub)

Add SSO on top of (or instead of) password login via the `sso` block.
For SSO-only, set `password_enabled: false`.

```yaml
auth:
  mode: login
  login:
    password_enabled: true
    signing_key_env: OPSRAG_SESSION_SIGNING_KEY
    # MUST match the redirect URI you register with each IdP. This is the
    # external base URL the browser uses, including any reverse-proxy
    # prefix (e.g. the UI proxy's /api). The callback path appended is
    # /auth/sso/{provider}/callback.
    sso_callback_base: https://opsrag.example.com/api
  sso:
    google:
      enabled: true
      client_id: "<google-oauth-client-id>.apps.googleusercontent.com"
      client_secret_env: OPSRAG_SSO_GOOGLE_SECRET
      # scopes default to openid/email/profile if omitted
    microsoft:
      enabled: true
      client_id: "<entra-app-client-id>"
      client_secret_env: OPSRAG_SSO_MICROSOFT_SECRET
      # Single-tenant? Pin the tenant issuer metadata (else the common
      # multi-tenant endpoint is used):
      server_metadata_url: "https://login.microsoftonline.com/<tenant-id>/v2.0/.well-known/openid-configuration"
    github:
      enabled: true
      client_id: "<github-oauth-app-client-id>"
      client_secret_env: OPSRAG_SSO_GITHUB_SECRET
```

Client SECRETS come from the named env var only, never inline:

```sh
OPSRAG_SSO_GOOGLE_SECRET=<google-oauth-client-secret>
OPSRAG_SSO_MICROSOFT_SECRET=<entra-client-secret>
OPSRAG_SSO_GITHUB_SECRET=<github-oauth-app-secret>
```

**Registering the OAuth apps.** In each provider's console, register an
OAuth/OIDC application and set the Authorized redirect URI to:

```
<sso_callback_base>/auth/sso/<provider>/callback
```

e.g. `https://opsrag.example.com/api/auth/sso/google/callback`,
`.../auth/sso/microsoft/callback`, `.../auth/sso/github/callback`. The
`sso_callback_base` MUST be the URL the BROWSER can reach (including any
reverse-proxy path prefix). If `sso_callback_base` is unset, OpsRAG
derives the redirect from the inbound request, which is only correct when
the API is hit directly (not behind a path-stripping proxy) -- set it
explicitly in production.

- **google** / **microsoft** are OIDC: discovered via
  `server_metadata_url` (defaults to the provider's well-known endpoint;
  Microsoft defaults to the multi-tenant `common` endpoint -- override
  `server_metadata_url` for single-tenant). The returned `id_token` is
  validated (signature, `iss`, `aud`, `exp`, and the `nonce`).
- **github** is plain OAuth2 (no `id_token`): OpsRAG exchanges the code,
  then reads `/user` + `/user/emails` to find the PRIMARY VERIFIED email.

The SSO flow is terminated server-side (Authorization Code, no browser
PKCE) and protected by a signed single-use `state` (CSRF) and an OIDC
`nonce` (replay protection).

**Account-takeover guard (important).** A federated identity is
auto-linked to an existing local account by email ONLY when the IdP
asserts `email_verified`. If the IdP does not assert a verified email,
OpsRAG refuses to link to an existing account and instead creates a fresh
federated-only account (or returns a clear error on an email clash). This
prevents an unverified-email IdP response from taking over a
password/local account.

A brand-new SSO user is created with the default `member_investigate`
role (chat + investigate); promote them via the admin Users & Roles view
(see below).

---

## Scopes and roles

Authorization is by **scope**, not role. Handlers gate on scopes; roles
are just named bundles of scopes. The single source of truth is
`opsrag/auth/scopes.py`.

### The four scopes

| Scope | Grants | Example endpoints |
|---|---|---|
| `chat` | ask questions, read/delete your OWN sessions, submit corrections | `POST /query`, `GET/DELETE /sessions/{id}`, `POST /correction` |
| `investigate` | run incident investigations (also implies `chat`) | `POST /investigations/...` |
| `mcp` | mint / list / revoke your own MCP bearer tokens | `POST/GET/DELETE /api/mcp/tokens` |
| `admin` | everything: usage/cost org-wide, indexing, correction review, cache purge, user & role management, MCP audit log | `GET /admin/usage`, `POST /index/repo`, `GET/PUT /admin/...`, `GET /admin/users`, `PUT /admin/users/{id}/roles` |

A missing scope returns **403** with
`{"error": "forbidden", "reason": "missing_scope", "scope": "<scope>"}`
-- distinct from the **401** the auth layer raises for an
unauthenticated request.

### The role -> scope map

| Role | Scopes |
|---|---|
| `admin` | `chat` + `investigate` + `mcp` + `admin` |
| `member_investigate` (the default) | `chat` + `investigate` |
| `member_chat` | `chat` |
| `member_mcp` | `mcp` |

`member_investigate` is the default role for an authenticated user with no
explicit mapping -- a signed-in user is never left with zero scopes.
Unknown role names contribute no scopes (default-deny).

### How users get roles

- **oidc mode**: roles are derived from the token's `groups` (or `roles`)
  claim via `auth.role_mappings` (`{group: [roles]}`). A user whose groups
  match nothing gets `member_investigate`. Example:

  ```yaml
  auth:
    mode: oidc
    issuer: https://your-org.okta.com/oauth2/default
    audience: api://opsrag
    role_mappings:
      sre-admins:    [admin]
      oncall:        [member_investigate]
      readonly-team: [member_chat]
  ```

  Make sure your IdP actually emits the matching group values in a
  `groups` (or `roles`) claim on the access token.

- **login mode**: roles are stored on the user record. The seeded admin
  gets `admin`; new password and SSO users get `member_investigate`.
  Admins change roles in the UI's Users & Roles view, which calls:

  ```sh
  # List users with their roles + derived scopes (admin scope required).
  curl -sf http://localhost:8080/admin/users -b cookies.txt | jq

  # Replace a user's roles (admin scope required).
  curl -sf -X PUT http://localhost:8080/admin/users/<user-id>/roles \
    -b cookies.txt -H 'Content-Type: application/json' \
    -d '{"roles": ["member_investigate", "member_mcp"]}' | jq
  ```

  Role names are validated against the catalog (unknown -> 400). An admin
  cannot strip their OWN `admin` role (self-lockout guard). Changing a
  user's roles revokes their refresh sessions, so the new roles take
  effect on their next sign-in / refresh rather than only after the old
  (15-min) session cookie expires.

Because the UI's nav gating and the server-side guards both read
`has_scope` from the same scope model, the UI never shows a control the
server then 403s on.

---

## MCP token auth

External MCP clients (Claude Code's `mcp-remote`, Cursor, etc.) connect
to OpsRAG's MCP wire protocol (`GET /api/mcp/sse`, `POST /api/mcp/messages`)
using a dedicated bearer token, NOT a session cookie or OIDC JWT. These
`opsrag_`-prefixed tokens are minted by an authenticated user who holds
the `mcp` scope.

### Minting a token

`POST /api/mcp/tokens` is browser/session-authed (it requires the `mcp`
scope), so call it with whatever credential your mode uses -- a session
cookie in `login` mode, an OIDC bearer in `oidc` mode:

```sh
# login mode (session cookie):
curl -sf -X POST http://localhost:8080/api/mcp/tokens -b cookies.txt \
  -H 'Content-Type: application/json' \
  -d '{"name": "claude-code-laptop", "expires_in_days": 90}' | jq

# oidc mode (OIDC bearer):
curl -sf -X POST http://localhost:8080/api/mcp/tokens \
  -H "Authorization: Bearer $OIDC_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"name": "ci-runner", "expires_in_days": 30}' | jq
```

Request body:

- `name` (required) -- human label shown in the token list (1-120 chars).
- `expires_in_days` (optional) -- 1-365; `null` means never expires.

The response includes the plaintext token EXACTLY ONCE:

```json
{
  "id": "<uuid>",
  "name": "claude-code-laptop",
  "token": "opsrag_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "created_at": "2026-06-16T12:00:00Z",
  "expires_at": "2026-09-14T12:00:00Z"
}
```

Store the `token` value immediately -- it is hashed at rest and is NOT
retrievable again. (Minting requires a concrete identified user holding
the `mcp` scope; an unauthenticated caller is rejected by the auth layer
before it reaches the handler.)

### Listing and revoking

```sh
# List your tokens (metadata only, no plaintext):
curl -sf http://localhost:8080/api/mcp/tokens -b cookies.txt | jq

# Revoke one of your tokens by id (204 on success):
curl -sf -X DELETE http://localhost:8080/api/mcp/tokens/<token-id> -b cookies.txt
```

You can only list/revoke your OWN tokens. Admins can review usage across
the org via the MCP audit log (admin scope).

### Using the token

Point your MCP client at the SSE endpoint with the token as a Bearer:

```
Authorization: Bearer opsrag_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

The wire-protocol endpoints validate the token in-handler (against
`MCPTokenStore`) and 401 on missing / invalid / revoked / expired tokens;
they bypass the global session/OIDC enforcement because they are
self-protecting.

---

## Per-session ownership

Authentication answers "who is calling"; ownership answers "what they may
touch". A conversation thread (session) is owned by the verified identity
that created it, and only that owner can read or delete it.

The owner is the caller's stable subject id (`current_user.oid`, the JWT
`sub` in `oidc` mode or the session user id in `login` mode), recorded
into the thread's metadata when it is first written.

On every single-thread read or delete (e.g. `GET /sessions/{id}`,
`GET /sessions/{id}/messages`, `DELETE /sessions/{id}`), OpsRAG compares
the thread's recorded owner to the caller
(`opsrag/api/routes.py: _deny_if_not_owner`). Three deliberate
properties:

- **Cross-user access returns 404, not 403.** A non-owner gets "session
  not found" -- the same response as a thread that does not exist.
  Returning 403 would be an existence oracle.
- **Every protected request has a real owner.** Because authentication is
  always enforced, any caller that reaches a session read/delete route is
  an authenticated identity with a stable `oid`, so the owner check always
  runs. (The ownership guard short-circuits only for an anonymous caller,
  which can no longer reach these routes -- there is no open mode.)
- **Legacy anonymous-owned threads are grandfathered.** Threads created
  before owner binding existed (empty or `"anonymous"` owner) stay
  readable/deletable by anyone; only threads with a real authenticated
  owner are guarded.

Ownership binds to the VERIFIED id. A client may still pass a `user_id`
in the request body for memory/personalization, but it never overrides
the owner binding.

> Note: **investigations are shared, not owner-scoped.** They are gated by
> the `investigate` scope and visible to everyone who holds it -- by
> design, since an incident is a team artifact. Sessions and usage are
> owner-scoped; investigations are scope-gated and shared.

---

## Rate limiting

OpsRAG applies two independent throttles, both backed by a pluggable
storage seam (`opsrag/api/rate_limit_backend.py`):

1. **Per-request rate limit** -- `RateLimitMiddleware` caps requests per
   client to `api.rate_limit_rpm` per minute. Over the limit returns HTTP
   429 with a `Retry-After`.
2. **Login lockout** -- `LoginRateLimiter` (used only in `auth.mode:
   login`) throttles failed `POST /auth/login` attempts. After
   `login_max_attempts` failures within `login_window_seconds`, the key
   (email + IP) is locked for `login_lockout_seconds`; a successful login
   clears the counter.

### Backend selection: memory vs redis

```yaml
api:
  rate_limit_enabled: true
  rate_limit_rpm: 60
  rate_limit_backend: memory        # "memory" (default) or "redis"
  redis_url_env: OPSRAG_REDIS_URL   # env var holding the Redis URL
```

- `memory` (default) -- in-process counters. Correct and dependency-free
  for a SINGLE replica, but per-process: with multiple replicas each
  enforces its own limit (effective aggregate = `rpm x replicas`) and a
  login lockout on one pod is not seen by the others.
- `redis` -- shared state across replicas. Use this for any
  horizontally-scaled deployment.

**Redis is REQUIRED when selected.** With `rate_limit_backend: redis`,
OpsRAG reads the URL from `redis_url_env`, `PING`s at startup, and FAILS
FAST if it is unreachable -- there is no silent fallback to in-memory.
The login lockout thresholds live under `auth.login`
(`login_max_attempts` / `login_window_seconds` / `login_lockout_seconds`)
and ride the same backend when it is `redis`.

See [`operations.md`](./operations.md#rate-limiting-across-replicas) for
the multi-replica checklist.

---

## Production hardening checklist

General

- [ ] Pick `auth.mode` deliberately (`login` default, or `oidc`).
      Authentication is always enforced -- there is no `open` mode to
      disable -- but verify the seeded admin password / IdP config is
      production-grade before exposing the deployment.
- [ ] Terminate TLS in front of OpsRAG; serve every route over HTTPS.
- [ ] Restrict the always-open paths (`/docs`, `/redoc`, `/openapi.json`)
      at the gateway if you do not want the schema public.
- [ ] Keep secrets in env vars or mounted secret files -- never inline in
      `config.yaml`. Tokens already live behind `*_env` indirection.

oidc mode

- [ ] `issuer` and `audience` are set and confirmed against a DECODED
      sample token (`iss`/`aud` match exactly).
- [ ] Use a CUSTOM authorization server / API registration so you control
      the `aud` (Okta, Entra, Auth0, Keycloak audience mapper).
- [ ] Configure `role_mappings` for least privilege (don't leave everyone
      at the default `member_investigate` if some should be admins or
      chat-only).
- [ ] Confirm your IdP emits the `groups` (or `roles`) claim that your
      `role_mappings` keys on.

login mode

- [ ] `OPSRAG_SESSION_SIGNING_KEY` is a fresh, high-entropy value (>= 32
      bytes), distinct per environment, and rotated periodically (prefer
      `signing_key_path` to a mounted secret).
- [ ] `OPSRAG_ADMIN_PASSWORD` is strong and supplied via a secret;
      consider changing it (or disabling password and using SSO-only)
      after first boot.
- [ ] `session.provider: postgres` (with `POSTGRES_DSN`) so users and
      sessions survive restarts and are shared across replicas.
- [ ] `cookie_secure: true` (default) and an appropriate `cookie_samesite`
      (`lax` for typical, `strict` if no cross-site flows). Use `none`
      only with `Secure` and a deliberate cross-site setup.
- [ ] `sso_callback_base` is set to the exact external URL the browser
      uses (incl. any `/api` proxy prefix), and matches the redirect URIs
      registered with each IdP.
- [ ] SSO client SECRETS come from env (`client_secret_env`), never
      inline.
- [ ] Keep the account-takeover guard intact (don't disable the
      `email_verified` requirement for SSO linking).

Multi-replica / scaling

- [ ] `api.rate_limit_backend: redis` with a reachable `OPSRAG_REDIS_URL`,
      so the per-request limit and the login lockout are enforced
      cluster-wide (it fails fast if Redis is unreachable).

Operational

- [ ] Confirm the Bearer token and `sub` are never logged (default
      behavior); enable per-user telemetry deliberately only where needed.
- [ ] Review MCP tokens: set `expires_in_days`, revoke unused tokens, and
      watch the admin MCP audit log.
- [ ] Tighten `role_mappings` / user roles toward least privilege; audit
      who holds the `admin` scope.

---

## See also

- [`configuration.md`](./configuration.md) -- the full `auth` and `api`
  config blocks and env precedence.
- [`operations.md`](./operations.md) -- day-2 ops, including rate limiting
  across replicas and the security hardening checklist.
- [`architecture.md`](./architecture.md) -- where the verifier and
  middleware sit in the request flow.
- [`mcp-integrations.md`](./mcp-integrations.md) -- the read-only
  integrations exposed over the MCP wire protocol.
</content>
</invoke>
