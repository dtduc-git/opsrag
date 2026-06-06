# Authentication

opsrag authenticates every API request with an OIDC Bearer token. There is
no API-key path: callers obtain a JWT from your identity provider (IdP) and
send it on each request. This document explains how verification works and
how to point opsrag at common identity providers.

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
