# Contract: HTTP API

**Status**: Frozen for this feature. Inherited from upstream; auth model
updated to OIDC-only (FR-016).

## Authentication

- Every endpoint except `/healthz` and `/readyz` requires
  `Authorization: Bearer <token>`.
- Token is validated against `auth.issuer`'s JWKS endpoint; `iss`, `aud`,
  and `exp` claims are enforced.
- Rejected requests return `401` with body
  `{"error":"unauthenticated","reason":"<one of: missing_bearer | invalid_signature | issuer_mismatch | audience_mismatch | expired>"}`.
- Successful requests propagate the token's `sub` claim into request-scoped
  context for usage attribution; the claim is NEVER logged in cleartext or
  returned in responses.

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/healthz` | none | Liveness — returns 200 if process is alive |
| GET | `/readyz` | none | Readiness — returns 200 only if config validated AND all enabled integrations probed reachable |
| POST | `/query` | OIDC | Synchronous or SSE-streamed agent query |
| GET | `/usage` | OIDC | Per-caller token + cost summary |
| GET | `/usage/{session_id}` | OIDC | Per-session usage detail |
| GET | `/sessions/{user_id}` | OIDC | List sessions for a user (subject to RBAC if introduced later) |
| DELETE | `/sessions/{thread_id}` | OIDC | Delete a session |
| GET | `/indexing/status` | OIDC | Per-source indexing progress |
| POST | `/index/repo` | OIDC | Trigger repository indexing |
| POST | `/webhook/gitlab` | webhook secret | GitLab push webhook (HMAC validated) |
| POST | `/webhook/github` | webhook secret | GitHub push webhook (HMAC validated) |
| GET | `/docs` | OIDC | Swagger UI (gated to avoid public exposure of schema) |
| GET | `/openapi.json` | OIDC | Machine-readable schema |

## Request / response shapes

`POST /query` request:

```json
{
  "query": "string",
  "session_id": "string|null",
  "user_id": "string|null",
  "stream": false,
  "agent": "default|investigation"
}
```

`POST /query` synchronous response:

```json
{
  "answer": "string",
  "citations": [{"source": "string", "title": "string", "snippet": "string"}],
  "session_id": "string",
  "trace_id": "string|null"
}
```

`POST /query` streaming response (`text/event-stream`):

```
event: status
data: {"phase":"retrieving"}

event: chunk
data: {"delta":"string"}

event: done
data: {"session_id":"string","trace_id":"string"}

event: close
data: {}
```

## Error envelope

Non-2xx responses use a stable envelope:

```json
{"error": "<machine_code>", "reason": "<human readable>", "request_id": "<uuid>"}
```

Machine codes: `unauthenticated`, `forbidden`, `bad_request`,
`not_found`, `conflict`, `mcp_misconfigured`, `mcp_upstream_failure`,
`agent_timeout`, `internal`.

## Compatibility

The endpoint set and request/response shapes are inherited from upstream
to keep the existing UI and Slack bot working after the port. The only
breaking change is removal of `X-API-Key` header support — all callers
must move to Bearer tokens.

## Contract tests (CI gates)

- `tests/contract/test_openapi_shape.py` — pulls `/openapi.json`, asserts
  every documented endpoint is present and every endpoint has documented
  auth requirements.
- `tests/contract/test_auth_required.py` — asserts every non-health
  endpoint returns 401 without `Authorization`.
- `tests/contract/test_error_envelope.py` — asserts every 4xx/5xx
  response matches the envelope shape.
