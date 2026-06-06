# Quickstart: opsrag in fifteen minutes

**Branch**: `001-port-opsrag-opensource` | **Date**: 2026-05-28

This walkthrough satisfies User Story 1's acceptance test: a new evaluator
clones the repository, sets one secret, runs one command, and answers a
question about an indexed sample runbook. Total time budget: 15 minutes.

## Prerequisites

You need:

- Docker Desktop or Docker Engine ≥ 24, with Compose v2
- ~6 GB of free disk for images and a Postgres / Qdrant volume
- An API key for one supported LLM provider (default: Anthropic;
  swap by editing `config.yaml`)
- `curl` and `jq`

You do NOT need:

- Kubernetes (Helm install is a separate path, covered by User Story 3)
- An external OIDC identity provider (the local compose stack bundles
  Dex — see FR-017)
- Neo4j or any cloud account (default config uses the null graph
  backend — FR-019 — and zero MCP integrations)

## Steps

### 1. Clone

```bash
git clone https://github.com/<org>/opsrag.git
cd opsrag
```

### 2. Set your LLM key

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=... (or your provider's equivalent).
# Every other value can stay at its placeholder.
```

### 3. Start the stack

```bash
docker compose -f deploy/compose/docker-compose.yaml up -d
```

The compose file starts:

| Service | Image | Purpose |
|---|---|---|
| `opsrag-api` | `ghcr.io/<org>/opsrag:<tag>` | FastAPI backend |
| `opsrag-ui`  | `ghcr.io/<org>/opsrag-ui:<tag>` | React SPA |
| `qdrant`     | `qdrant/qdrant:v1.13` | Vector store |
| `postgres`   | `postgres:16` | Sessions / memory / checkpoints |
| `dex`        | `dexidp/dex:v2.41` | Bundled local OIDC (FR-017) |
| `phoenix`    | `arizephoenix/phoenix:<tag>` | Local trace viewer |

Expected ready time: under 30 seconds (SC-005).

### 4. Verify health

```bash
curl -sf http://localhost:8080/healthz
curl -sf http://localhost:8080/readyz
```

`/readyz` returns 200 once Postgres is migrated and Qdrant accepts
connections.

### 5. Index the bundled sample corpus

```bash
docker compose exec opsrag-api scripts/seed-sample-corpus.sh
```

This indexes ~12 synthetic runbooks / postmortems / manifests for the
fictional product "Acme Notes" — used everywhere this walkthrough mentions
sample data.

### 6. Get an OIDC token

The bundled Dex ships with one static user (`evaluator@example.com` /
`evaluator`). To exchange those credentials for a token:

```bash
TOKEN=$(curl -sf -X POST \
  -d 'grant_type=password' \
  -d 'username=evaluator@example.com' \
  -d 'password=evaluator' \
  -d 'client_id=opsrag-local' \
  -d 'client_secret=local-secret' \
  -d 'scope=openid profile email' \
  http://localhost:5556/dex/token | jq -r .access_token)
```

(Outside the compose stack, your real IdP issues this token however you
normally acquire one. The backend treats `localhost:5556/dex` as a normal
OIDC issuer.)

### 7. Ask a question

```bash
curl -sf -X POST http://localhost:8080/query \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"query":"How do I roll back an Acme Notes deployment?"}' | jq
```

Expected response shape (abridged):

```json
{
  "answer": "Follow the Acme Notes rollback runbook: ...",
  "citations": [
    {
      "source": "samples/runbooks/002-acme-notes-db-failover.md",
      "title": "Acme Notes — DB failover & rollback",
      "snippet": "..."
    }
  ],
  "session_id": "...",
  "trace_id": "..."
}
```

Open the Phoenix UI at <http://localhost:6006> to see the full LangGraph
trace, including retrieval, grading, and generation spans.

### 8. (Optional) Try the UI

Browse to <http://localhost:5173>. The UI proxies to the same backend and
performs its own OIDC handshake against Dex.

## What's NOT happening (yet)

By design, **no MCP integration is active**. The agent answers from the
indexed sample corpus only. To activate, e.g., Kubernetes inspection:

```bash
# In config.yaml:
mcp:
  kubernetes:
    enabled: true
# Then provide credentials via env:
echo "KUBECONFIG=/path/to/kubeconfig" >> .env
docker compose restart opsrag-api
```

If `enabled: true` but `KUBECONFIG` is unset, the API refuses to start
with `MCP_MISCONFIGURED:kubernetes:KUBECONFIG` (FR-004).

## Failure modes you may hit (and what they mean)

| Symptom | Cause | Fix |
|---|---|---|
| `MCP_MISCONFIGURED:<name>:<env>` at startup | An MCP block has `enabled: true` but a required env var is unset | Either disable the MCP or set the variable |
| `AUTH_MISCONFIGURED:issuer_unreachable` | Backend started before Dex was healthy | `docker compose restart opsrag-api` |
| 401 on every request | `TOKEN` expired (default lifetime 1 h) | Re-run step 6 |
| `/readyz` flips between 200 and 503 | Postgres restart during migrations | Wait ~30 s, then re-curl |

## Move on

When this walkthrough succeeds end-to-end:

- **User Story 2** (enable an integration): pick one MCP from `config.yaml`
  and flip its `enabled` to `true`.
- **User Story 3** (Helm install): use `deploy/helm/opsrag/values.yaml`
  with your own credentials and run
  `helm install opsrag deploy/helm/opsrag`.
- **User Story 4** (audit): run `scripts/audit-vendor-neutrality.sh`.
