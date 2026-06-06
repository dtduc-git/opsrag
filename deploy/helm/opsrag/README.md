# opsrag Helm chart

Deploys the opsrag API (and optionally the UI and Slack-bot workers) to
Kubernetes. All 14 MCP integrations are present and disabled by default; each
`mcp.<name>.enabled` flag is wired to the api container as
`OPSRAG_MCP_<NAME>_ENABLED`.

## Install

```sh
helm install opsrag deploy/helm/opsrag \
  --namespace opsrag --create-namespace \
  -f my-values.yaml
```

A minimal `my-values.yaml`:

```yaml
image:
  repository: ghcr.io/OWNER/opsrag
  tag: "0.1.0"
auth:
  issuer: https://your-idp.example.com
  audience: opsrag
# Provide secrets (LLM key, POSTGRES_DSN, MCP creds) via an existing Secret:
api:
  envFromSecret: opsrag-secrets
mcp:
  prometheus:
    enabled: true   # provide KUBECONFIG via the secret above
```

Enabling an MCP without its required credentials makes the pod fail fast at
startup with `MCP_MISCONFIGURED:<name>:<env>`.

## Deploy on AWS (EKS / Bedrock)

Use the ready-made overlay [`values-aws.yaml`](./values-aws.yaml). It sets
`config.cloudProvider: aws` (Bedrock model bundle: Sonnet 4.6 reason/pro, Haiku
4.5 tools, Cohere Embed v4 = 1536-dim, Cohere Rerank 3.5 on Bedrock), an ALB
ingress, autoscaling, a PDB, and the index Job.

Prerequisites:

1. **IRSA** — create an IAM role trusted by the cluster OIDC provider, scoped to
   the `opsrag` ServiceAccount, with a policy granting `bedrock:InvokeModel`
   (and `bedrock:InvokeModelWithResponseStream`) on the chosen model ARNs.
   ```sh
   eksctl create iamserviceaccount --name opsrag --namespace opsrag \
     --cluster <cluster> --attach-policy-arn <bedrock-policy-arn> \
     --role-name opsrag-bedrock --approve
   ```
   Put the resulting role ARN in `serviceAccount.annotations`
   (`eks.amazonaws.com/role-arn`). No static AWS keys are used — the SDK reads
   the projected IRSA token; `AWS_REGION` is set via env.
2. **Secret** — credentials are referenced by `api.envFromSecret`, never inlined:
   ```sh
   kubectl create secret generic opsrag-secrets --namespace opsrag \
     --from-literal=POSTGRES_DSN='postgres://...' \
     --from-literal=OPSRAG_SESSION_SIGNING_KEY='<random>' \
     --from-literal=OPSRAG_ADMIN_PASSWORD='<bootstrap-admin>' \
     --from-literal=GITLAB_TOKEN='<token>' \
     --from-literal=CONFLUENCE_API_TOKEN='<token>'
   # add SSO client secrets as needed
   ```
3. Enable Bedrock **model access** in the region (Bedrock console → Model access).

```sh
helm upgrade --install opsrag deploy/helm/opsrag \
  --namespace opsrag --create-namespace \
  -f deploy/helm/opsrag/values-aws.yaml \
  --set image.repository=<your-ecr>/opsrag --set image.tag=<tag>
```

Edit the `<ACCOUNT_ID>`, region, ACM cert ARN, host, and Qdrant/Postgres values
in the overlay before applying.

## Deploy on GCP (GKE / Vertex AI)

Use [`values-gcp.yaml`](./values-gcp.yaml). It sets `config.cloudProvider: gcp`
(Vertex bundle: Gemini 2.5 Flash reason/tools, Gemini 2.5 Pro escalation,
`gemini-embedding-001` = 3072-dim, `semantic-ranker-default-004`), a GKE managed
Ingress, autoscaling, a PDB, and the index Job.

Prerequisites:

1. **Workload Identity** — bind a GCP service account (with
   `roles/aiplatform.user`) to the `opsrag` KSA:
   ```sh
   gcloud iam service-accounts create opsrag-vertex --project <PROJECT_ID>
   gcloud projects add-iam-policy-binding <PROJECT_ID> \
     --member "serviceAccount:opsrag-vertex@<PROJECT_ID>.iam.gserviceaccount.com" \
     --role roles/aiplatform.user
   gcloud iam service-accounts add-iam-policy-binding \
     opsrag-vertex@<PROJECT_ID>.iam.gserviceaccount.com \
     --role roles/iam.workloadIdentityUser \
     --member "serviceAccount:<PROJECT_ID>.svc.id.goog[opsrag/opsrag]"
   ```
   Put the GSA email in `serviceAccount.annotations`
   (`iam.gke.io/gcp-service-account`). No JSON keys are used — the GKE metadata
   server vends tokens; `GOOGLE_CLOUD_PROJECT`/`GOOGLE_CLOUD_REGION` are env.
2. **Secret** — same `kubectl create secret generic opsrag-secrets ...` as the
   AWS section (POSTGRES_DSN, OPSRAG_SESSION_SIGNING_KEY, OPSRAG_ADMIN_PASSWORD,
   GITLAB_TOKEN, CONFLUENCE_API_TOKEN, SSO secrets). Referenced via
   `api.envFromSecret`, never inlined.
3. Enable the **Vertex AI API** (`aiplatform.googleapis.com`); for the managed
   Ingress, reserve a global static IP and create a `ManagedCertificate` named
   to match the overlay annotations.

```sh
helm upgrade --install opsrag deploy/helm/opsrag \
  --namespace opsrag --create-namespace \
  -f deploy/helm/opsrag/values-gcp.yaml \
  --set image.repository=<region>-docker.pkg.dev/<project>/opsrag/opsrag \
  --set image.tag=<tag>
```

Edit `<PROJECT_ID>`, region/location, host, cert/IP names, and Qdrant/Postgres
values in the overlay before applying.

> The embedding `dimension` MUST match both the embedding model and the Qdrant
> collection. Switching embedding models against an existing collection requires
> a reindex with `config.vectorStore.allowDimensionChange: true`.

## Values reference

| Key | Default | Purpose |
|---|---|---|
| `image.repository` / `image.tag` | `ghcr.io/OWNER/opsrag` / `0.1.0` | API image (required) |
| `auth.issuer` / `auth.audience` | example values | OIDC settings (required) |
| `api.replicaCount` | `2` | API replicas (ignored when `autoscaling.enabled`) |
| `api.envFromSecret` | `""` | Existing Secret injected as env (LLM/MCP creds) |
| `ui.enabled` | `true` | Deploy the React UI + its Service |
| `slackBot.enabled` | `false` | Deploy the Socket-Mode Slack worker |
| `serviceAccount.create` | `true` | Create a ServiceAccount |
| `ingress.enabled` | `false` | Create an Ingress for the API |
| `config.*` | see `values.yaml` | Rendered into a ConfigMap mounted at `/etc/opsrag/config.yaml` |
| `config.cloudProvider` | `""` | `aws` / `gcp` / `""`. Fills unset model slots from a Bedrock/Vertex bundle |
| `config.embedding.dimension` | `""` | Embedding dim; MUST match the model + Qdrant collection |
| `config.{llm,embedding,reranker}.awsRegion` | `""` | Bedrock region (emitted only when set) |
| `config.{llm,embedding,reranker}.project` / `.location` | `""` | Vertex project/location (emitted only when set) |
| `config.vectorStore.allowDimensionChange` | `false` | Allow embed-dim change (intentional reindex only) |
| `serviceAccount.annotations` | `{}` | SA annotations (IRSA `role-arn` / WI `gcp-service-account`) |
| `mcp.<name>.enabled` | `false` (all 14) | Enable an integration; wired to `OPSRAG_MCP_<NAME>_ENABLED` |
| `mcp.<name>.secretRef` | `""` | Secret carrying that integration's credentials |
| `secret.create` / `secret.data` | `false` / `{}` | Opt-in chart-managed Secret |
| `networkPolicy.enabled` | `false` | Egress allowlist (DNS + intra-namespace + extras) |
| `podDisruptionBudget.enabled` | `false` | PDB for the API |
| `autoscaling.enabled` | `false` | HPA for the API (CPU target) |

The full, authoritative schema is `values.schema.json` (validated at
`helm install` time; unknown top-level or `mcp:` keys are rejected).

## What gets created

Always: API `Deployment` + `Service`, a `ConfigMap`, and (by default) a
`ServiceAccount` and the UI `Deployment` + `Service`. Opt-in: `Ingress`,
`Secret`, `NetworkPolicy`, `PodDisruptionBudget`, `HorizontalPodAutoscaler`,
and the Slack-bot `Deployment`.

## Test the release

```sh
helm test opsrag --namespace opsrag
```

runs a pod that curls the API `/healthz`.

## Upgrade / uninstall

```sh
helm upgrade opsrag deploy/helm/opsrag -f my-values.yaml
helm uninstall opsrag --namespace opsrag
```

The chart version (`Chart.yaml: version`) tracks chart changes independently of
the application `appVersion`.
