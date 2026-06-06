#!/usr/bin/env bash
# Integration test (T125): install the chart on a kind cluster and curl /healthz.
#
# Stack-gated: requires `kind`, `kubectl`, `helm`, and OPSRAG_KIND_E2E=1.
# Skipped (exit 0) otherwise so it never blocks the default suite. CI runs it
# in a dedicated job that provisions kind first.
set -euo pipefail

if [[ "${OPSRAG_KIND_E2E:-}" != "1" ]]; then
  echo "SKIP: set OPSRAG_KIND_E2E=1 (and have kind/kubectl/helm) to run"; exit 0
fi
for bin in kind kubectl helm; do
  command -v "$bin" >/dev/null || { echo "SKIP: $bin not installed"; exit 0; }
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLUSTER="opsrag-e2e"
NS="opsrag"

cleanup() { kind delete cluster --name "$CLUSTER" >/dev/null 2>&1 || true; }
trap cleanup EXIT

kind create cluster --name "$CLUSTER"
# Build + load the image if a local Dockerfile build is desired; here we assume
# image.repository:tag is pullable, or pre-loaded by the CI job.
helm install opsrag "${ROOT}/deploy/helm/opsrag" \
  --namespace "$NS" --create-namespace \
  --set auth.issuer=http://dex.local/dex --set auth.audience=opsrag \
  --wait --timeout 180s

kubectl -n "$NS" rollout status deploy/opsrag --timeout=120s
# Exec into the api pod and curl its own /healthz.
POD="$(kubectl -n "$NS" get pod -l app.kubernetes.io/component=api -o jsonpath='{.items[0].metadata.name}')"
kubectl -n "$NS" exec "$POD" -- curl -sf "http://localhost:8080/healthz"
echo "kind e2e: /healthz OK"
