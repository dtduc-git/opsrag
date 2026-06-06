#!/usr/bin/env bash
# Contract test (T105): an unknown mcp.<name> key fails schema validation.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
command -v helm >/dev/null || { echo "SKIP: helm not installed"; exit 0; }
# template runs values.schema.json validation; an unknown mcp key must fail.
if helm template opsrag "${ROOT}/deploy/helm/opsrag" \
     --set mcp.bogus_mcp.enabled=false >/dev/null 2>/tmp/helm_schema_err; then
  echo "FAIL: unknown mcp key was accepted"; exit 1
fi
grep -qiE "additional|not allow|valid" /tmp/helm_schema_err || {
  echo "FAIL: rejected, but not with a schema-validation message:"; cat /tmp/helm_schema_err; exit 1; }
echo "helm schema rejects unknown mcp key OK"
