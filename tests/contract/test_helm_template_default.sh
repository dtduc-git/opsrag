#!/usr/bin/env bash
# Contract test (T102): default render produces parseable YAML.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
command -v helm >/dev/null || { echo "SKIP: helm not installed"; exit 0; }
out="$(helm template opsrag "${ROOT}/deploy/helm/opsrag")"
echo "$out" | python3 -c "import sys,yaml; list(yaml.safe_load_all(sys.stdin))"
echo "helm template default OK"
