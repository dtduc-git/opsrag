#!/usr/bin/env bash
# Contract test (T101): the chart lints clean.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
command -v helm >/dev/null || { echo "SKIP: helm not installed"; exit 0; }
helm lint "${ROOT}/deploy/helm/opsrag"
echo "helm lint OK"
