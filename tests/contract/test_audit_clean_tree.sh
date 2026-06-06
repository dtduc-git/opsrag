#!/usr/bin/env bash
# Contract test (T126): the audit passes on the current tree (exit 0).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
"${ROOT}/scripts/audit-vendor-neutrality.sh" >/dev/null 2>&1
rc=$?
if [[ $rc -ne 0 ]]; then echo "FAIL: audit did not pass clean tree (rc=$rc)"; exit 1; fi
echo "audit clean-tree OK"
