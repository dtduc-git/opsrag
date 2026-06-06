#!/usr/bin/env bash
# Contract test (T129): a non-allowlisted hardcoded host is caught (exit non-zero).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
probe="opsrag/_audit_probe_$$.yaml"
cleanup() { git reset -q -- "$probe" 2>/dev/null || true; rm -f "$probe"; }
trap cleanup EXIT
# Assemble the host at runtime so no literal hostname appears in this test source.
host="my-real-host"".""com"
printf 'endpoint: "https://%s/api"\n' "$host" > "$probe"
git add -N "$probe"
scripts/audit-vendor-neutrality.sh >/dev/null 2>&1
rc=$?
[[ $rc -ne 0 ]] || { echo "FAIL: hardcoded host not caught"; exit 1; }
echo "audit catches hardcoded host OK"
