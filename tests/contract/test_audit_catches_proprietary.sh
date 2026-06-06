#!/usr/bin/env bash
# Contract test (T127): a proprietary denylist token is caught (exit non-zero).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
probe="opsrag/_audit_probe_$$.py"
cleanup() { git reset -q -- "$probe" 2>/dev/null || true; rm -f "$probe"; }
trap cleanup EXIT
# Assemble the denylisted company-name placeholder so this script stays clean itself.
tok="Example-""Corp"
printf '# vendor reference: %s internal\n' "$tok" > "$probe"
git add -N "$probe"
scripts/audit-vendor-neutrality.sh >/dev/null 2>&1
rc=$?
[[ $rc -ne 0 ]] || { echo "FAIL: proprietary token not caught"; exit 1; }
echo "audit catches proprietary OK"
