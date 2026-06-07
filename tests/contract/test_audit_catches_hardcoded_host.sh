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
# -f: `_audit_probe_*` is gitignored (a guard against committing probe fixtures),
# so a plain `git add -N` is rejected and the probe lands in neither
# `git ls-files` nor the `--others --exclude-standard` pass -> the audit never
# sees it. Force the intent-to-add so it shows in `git ls-files` and IS scanned.
git add -f -N "$probe"
scripts/audit-vendor-neutrality.sh >/dev/null 2>&1
rc=$?
[[ $rc -ne 0 ]] || { echo "FAIL: hardcoded host not caught"; exit 1; }
echo "audit catches hardcoded host OK"
