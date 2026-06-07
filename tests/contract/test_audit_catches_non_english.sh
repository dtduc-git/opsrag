#!/usr/bin/env bash
# Contract test (T128): non-English (non-ASCII) text is caught (exit non-zero).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
probe="opsrag/_audit_probe_$$.py"
cleanup() { git reset -q -- "$probe" 2>/dev/null || true; rm -f "$probe"; }
trap cleanup EXIT
# Build a Vietnamese sentence at runtime so this .sh source stays ASCII.
python3 -c "open('$probe','w',encoding='utf-8').write('# '+'Ti'+chr(0x1EBF)+'ng Vi'+chr(0x1EC7)+'t test\n')"
# -f: `_audit_probe_*` is gitignored, so a plain `git add -N` is rejected and the
# probe is invisible to the audit's `git ls-files` scan -> force the intent-to-add.
git add -f -N "$probe"
scripts/audit-vendor-neutrality.sh >/dev/null 2>&1
rc=$?
[[ $rc -ne 0 ]] || { echo "FAIL: non-English text not caught"; exit 1; }
echo "audit catches non-English OK"
