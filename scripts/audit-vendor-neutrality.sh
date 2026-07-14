#!/usr/bin/env bash
# audit-vendor-neutrality.sh -- scanner only.
#
# Three vendor-neutrality checks per
# `specs/001-port-opsrag-opensource/contracts/audit-cli.md`:
#   1. Proprietary names (denylist from rules file)
#   2. Non-English text (structural regex; exemptions from rules file)
#   3. Hardcoded hosts (regex match minus allowlist from rules file)
#
# Rules live in `scripts/audit-rules.yaml`. This script intentionally
# contains no denylist or allowlist content -- fill the rules file.
#
# Usage:
#   scripts/audit-vendor-neutrality.sh [--json] [--fix-suggestions]
#                                      [--exclude <path>]...
#                                      [--rules <path>]
#
# Exit code: 0 when all three checks pass, non-zero on any violation.

set -eu
export LC_ALL=C

# -----------------------------------------------------------------------------
# Bootstrap: locate the script and the default rules file.
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RULES_FILE="${SCRIPT_DIR}/audit-rules.yaml"

OUTPUT_MODE="human"          # human | json
EMIT_FIX_SUGGESTIONS=false
declare -a EXTRA_EXCLUDES=()

usage() {
    sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# -----------------------------------------------------------------------------
# CLI parsing.
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --json)              OUTPUT_MODE="json"; shift ;;
        --fix-suggestions)   EMIT_FIX_SUGGESTIONS=true; shift ;;
        --exclude)           EXTRA_EXCLUDES+=("$2"); shift 2 ;;
        --rules)             RULES_FILE="$2"; shift 2 ;;
        -h|--help)           usage; exit 0 ;;
        *)                   echo "unknown flag: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! -f "${RULES_FILE}" ]]; then
    echo "audit: rules file not found at ${RULES_FILE}" >&2
    exit 2
fi

# -----------------------------------------------------------------------------
# Rules loader.
#
# Delegates YAML parsing to python3 (pyyaml is in the project's dev deps).
# Each helper prints one value per line on stdout. Empty sections print
# nothing -- a freshly-initialised rules file produces zero findings.
# -----------------------------------------------------------------------------
load_list() {
    local key="$1"
    python3 - "$RULES_FILE" "$key" <<'PY'
import sys, yaml, pathlib
path, key = sys.argv[1], sys.argv[2]
data = yaml.safe_load(pathlib.Path(path).read_text()) or {}
for item in data.get(key) or []:
    if item is None:
        continue
    print(str(item))
PY
}

# -----------------------------------------------------------------------------
# File enumeration.
#
# Scan tracked files PLUS untracked-but-not-gitignored files. The
# `--others --exclude-standard` pass is essential: a freshly-ported tree can be
# almost entirely untracked, and a tracked-only audit (plain `git ls-files`)
# silently skips all of it -- exactly the blind spot that let internal tokens
# survive a "clean" audit. `.gitignore` is still honoured, so git-ignored
# real-value files (config-local.yaml, .env, design-scratch/, ...) are NEVER
# scanned, preserving the secret-isolation contract.
# -----------------------------------------------------------------------------
enumerate_files() {
    {
        git -C "${REPO_ROOT}" ls-files
        git -C "${REPO_ROOT}" ls-files --others --exclude-standard
    } | sort -u
}

is_excluded_path() {
    # $1 = candidate path (repo-relative)
    # $2 = newline-separated allowlist (path prefixes)
    local path="$1"
    local allowlist="$2"
    local pat
    while IFS= read -r pat; do
        [[ -z "$pat" ]] && continue
        if [[ "$path" == "$pat" || "$path" == "$pat"* || "$path" == "$pat"/* ]]; then
            return 0
        fi
    done <<< "$allowlist"
    if [[ ${#EXTRA_EXCLUDES[@]} -gt 0 ]]; then
        for pat in "${EXTRA_EXCLUDES[@]}"; do
            [[ -z "$pat" ]] && continue
            if [[ "$path" == "$pat" || "$path" == "$pat"* ]]; then
                return 0
            fi
        done
    fi
    return 1
}

# -----------------------------------------------------------------------------
# Findings accumulator (in-memory).
#
# Each finding is a tab-separated record:
#   check<TAB>file<TAB>line<TAB>snippet<TAB>matched-token<TAB>suggestion
# -----------------------------------------------------------------------------
declare -a FINDINGS=()

record() {
    local check="$1" file="$2" line="$3" snippet="$4" token="$5" suggestion="${6:-}"
    FINDINGS+=("${check}	${file}	${line}	${snippet}	${token}	${suggestion}")
}

# -----------------------------------------------------------------------------
# Check 1 -- proprietary names.
# -----------------------------------------------------------------------------
check_proprietary_names() {
    local denylist allowlist files pattern path
    denylist="$(load_list proprietary_names_denylist)"
    allowlist="$(load_list proprietary_names_allowlist_paths)"

    if [[ -z "$denylist" ]]; then
        return 0
    fi

    pattern="$(printf '%s' "$denylist" | paste -sd'|' -)"

    while IFS= read -r path; do
        [[ -z "$path" || ! -f "${REPO_ROOT}/${path}" ]] && continue
        is_excluded_path "$path" "$allowlist" && continue
        # Skip binary files (logos, favicons, fonts): a text name-scan on
        # binary bytes only yields false positives.
        grep -Iq . "${REPO_ROOT}/${path}" || continue
        while IFS=: read -r lineno content; do
            [[ -z "$lineno" ]] && continue
            match="$(printf '%s' "$content" \
                     | grep -oEi -- "${pattern}" | head -n1)"
            record proprietary_names "$path" "$lineno" \
                   "$content" \
                   "${match,,}" \
                   "rename or move to an exempt path"
        done < <(grep -nEi -- "${pattern}" "${REPO_ROOT}/${path}" 2>/dev/null)
    done < <(enumerate_files)
}

# -----------------------------------------------------------------------------
# Check 2 -- non-English text.
#
# Structural: any codepoint outside printable 7-bit ASCII + tab/newline is
# flagged. The pattern is fixed (matches the contract); only the exemption
# path list comes from the rules file. An opt-in line-1 marker is honoured:
#   # audit:allow-non-ascii -- <reason>
# -----------------------------------------------------------------------------
check_non_english_text() {
    local exempt files path first_line
    exempt="$(load_list non_english_exempt_paths)"

    while IFS= read -r path; do
        [[ -z "$path" || ! -f "${REPO_ROOT}/${path}" ]] && continue
        case "$path" in
            *.py|*.ts|*.tsx|*.js|*.jsx|*.md|*.yaml|*.yml|*.sh|*.tpl|Dockerfile*) ;;
            *) continue ;;
        esac
        is_excluded_path "$path" "$exempt" && continue

        first_line="$(head -n1 "${REPO_ROOT}/${path}" 2>/dev/null || true)"
        if [[ "$first_line" == *"audit:allow-non-ascii"* ]]; then
            continue
        fi

        while IFS=: read -r lineno content; do
            [[ -z "$lineno" ]] && continue
            record non_english_text "$path" "$lineno" \
                   "$content" \
                   "non-ascii" \
                   "translate to English or add a tests/fixtures/i18n exemption"
        done < <(python3 -c '
import sys, unicodedata
path = sys.argv[1]
try:
    with open(path, "rb") as f:
        for i, raw in enumerate(f, 1):
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError:
                print(f"{i}:<non-utf8 bytes>")
                continue
            for ch in line:
                o = ord(ch)
                if o == 9 or o == 10 or o == 13 or 0x20 <= o <= 0x7e:
                    continue
                # The rule targets FOREIGN-LANGUAGE text, i.e. actual words in
                # a non-Latin script -- NOT the Unicode symbols, math signs,
                # emoji, marks, or typographic punctuation that routinely appear
                # in English prose/comments/UX (em/en dashes, smart quotes,
                # ellipsis, arrows, bullets, middle dots, section signs, >=/<=,
                # 👍👎⚠️ and other load-bearing glyphs). Only a codepoint that is
                # a LETTER (Unicode category L*) AND not a Latin-script letter
                # (accented Latin like é/ü stays fine) is flagged.
                cat = unicodedata.category(ch)
                if not cat.startswith("L"):
                    continue
                try:
                    if unicodedata.name(ch).startswith("LATIN"):
                        continue
                except ValueError:
                    continue  # unnamed letter -> not a vendor-neutrality concern
                stripped = line.rstrip("\n")
                print(f"{i}:{stripped}")
                break
except OSError:
    pass
' "${REPO_ROOT}/${path}" 2>/dev/null)
    done < <(enumerate_files)
}

# -----------------------------------------------------------------------------
# Check 3 -- hardcoded hosts.
# -----------------------------------------------------------------------------
check_hardcoded_hosts() {
    local allowlist exempt files path
    allowlist="$(load_list hardcoded_hosts_allowlist)"
    exempt="$(load_list hardcoded_hosts_exempt_paths)"
    # Host detection is delegated to a python scanner that skips Python /
    # JS / etc. code-attribute access (e.g. `request.app.state`), which
    # the previous pure-regex approach false-flagged. The scanner emits
    # `lineno:host` for matches that look like hostnames in data context
    # (URLs, quoted strings, comments, config values), not code.
    # The scanner takes the allowlist as argv[2] and does exact + fnmatch
    # filtering IN-PROCESS, emitting only NON-allowlisted hosts as
    # `lineno:host`. This keeps the audit to ONE python invocation per file --
    # the previous design spawned a python3 fnmatch per matched host, which on
    # uv.lock's thousands of registry URLs made a full run take many minutes.
    local host_scanner='
import re, sys, fnmatch
PATTERN = re.compile(
    r"\b(?:[a-z0-9][a-z0-9-]*\.)+(?:com|net|io|org|dev|cloud|app)\b",
    re.IGNORECASE,
)
path = sys.argv[1]
allow_raw = sys.argv[2] if len(sys.argv) > 2 else ""
allow_exact = set()
allow_glob = []
for p in allow_raw.splitlines():
    p = p.strip().lower()
    if not p:
        continue
    if any(c in p for c in "*?["):
        allow_glob.append(p)
    else:
        allow_exact.add(p)

def allowed(host):
    if host in allow_exact:
        return True
    return any(fnmatch.fnmatch(host, g) for g in allow_glob)

try:
    with open(path, "rb") as f:
        for i, raw in enumerate(f, 1):
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            for m in PATTERN.finditer(line):
                host = m.group(0)
                # Skip code-attribute access: host followed by another
                # `.identifier` segment, e.g. `request.app.state`.
                tail = line[m.end():m.end()+2]
                if tail.startswith(".") and len(tail) > 1 and (tail[1].isalpha() or tail[1] == "_"):
                    continue
                lead = line[max(0, m.start()-3):m.start()]
                ch = line[m.start()-1] if m.start() > 0 else ""
                url_like = lead.endswith("://") or (lead and lead[-1] == "@")
                data_like = ch == "" or ch in " \t\"\x27`<>([{=:,;|/"
                if not (url_like or data_like):
                    continue  # looks like code identifier access
                if allowed(host.lower()):
                    continue  # allowlisted -> not a violation
                print(f"{i}:{host}")
except OSError:
    pass
'

    while IFS= read -r path; do
        [[ -z "$path" || ! -f "${REPO_ROOT}/${path}" ]] && continue
        is_excluded_path "$path" "$exempt" && continue
        grep -Iq . "${REPO_ROOT}/${path}" || continue  # skip binary files

        while IFS=: read -r lineno host; do
            [[ -z "$lineno" ]] && continue
            context="$(sed -n "${lineno}p" "${REPO_ROOT}/${path}" 2>/dev/null)"
            record hardcoded_hosts "$path" "$lineno" \
                   "$context" \
                   "${host,,}" \
                   "replace with example.com or move to config"
        done < <(python3 -c "$host_scanner" "${REPO_ROOT}/${path}" "$allowlist" 2>/dev/null)
    done < <(enumerate_files)
}

# -----------------------------------------------------------------------------
# Output.
# -----------------------------------------------------------------------------
emit_human() {
    local check counts_p counts_n counts_h
    counts_p=0; counts_n=0; counts_h=0
    for f in "${FINDINGS[@]:-}"; do
        [[ -z "$f" ]] && continue
        # NB: use $((x+1)) assignment, not ((x++)) -- the latter returns the
        # pre-increment value as its exit status, so the first increment
        # (0 -> 1) exits 1 and `set -e` would abort emit_human before it can
        # print the violations.
        case "${f%%	*}" in
            proprietary_names) counts_p=$((counts_p + 1)) ;;
            non_english_text)  counts_n=$((counts_n + 1)) ;;
            hardcoded_hosts)   counts_h=$((counts_h + 1)) ;;
        esac
    done

    printf "audit summary:\n"
    printf "  proprietary_names : %s\n" "$([ "$counts_p" -eq 0 ] && echo OK || echo "${counts_p} violation(s)")"
    printf "  non_english_text  : %s\n" "$([ "$counts_n" -eq 0 ] && echo OK || echo "${counts_n} violation(s)")"
    printf "  hardcoded_hosts   : %s\n" "$([ "$counts_h" -eq 0 ] && echo OK || echo "${counts_h} violation(s)")"

    if (( counts_p + counts_n + counts_h == 0 )); then
        return 0
    fi

    printf '\nviolations:\n'
    printf '%s\n' "${FINDINGS[@]}" \
        | sort -t$'\t' -k1,1 -k2,2 -k3,3n \
        | while IFS=$'\t' read -r check file line snippet token suggestion; do
              [[ -z "$check" ]] && continue
              # Truncate snippet for display at 200 *characters* (not bytes) so
              # multi-byte UTF-8 sequences aren't split mid-character.
              short_snippet="$(printf '%s' "$snippet" \
                  | python3 -c 'import sys; print(sys.stdin.read()[:200], end="")')"
              printf '  [%s] %s:%s  %s\n' "$check" "$file" "$line" "$short_snippet"
              printf '    matched: %s\n' "$token"
              if $EMIT_FIX_SUGGESTIONS && [[ -n "$suggestion" ]]; then
                  printf '    suggest: %s\n' "$suggestion"
              fi
          done
}

emit_json() {
    if [[ ${#FINDINGS[@]} -eq 0 ]]; then
        printf '{"violations": [], "summary": {"proprietary_names": 0, "non_english_text": 0, "hardcoded_hosts": 0}}\n'
        return 0
    fi
    printf '%s\n' "${FINDINGS[@]}" | python3 -c '
import json, sys
violations = []
for raw in sys.stdin:
    raw = raw.rstrip("\n")
    if not raw:
        continue
    parts = raw.split("\t")
    if len(parts) < 5:
        continue
    check, file_, line, snippet, token = parts[:5]
    suggestion = parts[5] if len(parts) > 5 else ""
    violations.append({
        "check": check,
        "file": file_,
        "line": int(line) if line.isdigit() else None,
        "snippet": snippet,
        "matched": token,
        "suggestion": suggestion or None,
    })
violations.sort(key=lambda v: (v["check"], v["file"], v["line"] or 0))
summary = {k: 0 for k in ("proprietary_names", "non_english_text", "hardcoded_hosts")}
for v in violations:
    summary[v["check"]] = summary.get(v["check"], 0) + 1
print(json.dumps({"violations": violations, "summary": summary}, ensure_ascii=False))
'
}

# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------
main() {
    if ! git -C "${REPO_ROOT}" rev-parse --git-dir >/dev/null 2>&1; then
        echo "audit: ${REPO_ROOT} is not a git working tree" >&2
        exit 2
    fi

    check_proprietary_names
    check_non_english_text
    check_hardcoded_hosts

    if [[ "$OUTPUT_MODE" == "json" ]]; then
        emit_json
    else
        emit_human
    fi

    [[ ${#FINDINGS[@]} -eq 0 ]] && exit 0 || exit 1
}

main "$@"
