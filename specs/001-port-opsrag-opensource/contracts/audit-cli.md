# Contract: `scripts/audit-vendor-neutrality.sh`

**Status**: New for this feature. Implements FR-014.

## Synopsis

```
audit-vendor-neutrality.sh [--json] [--fix-suggestions] [--exclude <path>]...
```

## Behaviour

The script scans tracked files (`git ls-files`) in the working tree for
three classes of violation and exits with status 0 only if all three
checks pass.

### Check 1 — proprietary names

- Case-insensitive `grep -E` for a denylist defined inline in the script:
  `acme`, `acme.com`, `acme.io`, plus any internal account
  IDs or known internal hostnames the maintainers add over time.
- Allowed exceptions (passed via `--exclude` or hard-coded in the script):
  - `CHANGELOG.md` (one historical reference is permitted by FR-001).
  - `samples/` (synthetic fixtures using the `acme-notes` placeholder).
  - `specs/` (project-management documents that legitimately describe
    what is being removed; not a shipped runtime artefact). The
    exemption is justified in `plan.md` §"Post-design re-check".
  - `.specify/` (Spec Kit scaffolding, not shipped at runtime).
- A violation is a `(file, line, snippet, matched-token)` tuple.

### Check 2 — non-English text

- `LC_ALL=C grep -lE '[<extended-ASCII-and-CJK-ranges>]'` across:
  `*.py`, `*.tsx`, `*.ts`, `*.jsx`, `*.js`, `*.md`, `*.yaml`, `*.yml`,
  `*.sh`, `*.tpl`, `Dockerfile*`.
- Allowed exceptions:
  - Files under `tests/fixtures/i18n/`.
  - Files containing an opt-in marker comment on line 1:
    `# audit:allow-non-ascii — <reason>`.
- A violation is a `(file, line, snippet, rule)` tuple.

### Check 3 — hardcoded hosts

- `grep -E '\b[a-z0-9-]+\.(com|net|io|org|dev|cloud|app)\b'` minus an
  allowlist of recognised placeholder domains:
  `example.com`, `example.org`, `example.net`, `example.dev`,
  `*.example.com`, `*.invalid`, `localhost`, `127.0.0.1`, `0.0.0.0`,
  `dex.local`, plus a handful of well-known docs hosts that legitimately
  appear in comments / README links (e.g. `helm.sh`, `kubernetes.io`,
  `python.org`).
- A violation is a `(file, line, snippet, matched-host)` tuple.

## Outputs

### Default (human)

Prints, per check, either `OK` or a table of violations grouped by file.
Exit code = sum of failing checks.

### `--json`

Emits the `AuditReport` shape documented in `data-model.md` §5. Single
JSON object on stdout, nothing else.

### `--fix-suggestions`

For each violation, prints a one-line suggestion (e.g. "replace
`acme-corp.com` with `example.com` or move into config").

## Determinism

- Violations are sorted by `(check, file, line)` so diffs between
  consecutive runs are stable.
- The script does NOT shell out to grep with locale-dependent
  defaults — `LC_ALL=C` is set explicitly.

## CI integration

- Runs as a GitHub Actions job named `audit` on every PR.
- Same script runs as a `pre-commit` hook locally (opt-in).
- Failure includes the offending line in the CI annotation so reviewers
  can act without leaving the PR view.

## Contract tests (CI gates)

- `tests/contract/test_audit_clean_tree.sh` — runs the audit on the
  current working tree; expects exit 0. This is the canonical "no
  regressions" gate.
- `tests/contract/test_audit_catches_proprietary.sh` — writes a temp file
  containing the literal string `Acme` under a path that is not
  exempt, runs the audit, asserts exit non-zero.
- `tests/contract/test_audit_catches_non_english.sh` — writes a temp file
  with a Vietnamese sentence, runs the audit, asserts exit non-zero.
- `tests/contract/test_audit_catches_hardcoded_host.sh` — writes a temp
  config containing `my-real-host.com`, runs the audit, asserts exit
  non-zero.
- `tests/contract/test_audit_json_shape.py` — runs with `--json`,
  parses stdout, asserts shape matches `data-model.md` §5.
