# Vendor-neutrality audit

opsrag ships no organization-specific content. A scripted audit
(`scripts/audit-vendor-neutrality.sh`) enforces this on every change, and
CI fails any PR that introduces a violation.

## What it checks

The audit scans tracked files (`git ls-files`) for three classes of
violation and exits non-zero if any are found:

1. **Proprietary names** - a case-insensitive denylist of company / internal
   service / repo / ticket-prefix tokens. The denylist (and the paths that
   are exempt, such as `CHANGELOG.md`, `samples/`, and the spec documents)
   lives in `scripts/audit-rules.yaml`, not in the script.
2. **Non-English text** - any byte outside printable 7-bit ASCII (plus tab /
   newline) in source, config, and docs. The English-only invariant applies
   to runtime-ingested content too (the sample corpus is plain ASCII). Exempt
   paths (for example `tests/fixtures/i18n/`) are listed in the rules file; a
   per-file opt-in marker on line 1 is also honoured.
3. **Hardcoded hosts** - hostnames ending in common TLDs that are not on the
   allowlist of placeholder / public docs / public-SaaS domains (for example
   `example.com`, package registries, and the public API hosts the
   integrations legitimately call). Deployment-specific hosts belong in
   config, not code.

All matching uses `LC_ALL=C` for deterministic, locale-independent results,
and violations are sorted by `(check, file, line)` so diffs are stable.

## Running it

```sh
# Human-readable summary; exit 0 only if all three checks pass.
scripts/audit-vendor-neutrality.sh

# Machine-readable report (AuditReport JSON on stdout).
scripts/audit-vendor-neutrality.sh --json

# Add a one-line remediation hint per violation.
scripts/audit-vendor-neutrality.sh --fix-suggestions

# Exclude an extra path for a one-off run.
scripts/audit-vendor-neutrality.sh --exclude some/generated/dir
```

It is also wired as an opt-in `pre-commit` hook
(`.pre-commit-config.yaml`) and runs as the `audit` job in CI on every PR.

## When the audit flags your change

- **A real leak** (a company name, a non-English string, a deployment host):
  remove or genericize it. Service / environment / repo names in examples
  should use placeholder shapes (`<service>`, `<env>`, `<repo>`); deployment
  facts should come from `DeploymentContext` (config), not code.
- **A false positive** (a public host, an intentional i18n fixture): add it
  to the appropriate allowlist / exempt-path list in
  `scripts/audit-rules.yaml`, with a comment explaining why. Do not weaken a
  check globally to silence one case.

## Tests

The audit's own behaviour is covered by contract tests under
`tests/contract/`: a clean-tree pass, one test per check that stages a probe
file containing a synthetic violation and asserts a non-zero exit, and a JSON
shape test. These run in CI so the guardrail itself cannot silently regress.
