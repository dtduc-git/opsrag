# Contributing to opsrag

Thank you for wanting to contribute. This document covers the development
workflow, the things CI will check on your PR, and the project's
non-negotiable guardrails.

## Before you start

1. Read the [project constitution](.specify/memory/constitution.md) — five
   principles every change must respect (vendor-neutrality, pluggable
   integrations, container-first, test discipline, observability and
   secret hygiene).
2. For non-trivial work, open an issue describing the change first so the
   approach can be agreed before code is written.

## Development setup

Prerequisites:

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Node 20+ for the UI
- Docker + Docker Compose for the local stack
- `pre-commit` for the local hooks (`pip install pre-commit` or `uv tool install pre-commit`)

```sh
uv sync --extra dev
pre-commit install
```

The compose stack (backend, UI, Qdrant, Postgres, Dex, Phoenix) is
documented in `specs/001-port-opsrag-opensource/quickstart.md`.

## PR workflow

1. Fork the repository and create a topic branch from `master`.
2. Make focused changes; one logical change per PR.
3. Run the local checks before pushing:

   ```sh
   uv run ruff check .
   uv run mypy opsrag
   uv run pytest
   scripts/audit-vendor-neutrality.sh
   ```

4. Push and open a PR against `master`. Fill in the PR template; reference
   the issue (if any) and the user-story IDs from `tasks.md` that the change
   completes.
5. Address review feedback by pushing additional commits — do not force-push
   while a review is in flight.

CI will run the full lint / type / unit / contract / integration / helm-lint
/ audit / eval suite on every PR. A PR cannot merge with any red check.

## Mandatory checks

Every PR must pass:

- **Lint** — `ruff check`
- **Types** — `mypy` on changed Python files (strict on changed files)
- **Tests** — unit, contract, and the integration tests for any MCP or
  agent code your change touches; new code must come with tests
- **Audit** — `scripts/audit-vendor-neutrality.sh` returns zero (no
  proprietary names, no non-English text in shipped artefacts, no
  hardcoded internal hosts)
- **Helm lint + template** — if you touched anything under
  `deploy/helm/opsrag/`
- **Eval regression** — if you touched anything under `opsrag/agent/`,
  `opsrag/agents/`, `opsrag/mcp/`, `opsrag/eval/`, or any prompt file

## What we will not accept

- Hardcoded internal hostnames, account IDs, Slack channel IDs, or
  organization-specific identifiers in shipped code or configuration.
- Non-English text in shipped artefacts (prompts, log messages, comments,
  UI strings). Test fixtures under `tests/fixtures/i18n/` are the only
  permitted exception.
- A new MCP integration without an `mcp.<name>.enabled` flag defaulting
  to `false`, fail-fast validation when enabled, and an integration test
  against a fake backend.
- Live network calls in tests. Use the per-MCP fake backends established
  in `tests/integration/`.
- Secrets in `values.yaml`, `config.yaml`, or any file under version
  control. Secrets are resolved at runtime from environment variables.
- Code committed without sign-off (`git commit -s`) — see DCO below.

## Commit and DCO

Sign your commits with the [Developer Certificate of Origin](https://developercertificate.org)
(`git commit -s`). The bot will check this on every PR.

Use [Conventional Commits](https://www.conventionalcommits.org/) where it
helps clarity (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`,
`audit:`). One-line summaries should be under 72 characters.

## Reporting bugs

Open an issue with:

- The version (commit SHA, image tag, or chart version).
- The relevant section of `config.yaml` with secrets redacted.
- Steps to reproduce.
- What you expected and what happened.
- Logs (with secrets redacted) and, if applicable, `/healthz` and
  `/readyz` output.

## Releases and security

Maintainers cut releases; the workflow is documented in
`.github/workflows/release.yml`. For security vulnerabilities, do **not**
open a public issue — follow the process in [SECURITY.md](SECURITY.md).
