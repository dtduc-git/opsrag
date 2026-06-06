# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial public opensource fork of the opsrag agentic-GraphRAG project,
  released under the Apache License 2.0. The codebase previously lived as
  a vendor-internal toolchain at its founding organization; that
  organization is intentionally not named in shipped artefacts. The
  public fork has:
  - Removed every organization-specific identifier, hostname, account ID,
    Slack channel ID, and runbook URL from shipped code and configuration.
  - Translated every non-English prompt, log message, and comment to English.
  - Placed each of the fourteen MCP integrations (`cartography`,
    `cloudflare`, `cloudsql`, `code`, `datadog`, `elasticsearch`, `gitlab`,
    `knowledge`, `kubernetes`, `prometheus`, `rootly`, `runbooks`, `slack`,
    `tool_cache`) behind an explicit `mcp.<name>.enabled` flag whose default
    is `false`. Missing credentials on an enabled integration cause a
    named, fail-fast startup error.
  - Replaced the upstream Pomerium-specific JWT verification path with a
    generic OIDC Bearer-token middleware. A bundled local Dex issuer in
    the development compose stack keeps the fifteen-minute new-evaluator
    bring-up timeline.
  - Introduced a built-in null knowledge-graph backend so a minimal
    deployment requires only an LLM key, a vector store, and the
    OIDC issuer — no Neo4j needed.
  - Reworked the Helm chart at `deploy/helm/opsrag/` to follow standard
    chart conventions (`values.schema.json`, NetworkPolicy, PDB,
    `helm test` hook, NOTES.txt) and to expose every MCP flag through
    `values.yaml`.
  - Added an automated vendor-neutrality audit script
    (`scripts/audit-vendor-neutrality.sh`) that scans for proprietary
    names, non-English text, and hardcoded hosts; CI fails the build on
    any violation.
  - Introduced a `DeploymentContext` model (Constitution Principle VI):
    the engine carries no organization knowledge, and agent prompts render
    operator-supplied facts (services, environments, clusters, repos,
    ticket prefix, source URLs) at runtime via a prompt-render helper.
    Concrete deployment anecdotes were removed from the agent and MCP code
    or rewritten as placeholder shapes.
  - Shipped a synthetic sample corpus (the fictional "Acme Notes" product:
    runbooks, postmortems, K8s manifests, Terraform) plus a local-filesystem
    indexer and `scripts/seed-sample-corpus.sh` for the quickstart.
  - Added a fake backend and integration test for every MCP integration so
    the tool surface is testable offline, and an end-to-end investigation
    agent test driven by a scripted LLM.
  - Added project documentation under `docs/` (architecture,
    MCP integrations, Helm chart, OIDC auth setup, and the audit guide).

[Unreleased]: https://github.com/OWNER/opsrag/commits/master
