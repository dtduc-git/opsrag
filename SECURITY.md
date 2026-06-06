# Security Policy

## Supported versions

Security fixes are issued only for the latest released minor version on the
default branch. Older versions are not patched; please upgrade.

| Version  | Supported          |
| -------- | ------------------ |
| latest   | :white_check_mark: |
| < latest | :x:                |

## Reporting a vulnerability

**Please do not open a public issue for security reports.**

Report suspected vulnerabilities privately via GitHub's
[Private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
feature on this repository, or by email to the address listed in the
repository's `MAINTAINERS` file (TODO: add `MAINTAINERS` and a security
contact address).

Include in your report:

- A description of the vulnerability and the impact you observed.
- Step-by-step instructions to reproduce, including a minimal config and
  the affected version (commit SHA, container image tag, or Helm chart
  version).
- Whether the vulnerability is already public.
- Any mitigations you have identified.

We aim to acknowledge new reports within **3 business days** and to provide
an initial assessment within **10 business days**. Coordinated-disclosure
timelines are negotiated on a case-by-case basis; 90 days is the default.

## Scope

In scope:

- The Python backend in `opsrag/`.
- The React UI in `ui/`.
- The Helm chart in `deploy/helm/opsrag/`.
- The compose stack in `deploy/compose/`.
- The container image published from this repository.
- The MCP integrations under `opsrag/mcp/` insofar as they affect the agent
  process running this code.

Out of scope:

- Vulnerabilities in third-party services (Datadog, GitLab, Kubernetes,
  etc.) that this project calls. Report those upstream.
- Findings against an evaluator's local LLM key, OIDC issuer, or sample
  corpus rather than against shipped code.
- Denial-of-service achieved purely by exhausting the configured LLM
  budget or vector-store capacity.

## Hardening expectations for operators

- Run the container as a non-root user (the shipped image uses UID 1000).
- Mount every secret via Kubernetes `Secret` or an external secret manager;
  never bake credentials into `values.yaml`.
- Restrict egress from the workload to the integration endpoints actually
  enabled in `config.yaml`.
- Configure the OIDC issuer to a trusted identity provider and verify the
  `aud` claim matches the deployment.
