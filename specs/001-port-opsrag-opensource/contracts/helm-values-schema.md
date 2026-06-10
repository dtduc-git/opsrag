# Contract: Helm `values.yaml` schema

**Status**: New for this feature. Authoritative schema lives in
`deploy/helm/opsrag/values.schema.json`; this document mirrors it.

## Source-of-truth invariants

1. **Every MCP integration has a flag.** The set of keys under `mcp:`
   MUST be exactly the set of names in the `MCPIntegration` registry
   (`data-model.md` §1). Drift is a CI failure.
2. **Every flag defaults `false`.** No exception.
3. **Schema rejects unknown top-level keys.** Catches typos at
   `helm install` time, not at pod startup.

## Schema outline (`values.schema.json`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "additionalProperties": false,
  "required": ["image", "auth", "api", "mcp"],
  "properties": {
    "image": {
      "type": "object",
      "required": ["repository", "tag"],
      "properties": {
        "repository": { "type": "string" },
        "tag":        { "type": "string", "minLength": 1 },
        "pullPolicy": { "enum": ["Always", "IfNotPresent", "Never"] }
      }
    },
    "auth": {
      "type": "object",
      "required": ["issuer", "audience"],
      "properties": {
        "issuer":   { "type": "string", "format": "uri" },
        "audience": { "type": "string", "minLength": 1 }
      }
    },
    "api": { ... },
    "ui": { ... },
    "slackBot": { ... },
    "mcp": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "aws", "azure", "cloudflare", "code",
        "datadog", "elasticsearch", "gcp", "github",
        "gitlab", "grafana", "knowledge", "kubernetes",
        "loki", "prometheus", "rootly", "runbooks",
        "sentry", "slack", "splunk", "tool_cache"
      ],
      "patternProperties": {
        "^[a-z][a-z0-9_]*$": {
          "type": "object",
          "required": ["enabled"],
          "properties": {
            "enabled":   { "type": "boolean" },
            "secretRef": { "type": "string" }
          }
        }
      }
    },
    "networkPolicy": { ... },
    "podDisruptionBudget": { ... },
    "autoscaling": { ... }
  }
}
```

## Wiring from values to running container

Every `mcp.<name>.enabled` value MUST be visible to the running pod as the
env var `OPSRAG_MCP_<NAME>_ENABLED` (uppercased). This is what the
contract test asserts mechanically rather than scanning templates.

## Contract tests (CI gates)

- `tests/contract/test_helm_lint.sh` — runs
  `helm lint deploy/helm/opsrag`; non-zero exit fails the build.
- `tests/contract/test_helm_template_default.sh` — runs
  `helm template opsrag deploy/helm/opsrag` and parses YAML; non-zero
  exit fails the build.
- `tests/contract/test_helm_values_covers_all_mcps.py` — loads
  `values.yaml` and the `MCPIntegration` registry, asserts the set of
  keys under `mcp:` equals the registry exactly. Fails the build on
  drift either way.
- `tests/contract/test_helm_env_propagation.py` — renders the chart with
  one MCP enabled and asserts the resulting Deployment spec contains the
  expected `OPSRAG_MCP_<NAME>_ENABLED=true` env var on the api
  container.
- `tests/contract/test_helm_schema_rejects_unknown.sh` — runs `helm install
  --dry-run` against a values file with an unknown `mcp:` key; asserts
  failure with a `does not allow additional properties` style message.
