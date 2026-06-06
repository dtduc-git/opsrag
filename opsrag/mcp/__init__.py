"""MCP-style tool integrations for OpsRAG.

This package hosts Python ports of MCP (Model Context Protocol) servers
that the agent uses for live-system enrichment. The first port is the
GitLab read-only tool surface adapted from `@zereight/mcp-gitlab`. We
keep the MCP-shape (name + description + JSON-schema inputs + async
handler) so a future migration to the real MCP protocol is mechanical.

Per the Phase 03 roadmap, additional ports are planned for Datadog,
Kubernetes, Rootly, and GCP Cloud SQL -- they share the same `MCPTool`
shape from this package.
"""
from opsrag.mcp.aws import AWS_TOOLS, AWSMCPError
from opsrag.mcp.azure import AZURE_TOOLS, AzureMCPError
from opsrag.mcp.cloudflare import (
    CLOUDFLARE_TOOLS,
    MCPCloudflareError,
)
from opsrag.mcp.cloudflare import (
    bind as bind_cloudflare,
)
from opsrag.mcp.code import CODE_TOOLS
from opsrag.mcp.code import bind_scm as bind_code_scm
from opsrag.mcp.datadog import DATADOG_TOOLS
from opsrag.mcp.elasticsearch import (
    ES_TOOLS,
    MCPElasticsearchError,
)
from opsrag.mcp.elasticsearch import (
    bind as bind_elasticsearch,
)
from opsrag.mcp.gcp import GCP_TOOLS, GCPMCPError
from opsrag.mcp.github import GITHUB_TOOLS, GitHubMCPError
from opsrag.mcp.gitlab import (
    GITLAB_TOOLS,
    GitLabClient,
    GitLabMCPError,
    MCPTool,
)
from opsrag.mcp.grafana import GRAFANA_TOOLS, GrafanaMCPError
from opsrag.mcp.knowledge import KNOWLEDGE_TOOLS
from opsrag.mcp.knowledge import bind as bind_knowledge
from opsrag.mcp.kubernetes import KUBERNETES_TOOLS
from opsrag.mcp.loki import LOKI_TOOLS, LokiMCPError
from opsrag.mcp.prometheus import PROMETHEUS_TOOLS
from opsrag.mcp.rootly import ROOTLY_TOOLS
from opsrag.mcp.runbooks import RUNBOOK_TOOLS
from opsrag.mcp.sentry import SENTRY_TOOLS, SentryMCPError
from opsrag.mcp.slack import SLACK_TOOLS, SlackMCPError
from opsrag.mcp.splunk import SPLUNK_TOOLS, SplunkMCPError

# Single registry for all MCP-style tools.
#
# NOTE: when adding a new MCP family, also register a `<prefix>_` ->
# user-facing label entry in `opsrag.agent.nodes.multi_agent._MCP_PREFIX_LABELS`
# so the tool_caller SSE label reads e.g. "Calling Runbook tools..."
# instead of falling back to "Calling live tools...". The `runbook_` prefix
# mapping will be wired in the final multi_agent pass.
ALL_MCP_TOOLS: list[MCPTool] = (
    list(GITLAB_TOOLS)
    + list(GITHUB_TOOLS)
    + list(KUBERNETES_TOOLS)
    + list(PROMETHEUS_TOOLS)
    + list(GRAFANA_TOOLS)
    + list(LOKI_TOOLS)
    + list(ROOTLY_TOOLS)
    + list(DATADOG_TOOLS)
    + list(SENTRY_TOOLS)
    + list(SPLUNK_TOOLS)
    + list(AWS_TOOLS)
    + list(GCP_TOOLS)
    + list(AZURE_TOOLS)
    + list(SLACK_TOOLS)
    + list(ES_TOOLS)
    + list(KNOWLEDGE_TOOLS)
    + list(RUNBOOK_TOOLS)
    + list(CODE_TOOLS)
    + list(CLOUDFLARE_TOOLS)
)

__all__ = [
    "GitLabClient",
    "GitLabMCPError",
    "MCPTool",
    "GITLAB_TOOLS",
    "KUBERNETES_TOOLS",
    "PROMETHEUS_TOOLS",
    "ROOTLY_TOOLS",
    "DATADOG_TOOLS",
    "GITHUB_TOOLS",
    "GitHubMCPError",
    "SENTRY_TOOLS",
    "SentryMCPError",
    "GRAFANA_TOOLS",
    "GrafanaMCPError",
    "LOKI_TOOLS",
    "LokiMCPError",
    "SPLUNK_TOOLS",
    "SplunkMCPError",
    "AWS_TOOLS",
    "AWSMCPError",
    "GCP_TOOLS",
    "GCPMCPError",
    "AZURE_TOOLS",
    "AzureMCPError",
    "SLACK_TOOLS",
    "SlackMCPError",
    "ES_TOOLS",
    "MCPElasticsearchError",
    "bind_elasticsearch",
    "KNOWLEDGE_TOOLS",
    "bind_knowledge",
    "RUNBOOK_TOOLS",
    "CODE_TOOLS",
    "bind_code_scm",
    "CLOUDFLARE_TOOLS",
    "MCPCloudflareError",
    "bind_cloudflare",
    "ALL_MCP_TOOLS",
]
