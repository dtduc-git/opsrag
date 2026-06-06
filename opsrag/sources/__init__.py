"""Phase 2 -- non-git sources.

Each subpackage implements `SourceProtocol` (`opsrag.interfaces.source`)
for one external system: Confluence, Rootly, Slack, Datadog, PagerDuty,
Jira. Wired into `IngestionPipeline` via the `sources` dict on the
constructor.
"""
