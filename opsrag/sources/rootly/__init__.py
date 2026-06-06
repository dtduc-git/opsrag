"""Rootly incidents + post-mortems connector -- Phase 2.

Read-only ingestion of resolved/mitigated incidents and their
post-mortems. Alerts are deliberately NOT indexed -- config noise, not
knowledge. Per the Phase 2 scope decision (2026-05-06), Datadog &
PagerDuty also stay out of RAG and only land in Phase 4 as MCP tools.
"""
