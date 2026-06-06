"""Slack channel archive connector -- Phase 2.

Read-only ingestion of selected Slack channels via the Web API. Each
thread becomes one Markdown document so retrieval surfaces a coherent
back-and-forth (problem -> debugging -> resolution) instead of single
messages out of context.
"""
