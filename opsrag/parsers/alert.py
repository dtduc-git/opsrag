"""Alert definition parser -- Prometheus rules and Datadog monitors.

Handles Prometheus PrometheusRule CRDs and plain YAML alert files.
Each alert group / monitor becomes a section.
"""
from __future__ import annotations

import yaml

from opsrag.interfaces.parser import DocSection, DocType, ParsedDocument
from opsrag.interfaces.scm import RepoFile

_ALERT_PATH_HINTS = ("alert", "monitor", "prometheus", "rules/")


class AlertParser:
    def supports(self, file_path: str, content: str) -> bool:
        if not file_path.lower().endswith((".yaml", ".yml")):
            return False
        low = file_path.lower()
        if any(h in low for h in _ALERT_PATH_HINTS):
            return True
        return "PrometheusRule" in content or "alert:" in content

    def detect_doc_type(self, file: RepoFile) -> DocType:
        return DocType.ALERT_DEFINITION

    def parse(self, file: RepoFile) -> ParsedDocument:
        try:
            data = yaml.safe_load(file.content) or {}
        except Exception:
            data = {}

        sections: list[DocSection] = []

        if isinstance(data, dict):
            kind = data.get("kind", "")
            if kind == "PrometheusRule":
                sections = self._parse_prometheus_rule(data)
            elif "groups" in data:
                sections = self._parse_prometheus_groups(data["groups"])
            elif "monitors" in data:
                sections = self._parse_datadog_monitors(data["monitors"])
            else:
                sections = self._parse_flat_alerts(data)

        if not sections:
            sections = [DocSection(
                heading=file.path, content=file.content, level=0,
                section_type="alert_raw",
            )]

        title = sections[0].heading if sections else file.path
        return ParsedDocument(
            content=file.content,
            doc_type=DocType.ALERT_DEFINITION,
            title=title,
            source=file,
            metadata={
                "repo": file.repo,
                "branch": file.branch,
                "path": file.path,
                "sha": file.sha,
                "alert_count": len(sections),
            },
            sections=sections,
            references=[],
        )

    def _parse_prometheus_rule(self, data: dict) -> list[DocSection]:
        spec = data.get("spec", {}) or {}
        groups = spec.get("groups", [])
        meta = data.get("metadata", {}) or {}
        name = meta.get("name", "")

        sections: list[DocSection] = []
        if name:
            sections.append(DocSection(
                heading=f"PrometheusRule: {name}",
                content=f"namespace: {meta.get('namespace', 'default')}",
                level=1,
                section_type="alert_rule_meta",
            ))
        sections.extend(self._parse_prometheus_groups(groups))
        return sections

    @staticmethod
    def _parse_prometheus_groups(groups: list) -> list[DocSection]:
        sections: list[DocSection] = []
        for group in groups:
            if not isinstance(group, dict):
                continue
            group_name = group.get("name", "unnamed")
            rules = group.get("rules", [])
            for rule in rules:
                if not isinstance(rule, dict):
                    continue
                alert_name = rule.get("alert", rule.get("record", ""))
                if not alert_name:
                    continue
                expr = rule.get("expr", "")
                severity = (rule.get("labels", {}) or {}).get("severity", "")
                summary = (rule.get("annotations", {}) or {}).get("summary", "")
                duration = rule.get("for", "")

                body_parts = [f"group: {group_name}"]
                if expr:
                    body_parts.append(f"expr: {expr}")
                if severity:
                    body_parts.append(f"severity: {severity}")
                if duration:
                    body_parts.append(f"for: {duration}")
                if summary:
                    body_parts.append(f"summary: {summary}")

                sections.append(DocSection(
                    heading=f"alert: {alert_name}",
                    content="\n".join(body_parts),
                    level=2,
                    section_type="alert_rule",
                ))
        return sections

    @staticmethod
    def _parse_datadog_monitors(monitors: list) -> list[DocSection]:
        sections: list[DocSection] = []
        for mon in monitors:
            if not isinstance(mon, dict):
                continue
            name = mon.get("name", "unnamed")
            mtype = mon.get("type", "")
            query = mon.get("query", "")
            msg = mon.get("message", "")
            body = yaml.dump(mon, default_flow_style=False).strip()
            sections.append(DocSection(
                heading=f"monitor: {name}",
                content=body,
                level=2,
                section_type="alert_monitor",
            ))
        return sections

    @staticmethod
    def _parse_flat_alerts(data: dict) -> list[DocSection]:
        sections: list[DocSection] = []
        if "alert" in data:
            name = data["alert"]
            body = yaml.dump(data, default_flow_style=False).strip()
            sections.append(DocSection(
                heading=f"alert: {name}",
                content=body,
                level=1,
                section_type="alert_rule",
            ))
        return sections
