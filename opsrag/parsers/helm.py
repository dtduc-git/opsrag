"""Helm parser -- handles Chart.yaml and values.yaml files.

Extracts chart metadata (name, version, dependencies) and values
structure as sections for chunking.
"""
from __future__ import annotations

import yaml

from opsrag.ingestion.metadata import apply_provenance
from opsrag.interfaces.parser import DocSection, DocType, ParsedDocument
from opsrag.interfaces.scm import RepoFile

_HELM_FILES = ("chart.yaml", "chart.yml", "values.yaml", "values.yml")
_HELM_PATH_HINTS = ("charts/", "helm/", "/templates/")


class HelmParser:
    def supports(self, file_path: str, content: str) -> bool:
        low = file_path.lower()
        basename = low.rsplit("/", 1)[-1]
        if basename in _HELM_FILES:
            return True
        if any(h in low for h in _HELM_PATH_HINTS) and low.endswith((".yaml", ".yml")):
            return True
        return False

    def detect_doc_type(self, file: RepoFile) -> DocType:
        return DocType.HELM

    def parse(self, file: RepoFile) -> ParsedDocument:
        basename = file.path.lower().rsplit("/", 1)[-1]
        is_chart = basename in ("chart.yaml", "chart.yml")

        try:
            data = yaml.safe_load(file.content) or {}
        except Exception:
            data = {}

        if is_chart:
            sections, title, refs = self._parse_chart(data, file)
        else:
            sections, title, refs = self._parse_values(data, file)

        metadata = {
            "repo": file.repo,
            "branch": file.branch,
            "path": file.path,
            "sha": file.sha,
            "helm_file_type": "chart" if is_chart else "values",
        }
        # Chart.yaml carries the canonical service name + version. Use them
        # as the scalar `service`/`version` facets (path-derived fallback in
        # apply_provenance still applies when these are absent).
        if is_chart and isinstance(data, dict):
            if data.get("name"):
                metadata["service"] = str(data["name"])
            ver = data.get("version") or data.get("appVersion")
            if ver:
                metadata["version"] = str(ver)
        apply_provenance(metadata, file)
        return ParsedDocument(
            content=file.content,
            doc_type=DocType.HELM,
            title=title,
            source=file,
            metadata=metadata,
            sections=sections,
            references=refs,
        )

    @staticmethod
    def _parse_chart(data: dict, file: RepoFile) -> tuple[list[DocSection], str, list[str]]:
        name = data.get("name", file.path)
        version = data.get("version", "")
        app_version = data.get("appVersion", "")
        description = data.get("description", "")

        sections = [
            DocSection(
                heading=f"Chart: {name}",
                content=f"name: {name}\nversion: {version}\n"
                        f"appVersion: {app_version}\ndescription: {description}",
                level=1,
                section_type="chart_metadata",
            )
        ]

        deps = data.get("dependencies", [])
        if deps:
            dep_lines = []
            refs = []
            for d in deps:
                dep_lines.append(
                    f"- {d.get('name', '?')} ({d.get('version', '?')}) "
                    f"from {d.get('repository', '?')}"
                )
                if repo := d.get("repository"):
                    refs.append(repo)
            sections.append(DocSection(
                heading="Dependencies",
                content="\n".join(dep_lines),
                level=2,
                section_type="chart_dependencies",
            ))
        else:
            refs = []

        return sections, name, refs

    @staticmethod
    def _parse_values(data: dict, file: RepoFile) -> tuple[list[DocSection], str, list[str]]:
        title = file.path.rsplit("/", 1)[-1]
        if not isinstance(data, dict):
            return [DocSection(heading=title, content=file.content, level=0)], title, []

        sections: list[DocSection] = []
        for key, value in data.items():
            # Dump {key: value}, NOT bare value -- otherwise the top-level key
            # (`resources`, `image`, `replicas`) appears nowhere in the indexed
            # content, only in metadata, so "resource limits for X" loses both
            # the BM25 anchor and the semantic anchor. generic.py already does
            # this; Helm (the most common file type) silently didn't.
            if isinstance(value, (dict, list)):
                content = yaml.dump(
                    {key: value}, default_flow_style=False, sort_keys=False
                ).strip()
            else:
                content = f"{key}: {value}"

            section_type = "values_section"
            low = key.lower()
            if low in ("image", "container", "containers"):
                section_type = "values_image"
            elif low in ("resources", "limits", "requests"):
                section_type = "values_resources"
            elif low in ("ingress", "service", "networking"):
                section_type = "values_networking"
            elif low in ("env", "environment", "config"):
                section_type = "values_config"

            sections.append(DocSection(
                heading=key,
                content=content,
                level=1,
                section_type=section_type,
            ))

        return sections, title, []
