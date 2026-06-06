"""Terraform parser -- extracts resource/module/variable/output blocks as sections.

Supports .tf and .hcl files. Each top-level block becomes a DocSection
so the chunker can produce per-resource chunks.
"""
from __future__ import annotations

import re

from opsrag.interfaces.parser import DocSection, DocType, ParsedDocument
from opsrag.interfaces.scm import RepoFile

_BLOCK_RE = re.compile(
    r'^(resource|data|module|variable|output|provider|locals|terraform)\s+'
    r'"?([^"\s{]*)"?\s*(?:"([^"]*)")?\s*\{',
    re.MULTILINE,
)
_TF_EXTENSIONS = (".tf", ".hcl")
_NON_TF_HCL = ("terragrunt.hcl",)


class TerraformParser:
    def supports(self, file_path: str, content: str) -> bool:
        low = file_path.lower()
        if not low.endswith(_TF_EXTENSIONS):
            return False
        if any(low.endswith(exc) for exc in _NON_TF_HCL):
            return False
        return True

    def detect_doc_type(self, file: RepoFile) -> DocType:
        return DocType.TERRAFORM

    def parse(self, file: RepoFile) -> ParsedDocument:
        sections = self._extract_blocks(file.content)
        title = file.path.rsplit("/", 1)[-1]
        references = self._find_module_sources(file.content)

        return ParsedDocument(
            content=file.content,
            doc_type=DocType.TERRAFORM,
            title=title,
            source=file,
            metadata={
                "repo": file.repo,
                "branch": file.branch,
                "path": file.path,
                "sha": file.sha,
                "language": "hcl",
            },
            sections=sections,
            references=references,
        )

    @staticmethod
    def _extract_blocks(content: str) -> list[DocSection]:
        matches = list(_BLOCK_RE.finditer(content))
        if not matches:
            return [DocSection(heading=content[:60].strip(), content=content, level=0,
                               section_type="terraform_file")]

        sections: list[DocSection] = []
        for i, m in enumerate(matches):
            block_type = m.group(1)
            name_parts = [p for p in [m.group(2), m.group(3)] if p]
            heading = f'{block_type} {".".join(name_parts)}' if name_parts else block_type

            body_start = m.start()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            body = content[body_start:body_end].strip()

            section_type = {
                "resource": "tf_resource",
                "data": "tf_data",
                "module": "tf_module",
                "variable": "tf_variable",
                "output": "tf_output",
                "provider": "tf_provider",
                "locals": "tf_locals",
                "terraform": "tf_settings",
            }.get(block_type, "tf_block")

            sections.append(DocSection(
                heading=heading,
                content=body,
                level=1,
                section_type=section_type,
            ))
        return sections

    @staticmethod
    def _find_module_sources(content: str) -> list[str]:
        return re.findall(r'source\s*=\s*"([^"]+)"', content)
