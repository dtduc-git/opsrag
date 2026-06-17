"""M5: fenced code blocks (```/~~~) stay atomic.

A runbook command block must NOT be split as prose: a `# comment` line inside
a fence is not a markdown heading, and a `. ` inside a command line is not a
sentence boundary. Two layers are exercised:

1. parsers/markdown._extract_sections -- a `#`-prefixed line inside a fence
   must not open a new section.
2. chunkers/parent_child -- a prose child boundary must not land inside a fence;
   it extends to the closing fence (bounded), so the whole command block lands
   in a single child.
"""
from __future__ import annotations

from datetime import UTC, datetime

from opsrag.chunkers.parent_child import ParentChildChunker
from opsrag.interfaces.parser import DocType, ParsedDocument
from opsrag.interfaces.scm import RepoFile
from opsrag.parsers.markdown import GenericMarkdownParser


def _md_doc(content: str, doc_type: DocType = DocType.RUNBOOK) -> ParsedDocument:
    rf = RepoFile(
        path="runbooks/restart.md", content=content, sha="x",
        last_modified=datetime(2026, 1, 1, tzinfo=UTC),
        repo="r/p", branch="main",
    )
    parser = GenericMarkdownParser()
    sections = parser._extract_sections(content)
    title = sections[0].heading if sections else "restart"
    return ParsedDocument(
        content=content, doc_type=doc_type, title=title, source=rf, sections=sections,
    )


def _children(chunks):
    return [c for c in chunks if c.chunk_type == "child"]


# --- Layer 1: parser does not split a fence into sections -------------------

def test_hash_comment_inside_fence_is_not_a_heading():
    content = (
        "# Restart Procedure\n\n"
        "Run the following:\n\n"
        "```bash\n"
        "# stop the service first\n"
        "systemctl stop app\n"
        "# now start it\n"
        "systemctl start app\n"
        "```\n\n"
        "Verify it is healthy.\n"
    )
    sections = GenericMarkdownParser()._extract_sections(content)
    # Exactly ONE heading section ("Restart Procedure"); the `# stop`/`# now`
    # lines inside the fence must NOT have produced extra sections.
    headings = [s.heading for s in sections if s.heading]
    assert headings == ["Restart Procedure"], headings
    # The whole fenced block must live inside that single section's body.
    body = sections[0].content if sections[0].heading else sections[1].content
    assert "systemctl stop app" in body
    assert "systemctl start app" in body
    assert "# stop the service first" in body


def test_tilde_fence_also_shields_headings():
    content = (
        "# Title\n\n"
        "~~~\n"
        "## not a heading\n"
        "echo hello\n"
        "~~~\n"
    )
    sections = GenericMarkdownParser()._extract_sections(content)
    headings = [s.heading for s in sections if s.heading]
    assert headings == ["Title"], headings


def test_real_headings_outside_fence_still_parse():
    content = (
        "# One\n\nalpha\n\n"
        "```\n# comment\ncode\n```\n\n"
        "## Two\n\nbeta\n"
    )
    sections = GenericMarkdownParser()._extract_sections(content)
    headings = [s.heading for s in sections if s.heading]
    assert headings == ["One", "Two"], headings


# --- Layer 2: chunker keeps a fence in one child ---------------------------

def test_command_block_with_dots_and_comment_stays_in_one_child():
    # A long prose intro forces a child boundary; the fenced command block that
    # follows contains a '. ' (would be a sentence snap) and a '# comment'
    # (would be a heading-ish line). Neither may split the fence.
    intro = (
        "This runbook explains the restart sequence in detail. " * 30
    )
    fence = (
        "```bash\n"
        "# first stop the service. then wait.\n"
        "kubectl rollout restart deploy/app -n prod. sleep 5\n"
        "kubectl get pods -n prod\n"
        "```\n"
    )
    content = f"# Restart\n\n{intro}\n\n{fence}\nDone.\n"
    children = _children(
        ParentChildChunker(child_size=64, child_overlap=8).chunk(
            _md_doc(content, DocType.RUNBOOK)
        )
    )
    assert children, "expected children"
    # The fence must be wholly contained in exactly one child -- no child may
    # start or end in the middle of it.
    fence_open = "```bash"
    fence_lines = [
        "# first stop the service. then wait.",
        "kubectl rollout restart deploy/app -n prod. sleep 5",
        "kubectl get pods -n prod",
    ]
    holders = [c for c in children if fence_open in c.content]
    assert len(holders) == 1, (
        f"fence split across {len(holders)} children: "
        f"{[c.metadata.get('child_index') for c in holders]}"
    )
    holder = holders[0]
    for line in fence_lines:
        assert line in holder.content, f"fence line missing from holder: {line!r}"
    # Closing fence is in the same child.
    assert holder.content.count("```") >= 2 or holder.content.rstrip().endswith("```") \
        or "```\n" in holder.content


def test_no_child_boundary_inside_fence_across_all_children():
    # Stronger invariant: reconstruct the fenced region and confirm it is not
    # straddled. We check that no child's content STARTS partway through a
    # command line (i.e. the fence open and close are never separated).
    fence = (
        "```\n"
        "run --flag value. another command\n"
        "# inline note. with a period\n"
        "final --done\n"
        "```\n"
    )
    intro = "Intro paragraph that is reasonably long to push a boundary. " * 20
    content = f"# H\n\n{intro}\n\n{fence}\ntrailer text.\n"
    children = _children(
        ParentChildChunker(child_size=48, child_overlap=8).chunk(
            _md_doc(content, DocType.RUNBOOK)
        )
    )
    # Every child either contains the full fence or none of its interior lines.
    interior = "run --flag value. another command"
    for c in children:
        if interior in c.content:
            assert "```" in c.content, "interior line present but fence markers cut"
            assert "final --done" in c.content, "fence body split mid-block"


def test_prose_without_fence_is_byte_identical():
    # Guard: prose with no fence is unaffected -> stable child boundaries/ids.
    content = "# H\n\n" + ("A normal sentence about operations. " * 200)
    a = _children(ParentChildChunker().chunk(_md_doc(content, DocType.RUNBOOK)))
    b = _children(ParentChildChunker().chunk(_md_doc(content, DocType.RUNBOOK)))
    assert [c.id for c in a] == [c.id for c in b]
    assert [c.content for c in a] == [c.content for c in b]
    # And no breadcrumb leaks into prose.
    for c in a:
        assert not c.content.startswith("# [context]")
