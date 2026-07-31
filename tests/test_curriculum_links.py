"""Tests for the curriculum's navigation system.

The curriculum is meant to be read like a book: one roadmap, twelve chapters,
five sections per chapter, always in the same order. Three things have to stay
true for that to work, and each is checked here:

* every relative link in every Markdown file resolves — including its ``#anchor``;
* the roadmap's "exact file order" list names files that exist, once each;
* every document in that order carries a navigation footer whose *previous* and
  *next* entries agree with the position the roadmap gives it.

These are structural checks only. They deliberately say nothing about the prose.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROADMAP = REPO_ROOT / "docs" / "curriculum" / "README.md"
NAV_START = "<!-- nav:start -->"
NAV_END = "<!-- nav:end -->"

#: ``[text](target)``, excluding image embeds, which are matched with a leading ``!``.
LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")
#: A checklist entry in the roadmap's spine, e.g. ``* [ ] [§1.1 Syllabus](week01/syllabus.md)``.
SPINE_PATTERN = re.compile(r"^\* \[[ x]\] (?:\*\*)?(?:\[([^\]]+)\]\(([^)]+)\))?")


def markdown_files() -> list[Path]:
    """Every Markdown file in the repository, excluding version control data."""
    return sorted(
        path
        for path in REPO_ROOT.rglob("*.md")
        if ".git" not in path.parts and "venv" not in path.parts
    )


def heading_anchors(path: Path) -> set[str]:
    """The GitHub-style anchors a Markdown file offers.

    Args:
        path: The Markdown file to scan.

    Returns:
        The set of anchors, lower-cased with spaces turned into hyphens.
    """
    anchors: set[str] = set()
    fenced = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        match = re.match(r"#{1,6}\s+(.*)", line)
        if match is None:
            continue
        heading = re.sub(r"[`*_]", "", match.group(1).strip().rstrip("#").strip())
        kept = "".join(ch for ch in heading.lower() if ch.isalnum() or ch in "- _")
        anchors.add(kept.strip().replace(" ", "-"))
    return anchors


def spine() -> list[Path]:
    """The reading order declared by the roadmap's "exact file order" section."""
    text = ROADMAP.read_text(encoding="utf-8")
    section = text.split("## The exact file order, start to finish", 1)[1]
    order: list[Path] = [ROADMAP]
    for line in section.splitlines():
        match = SPINE_PATTERN.match(line)
        if match is None or match.group(2) is None:
            continue
        order.append((ROADMAP.parent / match.group(2)).resolve())
    return order


def nav_row(path: Path) -> tuple[str, str, str]:
    """The previous / up / next cells of a document's navigation footer."""
    text = path.read_text(encoding="utf-8")
    assert NAV_START in text and NAV_END in text, f"{path} has no navigation footer"
    block = text.split(NAV_START, 1)[1].split(NAV_END, 1)[0]
    rows = [line for line in block.splitlines() if line.startswith("|")]
    cells = [cell.strip() for cell in rows[-1].strip("|").split("|")]
    assert len(cells) == 3, f"{path} has a malformed navigation row: {rows[-1]!r}"
    return cells[0], cells[1], cells[2]


def link_target(cell: str, path: Path) -> Path | None:
    """The file a navigation cell points at, or ``None`` if it points nowhere."""
    match = LINK_PATTERN.search(cell)
    if match is None:
        return None
    return (path.parent / match.group(1).split("#")[0]).resolve()


def test_relative_links_resolve() -> None:
    broken: list[str] = []
    for path in markdown_files():
        for match in LINK_PATTERN.finditer(path.read_text(encoding="utf-8")):
            target = match.group(1)
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            file_part, _, fragment = target.partition("#")
            resolved = (path.parent / file_part).resolve()
            where = path.relative_to(REPO_ROOT)
            if not resolved.exists():
                broken.append(f"{where} -> {target} (no such file)")
            elif fragment and resolved.suffix == ".md":
                if fragment not in heading_anchors(resolved):
                    broken.append(f"{where} -> {target} (no such anchor)")
    assert not broken, "broken links:\n" + "\n".join(broken)


def test_spine_lists_every_curriculum_document_once() -> None:
    order = spine()
    assert len(order) == len(set(order)), "the roadmap lists a document twice"
    curriculum = {
        path.resolve()
        for path in (REPO_ROOT / "docs" / "curriculum").rglob("*.md")
    }
    assert curriculum <= set(order), "the roadmap omits a curriculum document"


def test_spine_covers_all_twelve_chapters_in_order() -> None:
    chapters = [
        path for path in spine() if path.name == "README.md" and path.parent.name != "curriculum"
    ]
    assert [path.parent.name for path in chapters] == [f"week{n:02d}" for n in range(1, 13)]


def test_navigation_matches_the_spine() -> None:
    order = spine()
    for index, path in enumerate(order):
        previous_cell, up_cell, next_cell = nav_row(path)
        where = path.relative_to(REPO_ROOT)

        expected_previous = order[index - 1] if index > 0 else None
        expected_next = order[index + 1] if index + 1 < len(order) else None
        assert link_target(previous_cell, path) == expected_previous, (
            f"{where} has the wrong 'previous' link"
        )
        assert link_target(next_cell, path) == expected_next, (
            f"{where} has the wrong 'next' link"
        )
        if path == ROADMAP:
            continue
        up_targets = {
            (path.parent / match.group(1).split("#")[0]).resolve()
            for match in LINK_PATTERN.finditer(up_cell)
        }
        assert ROADMAP.resolve() in up_targets, f"{where} does not link back to the roadmap"
