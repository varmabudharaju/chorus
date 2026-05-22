"""Light integrity check: every internal Markdown link in our top-level docs
resolves to a file or anchor that actually exists.

Catches the most common docs regression: an anchor or filename gets renamed and
all the references quietly stop working. This is not a full Markdown link
checker (no external URLs, no relative ../ traversal beyond what we use). It
covers exactly the link patterns the F5 PR introduces.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files we check. Anything else is out of scope for this test.
TARGET_FILES = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "honest-tradeoffs.md",
]

# Match [text](path[#anchor]) where path is a relative file path (no scheme,
# no protocol-relative, no fragment-only links — those are tested separately
# by ensuring the anchor exists in the same file).
LINK_RE = re.compile(r"\[(?:[^\]]+)\]\(([^)]+)\)")


def _collect_anchors(md_text: str) -> set[str]:
    """Return the set of anchor ids defined in a Markdown file.

    Recognizes both:
    - explicit `<a id="foo"></a>` blocks
    - GitHub-style anchors from `## Heading` (lowercased, spaces to dashes,
      punctuation stripped). This is approximate but good enough for our docs.
    """
    anchors: set[str] = set()
    anchors.update(re.findall(r'<a\s+id="([^"]+)"', md_text))
    for line in md_text.splitlines():
        m = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if not m:
            continue
        # GitHub anchor: lowercase, strip punctuation, spaces to dashes
        slug = m.group(1).lower()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"\s+", "-", slug).strip("-")
        anchors.add(slug)
    return anchors


def test_internal_links_resolve():
    for md_path in TARGET_FILES:
        text = md_path.read_text()
        for link in LINK_RE.findall(text):
            # Skip external links and mailtos
            if link.startswith(("http://", "https://", "mailto:")):
                continue
            # Same-file anchor — verify it exists in this doc
            if link.startswith("#"):
                anchors = _collect_anchors(text)
                assert link.lstrip("#") in anchors, (
                    f"{md_path.name}: same-file anchor {link!r} not found. "
                    f"Available: {sorted(anchors)}"
                )
                continue
            # Split path and anchor
            if "#" in link:
                path_part, anchor = link.split("#", 1)
            else:
                path_part, anchor = link, ""
            # Resolve relative to the markdown file's directory
            target = (md_path.parent / path_part).resolve()
            assert target.exists(), (
                f"{md_path.name}: link {link!r} -> {target} does not exist"
            )
            if anchor:
                target_text = target.read_text()
                target_anchors = _collect_anchors(target_text)
                assert anchor in target_anchors, (
                    f"{md_path.name}: link {link!r} -> anchor #{anchor} not "
                    f"found in {target.name}. Available: {sorted(target_anchors)}"
                )
