"""
Streaming parser for eCFR title XML.

The XML hierarchy is DIV1 (title) > DIV2 (subtitle) > DIV3 (chapter) >
DIV4 (subchapter) > DIV5 (part) > DIV6 (subpart) > DIV7 (subject group) >
DIV8 (section) / DIV9 (appendix). Sections and appendices are the leaf
units we index; the ancestor headings are kept as metadata and as a context
header for each chunk.

iterparse + clear() keeps memory flat even for the largest titles
(hundreds of MB of XML).
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from lxml import etree

LEAF_TAGS = {"DIV8", "DIV9"}
CONTAINER_TAGS = {"DIV1", "DIV2", "DIV3", "DIV4", "DIV5", "DIV6", "DIV7"}
# Boilerplate that adds tokens but not retrievable meaning.
SKIP_CHILD_TAGS = {"HEAD", "CITA", "SECAUTH"}

_ws = re.compile(r"\s+")


def _text(el) -> str:
    return _ws.sub(" ", "".join(el.itertext())).strip()


@dataclass
class Section:
    title: int
    type: str            # SECTION | APPENDIX
    number: str          # e.g. "391.25"
    heading: str         # e.g. "§ 391.25 Annual inquiry and review..."
    hierarchy: dict = field(default_factory=dict)  # {"PART": ("391", "PART 391—..."), ...}
    paragraphs: list[str] = field(default_factory=list)

    @property
    def part(self) -> str | None:
        p = self.hierarchy.get("PART")
        return p[0] if p else None

    @property
    def breadcrumb(self) -> str:
        heads = [h for _, h in self.hierarchy.values() if h]
        return " > ".join(heads + [self.heading])


def iter_sections(path: Path, title_number: int) -> Iterator[Section]:
    stack: list[dict] = []
    for event, el in etree.iterparse(str(path), events=("start", "end"), huge_tree=True):
        tag = el.tag
        if event == "start":
            if tag in CONTAINER_TAGS:
                stack.append({"type": el.get("TYPE"), "n": el.get("N"), "head": None})
            continue

        if tag == "HEAD":
            parent = el.getparent()
            if parent is not None and parent.tag in CONTAINER_TAGS and stack and stack[-1]["head"] is None:
                stack[-1]["head"] = _text(el)
        elif tag in LEAF_TAGS:
            head_el = el.find("HEAD")
            paragraphs = [
                t for child in el
                if child.tag not in SKIP_CHILD_TAGS and (t := _text(child))
            ]
            if paragraphs:  # "[Reserved]" sections have no body
                yield Section(
                    title=title_number,
                    type=el.get("TYPE", "SECTION"),
                    number=el.get("N", ""),
                    heading=_text(head_el) if head_el is not None else "",
                    hierarchy={f["type"]: (f["n"], f["head"]) for f in stack},
                    paragraphs=paragraphs,
                )
            el.clear()
        elif tag in CONTAINER_TAGS:
            stack.pop()
            el.clear()
