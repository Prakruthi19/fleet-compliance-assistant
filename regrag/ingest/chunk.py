"""
Section-aware chunking.

Small sections stay whole (a rule stays with its exceptions). Large ones are
packed paragraph-by-paragraph up to CHUNK_MAX_CHARS, never splitting mid
paragraph unless a single paragraph is itself oversized.

Each chunk gets a contextual header ("Title 49 > ... > PART 391 > § 391.25")
prepended for embedding. This is a cheap retrieval-quality win: a chunk that
only says "the motor carrier shall..." becomes findable by its part/section
topic.
"""
import hashlib
from dataclasses import dataclass

from regrag.config import CHUNK_MAX_CHARS
from regrag.ingest.parse import Section


@dataclass
class Chunk:
    id: str
    title: int
    part: str | None
    section: str
    heading: str
    breadcrumb: str
    text: str
    hash: str

    @property
    def embed_text(self) -> str:
        return f"{self.breadcrumb}\n\n{self.text}"

    def payload(self) -> dict:
        return {
            "chunk_id": self.id,
            "title": self.title,
            "part": self.part,
            "section": self.section,
            "heading": self.heading,
            "breadcrumb": self.breadcrumb,
            "text": self.text,
        }


def _pieces(paragraphs: list[str], limit: int) -> list[str]:
    out, buf = [], ""
    for p in paragraphs:
        while len(p) > limit:  # oversized single paragraph (e.g. tables)
            if buf:
                out.append(buf)
                buf = ""
            cut = p.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            out.append(p[:cut])
            p = p[cut:].lstrip()
        if buf and len(buf) + 1 + len(p) > limit:
            out.append(buf)
            buf = ""
        buf = f"{buf}\n{p}" if buf else p
    if buf:
        out.append(buf)
    return out


def chunk_section(sec: Section) -> list[Chunk]:
    breadcrumb = sec.breadcrumb
    chunks = []
    for i, text in enumerate(_pieces(sec.paragraphs, CHUNK_MAX_CHARS)):
        cid = f"{sec.title}:{sec.type}:{sec.number}:{i}"
        digest = hashlib.sha1(f"{breadcrumb}\n{text}".encode()).hexdigest()
        chunks.append(Chunk(cid, sec.title, sec.part, sec.number, sec.heading,
                            breadcrumb, text, digest))
    return chunks
