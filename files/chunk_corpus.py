"""
Chunking strategy: SECTION-BASED.

Splits each raw regulation text file into one chunk per § section (e.g.
everything under "§ 396.17 Periodic inspection." becomes one chunk, up to
the next § marker or a named subpart/appendix heading).

Why section-based for this corpus: FMCSA regulation text is already
organized into legally meaningful units (sections), and questions in this
domain ("what does § 396.17 require?") map naturally onto that boundary.
Fixed-size chunking would risk splitting a rule from its own exceptions
mid-paragraph; section-based chunking respects t
he document's own structure
instead of imposing an arbitrary size limit.

Trade-off: some sections (e.g. Appendix A's brake-defect list, or § 383.51's
disqualification tables) are long and could arguably be split further at the
subsection or table level for finer retrieval granularity. That is a
reasonable next iteration, not done here to keep the strategy simple and
explainable.

Usage:
    python data/chunk_corpus.py
Reads every .txt file in data/raw/ and writes data/fmcsa_corpus.json
"""
import json
import re
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
OUTPUT_PATH = Path(__file__).parent / "fmcsa_corpus.json"

# Matches section headers like "§ 396.17 Periodic inspection."
# and appendix/part-level headers like "Appendix A to Part 396—..."
SECTION_HEADER_RE = re.compile(
    r"^(§\s*[\d\.]+[A-Za-z]?\s+.+|Appendix [A-Z] to Part \d+.*)$",
    re.MULTILINE,
)

# Infer domain from filename — a simple, explicit mapping rather than
# per-chunk guessing.
DOMAIN_BY_FILE = {
    "part_397.txt": "hazmat",
    "part_385_subpart_e.txt": "hazmat",
    "part_396.txt": "inspection",
    "part_391.txt": "driver_qualification",
    "part_383.txt": "driver_qualification",
}


def extract_source_url(text: str) -> str:
    """Pull the first SOURCE: line from the raw file header."""
    match = re.search(r"^SOURCE:\s*(\S+)", text, re.MULTILINE)
    return match.group(1) if match else ""


def extract_part_number(filename: str) -> str:
    match = re.search(r"part_(\d+)", filename)
    return match.group(1) if match else ""


def slugify(section_label: str) -> str:
    """Turn '§ 396.17 Periodic inspection.' into 'section-396-17'."""
    num_match = re.search(r"§\s*([\d\.]+[A-Za-z]?)", section_label)
    if num_match:
        return "section-" + num_match.group(1).replace(".", "-")
    # Appendix / Part-level headers
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", section_label).strip("-").lower()
    return slug[:60]


def chunk_file(filepath: Path) -> list[dict]:
    raw_text = filepath.read_text(encoding="utf-8")
    source_url = extract_source_url(raw_text)
    part_number = extract_part_number(filepath.name)
    domain = DOMAIN_BY_FILE.get(filepath.name, "unknown")

    # Strip the SOURCE: header lines before splitting into sections
    body = re.sub(r"^SOURCE:.*$", "", raw_text, flags=re.MULTILINE).strip()

    headers = list(SECTION_HEADER_RE.finditer(body))
    chunks = []

    for i, match in enumerate(headers):
        header_text = match.group(0).strip()
        start = match.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(body)
        section_body = body[start:end].strip()

        if not section_body or len(section_body) < 20:
            # Skip empty or trivially short sections (e.g. bare subpart titles)
            continue

        section_id = slugify(header_text)
        chunk_id = f"{part_number}-{section_id}"

        chunks.append(
            {
                "id": chunk_id,
                "part": part_number,
                "section": header_text,
                "domain": domain,
                "source_url": source_url,
                "title": header_text,
                "text": section_body,
            }
        )

    return chunks


def main():
    all_chunks = []
    for filepath in sorted(RAW_DIR.glob("*.txt")):
        file_chunks = chunk_file(filepath)
        print(f"{filepath.name}: {len(file_chunks)} chunks")
        all_chunks.extend(file_chunks)

    OUTPUT_PATH.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")
    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()