"""
Synthetic test set: sample real chunks per title, have the LLM write the
question a practitioner would ask that this chunk answers, plus a reference
answer. The source chunk is the ground truth for retrieval metrics.

    python -m regrag.eval.testset --per-title 10

Questions must not quote section numbers or distinctive headings, otherwise
keyword search finds them trivially and the eval overstates quality. Output
is saved once (evals/testset.jsonl) so every run is compared on the same set.
"""
import argparse
import json
import random

from qdrant_client import models

from regrag import catalog, llm
from regrag.config import COLLECTION, ROOT
from regrag.store import client

OUT = ROOT / "evals" / "testset.jsonl"

PROMPT = """Below is an excerpt from the US Code of Federal Regulations.

{breadcrumb}
{text}

Write ONE realistic question that a compliance officer, business owner or \
lawyer might ask, which this excerpt answers. Requirements:
- Phrase it the way a person would ask, in plain language, about their \
situation. Do NOT mention section/part numbers, the CFR title, or copy the \
heading wording.
- It must be answerable from this excerpt alone.
Also give a short reference answer (1-3 sentences) based only on the excerpt."""

SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}, "reference": {"type": "string"}},
    "required": ["question", "reference"],
}


def sample(title: int, n: int, rng: random.Random) -> list[dict]:
    pts, _ = client().scroll(
        COLLECTION, limit=5000, with_payload=True, with_vectors=False,
        scroll_filter=models.Filter(must=[models.FieldCondition(key="title", match=models.MatchValue(value=title))]),
    )
    # Skip stubs and definition lists: too short to ask about, or too generic.
    good = [p.payload for p in pts
            if 400 <= len(p.payload["text"]) and "definitions" not in p.payload["heading"].lower()]
    return rng.sample(good, min(n, len(good)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-title", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for title in sorted(catalog.load()):
            for chunk in sample(title, args.per_title, rng):
                prompt = PROMPT.format(breadcrumb=chunk["breadcrumb"], text=chunk["text"])
                gen = llm.chat_json([{"role": "user", "content": prompt}], SCHEMA)
                row = {"id": chunk["chunk_id"], "question": gen["question"].strip(),
                       "reference": gen["reference"].strip(), "title": title,
                       "section": chunk["section"], "chunk_id": chunk["chunk_id"],
                       "context": chunk["text"]}
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"[{title} {chunk['section']}] {row['question']}", flush=True)


if __name__ == "__main__":
    main()
