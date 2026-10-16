"""
The answer pipeline:  route -> filtered hybrid search -> rerank -> generate.

run() returns everything at once (eval, scripts); events() yields progress,
sources and answer tokens as they happen (the API/UI streams these). Both
record per-stage latency so evaluation can track speed alongside quality.
"""
import time
from functools import lru_cache
from typing import Iterator

from regrag import config, llm
from regrag.retrieval import search
from regrag.router import route

ANSWER_SYSTEM = """You answer questions about US federal regulations using \
ONLY the numbered sources provided. Rules:
- Cite every factual claim with its source number in brackets, e.g. [2].
- If the sources do not contain the answer, say so plainly; do not use \
outside knowledge or guess.
- Be concise: lead with the direct answer, then the key requirements."""


@lru_cache
def _reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(config.RERANK_MODEL)


def rerank(question: str, hits: list[dict], k: int) -> list[dict]:
    if not hits:
        return []
    scores = list(_reranker().rerank(question, [f"{h['breadcrumb']}\n{h['text']}" for h in hits]))
    for h, s in zip(hits, scores):
        h["rerank_score"] = float(s)
    return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)[:k]


def citation(hit: dict) -> str:
    return f"{hit['title']} CFR {hit['section']}"


def ecfr_url(hit: dict) -> str:
    return f"https://www.ecfr.gov/current/title-{hit['title']}/section-{hit['section']}"


def _messages(question: str, sources: list[dict]) -> list[dict]:
    ctx = "\n\n".join(
        f"[{i}] {citation(h)} — {h['heading']}\n{h['text']}" for i, h in enumerate(sources, 1)
    )
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": f"Sources:\n{ctx}\n\nQuestion: {question}"},
    ]


def retrieve(question: str, use_router: bool = True, candidates: int | None = None,
             k: int | None = None, timings: dict | None = None) -> tuple[dict, list[dict]]:
    timings = timings if timings is not None else {}
    t = time.perf_counter()
    scope = route(question, method=config.ROUTER) if use_router else {"titles": [], "parts": [], "method": "none"}
    timings["route_s"] = time.perf_counter() - t

    k = k or config.CONTEXT_K
    t = time.perf_counter()
    hits = search(question, top_k=(candidates or config.CANDIDATES) if config.RERANK else k,
                  titles=scope["titles"] or None, parts=scope["parts"] or None)
    timings["search_s"] = time.perf_counter() - t
    if not config.RERANK:
        return scope, hits

    t = time.perf_counter()
    top = rerank(question, hits, k)
    timings["rerank_s"] = time.perf_counter() - t
    return scope, top


def events(question: str, **opts) -> Iterator[dict]:
    t0 = time.perf_counter()
    timings: dict = {}
    yield {"type": "status", "stage": "route"}
    scope, sources = retrieve(question, timings=timings, **opts)
    yield {"type": "scope", **scope}
    yield {"type": "sources", "sources": [
        {"n": i, "citation": citation(h), "heading": h["heading"], "breadcrumb": h["breadcrumb"],
         "url": ecfr_url(h), "text": h["text"], "chunk_id": h["chunk_id"]}
        for i, h in enumerate(sources, 1)
    ]}

    t = time.perf_counter()
    first = None
    for piece in llm.stream(_messages(question, sources)):
        if first is None:
            first = time.perf_counter() - t0
        yield {"type": "token", "text": piece}
    timings["generate_s"] = time.perf_counter() - t
    timings["first_token_s"] = first
    timings["total_s"] = time.perf_counter() - t0
    yield {"type": "done", "timings": {k: round(v, 2) for k, v in timings.items() if v is not None}}


def run(question: str, **opts) -> dict:
    out = {"question": question, "answer": ""}
    for ev in events(question, **opts):
        if ev["type"] == "token":
            out["answer"] += ev["text"]
        elif ev["type"] == "scope":
            out["scope"] = {k: v for k, v in ev.items() if k != "type"}
        elif ev["type"] == "sources":
            out["sources"] = ev["sources"]
        elif ev["type"] == "done":
            out["timings"] = ev["timings"]
    return out


if __name__ == "__main__":
    import json
    import sys
    res = run(" ".join(sys.argv[1:]))
    print(res["answer"])
    print(json.dumps({"scope": res["scope"], "timings": res["timings"],
                      "sources": [s["citation"] for s in res["sources"]]}, indent=1))
