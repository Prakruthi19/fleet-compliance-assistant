"""
Up-front scope selection: decide which CFR titles (and parts, if the user
cited them) to search *before* searching, so all retrieval is one filtered,
parallel-friendly pass.

  - Explicit citations ("49 CFR 391.25", "part 396") are pulled out with a
    regex: free, instant, exact.
  - Otherwise one constrained LLM call picks titles from the catalog. The
    JSON schema enum means it cannot return a title that isn't indexed.
  - If routing fails or picks nothing, search everything rather than
    nothing: a wide search degrades precision, an empty one fails outright.
"""
import re

from regrag import catalog, llm

MAX_TITLES = 3

_cfr = re.compile(r"\b(\d{1,2})\s*C\.?F\.?R\.?\s*(?:(?:part|§|sec(?:tion)?\.?)\s*)?(\d{1,4})(?:\.\d+)?", re.I)
_part = re.compile(r"\b(?:part|§|section|sec\.)\s*(\d{1,4})(?:\.\d+)?\b", re.I)

ROUTER_PROMPT = """You route questions about US federal regulations (the Code \
of Federal Regulations) to the CFR titles most likely to contain the answer.

Available titles and the agencies/chapters inside them:
{catalog}

Pick the 1 to {max_titles} titles most likely to contain the answer, most \
relevant first. Only pick more than one when the question genuinely spans \
several regulatory areas.

Question: {question}"""


def _explicit(question: str) -> tuple[list[int], list[str]]:
    titles, parts = [], []
    for t, p in _cfr.findall(question):
        titles.append(int(t))
        parts.append(p)
    parts += _part.findall(question)
    return sorted(set(titles)), sorted(set(parts))


def route_vote(question: str, probe: int = 20, min_share: float = 0.2) -> list[int]:
    """
    LLM-free routing: run a cheap keyword probe over every title and keep
    the titles that win a meaningful share of reciprocal-rank weight. Costs
    one BM25 query (~tens of ms) instead of one LLM call (20-90 s on CPU).
    """
    from regrag.retrieval import keyword_search
    weight: dict[int, float] = {}
    for rank, hit in enumerate(keyword_search(question, probe), 1):
        weight[hit["title"]] = weight.get(hit["title"], 0) + 1 / rank
    total = sum(weight.values()) or 1
    ranked = sorted(weight, key=weight.get, reverse=True)
    return [t for t in ranked if weight[t] / total >= min_share][:MAX_TITLES]


def route(question: str, method: str = "vote") -> dict:
    available = sorted(catalog.load())
    titles, parts = _explicit(question)
    titles = [t for t in titles if t in available]
    if titles:
        return {"titles": titles, "parts": parts, "method": "citation"}

    if method == "citation":  # eval e1: guessing a scope cost more recall than it saved
        return {"titles": [], "parts": [], "method": "none"}

    if method == "vote":
        picked = route_vote(question)
        return {"titles": picked, "parts": parts if len(picked) == 1 else [],
                "method": "vote" if picked else "fallback"}

    schema = {
        "type": "object",
        "properties": {
            "titles": {"type": "array", "items": {"type": "integer", "enum": available},
                       "minItems": 1, "maxItems": MAX_TITLES},
        },
        "required": ["titles"],
    }
    prompt = ROUTER_PROMPT.format(catalog=catalog.describe(), question=question, max_titles=MAX_TITLES)
    try:
        picked = llm.chat_json([{"role": "user", "content": prompt}], schema)["titles"]
        picked = list(dict.fromkeys(t for t in picked if t in available))[:MAX_TITLES]
    except Exception as e:  # malformed output, LLM down, ...
        print(f"[router] falling back to unscoped search: {e!r}")
        picked = []
    # Parts without a title are ambiguous across titles, so only keep them
    # when the scope is a single title.
    return {"titles": picked, "parts": parts if len(picked) == 1 else [],
            "method": "llm" if picked else "fallback"}
