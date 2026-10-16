"""
Evaluation runner (main venv). Two modes:

  retrieval — compares retrieval strategies on the test set without
              generating answers (fast, no judge needed):
                baseline       hybrid search over everything, top K
                rerank         hybrid search over everything, top N -> rerank -> K
                router         router-scoped hybrid search, top K
                router+rerank  router-scoped, top N -> rerank -> K
              Metrics: router title accuracy, section hit@K, MRR, latency.

  answers   — runs the full pipeline and saves question/answer/contexts to
              evals/runs/<name>.jsonl for the DeepEval judge (evals/judge.py,
              separate venv), plus generation latency.

    python -m regrag.eval.run retrieval [--limit 20]
    python -m regrag.eval.run answers --name v1 [--limit 15]
"""
import argparse
import json
import statistics
import time
from datetime import datetime

from regrag import config, rag
from regrag.eval.testset import OUT as TESTSET
from regrag.retrieval import search
from regrag.router import route

RUNS = config.ROOT / "evals" / "runs"


def load_testset(limit: int | None) -> list[dict]:
    rows = [json.loads(l) for l in TESTSET.read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit:  # keep every title represented when subsetting
        by_title: dict[int, list] = {}
        for r in rows:
            by_title.setdefault(r["title"], []).append(r)
        rows, i = [], 0
        while len(rows) < limit and any(i < len(v) for v in by_title.values()):
            rows += [v[i] for v in by_title.values() if i < len(v)][: limit - len(rows)]
            i += 1
    return rows


def _rank(row: dict, hits: list[dict]) -> int | None:
    """1-based rank of the first hit from the gold section (chunks of one section count)."""
    for i, h in enumerate(hits, 1):
        if h["title"] == row["title"] and h["section"] == row["section"]:
            return i
    return None


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


def summarize(results: list[dict]) -> dict:
    out = {"n": len(results)}
    ranks = [r["rank"] for r in results]
    out["hit@K"] = round(sum(r is not None for r in ranks) / len(ranks), 3)
    out["MRR"] = round(sum(1 / r for r in ranks if r) / len(ranks), 3)
    if "title_ok" in results[0]:
        out["router_title_acc"] = round(sum(r["title_ok"] for r in results) / len(results), 3)
    lat = [r["latency_s"] for r in results]
    out["p50_s"], out["p95_s"] = round(statistics.median(lat), 2), round(pct(lat, 0.95), 2)
    by_title: dict = {}
    for r in results:
        by_title.setdefault(r["title"], []).append(r["rank"] is not None)
    out["hit@K_by_title"] = {t: round(sum(v) / len(v), 2) for t, v in sorted(by_title.items())}
    return out


def eval_retrieval(rows: list[dict], k: int, candidates: int, router: str) -> dict:
    strategies = {s: [] for s in ("baseline", "rerank", "router", "router+rerank")}
    for i, row in enumerate(rows, 1):
        q = row["question"]
        t = time.perf_counter()
        scope = route(q, method=router)
        route_s = time.perf_counter() - t
        titles = scope["titles"] or None
        parts = scope["parts"] or None
        title_ok = row["title"] in scope["titles"]

        for name in strategies:
            routed = name.startswith("router")
            t = time.perf_counter()
            if name.endswith("rerank"):
                hits = rag.rerank(q, search(q, candidates, titles if routed else None,
                                            parts if routed else None), k)
            else:
                hits = search(q, k, titles if routed else None, parts if routed else None)
            res = {"id": row["id"], "title": row["title"], "rank": _rank(row, hits),
                   "latency_s": time.perf_counter() - t + (route_s if routed else 0)}
            if routed:
                res["title_ok"] = title_ok
            strategies[name].append(res)
        print(f"{i}/{len(rows)} routed={scope['titles']} gold={row['title']} "
              + " ".join(f"{n}={strategies[n][-1]['rank']}" for n in strategies), flush=True)
    return {name: summarize(res) for name, res in strategies.items()}


def eval_answers(rows: list[dict], name: str) -> dict:
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"{name}.jsonl"
    timings = []
    with open(path, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            res = rag.run(row["question"])
            timings.append(res["timings"])
            f.write(json.dumps({
                "id": row["id"], "title": row["title"], "question": row["question"],
                "reference": row["reference"], "gold_context": row["context"],
                "answer": res["answer"], "scope": res["scope"], "timings": res["timings"],
                "rank": _rank(row, [{"title": int(s["citation"].split()[0]),
                                     "section": s["citation"].split()[-1]} for s in res["sources"]]),
                "contexts": [f"{s['citation']} {s['heading']}\n{s['text']}" for s in res["sources"]],
            }) + "\n")
            f.flush()
            print(f"{i}/{len(rows)} {res['timings']}", flush=True)
    keys = timings[0].keys()
    return {"run_file": str(path), "n": len(rows), **{
        f"{k}_p50": round(statistics.median(t[k] for t in timings), 2) for k in keys}, **{
        f"{k}_p95": round(pct([t[k] for t in timings], 0.95), 2) for k in ("first_token_s", "total_s")}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["retrieval", "answers"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--name", default=datetime.now().strftime("run-%Y%m%d-%H%M"))
    ap.add_argument("--k", type=int, default=config.CONTEXT_K)
    ap.add_argument("--candidates", type=int, default=config.CANDIDATES)
    ap.add_argument("--router", choices=["citation", "vote", "llm"], default=config.ROUTER)
    args = ap.parse_args()

    rows = load_testset(args.limit)
    summary = (eval_retrieval(rows, args.k, args.candidates, args.router) if args.mode == "retrieval"
               else eval_answers(rows, args.name))
    summary = {"mode": args.mode, "name": args.name, "k": args.k, "candidates": args.candidates,
               "llm": f"{config.LLM_PROVIDER}:{config.LLM_MODEL}", "router": args.router, "results": summary}
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{args.name}-{args.mode}-summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
