"""
Hybrid retrieval: dense + BM25 sparse candidates fused with reciprocal rank
fusion inside Qdrant (one round trip), optionally scoped to titles/parts.
Scoping is what keeps whole-CFR search precise: the router/agents narrow to
the relevant titles first, then search within them.
"""
import sqlite3
import time

from qdrant_client import models

from regrag.config import COLLECTION, STATE_DB
from regrag.store import DENSE, SPARSE, client, dense_model, sparse_model


def _filter(titles: list[int] | None, parts: list[str] | None) -> models.Filter | None:
    must = []
    if titles:
        must.append(models.FieldCondition(key="title", match=models.MatchAny(any=titles)))
    if parts:
        must.append(models.FieldCondition(key="part", match=models.MatchAny(any=parts)))
    return models.Filter(must=must) if must else None


_dense_cache: tuple[float, dict[int, bool]] = (0.0, {})


def _dense_status() -> dict[int, bool]:
    """{title: fully dense-embedded?} from ingest state, refreshed every minute."""
    global _dense_cache
    if time.time() - _dense_cache[0] > 60:
        db = sqlite3.connect(STATE_DB)
        _dense_cache = (time.time(), {n: bool(d) for n, d in db.execute("SELECT number, dense FROM titles")})
    return _dense_cache[1]


def use_dense(titles: list[int] | None) -> bool:
    """
    Only fuse in the dense ranking when every title in scope is fully
    dense-embedded. With a partially embedded scope, the dense list can
    only contain the embedded titles, so RRF gives them an unearned boost
    (seen in testing: title 1 "Fees" sections outranking the correct
    title 49 hits). Keyword-only is the honest fallback.
    """
    status = _dense_status()
    scope = titles or list(status)
    return bool(scope) and all(status.get(t, False) for t in scope)


def keyword_search(query: str, top_k: int = 20, titles: list[int] | None = None) -> list[dict]:
    """BM25 only, minimal payload: used for routing probes."""
    sp = next(sparse_model().query_embed(query))
    res = client().query_points(
        COLLECTION, query=models.SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist()),
        using=SPARSE, query_filter=_filter(titles, None), limit=top_k,
        with_payload=["title", "part", "section"],
    )
    return [{**p.payload, "score": p.score} for p in res.points]


def search(query: str, top_k: int = 8, titles: list[int] | None = None,
           parts: list[str] | None = None) -> list[dict]:
    sp = next(sparse_model().query_embed(query))
    sparse_q = models.SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist())
    flt = _filter(titles, parts)
    if not use_dense(titles):
        res = client().query_points(COLLECTION, query=sparse_q, using=SPARSE, query_filter=flt,
                                    limit=top_k, with_payload=True)
        return [{**p.payload, "score": p.score} for p in res.points]

    dense = next(dense_model().query_embed(query)).tolist()
    candidates = top_k * 4
    res = client().query_points(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=dense, using=DENSE, limit=candidates, filter=flt),
            models.Prefetch(query=sparse_q, using=SPARSE, limit=candidates, filter=flt),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return [{**p.payload, "score": p.score} for p in res.points]


if __name__ == "__main__":
    import sys
    for hit in search(" ".join(sys.argv[1:]) or "definition of agency"):
        print(f"{hit['score']:.3f}  [{hit['title']} CFR {hit['section']}] {hit['heading']}")
    client().close()
