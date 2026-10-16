"""
Incremental, tiered eCFR ingestion.

    python -m regrag.ingest.pipeline --all --sparse-only   # whole CFR, keyword-searchable in minutes
    python -m regrag.ingest.pipeline --titles 21,40,49     # add dense (semantic) vectors for hot titles
    python -m regrag.ingest.pipeline --all                 # dense everywhere (hours on CPU)
    python -m regrag.ingest.pipeline --titles 49 --force   # ignore saved state

Tiers: BM25 sparse vectors are ~250x cheaper to compute than dense ones on
CPU, so every title can be searchable by keyword immediately, and dense
vectors are added later only where they're worth the compute. Retrieval
fuses whatever vectors a chunk has, so nothing else changes.

Work is skipped at three levels, so re-runs (e.g. a nightly cron) cost
almost nothing:
  1. title level — latest_amended_on unchanged (and requested tier already
     present) -> skip before downloading.
  2. download level — XML cached on disk per issue date.
  3. chunk level — content hash per chunk; only new/changed chunks are
     re-embedded, and chunks that disappeared are deleted from the index.
"""
import argparse
import sqlite3
import time

from qdrant_client import models

from regrag.config import COLLECTION, DATA_DIR, EMBED_BATCH, STATE_DB
from regrag.ingest.chunk import Chunk, chunk_section
from regrag.ingest.ecfr import download_title, list_titles
from regrag.ingest.parse import iter_sections
from regrag.store import DENSE, SPARSE, client, dense_model, ensure_collection, point_id, sparse_model

FLUSH_EVERY = EMBED_BATCH * 4


def _db() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(STATE_DB)
    db.execute("CREATE TABLE IF NOT EXISTS titles "
               "(number INTEGER PRIMARY KEY, amended_on TEXT, chunks INTEGER, dense INTEGER)")
    db.execute("CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, title INTEGER, hash TEXT, dense INTEGER)")
    db.execute("CREATE INDEX IF NOT EXISTS chunks_title ON chunks(title)")
    return db


def _dense(batch: list[Chunk]) -> list[list[float]]:
    return [v.tolist() for v in dense_model().embed([c.embed_text for c in batch], batch_size=EMBED_BATCH)]


def _upsert(batch: list[Chunk], with_dense: bool) -> None:
    """Write new/changed chunks: payload + sparse, and dense if requested."""
    sparse = sparse_model().embed([c.embed_text for c in batch], batch_size=EMBED_BATCH)
    dense = _dense(batch) if with_dense else [None] * len(batch)
    points = []
    for c, s, d in zip(batch, sparse, dense):
        vector = {SPARSE: models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())}
        if d is not None:
            vector[DENSE] = d
        points.append(models.PointStruct(id=point_id(c.id), vector=vector, payload=c.payload()))
    client().upsert(COLLECTION, points=points, wait=False)


def _add_dense(batch: list[Chunk]) -> None:
    """Unchanged chunks that were ingested sparse-only: attach dense vectors."""
    client().update_vectors(COLLECTION, points=[
        models.PointVectors(id=point_id(c.id), vector={DENSE: d})
        for c, d in zip(batch, _dense(batch))
    ], wait=False)


def ingest_title(db: sqlite3.Connection, meta: dict, with_dense: bool) -> dict:
    number = meta["number"]
    t0 = time.perf_counter()
    path = download_title(number, meta["up_to_date_as_of"])
    t_dl = time.perf_counter() - t0

    known = {cid: (h, bool(d)) for cid, h, d in
             db.execute("SELECT id, hash, dense FROM chunks WHERE title = ?", (number,))}
    seen: set[str] = set()
    changed: list[Chunk] = []
    needs_dense: list[Chunk] = []
    stats = {"written": 0, "dense_added": 0}

    def flush(force: bool = False) -> None:
        if changed and (force or len(changed) >= FLUSH_EVERY):
            _upsert(changed, with_dense)
            db.executemany("INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?)",
                           [(c.id, number, c.hash, int(with_dense)) for c in changed])
            stats["written"] += len(changed)
            changed.clear()
        if needs_dense and (force or len(needs_dense) >= FLUSH_EVERY):
            _add_dense(needs_dense)
            db.executemany("UPDATE chunks SET dense = 1 WHERE id = ?", [(c.id,) for c in needs_dense])
            stats["dense_added"] += len(needs_dense)
            needs_dense.clear()
        db.commit()

    for sec in iter_sections(path, number):
        for ch in chunk_section(sec):
            if ch.id in seen:  # duplicate section numbers do occur; keep first
                continue
            seen.add(ch.id)
            old = known.get(ch.id)
            if old is None or old[0] != ch.hash:
                changed.append(ch)
            elif with_dense and not old[1]:
                needs_dense.append(ch)
            else:
                continue
            before = stats["written"] + stats["dense_added"]
            flush()
            if stats["written"] + stats["dense_added"] != before:
                print(f"  title {number}: {stats}", flush=True)
    flush(force=True)

    stale = [cid for cid in known if cid not in seen]
    if stale:
        client().delete(COLLECTION, points_selector=models.PointIdsList(points=[point_id(c) for c in stale]))
        db.executemany("DELETE FROM chunks WHERE id = ?", [(c,) for c in stale])

    (missing_dense,) = db.execute("SELECT COUNT(*) FROM chunks WHERE title = ? AND dense = 0", (number,)).fetchone()
    db.execute("INSERT OR REPLACE INTO titles VALUES (?, ?, ?, ?)",
               (number, meta["latest_amended_on"], len(seen), int(missing_dense == 0)))
    db.commit()
    return {"title": number, "chunks": len(seen), **stats, "deleted": len(stale),
            "download_s": round(t_dl, 1), "total_s": round(time.perf_counter() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--titles", help="comma-separated title numbers, e.g. 1,49")
    group.add_argument("--all", action="store_true")
    ap.add_argument("--sparse-only", action="store_true", help="BM25 only; skip dense embeddings")
    ap.add_argument("--force", action="store_true", help="re-check every chunk even if the title is unchanged")
    args = ap.parse_args()
    with_dense = not args.sparse_only

    db = _db()
    ensure_collection()
    titles = list_titles()
    if args.titles:
        wanted = {int(t) for t in args.titles.split(",")}
        titles = [t for t in titles if t["number"] in wanted]

    done = {n: (a, bool(d)) for n, a, d in db.execute("SELECT number, amended_on, dense FROM titles")}
    for meta in titles:
        prev = done.get(meta["number"])
        if not args.force and prev and prev[0] == meta["latest_amended_on"] and (prev[1] or not with_dense):
            print(f"title {meta['number']}: unchanged since {meta['latest_amended_on']}, skipped")
            continue
        tier = "sparse+dense" if with_dense else "sparse"
        print(f"title {meta['number']} ({meta['name']}): ingesting [{tier}]", flush=True)
        print(f"  {ingest_title(db, meta, with_dense)}", flush=True)
    client().close()


if __name__ == "__main__":
    main()
