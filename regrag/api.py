"""
HTTP API. Run: uvicorn regrag.api:app --port 8000

  POST /api/ask       {"question": "..."} -> text/event-stream of pipeline events
                      (scope, sources, answer tokens, timings)
  GET  /api/obligations  precomputed profile -> requirement cards (regrag.obligations)
  GET  /api/titles    indexed titles and whether semantic search is enabled for each
  POST /api/feedback  thumbs up/down + comment, appended to data/feedback.jsonl
                      (becomes eval cases)
  GET  /health

Latency measures here:
  - startup warms every model (LLM, dense, sparse, reranker) so no user pays
    for cold loads (~18 s for the LLM alone on CPU);
  - answers stream token by token;
  - exact-repeat questions are served from Redis. The cache key includes a
    corpus version, so re-ingesting a changed title invalidates old answers.
"""
import hashlib
import json
import sqlite3
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from regrag import catalog, config, llm, obligations, rag
from regrag.retrieval import _dense_status, search

FEEDBACK = config.DATA_DIR / "feedback.jsonl"
CACHE_TTL_S = 24 * 3600


def _redis():
    if not config.REDIS_URL:
        return None
    try:
        import redis
        r = redis.Redis.from_url(config.REDIS_URL, socket_timeout=5, socket_connect_timeout=5)
        r.ping()
        return r
    except Exception as e:
        print(f"[api] Redis unavailable, caching disabled: {e!r}")
        return None


def corpus_version() -> str:
    db = sqlite3.connect(config.STATE_DB)
    rows = db.execute("SELECT number, amended_on, dense FROM titles ORDER BY number").fetchall()
    return hashlib.sha1(repr(rows).encode()).hexdigest()[:12]


def cache_key(question: str) -> str:
    norm = " ".join(question.lower().split())
    return f"regrag:answer:{config.LLM_MODEL}:p{rag.PROMPT_VERSION}:{corpus_version()}:{hashlib.sha1(norm.encode()).hexdigest()}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = time.perf_counter()
    llm.warm()
    search("warm up", 1)
    if config.RERANK:
        rag.rerank("warm up", [{"breadcrumb": "", "text": "warm up"}], 1)
    print(f"[api] models warm in {time.perf_counter() - t:.1f}s")
    app.state.redis = _redis()
    yield


app = FastAPI(title="regrag", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class Feedback(BaseModel):
    question: str
    answer: str
    rating: int = Field(ge=-1, le=1)
    comment: str = ""
    sources: list[str] = []


_asks: dict[str, deque] = defaultdict(deque)


def _check_limit(request: Request) -> None:
    """In-memory sliding window per client IP; fine for a single-instance demo."""
    if not config.ASK_LIMIT_PER_HOUR:
        return
    fwd = request.headers.get("x-forwarded-for")
    ip = fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")
    now, q = time.time(), _asks[ip]
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= config.ASK_LIMIT_PER_HOUR:
        raise HTTPException(429, f"Demo limit: {config.ASK_LIMIT_PER_HOUR} questions per hour. Try again later.")
    q.append(now)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.post("/api/ask")
def ask(body: Ask, request: Request):
    r = app.state.redis
    key = cache_key(body.question)
    if r is None or not r.exists(key):  # cached answers cost no LLM quota
        _check_limit(request)

    def stream():
        if r is not None and (cached := r.get(key)):
            for ev in json.loads(cached):
                yield _sse(ev)
            yield _sse({"type": "cache", "hit": True})
            return
        events = []
        for ev in rag.events(body.question):
            events.append(ev)
            yield _sse(ev)
        if r is not None:
            # Store tokens merged into one event: replaying hundreds of
            # tiny events is pointless for a cached answer.
            answer = "".join(e["text"] for e in events if e["type"] == "token")
            compact = [e for e in events if e["type"] != "token"]
            compact.insert(-1, {"type": "token", "text": answer})
            r.setex(key, CACHE_TTL_S, json.dumps(compact))

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/titles")
def titles():
    dense = _dense_status()
    return [{"number": n, "name": t["name"], "semantic": dense.get(n, False)}
            for n, t in sorted(catalog.load().items())]


@app.get("/api/obligations")
def get_obligations():
    return obligations.load()


@app.post("/api/feedback")
def feedback(body: Feedback):
    FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK, "a", encoding="utf-8") as f:
        f.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), **body.model_dump()}) + "\n")
    return {"ok": True}


@app.get("/health")
def health():
    return {"ok": True, "llm": f"{config.LLM_PROVIDER}:{config.LLM_MODEL}",
            "corpus_version": corpus_version(), "cache": app.state.redis is not None}


# Production: serve the built UI (web/dist) from the same origin as the API.
# Mounted last so /api/* and /health take precedence.
_dist = config.ROOT / "web" / "dist"
if _dist.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=_dist, html=True), name="web")
