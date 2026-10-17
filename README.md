# Federal Rules Guide

Plain-English answers to U.S. federal regulations, with the exact rule cited —
plus a compliance checklist for common business types.

**Live demo:** https://federal-rules-guide.onrender.com
(free hosting: the first visit after 15 idle minutes takes ~30–60 s to wake up)

## What it does

- **My checklist** — pick a business type (trucking company, employer, food or
  supplement maker, industrial facility), answer a few yes/no questions, and get
  requirement cards: what you must do, deadlines, records to keep, each linked to
  the CFR section it comes from.
- **Ask a question** — a short answer plus labelled key points (Who, Deadline,
  Records…), every fact cited to a section you can open and read.

Coverage: CFR titles 1, 21 (Food & Drugs), 29 (Labor), 40 (Environment) and
49 (Transportation) — 122,269 chunks from the official eCFR. General
information, not legal advice.

## Architecture

```
eCFR API ──► ingest (parse → section chunks → BM25 sparse + bge-small dense)
                │   incremental: skips unchanged titles / chunks by content hash
                ▼
           Qdrant (named dense + sparse vectors per chunk, title/part filters)
                │
question ──► router (only scopes when a title/part is cited) ──► hybrid search
                │                                     (RRF fusion, server-side)
                ▼
           Groq LLM (Qwen 27B) ──► streamed answer with [n] citations
                │
           FastAPI (SSE) + React UI, Redis cache for repeat questions

checklist: curated part-scoped searches ──► LLM writes structured cards ONCE
           ──► data/obligations.json ──► served instantly, no LLM per visit
```

| Piece | Choice | Why |
|---|---|---|
| Vector DB | Qdrant (Docker locally, Qdrant Cloud in prod) | Same engine local and hosted; named dense+sparse vectors with server-side RRF; moved to the cloud as one snapshot |
| Embeddings | fastembed: `Qdrant/bm25` + `BAAI/bge-small-en-v1.5` (ONNX, CPU) | No API cost; BM25 is ~250× cheaper, so every title is searchable first and dense vectors are added per title |
| LLM | Groq, `qwen/qwen3.8-27b`, reasoning off | Local CPU Ollama took minutes per answer; Groq answers in ~1–5 s |
| API / UI | FastAPI + Server-Sent Events, React + Vite, lucide icons | Answers stream token by token |
| Hosting | Render (free, Docker) | Hugging Face Docker Spaces now need PRO; container needs only ~125 MB RAM |

## Evaluation

50 synthetic questions, each with one gold CFR section (`evals/testset.jsonl`).
Retrieval is scored by **hit@6** (gold section in the 6 chunks the LLM reads)
and **MRR** (how high it ranks). Answers are scored by a DeepEval judge that is a
*different, larger* model than the generator (`gpt-oss-120b` judging Qwen), to
avoid self-grading bias.

**Retrieval (keyword index, all 50 questions)**

| Strategy | hit@6 | MRR | Added latency | Decision |
|---|---|---|---|---|
| No routing (baseline) | **0.82** | 0.60 | — | **kept** |
| Keyword "vote" router | 0.78 | 0.59 | <1 s | dropped |
| LLM router (Groq) | 0.78 | 0.59 | ~0.4 s, title acc 92% | dropped — right title more often, but no retrieval gain |
| Cross-encoder rerank | 0.80 | 0.64 | +5–9 s on CPU | off until faster hardware |

**Hybrid vs keyword** (titles 1 and 21, which are dense-embedded so far; 20 questions):

| Search | hit@6 | MRR |
|---|---|---|
| Keyword (BM25) | 0.80 | 0.64 |
| Hybrid (BM25 + dense, RRF) | **0.90** | **0.75** |

**Answers** (same 15 questions, same judge):

| Answer format | Faithfulness | Relevancy | Citations valid | Median length |
|---|---|---|---|---|
| e2: free-form | 0.95 | 0.95 | 1.00 | 153 words |
| e3: short answer + labelled points | 0.95 | **0.97** | 1.00 | **96 words** |

Single-question latency on the live site is ~1–5 s. Under back-to-back eval load
it rises to ~22 s median because of Groq free-tier rate limits (8k tokens/min),
not model speed.

Run it yourself:
```bash
python -m regrag.eval.run retrieval --name myrun       # routing / rerank comparison
python -m regrag.eval.run dense --titles 1,21          # keyword vs hybrid
python -m regrag.eval.run answers --name myrun         # full pipeline → evals/runs/
venv-eval/Scripts/python.exe evals/judge.py evals/runs/myrun.jsonl --judge groq
```

## Limitations

- Hybrid search turns on only when every title in scope has dense vectors; titles
  29, 40 and 49 are still being embedded (≈2–3 chunks/s on a laptop CPU), so the
  live site is keyword-only for now.
- The test set is synthetic and small (50 questions); differences of a few
  points are within noise.
- The judge only scored a 15-question subset (titles 1 and 21) because of the
  judge model's daily token quota.
- Free tiers: Groq rate limits (live site caps each visitor at 20 questions/hour),
  Render cold starts.

## Run locally

```bash
python -m venv venv && venv/Scripts/pip install -r requirements.txt
cp .env.example .env                       # add LLM_API_KEY (Groq) etc.
docker compose up -d                       # Qdrant + Redis
python -m regrag.ingest.pipeline --all --sparse-only   # keyword index for every title
python -m regrag.ingest.pipeline --titles 49           # add dense vectors for a title
python -m regrag.obligations build                     # precompute checklist cards
cd web && npm ci && npm run build && cd ..
uvicorn regrag.api:app --port 8000                     # http://localhost:8000
```

Deploy: `render.yaml` (Render Blueprint) builds the `Dockerfile`; set
`QDRANT_URL`, `QDRANT_API_KEY` and `LLM_API_KEY` in the Render dashboard.

## Project layout

```
regrag/
  ingest/        eCFR download, XML parse, chunking, incremental pipeline
  retrieval.py   hybrid / keyword search over Qdrant
  router.py      title/part scoping (citation | vote | llm)
  rag.py         retrieve → prompt → stream answer
  obligations.py business profiles → precomputed requirement cards
  api.py         FastAPI: /api/ask (SSE), /api/obligations, /api/feedback
  eval/          test set generation + eval runner
evals/           DeepEval judge, test set, run results
web/             React UI
docs/legacy-fmcsa-demo.md   the earlier small FMCSA multi-hop demo (src/, app.py)
```
