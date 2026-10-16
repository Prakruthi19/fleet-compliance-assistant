"""
Central settings, read from environment / .env.

Everything defaults to free, local components:
  - Qdrant: server at QDRANT_URL if set, otherwise embedded on-disk mode at
    QDRANT_PATH (no Docker needed; fine for a few titles, use the server for
    the full CFR because embedded mode does brute-force search).
  - Embeddings: fastembed ONNX models on CPU, no API cost.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("REGRAG_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
STATE_DB = DATA_DIR / "state.db"

QDRANT_URL = os.environ.get("QDRANT_URL")  # e.g. http://localhost:6333
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")  # Qdrant Cloud
QDRANT_PATH = os.environ.get("QDRANT_PATH", str(DATA_DIR / "qdrant_local"))
COLLECTION = os.environ.get("QDRANT_COLLECTION", "ecfr")

DENSE_MODEL = os.environ.get("DENSE_MODEL", "BAAI/bge-small-en-v1.5")
SPARSE_MODEL = os.environ.get("SPARSE_MODEL", "Qdrant/bm25")

# ~2000 chars is ~450 tokens: under bge-small's 512-token limit once the
# hierarchy header is prepended.
CHUNK_MAX_CHARS = int(os.environ.get("CHUNK_MAX_CHARS", "2000"))
EMBED_BATCH = int(os.environ.get("EMBED_BATCH", "64"))

ECFR_API = "https://www.ecfr.gov/api/versioner/v1"

# LLM: "ollama" (local, default), "azure" (uses the AZURE_OPENAI_* vars), or
# "openai" (any OpenAI-compatible host such as Groq, via LLM_BASE_URL/LLM_API_KEY).
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.5:4b")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY")
# Hosted reasoning models think silently before the first token; on Groq,
# "low" (gpt-oss) or "none" (qwen) cut first-token time from ~8-11 s to ~1 s.
LLM_REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT")
# Cap answer length: Groq's free tier rejects requests whose expected output
# exceeds its per-minute output-token limit (1000 OTPM on qwen3.8-27b).
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "700"))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# Keep the model loaded between requests: first load costs ~18 s on CPU.
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")
# Smaller context window = less RAM and faster prompt processing on CPU.
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))

RERANK_MODEL = os.environ.get("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
CANDIDATES = int(os.environ.get("CANDIDATES", "30"))   # hybrid search results fed to the reranker
CONTEXT_K = int(os.environ.get("CONTEXT_K", "6"))      # reranked chunks given to the LLM

REDIS_URL = os.environ.get("REDIS_URL")  # e.g. redis://localhost:6380/0; unset disables caching
# Per-client /api/ask limit for a public demo: the free Groq quota is shared
# by everyone. 0 disables.
ASK_LIMIT_PER_HOUR = int(os.environ.get("ASK_LIMIT_PER_HOUR", "0"))
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")

# Scope selection before search. Eval e1 (50 q, keyword-only index):
#   no routing 82% hit@6 | vote router 78% | llm router ~90 s/call on this laptop.
# So the default only scopes when the user explicitly cites a title/part.
#   "citation" | "vote" (keyword probe, ~ms) | "llm" (one LLM call)
ROUTER = os.environ.get("ROUTER", "citation")
# Cross-encoder rerank: +0.04 MRR, -0.02 hit@6, +9 s on this CPU (eval e1).
# Off by default until it runs on faster hardware.
RERANK = os.environ.get("RERANK", "0") == "1"
