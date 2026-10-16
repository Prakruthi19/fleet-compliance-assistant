# One image serving the API and the built UI on port 8000.
#   docker compose --profile app up -d --build

FROM node:22-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/models
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Bake the search models into the image so a cold start doesn't download them.
RUN python -c "from fastembed import TextEmbedding, SparseTextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5'); SparseTextEmbedding('Qdrant/bm25')"
COPY regrag/ regrag/
COPY --from=web /web/dist web/dist
# Small runtime data: title catalog, ingest state (corpus version, dense
# status) and the precomputed obligation cards. The vectors live in Qdrant.
COPY data/catalog.json data/obligations.json data/
# Slim copy of data/state.db (titles table only; regenerate after re-ingest).
COPY data/state.deploy.db data/state.db
# Hugging Face Spaces runs the container as uid 1000; let it write feedback
# and model cache files.
RUN chmod -R a+rwX /app/data /models
# Hosts like Hugging Face Spaces / Render set PORT; default 8000 locally.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn regrag.api:app --host 0.0.0.0 --port ${PORT:-8000}"]


