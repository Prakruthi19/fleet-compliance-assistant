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
COPY regrag/ regrag/
COPY --from=web /web/dist web/dist
EXPOSE 8000
CMD ["uvicorn", "regrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
