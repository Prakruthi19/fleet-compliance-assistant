"""
Qdrant vector store: one collection holding a dense vector (semantic) and a
sparse BM25 vector (exact terms like "391.25" or "hazardous materials
table") per chunk, so retrieval is hybrid without a second search engine.

Memory/cost savers for a whole-CFR index:
  - dense vectors kept on disk, with an int8-quantized copy in RAM (~4x
    smaller) used for the HNSW search; originals only used for rescoring.
  - payload indexes on title/part so filtered searches stay fast.
"""
import uuid
from functools import lru_cache

from qdrant_client import QdrantClient, models

from regrag.config import COLLECTION, DENSE_MODEL, QDRANT_API_KEY, QDRANT_PATH, QDRANT_URL, SPARSE_MODEL

DENSE = "dense"
SPARSE = "bm25"


@lru_cache
def client() -> QdrantClient:
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
    return QdrantClient(path=QDRANT_PATH)


@lru_cache
def dense_model():
    from fastembed import TextEmbedding
    return TextEmbedding(DENSE_MODEL)


@lru_cache
def sparse_model():
    from fastembed import SparseTextEmbedding
    return SparseTextEmbedding(SPARSE_MODEL)


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def ensure_collection() -> None:
    c = client()
    if c.collection_exists(COLLECTION):
        return
    dim = dense_model().embedding_size
    c.create_collection(
        COLLECTION,
        vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE, on_disk=True)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)
        ),
    )
    c.create_payload_index(COLLECTION, "title", models.PayloadSchemaType.INTEGER)
    c.create_payload_index(COLLECTION, "part", models.PayloadSchemaType.KEYWORD)
