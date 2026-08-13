"""
Hybrid (vector + keyword) retrieval against fleet-compliance-index.
Hybrid search matters here because FMCSA questions often reference exact
terms (e.g. "391.25", "periodic inspection") where keyword matching beats
pure vector similarity, alongside paraphrased questions where vector
similarity wins. Combining both is a deliberate, defensible design choice.

Embeddings use Azure OpenAI (not GitHub Models) — GitHub Models serves
embeddings through a separate endpoint (models.github.ai) with an
"openai/" model prefix, which the older chat-completions-style endpoint
used elsewhere in this project doesn't support. Using Azure OpenAI here
keeps embedding generation on the same provider as index-build time,
which matters: query-time and index-time embeddings must come from the
same model to be comparable in vector search.
"""
import os

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI

load_dotenv()

SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_API_KEY = os.environ["AZURE_SEARCH_API_KEY"]
INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME", "fleet-compliance-index")

AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ.get(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
)

_search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_API_KEY),
)
_embed_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)


def embed(text: str) -> list[float]:
    resp = _embed_client.embeddings.create(
        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT, input=text
    )
    return resp.data[0].embedding


def retrieve(query: str, top_k: int = 3, domain: str = None) -> list[dict]:
    """
    Hybrid search: BM25 keyword search + vector similarity, fused by Azure
    AI Search.

    domain: optional filter, e.g. "hazmat", "inspection",
    "driver_qualification". When set, only chunks tagged with that domain
    are eligible to be returned — this uses Azure AI Search's OData
    $filter, applied before scoring, not a post-hoc Python filter.
    """
    vector_query = VectorizedQuery(
        vector=embed(query), k_nearest_neighbors=top_k, fields="text_vector"
    )
    search_kwargs = dict(
        search_text=query,              # keyword/BM25 side
        vector_queries=[vector_query],  # vector side
        select=["id", "part", "section", "domain", "title", "text"],
        top=top_k,
    )
    if domain:
        search_kwargs["filter"] = f"domain eq '{domain}'"

    results = _search_client.search(**search_kwargs)
    return [
        {
            "id": r["id"],
            "part": r["part"],
            "section": r["section"],
            "domain": r["domain"],
            "title": r["title"],
            "text": r["text"],
            "score": r["@search.score"],
        }
        for r in results
    ]