"""
Builds the fleet-compliance-index inside the existing fitadvisor-search
Azure AI Search service, and uploads embedded FMCSA regulation chunks.

Run once (or whenever data/fmcsa_corpus.json changes):
    python src/build_index.py
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
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
EMBEDDING_DIM = 1536

CORPUS_PATH = Path(__file__).parent.parent / "data" / "fmcsa_corpus.json"

client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)


def embed(text: str) -> list[float]:
    resp = client.embeddings.create(
        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT, input=text
    )
    return resp.data[0].embedding


def create_index(index_client: SearchIndexClient):
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="part", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="section", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="domain", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="source_url", type=SearchFieldDataType.String),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SearchableField(name="text", type=SearchFieldDataType.String),
        SearchField(
            name="text_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIM,
            vector_search_profile_name="default-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="default-hnsw")],
        profiles=[
            VectorSearchProfile(
                name="default-profile", algorithm_configuration_name="default-hnsw"
            )
        ],
    )

    index = SearchIndex(name=INDEX_NAME, fields=fields, vector_search=vector_search)
    index_client.create_or_update_index(index)
    print(f"Index '{INDEX_NAME}' created/updated.")


def upload_documents(search_client: SearchClient):
    corpus = json.loads(CORPUS_PATH.read_text())
    docs = []
    for chunk in corpus:
        vector = embed(chunk["text"])
        docs.append(
            {
                "id": chunk["id"],
                "part": chunk["part"],
                "section": chunk["section"],
                "domain": chunk["domain"],
                "source_url": chunk.get("source_url", ""),
                "title": chunk["title"],
                "text": chunk["text"],
                "text_vector": vector,
            }
        )
    result = search_client.upload_documents(documents=docs)
    succeeded = sum(1 for r in result if r.succeeded)
    print(f"Uploaded {succeeded}/{len(docs)} chunks to '{INDEX_NAME}'.")


if __name__ == "__main__":
    credential = AzureKeyCredential(SEARCH_API_KEY)
    index_client = SearchIndexClient(endpoint=SEARCH_ENDPOINT, credential=credential)
    create_index(index_client)

    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=credential
    )
    upload_documents(search_client)