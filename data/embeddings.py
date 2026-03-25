"""
Vector embedding via Azure OpenAI text-embedding-3-large (dimensions=1024).
"""

import os

from openai import AzureOpenAI
from dotenv import load_dotenv

from data import db

load_dotenv()

EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT", "azure-text-embedding-3-large")
EMBEDDING_DIMENSIONS = 1024

_client: AzureOpenAI | None = None


def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = _get_client().embeddings.create(
        model=EMBEDDING_DEPLOYMENT,
        input=texts,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return [item.embedding for item in response.data]


def similarity_search(
    query_text: str,
    top_k: int = 5,
    filters: dict | None = None,
) -> list[dict]:
    """
    Embed query_text and find nearest neighbours in pgvector.
    Optional filters: colour_group_name, product_type_name, garment_group_name.
    """
    query_vec = embed_texts([query_text])[0]
    return db.similarity_search(query_vec, top_k=top_k, filters=filters)
