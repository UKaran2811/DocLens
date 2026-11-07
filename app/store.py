"""Qdrant + embeddings wiring."""
from functools import lru_cache

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from .config import COLLECTION, EMBED_MODEL, QDRANT_API_KEY, QDRANT_PATH, QDRANT_URL


@lru_cache
def embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=EMBED_MODEL)


@lru_cache
def client() -> QdrantClient:
    # Qdrant Cloud / Docker when QDRANT_URL is set; otherwise embedded local storage
    # (embedded allows only ONE process at a time).
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    return QdrantClient(path=QDRANT_PATH)


def vector_store() -> QdrantVectorStore:
    c = client()
    if not c.collection_exists(COLLECTION):
        dim = len(embeddings().embed_query("dimension probe"))
        c.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        c.create_payload_index(
            COLLECTION,
            field_name="metadata.source",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    return QdrantVectorStore(client=c, collection_name=COLLECTION, embedding=embeddings())


def source_filter(source: str | None) -> models.Filter | None:
    if not source:
        return None
    return models.Filter(
        must=[models.FieldCondition(key="metadata.source", match=models.MatchValue(value=source))]
    )
