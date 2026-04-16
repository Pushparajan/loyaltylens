"""Retrieve semantically similar documents from Weaviate for RAG context."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from rag_retrieval.weaviate_client import WeaviateClient, _CLASS_NAME
from shared.logger import get_logger

logger = get_logger(__name__)

_EMBED_MODEL = "all-MiniLM-L6-v2"


class VectorRetriever:
    """Encode a query and return the top-k nearest documents from Weaviate."""

    def __init__(self, weaviate: WeaviateClient | None = None, top_k: int = 5) -> None:
        self._weaviate = weaviate or WeaviateClient()
        self._embed = SentenceTransformer(_EMBED_MODEL)
        self._top_k = top_k

    def retrieve(self, query: str) -> list[dict[str, str]]:
        """Return top-k documents most similar to *query*."""
        vector = self._embed.encode(query).tolist()
        collection = self._weaviate.client.collections.get(_CLASS_NAME)
        result = collection.query.near_vector(
            near_vector=vector,
            limit=self._top_k,
            return_properties=["text", "metadata"],
        )
        docs = [
            {"text": obj.properties["text"], "metadata": obj.properties.get("metadata", "")}
            for obj in result.objects
        ]
        logger.info("documents_retrieved", query_len=len(query), count=len(docs))
        return docs
