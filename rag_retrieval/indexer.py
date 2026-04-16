"""Index loyalty documents with pre-computed embeddings into Weaviate."""

from __future__ import annotations

from typing import Any

from sentence_transformers import SentenceTransformer

from rag_retrieval.weaviate_client import WeaviateClient, _CLASS_NAME
from shared.logger import get_logger

logger = get_logger(__name__)

_EMBED_MODEL = "all-MiniLM-L6-v2"


class DocumentIndexer:
    """Embed documents with sentence-transformers and upsert into Weaviate."""

    def __init__(self, weaviate: WeaviateClient | None = None) -> None:
        self._weaviate = weaviate or WeaviateClient()
        self._embed = SentenceTransformer(_EMBED_MODEL)
        self._weaviate.ensure_collection()

    def index(self, documents: list[dict[str, Any]]) -> int:
        """
        Embed and insert documents. Each dict must have 'text' and optionally 'metadata'.
        Returns the count of indexed documents.
        """
        collection = self._weaviate.client.collections.get(_CLASS_NAME)
        texts = [d["text"] for d in documents]
        vectors = self._embed.encode(texts, show_progress_bar=False).tolist()

        with collection.batch.dynamic() as batch:
            for doc, vec in zip(documents, vectors):
                batch.add_object(
                    properties={"text": doc["text"], "metadata": str(doc.get("metadata", {}))},
                    vector=vec,
                )

        logger.info("documents_indexed", count=len(documents))
        return len(documents)
