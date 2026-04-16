"""LangChain PGVector retriever with propensity-score filtering."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from shared.config import get_settings
from shared.logger import get_logger

logger = get_logger(__name__)

_COLLECTION = "lc_offer_embeddings"
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_OFFERS_PATH = Path(__file__).parent / "data" / "offers.json"


@dataclass
class OfferResult:
    offer_id: str
    title: str
    description: str
    category: str
    score: float


class LangChainOfferRetriever:
    def __init__(self, offers_path: Path = _OFFERS_PATH) -> None:
        settings = get_settings()
        self._embeddings = HuggingFaceEmbeddings(model_name=_MODEL_NAME)
        self._conn = settings.postgres_url

        # Build the LangChain-managed collection if it is empty
        self._store = PGVector(
            connection_string=self._conn,
            embedding_function=self._embeddings,
            collection_name=_COLLECTION,
            pre_delete_collection=False,
        )
        if self._collection_empty():
            self._index_offers(offers_path)

    # ------------------------------------------------------------------

    def _collection_empty(self) -> bool:
        try:
            return len(self._store.similarity_search("test", k=1)) == 0
        except Exception:
            return True

    def _index_offers(self, offers_path: Path) -> None:
        offers = json.loads(offers_path.read_text(encoding="utf-8"))
        docs = [
            Document(
                page_content=o["description"],
                metadata={
                    "id": o["id"],
                    "title": o["title"],
                    "category": o["category"],
                    "channel": o["channel"],
                    "min_propensity_threshold": o["min_propensity_threshold"],
                    "discount_pct": o["discount_pct"],
                    "expiry_days": o["expiry_days"],
                },
            )
            for o in offers
        ]
        PGVector.from_documents(
            documents=docs,
            embedding=self._embeddings,
            collection_name=_COLLECTION,
            connection_string=self._conn,
            pre_delete_collection=True,
        )
        logger.info("langchain_indexed", count=len(docs))

    # ------------------------------------------------------------------

    def retrieve(
        self,
        customer_context: str,
        propensity: float,
        k: int = 5,
    ) -> list[OfferResult]:
        candidates = self._store.similarity_search_with_score(
            customer_context,
            k=k * 4,
        )

        results: list[OfferResult] = []
        for doc, score in candidates:
            meta: dict[str, Any] = doc.metadata
            threshold = float(meta.get("min_propensity_threshold", 0.0))
            if propensity < threshold:
                continue
            results.append(
                OfferResult(
                    offer_id=meta.get("id", ""),
                    title=meta.get("title", ""),
                    description=doc.page_content,
                    category=meta.get("category", ""),
                    score=float(score),
                )
            )
            if len(results) == k:
                break

        logger.info(
            "langchain_retrieve",
            context_snippet=customer_context[:60],
            propensity=propensity,
            returned=len(results),
        )
        return results

    def retrieve_with_qa(
        self,
        customer_context: str,
        propensity: float,
        k: int = 5,
    ) -> dict[str, Any]:
        results = self.retrieve(customer_context, propensity, k)
        return {
            "answer": f"Retrieved {len(results)} offers for context.",
            "filtered_offers": [r.description for r in results],
        }
