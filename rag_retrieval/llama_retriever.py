"""LlamaIndex VectorStoreIndex retriever with propensity-score filtering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from rag_retrieval.langchain_retriever import OfferResult
from shared.logger import get_logger

logger = get_logger(__name__)

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_OFFERS_PATH = Path(__file__).parent / "data" / "offers.json"


class LlamaOfferRetriever:
    def __init__(self, offers_path: Path = _OFFERS_PATH) -> None:
        Settings.embed_model = HuggingFaceEmbedding(model_name=_MODEL_NAME)
        Settings.llm = None  # disable LLM; retrieval only

        offers: list[dict[str, Any]] = json.loads(
            offers_path.read_text(encoding="utf-8")
        )
        docs = [
            Document(
                text=o["description"],
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
        self._index = VectorStoreIndex.from_documents(docs, show_progress=False)

    def retrieve(
        self,
        customer_context: str,
        propensity: float,
        k: int = 5,
    ) -> list[OfferResult]:
        retriever = self._index.as_retriever(similarity_top_k=k * 4)
        nodes = retriever.retrieve(customer_context)

        results: list[OfferResult] = []
        for node in nodes:
            meta: dict[str, Any] = node.metadata
            threshold = float(meta.get("min_propensity_threshold", 0.0))
            if propensity < threshold:
                continue
            results.append(
                OfferResult(
                    offer_id=meta.get("id", ""),
                    title=meta.get("title", ""),
                    description=node.get_content(),
                    category=meta.get("category", ""),
                    score=float(node.score or 0.0),
                )
            )
            if len(results) == k:
                break

        logger.info(
            "llama_retrieve",
            context_snippet=customer_context[:60],
            propensity=propensity,
            returned=len(results),
        )
        return results
