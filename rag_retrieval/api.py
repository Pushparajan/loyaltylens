"""FastAPI for offer retrieval: POST /retrieve and GET /offers/stats."""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_retrieval.langchain_retriever import LangChainOfferRetriever, OfferResult
from shared.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="LoyaltyLens RAG Retrieval", version="1.0.0")

_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_VECTOR_DB = "pgvector + weaviate"


# ------------------------------------------------------------------
# Request / response models
# ------------------------------------------------------------------

class RetrieveRequest(BaseModel):
    customer_id: str
    propensity_score: float = Field(ge=0.0, le=1.0)
    context_text: str


class OfferOut(BaseModel):
    offer_id: str
    title: str
    description: str
    category: str
    score: float


class RetrieveResponse(BaseModel):
    offers: list[OfferOut]
    retriever_used: str
    latency_ms: float


class StatsResponse(BaseModel):
    total_indexed: int
    embedding_model: str
    vector_db: str


# ------------------------------------------------------------------
# Dependency
# ------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_retriever() -> LangChainOfferRetriever:
    return LangChainOfferRetriever()


def _count_indexed() -> int:
    try:
        import psycopg2
        from shared.config import get_settings

        conn = psycopg2.connect(get_settings().postgres_url)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM offer_embeddings;")
            count: int = cur.fetchone()[0]
        conn.close()
        return count
    except Exception:
        return -1


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    try:
        retriever = _get_retriever()
    except Exception as exc:
        logger.error("retriever_init_failed", error=str(exc))
        raise HTTPException(status_code=503, detail="Retriever unavailable") from exc

    t0 = time.perf_counter()
    results: list[OfferResult] = retriever.retrieve(
        req.context_text,
        propensity=req.propensity_score,
        k=5,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "retrieve_request",
        customer_id=req.customer_id,
        propensity=req.propensity_score,
        returned=len(results),
        latency_ms=round(latency_ms, 2),
    )

    return RetrieveResponse(
        offers=[
            OfferOut(
                offer_id=r.offer_id,
                title=r.title,
                description=r.description,
                category=r.category,
                score=r.score,
            )
            for r in results
        ],
        retriever_used="langchain-pgvector",
        latency_ms=round(latency_ms, 2),
    )


@app.get("/offers/stats", response_model=StatsResponse)
def offers_stats() -> StatsResponse:
    return StatsResponse(
        total_indexed=_count_indexed(),
        embedding_model=_EMBEDDING_MODEL,
        vector_db=_VECTOR_DB,
    )
