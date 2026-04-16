"""FastAPI service exposing feature vectors from the DuckDB feature store.

Run with:
    uvicorn feature_store.api:app --reload --port 8001
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from feature_store.store import FeatureStore, FeatureValidationError
from shared.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="LoyaltyLens Feature Store API",
    description="Serve per-customer feature vectors from the DuckDB feature store.",
    version="0.1.0",
)

# Single shared store instance reused across requests.
_store: FeatureStore | None = None


def _get_store() -> FeatureStore:
    global _store
    if _store is None:
        _store = FeatureStore()
    return _store


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class FeatureVector(BaseModel):
    customer_id: str
    version: str
    recency_days: int | None
    frequency_30d: int | None
    monetary_90d: float | None
    offer_redemption_rate: float | None
    channel_preference: str | None
    engagement_score: float | None


class ColumnStats(BaseModel):
    column_name: str
    min: float | None
    max: float | None
    mean: float | None
    std: float | None
    count: int | None


class StatsResponse(BaseModel):
    version: str
    stats: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/features/stats",
    response_model=StatsResponse,
    summary="Summary statistics for the latest feature version",
    description=(
        "Run a DuckDB ``SUMMARIZE`` over numeric feature columns for the latest version "
        "and return the result as a list of per-column stat records."
    ),
)
def get_feature_stats() -> StatsResponse:
    store = _get_store()
    versions = store.list_versions()
    if not versions:
        raise HTTPException(status_code=404, detail="No feature versions found in the store")
    latest = versions[0]
    stats_df = store.get_feature_stats(latest)
    logger.info("stats_served", version=latest, columns=len(stats_df))
    return StatsResponse(version=latest, stats=stats_df.to_dict(orient="records"))


@app.get(
    "/features/{customer_id}",
    response_model=FeatureVector,
    summary="Get feature vector for a customer",
    description=(
        "Return the most recently written feature row for the given ``customer_id``. "
        "Raises **404** if the customer is not found in any feature version."
    ),
)
def get_customer_features(customer_id: str) -> FeatureVector:
    store = _get_store()
    row = store.read_latest(customer_id)
    if row is None:
        logger.info("customer_not_found", customer_id=customer_id)
        raise HTTPException(status_code=404, detail=f"No features found for customer '{customer_id}'")
    logger.info("features_served", customer_id=customer_id, version=row.get("version"))
    return FeatureVector(**row)
