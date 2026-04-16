"""FastAPI service wrapping PropensityPredictor."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from propensity_model.predictor import PropensityPredictor, PropensityResult
from shared.config import get_settings
from shared.logger import get_logger

logger = get_logger(__name__)

_predictor: PropensityPredictor | None = None


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _predictor
    settings = get_settings()
    _predictor = PropensityPredictor()
    try:
        _predictor.load(
            version=settings.propensity_model_version,
            models_dir=settings.propensity_models_dir,
        )
        logger.info("api_model_loaded", version=settings.propensity_model_version)
    except FileNotFoundError:
        logger.warning(
            "api_model_not_found",
            version=settings.propensity_model_version,
            models_dir=settings.propensity_models_dir,
        )
    yield


app = FastAPI(title="LoyaltyLens Propensity API", lifespan=_lifespan)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class FeaturePayload(BaseModel):
    customer_id: str = ""
    recency_days: float
    frequency_30d: float
    monetary_90d: float
    offer_redemption_rate: float
    channel_preference: str = "web"
    engagement_score: float


class PropensityResponse(BaseModel):
    customer_id: str
    propensity_score: float
    label: int
    threshold: float
    model_version: str

    @classmethod
    def from_result(cls, r: PropensityResult) -> "PropensityResponse":
        return cls(
            customer_id=r.customer_id,
            propensity_score=r.propensity_score,
            label=r.label,
            threshold=r.threshold,
            model_version=r.model_version,
        )


class ModelInfoResponse(BaseModel):
    version: str
    val_auc: float
    loaded: bool
    config: dict[str, Any] | None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _require_predictor() -> PropensityPredictor:
    if _predictor is None or not _predictor.version:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _predictor


@app.post("/predict", response_model=PropensityResponse)
def predict(payload: FeaturePayload) -> PropensityResponse:
    """Score a single customer feature vector."""
    predictor = _require_predictor()
    result = predictor.predict(payload.model_dump())
    logger.info("predict_request", customer_id=payload.customer_id, score=result.propensity_score)
    return PropensityResponse.from_result(result)


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Return current loaded model version and validation AUC."""
    loaded = _predictor is not None and bool(_predictor.version)
    cfg = _predictor.config if loaded and _predictor else None
    cfg_dict: dict[str, Any] | None = None
    if cfg is not None:
        from dataclasses import asdict
        cfg_dict = asdict(cfg)
    return ModelInfoResponse(
        version=_predictor.version if loaded and _predictor else "",
        val_auc=_predictor.val_auc if loaded and _predictor else 0.0,
        loaded=loaded,
        config=cfg_dict,
    )
