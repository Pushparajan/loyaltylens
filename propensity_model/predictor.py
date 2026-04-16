"""Inference interface for the TabTransformer propensity model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from propensity_model.model import PropensityModel, TabTransformerConfig
from shared.logger import get_logger

logger = get_logger(__name__)

_NUMERICAL_COLS = [
    "recency_days",
    "frequency_30d",
    "monetary_90d",
    "offer_redemption_rate",
    "channel_preference",
    "engagement_score",
]
_CHANNEL_MAP = {"in-store": 0, "mobile": 1, "web": 2}


@dataclass(frozen=True)
class PropensityResult:
    customer_id: str
    propensity_score: float  # raw sigmoid output in [0, 1]
    label: int               # 1 if score >= threshold, else 0
    threshold: float
    model_version: str


def _encode_channel(series: pd.Series) -> pd.Series:
    return series.str.lower().map(_CHANNEL_MAP).fillna(_CHANNEL_MAP["web"]).astype(float)


class PropensityPredictor:
    """Load a saved TabTransformer checkpoint and run single / batch inference."""

    def __init__(self) -> None:
        self._model: PropensityModel | None = None
        self._version: str = ""
        self._val_auc: float = 0.0
        self._threshold: float = 0.5

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, version: str, models_dir: Path | str = "models") -> "PropensityPredictor":
        """Load model weights and metadata for *version*."""
        base = Path(models_dir)
        model_path = base / f"propensity_v{version}.pt"
        meta_path = base / f"propensity_v{version}_meta.json"

        model = PropensityModel()
        model.load(model_path)
        self._model = model
        self._version = version

        if meta_path.exists():
            meta: dict[str, Any] = json.loads(meta_path.read_text())
            self._val_auc = float(meta.get("val_auc", 0.0))
            self._threshold = float(meta.get("threshold", 0.5))

        logger.info("predictor_loaded", version=version, val_auc=self._val_auc)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, feature_dict: dict[str, Any]) -> PropensityResult:
        """Score a single customer from a feature dictionary."""
        if self._model is None:
            raise RuntimeError("No model loaded. Call load() first.")

        row = pd.DataFrame([feature_dict])
        return self.predict_batch(row)[0]

    def predict_batch(self, df: pd.DataFrame) -> list[PropensityResult]:
        """Score a DataFrame of customers; returns one PropensityResult per row."""
        if self._model is None:
            raise RuntimeError("No model loaded. Call load() first.")

        df = df.copy()
        if "channel_preference" in df.columns:
            df["channel_preference"] = _encode_channel(df["channel_preference"])

        X = df[_NUMERICAL_COLS].to_numpy(dtype=np.float32)
        scores: np.ndarray = self._model.predict_proba(X)

        customer_ids: list[str] = (
            df["customer_id"].astype(str).tolist()
            if "customer_id" in df.columns
            else [str(i) for i in range(len(df))]
        )

        results = [
            PropensityResult(
                customer_id=cid,
                propensity_score=float(score),
                label=int(score >= self._threshold),
                threshold=self._threshold,
                model_version=self._version,
            )
            for cid, score in zip(customer_ids, scores)
        ]
        logger.info("batch_scored", n=len(results), version=self._version)
        return results

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def version(self) -> str:
        return self._version

    @property
    def val_auc(self) -> float:
        return self._val_auc

    @property
    def config(self) -> TabTransformerConfig | None:
        return self._model.config if self._model else None
