"""XGBoost-based propensity model wrapper with MLflow artifact support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from shared.logger import get_logger

logger = get_logger(__name__)


class PropensityModel:
    """Thin wrapper around an XGBoost classifier for churn / upsell scoring."""

    DEFAULT_PARAMS: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "n_estimators": 300,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._params = {**self.DEFAULT_PARAMS, **(params or {})}
        self._clf = xgb.XGBClassifier(**self._params)
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PropensityModel":
        self._clf.fit(X, y)
        self._is_fitted = True
        logger.info("model_fitted", n_samples=len(X))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return self._clf.predict_proba(X)[:, 1]

    def save(self, path: Path | str) -> None:
        self._clf.save_model(str(path))
        logger.info("model_saved", path=str(path))

    def load(self, path: Path | str) -> "PropensityModel":
        self._clf.load_model(str(path))
        self._is_fitted = True
        logger.info("model_loaded", path=str(path))
        return self
