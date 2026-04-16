"""Offline evaluation metrics for the propensity model."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from propensity_model.model import PropensityModel
from shared.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Compute standard binary-classification metrics for a fitted PropensityModel."""

    def evaluate(
        self,
        model: PropensityModel,
        X: np.ndarray,
        y_true: np.ndarray,
    ) -> dict[str, float]:
        y_prob = model.predict_proba(X)
        metrics = {
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "avg_precision": float(average_precision_score(y_true, y_prob)),
            "mean_score": float(np.mean(y_prob)),
        }
        logger.info("model_evaluated", **metrics)
        return metrics

    def passes_threshold(
        self,
        model: PropensityModel,
        X: np.ndarray,
        y_true: np.ndarray,
        threshold: float = 0.75,
    ) -> bool:
        metrics = self.evaluate(model, X, y_true)
        passed = metrics["roc_auc"] >= threshold
        logger.info("eval_gate", roc_auc=metrics["roc_auc"], threshold=threshold, passed=passed)
        return passed
