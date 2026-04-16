"""Unit tests for propensity_model.model and evaluator."""

from __future__ import annotations

import numpy as np
import pytest

from propensity_model.evaluator import ModelEvaluator
from propensity_model.model import PropensityModel


@pytest.fixture()
def trained_model() -> PropensityModel:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 5))
    y = (X[:, 0] > 0).astype(int)
    model = PropensityModel()
    model.fit(X, y)
    return model


@pytest.fixture()
def eval_data(trained_model: PropensityModel) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    X = rng.standard_normal((50, 5))
    y = (X[:, 0] > 0).astype(int)
    return X, y


class TestPropensityModel:
    def test_fit_returns_self(self) -> None:
        rng = np.random.default_rng(0)
        X, y = rng.standard_normal((50, 5)), rng.integers(0, 2, 50)
        m = PropensityModel()
        assert m.fit(X, y) is m

    def test_predict_proba_shape(self, trained_model: PropensityModel, eval_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, _ = eval_data
        probs = trained_model.predict_proba(X)
        assert probs.shape == (50,)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_unfitted_raises(self) -> None:
        m = PropensityModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            m.predict_proba(np.zeros((5, 3)))


class TestModelEvaluator:
    def test_evaluate_returns_metrics(
        self, trained_model: PropensityModel, eval_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = eval_data
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(trained_model, X, y)
        assert "roc_auc" in metrics
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_passes_threshold(
        self, trained_model: PropensityModel, eval_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = eval_data
        evaluator = ModelEvaluator()
        # A model trained on a separable dataset should easily exceed 0.5
        assert evaluator.passes_threshold(trained_model, X, y, threshold=0.5)
