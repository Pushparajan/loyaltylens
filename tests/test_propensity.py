"""Tests for TabTransformer model, predictor, and API endpoints."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from propensity_model.api import app
from propensity_model.model import PropensityModel, TabTransformerConfig, TabTransformerNet
from propensity_model.predictor import PropensityPredictor, PropensityResult

N_FEATURES = 6
BATCH = 32


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cfg() -> TabTransformerConfig:
    return TabTransformerConfig(epochs=2, batch_size=16, early_stopping_patience=1)


@pytest.fixture()
def net(cfg: TabTransformerConfig) -> TabTransformerNet:
    return TabTransformerNet(cfg)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def sample_X(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((BATCH, N_FEATURES)).astype(np.float32)


@pytest.fixture()
def sample_y(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, BATCH).astype(np.float32)


@pytest.fixture()
def fitted_model(cfg: TabTransformerConfig, sample_X: np.ndarray, sample_y: np.ndarray) -> PropensityModel:
    model = PropensityModel(cfg)
    model.fit(sample_X, sample_y)
    return model


@pytest.fixture()
def saved_model_dir(fitted_model: PropensityModel, tmp_path: Path) -> Path:
    """Save a fitted model + metadata to a temp dir and return the dir."""
    model_path = tmp_path / "propensity_v1.pt"
    fitted_model.save(model_path)
    meta: dict[str, Any] = {
        "version": "1",
        "trained_at": "2026-01-01T00:00:00+00:00",
        "val_auc": 0.81,
        "feature_names": [
            "recency_days", "frequency_30d", "monetary_90d",
            "offer_redemption_rate", "channel_preference", "engagement_score",
        ],
        "threshold": 0.5,
    }
    (tmp_path / "propensity_v1_meta.json").write_text(json.dumps(meta))
    return tmp_path


# ---------------------------------------------------------------------------
# TabTransformerNet — forward pass
# ---------------------------------------------------------------------------


class TestTabTransformerNet:
    def test_output_shape(self, net: TabTransformerNet, sample_X: np.ndarray) -> None:
        x = torch.tensor(sample_X)
        out = net(x)
        assert out.shape == (BATCH,), f"Expected ({BATCH},), got {out.shape}"

    def test_output_bounds(self, net: TabTransformerNet, sample_X: np.ndarray) -> None:
        x = torch.tensor(sample_X)
        with torch.no_grad():
            out = net(x).numpy()
        assert np.all(out >= 0.0) and np.all(out <= 1.0), "Scores outside [0, 1]"

    def test_single_sample(self, net: TabTransformerNet) -> None:
        x = torch.zeros(1, N_FEATURES)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (1,)
        assert 0.0 <= out.item() <= 1.0

    def test_deterministic_eval(self, net: TabTransformerNet, sample_X: np.ndarray) -> None:
        net.eval()
        x = torch.tensor(sample_X)
        with torch.no_grad():
            out1 = net(x).numpy()
            out2 = net(x).numpy()
        np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# PropensityModel — wrapper
# ---------------------------------------------------------------------------


class TestPropensityModel:
    def test_fit_returns_self(self, sample_X: np.ndarray, sample_y: np.ndarray) -> None:
        model = PropensityModel(TabTransformerConfig(epochs=1))
        result = model.fit(sample_X, sample_y)
        assert result is model

    def test_predict_proba_shape(self, fitted_model: PropensityModel, sample_X: np.ndarray) -> None:
        scores = fitted_model.predict_proba(sample_X)
        assert scores.shape == (BATCH,)

    def test_predict_proba_bounds(self, fitted_model: PropensityModel, sample_X: np.ndarray) -> None:
        scores = fitted_model.predict_proba(sample_X)
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)

    def test_unfitted_raises(self) -> None:
        model = PropensityModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(np.zeros((5, N_FEATURES), dtype=np.float32))

    def test_save_load_roundtrip(
        self, fitted_model: PropensityModel, sample_X: np.ndarray, tmp_path: Path
    ) -> None:
        path = tmp_path / "model.pt"
        fitted_model.save(path)

        loaded = PropensityModel()
        loaded.load(path)

        orig_scores = fitted_model.predict_proba(sample_X)
        loaded_scores = loaded.predict_proba(sample_X)
        np.testing.assert_allclose(orig_scores, loaded_scores, rtol=1e-5)

    def test_batch_consistency(self, fitted_model: PropensityModel, sample_X: np.ndarray) -> None:
        """Batch and individual predictions must agree."""
        batch_scores = fitted_model.predict_proba(sample_X)
        single_scores = np.array([
            fitted_model.predict_proba(sample_X[i : i + 1])[0]
            for i in range(len(sample_X))
        ])
        np.testing.assert_allclose(batch_scores, single_scores, rtol=1e-5)


# ---------------------------------------------------------------------------
# PropensityPredictor
# ---------------------------------------------------------------------------


class TestPropensityPredictor:
    def test_load_and_predict(self, saved_model_dir: Path) -> None:
        predictor = PropensityPredictor()
        predictor.load(version="1", models_dir=saved_model_dir)
        result = predictor.predict({
            "customer_id": "cust_001",
            "recency_days": 10.0,
            "frequency_30d": 5.0,
            "monetary_90d": 100.0,
            "offer_redemption_rate": 0.4,
            "channel_preference": "mobile",
            "engagement_score": 0.6,
        })
        assert isinstance(result, PropensityResult)
        assert 0.0 <= result.propensity_score <= 1.0
        assert result.label in (0, 1)
        assert result.model_version == "1"

    def test_predict_batch_returns_one_per_row(self, saved_model_dir: Path, rng: np.random.Generator) -> None:
        import pandas as pd
        predictor = PropensityPredictor().load(version="1", models_dir=saved_model_dir)
        df = pd.DataFrame({
            "customer_id": [f"c{i}" for i in range(10)],
            "recency_days": rng.integers(1, 30, 10).astype(float),
            "frequency_30d": rng.integers(0, 20, 10).astype(float),
            "monetary_90d": rng.uniform(0, 500, 10),
            "offer_redemption_rate": rng.uniform(0, 1, 10),
            "channel_preference": ["mobile"] * 10,
            "engagement_score": rng.uniform(0, 1, 10),
        })
        results = predictor.predict_batch(df)
        assert len(results) == 10
        for r in results:
            assert 0.0 <= r.propensity_score <= 1.0

    def test_unloaded_predict_raises(self) -> None:
        predictor = PropensityPredictor()
        with pytest.raises(RuntimeError, match="No model loaded"):
            predictor.predict({"recency_days": 1.0})

    def test_metadata_loaded(self, saved_model_dir: Path) -> None:
        predictor = PropensityPredictor().load(version="1", models_dir=saved_model_dir)
        assert predictor.val_auc == pytest.approx(0.81)
        assert predictor.version == "1"


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@pytest.fixture()
def api_client(saved_model_dir: Path) -> TestClient:
    """TestClient with the predictor pre-loaded via monkeypatching."""
    from propensity_model import api as api_module

    predictor = PropensityPredictor().load(version="1", models_dir=saved_model_dir)
    with patch.object(api_module, "_predictor", predictor):
        yield TestClient(app)


class TestAPI:
    _FEATURE_PAYLOAD = {
        "customer_id": "cust_test",
        "recency_days": 7.0,
        "frequency_30d": 8.0,
        "monetary_90d": 150.0,
        "offer_redemption_rate": 0.35,
        "channel_preference": "web",
        "engagement_score": 0.55,
    }

    def test_predict_status_200(self, api_client: TestClient) -> None:
        resp = api_client.post("/predict", json=self._FEATURE_PAYLOAD)
        assert resp.status_code == 200

    def test_predict_response_schema(self, api_client: TestClient) -> None:
        resp = api_client.post("/predict", json=self._FEATURE_PAYLOAD)
        body = resp.json()
        assert "propensity_score" in body
        assert "label" in body
        assert "threshold" in body
        assert "model_version" in body
        assert 0.0 <= body["propensity_score"] <= 1.0
        assert body["label"] in (0, 1)

    def test_model_info_schema(self, api_client: TestClient) -> None:
        resp = api_client.get("/model/info")
        assert resp.status_code == 200
        body = resp.json()
        assert "version" in body
        assert "val_auc" in body
        assert "loaded" in body
        assert body["loaded"] is True
        assert body["version"] == "1"

    def test_predict_missing_field_422(self, api_client: TestClient) -> None:
        resp = api_client.post("/predict", json={"customer_id": "x"})
        assert resp.status_code == 422
