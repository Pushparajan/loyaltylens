"""Lightweight training wrapper: features → PropensityModel → MLflow run."""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from propensity_model.model import PropensityModel, TabTransformerConfig
from shared.config import get_settings
from shared.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Orchestrates feature assembly, training, and MLflow experiment tracking."""

    def __init__(self, config: TabTransformerConfig | None = None) -> None:
        self._config = config or TabTransformerConfig()
        settings = get_settings()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

    def train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        run_name: str = "propensity-train",
    ) -> PropensityModel:
        X = features.to_numpy(dtype=np.float32)
        y = labels.to_numpy(dtype=np.float32)

        with mlflow.start_run(run_name=run_name):
            from dataclasses import asdict
            mlflow.log_params(asdict(self._config))

            model = PropensityModel(self._config)
            model.fit(X, y)

            scores = model.predict_proba(X)
            mlflow.log_metric("train_mean_score", float(np.mean(scores)))

            with tempfile.TemporaryDirectory() as tmp:
                artifact_path = Path(tmp) / "propensity.pt"
                model.save(artifact_path)
                mlflow.log_artifact(str(artifact_path))

            logger.info("training_run_complete", run_name=run_name)

        return model
