"""Training pipeline: pull features, train PropensityModel, log to MLflow."""

from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd

from propensity_model.model import PropensityModel
from shared.config import get_settings
from shared.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Orchestrates feature assembly, training, and MLflow experiment tracking."""

    def __init__(self) -> None:
        settings = get_settings()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

    def train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        run_name: str = "propensity-train",
    ) -> PropensityModel:
        X = features.to_numpy()
        y = labels.to_numpy()

        with mlflow.start_run(run_name=run_name):
            model = PropensityModel()
            model.fit(X, y)
            mlflow.log_params(model._params)
            scores = model.predict_proba(X)
            mlflow.log_metric("train_mean_score", float(np.mean(scores)))
            mlflow.xgboost.log_model(model._clf, artifact_path="model")
            logger.info("training_run_complete", run_name=run_name)

        return model
