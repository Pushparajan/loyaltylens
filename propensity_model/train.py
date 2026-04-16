"""Full training pipeline: feature store → TabTransformer → MLflow → saved artifact."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from feature_store.store import FeatureStore
from propensity_model.model import (
    FEATURE_NAMES,
    PropensityModel,
    TabTransformerConfig,
    TabTransformerNet,
)
from shared.config import get_settings
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
_CHANNEL_CATEGORIES = ["in-store", "mobile", "web"]


def _load_all_features() -> pd.DataFrame:
    store = FeatureStore()
    versions = store.list_versions()
    if not versions:
        raise ValueError("No feature versions found in feature store")
    df: pd.DataFrame = store._conn.execute(
        "SELECT * FROM features WHERE version = ?", [versions[0]]
    ).fetchdf()
    store.close()
    logger.info("features_loaded", rows=len(df), version=versions[0])
    return df


def _encode_features(df: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()
    le.fit(_CHANNEL_CATEGORIES)
    out = df.copy()
    out["channel_preference"] = le.transform(
        out["channel_preference"].fillna("web").str.lower()
    )
    return out


def _make_tensors(
    X: np.ndarray, y: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )


def _eval_split(
    net: TabTransformerNet,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    criterion: nn.BCELoss,
    threshold: float,
) -> dict[str, float]:
    net.eval()
    all_probs: list[float] = []
    all_true: list[int] = []
    losses: list[float] = []
    with torch.no_grad():
        for xb, yb in loader:
            preds = net(xb)
            losses.append(criterion(preds, yb).item())
            all_probs.extend(preds.numpy().tolist())
            all_true.extend(yb.numpy().astype(int).tolist())
    probs_arr = np.array(all_probs)
    true_arr = np.array(all_true)
    preds_bin = (probs_arr >= threshold).astype(int)
    return {
        "loss": float(np.mean(losses)),
        "auc": float(roc_auc_score(true_arr, probs_arr)),
        "precision": float(precision_score(true_arr, preds_bin, zero_division=0)),
        "recall": float(recall_score(true_arr, preds_bin, zero_division=0)),
    }


def train(
    version: str = "1",
    cfg: TabTransformerConfig | None = None,
) -> tuple[PropensityModel, dict[str, object]]:
    """
    Load features, train TabTransformer with early stopping, log to MLflow.

    Returns the best-checkpoint PropensityModel and the metadata dict written
    to models/propensity_v{version}_meta.json.
    """
    cfg = cfg or TabTransformerConfig()
    settings = get_settings()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    df = _load_all_features()
    df = _encode_features(df)
    X = df[_NUMERICAL_COLS].to_numpy(dtype=np.float32)
    y = (df["offer_redemption_rate"] > cfg.label_threshold).astype(int).to_numpy()

    # Stratified 70/15/15 split
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=cfg.test_split, stratify=y, random_state=42
    )
    val_ratio = cfg.val_split / (1.0 - cfg.test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_ratio, stratify=y_tmp, random_state=42
    )

    train_loader: DataLoader[tuple[torch.Tensor, ...]] = DataLoader(
        TensorDataset(*_make_tensors(X_train, y_train)),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_loader: DataLoader[tuple[torch.Tensor, ...]] = DataLoader(
        TensorDataset(*_make_tensors(X_val, y_val)),
        batch_size=cfg.batch_size,
    )

    net = TabTransformerNet(cfg)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    best_val_auc = 0.0
    best_state: dict[str, torch.Tensor] = {}
    patience_counter = 0

    with mlflow.start_run(run_name=f"propensity-tabtransformer-v{version}"):
        mlflow.log_params(
            {
                "d_model": cfg.d_model,
                "nhead": cfg.nhead,
                "num_encoder_layers": cfg.num_encoder_layers,
                "dim_feedforward": cfg.dim_feedforward,
                "dropout": cfg.dropout,
                "lr": cfg.lr,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "early_stopping_patience": cfg.early_stopping_patience,
                "label_threshold": cfg.label_threshold,
            }
        )

        for epoch in range(cfg.epochs):
            net.train()
            train_losses: list[float] = []
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = criterion(net(xb), yb)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            val_metrics = _eval_split(net, val_loader, criterion, cfg.decision_threshold)

            mlflow.log_metrics(
                {
                    "train_loss": float(np.mean(train_losses)),
                    "val_loss": val_metrics["loss"],
                    "val_auc": val_metrics["auc"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                },
                step=epoch,
            )
            logger.info(
                "epoch_complete",
                epoch=epoch,
                train_loss=round(float(np.mean(train_losses)), 4),
                val_auc=round(val_metrics["auc"], 4),
            )

            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stopping_patience:
                    logger.info(
                        "early_stopping", epoch=epoch, best_val_auc=round(best_val_auc, 4)
                    )
                    break

        net.load_state_dict(best_state)

        # Persist model
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / f"propensity_v{version}.pt"

        model = PropensityModel(cfg)
        model.net = net
        model._is_fitted = True
        model.save(model_path)
        mlflow.log_artifact(str(model_path))

        # Test-set AUC
        net.eval()
        with torch.no_grad():
            test_probs = net(torch.tensor(X_test, dtype=torch.float32)).numpy()
        test_auc = float(roc_auc_score(y_test, test_probs))
        mlflow.log_metric("test_auc", test_auc)

        # Metadata
        meta: dict[str, object] = {
            "version": version,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "val_auc": round(best_val_auc, 4),
            "feature_names": FEATURE_NAMES,
            "threshold": cfg.decision_threshold,
        }
        meta_path = models_dir / f"propensity_v{version}_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        mlflow.log_artifact(str(meta_path))

        logger.info(
            "training_complete",
            version=version,
            val_auc=round(best_val_auc, 4),
            test_auc=round(test_auc, 4),
        )

    return model, meta


if __name__ == "__main__":
    train()
