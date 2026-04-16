"""TabTransformer-lite propensity model in PyTorch with sklearn-compatible interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from shared.logger import get_logger

logger = get_logger(__name__)

FEATURE_NAMES: list[str] = [
    "recency_days",
    "frequency_30d",
    "monetary_90d",
    "offer_redemption_rate",
    "channel_preference",
    "engagement_score",
]


@dataclass
class TabTransformerConfig:
    """All hyperparameters in one place — no magic numbers elsewhere."""

    # Architecture
    n_features: int = 6
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    mlp_hidden: int = 64

    # Training
    lr: float = 1e-3
    epochs: int = 20
    batch_size: int = 64
    early_stopping_patience: int = 3

    # Data
    val_split: float = 0.15
    test_split: float = 0.15
    label_threshold: float = 0.3  # offer_redemption_rate > this → positive class

    # Inference
    decision_threshold: float = 0.5


class TabTransformerNet(nn.Module):
    """
    TabTransformer-lite: per-feature linear embedding → TransformerEncoder → MLP head.

    Each of the n_features scalar inputs is independently projected to d_model via its
    own Linear layer, forming a sequence of length n_features for the Transformer.
    """

    def __init__(self, cfg: TabTransformerConfig) -> None:
        super().__init__()
        self.feature_embeddings = nn.ModuleList(
            [nn.Linear(1, cfg.d_model) for _ in range(cfg.n_features)]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.num_encoder_layers
        )
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_features * cfg.d_model, cfg.mlp_hidden),
            nn.ReLU(),
            nn.Linear(cfg.mlp_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_features) float tensor

        Returns:
            (batch_size,) propensity scores in [0, 1]
        """
        embeds = [emb(x[:, i : i + 1]) for i, emb in enumerate(self.feature_embeddings)]
        seq = torch.stack(embeds, dim=1)  # (batch, n_features, d_model)
        seq = self.transformer(seq)
        out = seq.flatten(1)  # (batch, n_features * d_model)
        return self.mlp(out).squeeze(-1)  # (batch,)


class PropensityModel:
    """Sklearn-style wrapper around TabTransformerNet for compatibility with evaluator."""

    def __init__(self, config: TabTransformerConfig | None = None) -> None:
        self.config = config or TabTransformerConfig()
        self.net = TabTransformerNet(self.config)
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PropensityModel":
        """Basic training loop (no MLflow / early stopping). Use train.py for full pipeline."""
        self.net.train()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr, weight_decay=1e-4)
        criterion = nn.BCELoss()
        x_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        for _ in range(self.config.epochs):
            optimizer.zero_grad()
            loss = criterion(self.net(x_t), y_t)
            loss.backward()
            optimizer.step()
        self._is_fitted = True
        logger.info("model_fitted", n_samples=len(X), epochs=self.config.epochs)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return propensity scores in [0, 1] with shape (n_samples,)."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        self.net.eval()
        with torch.no_grad():
            probs = self.net(torch.tensor(X, dtype=torch.float32))
        return probs.numpy()

    def save(self, path: Path | str) -> None:
        torch.save({"state_dict": self.net.state_dict(), "config": self.config}, str(path))
        logger.info("model_saved", path=str(path))

    def load(self, path: Path | str) -> "PropensityModel":
        checkpoint: dict[str, Any] = torch.load(str(path), map_location="cpu", weights_only=False)
        self.config = checkpoint["config"]
        self.net = TabTransformerNet(self.config)
        self.net.load_state_dict(checkpoint["state_dict"])
        self._is_fitted = True
        logger.info("model_loaded", path=str(path))
        return self
