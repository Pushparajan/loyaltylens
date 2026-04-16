"""Emit retraining signals from aggregated feedback to the propensity model."""

from __future__ import annotations

from typing import Any

import pandas as pd

from feedback_loop.processor import FeedbackProcessor
from shared.logger import get_logger

logger = get_logger(__name__)

_CHURN_THRESHOLD = 0.5


class ModelUpdater:
    """Convert feedback signals into pseudo-labels and trigger model retraining."""

    def __init__(self, processor: FeedbackProcessor | None = None) -> None:
        self._processor = processor or FeedbackProcessor()

    def build_training_signal(self, since_iso: str | None = None) -> pd.DataFrame:
        """
        Return a DataFrame with columns [customer_id, label] where label=1 means
        the customer showed churn-like behaviour (high negative, low positive).
        """
        aggregates = self._processor.aggregate(since_iso=since_iso)
        records: list[dict[str, Any]] = []
        for agg in aggregates:
            total = agg["positive"] + agg["negative"]
            churn_prob = agg["negative"] / total if total > 0 else 0.5
            records.append(
                {
                    "customer_id": agg["customer_id"],
                    "positive": agg["positive"],
                    "negative": agg["negative"],
                    "churn_prob": churn_prob,
                    "label": int(churn_prob >= _CHURN_THRESHOLD),
                }
            )
        df = pd.DataFrame(records)
        logger.info("training_signal_built", n_rows=len(df))
        return df

    def should_retrain(self, since_iso: str | None = None, min_samples: int = 100) -> bool:
        df = self.build_training_signal(since_iso)
        return len(df) >= min_samples
