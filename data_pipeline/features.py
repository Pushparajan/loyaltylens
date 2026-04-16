"""Compute per-customer RFM features from raw events and write to Parquet.

Features produced
-----------------
recency_days          Days since the customer's most recent event.
frequency_30d         Total event count in the last 30 days.
monetary_90d          Sum of purchase amounts in the last 90 days.
offer_redemption_rate redeems / offer_views  (0.0 when no offer_views).
channel_preference    Modal channel across all events.
engagement_score      Weighted composite normalised to [0, 1].

Usage:
    python -m data_pipeline.features
    python data_pipeline/features.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from shared.config import get_settings
from shared.logger import get_logger

logger = get_logger(__name__)

# Weights used to compute engagement_score (must sum to 1.0)
_ENGAGEMENT_WEIGHTS = {
    "frequency_30d": 0.30,
    "monetary_90d": 0.40,
    "offer_redemption_rate": 0.30,
}


def _safe_mode(series: pd.Series) -> str:
    """Return the mode of *series*, falling back to the first value."""
    modes = series.mode()
    return str(modes.iloc[0]) if len(modes) > 0 else str(series.iloc[0])


def _minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def compute_features(events: pd.DataFrame, reference_ts: pd.Timestamp | None = None) -> pd.DataFrame:
    """Derive per-customer features from *events*.

    Parameters
    ----------
    events:
        Raw events DataFrame (output of ``generate_events``).
    reference_ts:
        Timestamp treated as "now". Defaults to ``pd.Timestamp.utcnow()``.
    """
    now = reference_ts or pd.Timestamp.utcnow().tz_localize(None)

    # Normalise timestamps to tz-naive for arithmetic
    ts = events["timestamp"]
    if ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)
    events = events.copy()
    events["_ts"] = ts

    cutoff_30d = now - pd.Timedelta(days=30)
    cutoff_90d = now - pd.Timedelta(days=90)

    # ── recency_days ──────────────────────────────────────────────────────
    recency = (
        events.groupby("customer_id")["_ts"]
        .max()
        .apply(lambda t: max((now - t).days, 0))
        .rename("recency_days")
    )

    # ── frequency_30d ─────────────────────────────────────────────────────
    freq_30d = (
        events[events["_ts"] >= cutoff_30d]
        .groupby("customer_id")
        .size()
        .rename("frequency_30d")
    )

    # ── monetary_90d ──────────────────────────────────────────────────────
    purchases_90d = events[(events["_ts"] >= cutoff_90d) & (events["event_type"] == "purchase")]
    monetary_90d = (
        purchases_90d.groupby("customer_id")["amount"]
        .sum()
        .rename("monetary_90d")
    )

    # ── offer_redemption_rate ─────────────────────────────────────────────
    offer_views = (
        events[events["event_type"] == "offer_view"]
        .groupby("customer_id")
        .size()
        .rename("offer_views")
    )
    redeems = (
        events[events["event_type"] == "redeem"]
        .groupby("customer_id")
        .size()
        .rename("redeems")
    )
    offer_df = pd.concat([offer_views, redeems], axis=1).fillna(0)
    redemption_rate = (
        (offer_df["redeems"] / offer_df["offer_views"].replace(0, np.nan))
        .fillna(0.0)
        .clip(0.0, 1.0)
        .rename("offer_redemption_rate")
    )

    # ── channel_preference ────────────────────────────────────────────────
    channel_pref = (
        events.groupby("customer_id")["channel"]
        .agg(_safe_mode)
        .rename("channel_preference")
    )

    # ── assemble base frame ───────────────────────────────────────────────
    all_customers = events["customer_id"].unique()
    features = pd.DataFrame(index=pd.Index(all_customers, name="customer_id"))
    features = features.join(recency).join(freq_30d).join(monetary_90d).join(
        redemption_rate
    ).join(channel_pref)
    features["recency_days"] = features["recency_days"].fillna(0).astype(int)
    features["frequency_30d"] = features["frequency_30d"].fillna(0).astype(int)
    features["monetary_90d"] = features["monetary_90d"].fillna(0.0)
    features["offer_redemption_rate"] = features["offer_redemption_rate"].fillna(0.0)
    features = features.reset_index()

    # ── engagement_score ──────────────────────────────────────────────────
    # Invert recency so that more-recent == higher score
    recency_inv = features["recency_days"].max() - features["recency_days"]
    normed = pd.DataFrame(
        {
            "frequency_30d": _minmax(features["frequency_30d"].astype(float)),
            "monetary_90d": _minmax(features["monetary_90d"]),
            "offer_redemption_rate": _minmax(features["offer_redemption_rate"]),
            "recency_inv": _minmax(recency_inv.astype(float)),
        }
    )
    # Blend recency_inv into the frequency weight
    features["engagement_score"] = (
        normed["frequency_30d"] * _ENGAGEMENT_WEIGHTS["frequency_30d"]
        + normed["monetary_90d"] * _ENGAGEMENT_WEIGHTS["monetary_90d"]
        + normed["offer_redemption_rate"] * _ENGAGEMENT_WEIGHTS["offer_redemption_rate"]
    ).clip(0.0, 1.0).round(6)

    return features


def main() -> Path:
    settings = get_settings()
    src = Path(settings.raw_events_path)
    dst = Path(settings.processed_features_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    logger.info("reading_events", path=str(src))
    events = pd.read_parquet(src)
    logger.info("computing_features", n_events=len(events))
    features = compute_features(events)
    features.to_parquet(dst, index=False)
    logger.info("features_written", path=str(dst), n_customers=len(features))
    return dst


if __name__ == "__main__":
    main()
