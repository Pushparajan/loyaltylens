"""Generate 50,000 synthetic loyalty event records and write to Parquet.

Usage:
    python -m data_pipeline.generate
    python data_pipeline/generate.py
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from shared.config import get_settings
from shared.logger import get_logger

logger = get_logger(__name__)

_SEED = 42
_N_EVENTS = 50_000
_N_CUSTOMERS = 5_000
_HORIZON_DAYS = 180

_EVENT_TYPES = ["purchase", "app_open", "offer_view", "redeem"]
_EVENT_WEIGHTS = [0.35, 0.40, 0.15, 0.10]

_CHANNELS = ["mobile", "web", "in-store"]
_CHANNEL_WEIGHTS = [0.55, 0.25, 0.20]


def generate_events(
    n_events: int = _N_EVENTS,
    n_customers: int = _N_CUSTOMERS,
    horizon_days: int = _HORIZON_DAYS,
    seed: int = _SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_uuids = [str(uuid.UUID(bytes=rng.integers(0, 256, size=16, dtype=np.uint8).tobytes())) for _ in range(n_customers)]
    offer_pool = [str(uuid.UUID(bytes=rng.integers(0, 256, size=16, dtype=np.uint8).tobytes())) for _ in range(200)]

    customer_ids = rng.choice(customer_uuids, size=n_events)
    event_types = rng.choice(_EVENT_TYPES, size=n_events, p=_EVENT_WEIGHTS)
    channels = rng.choice(_CHANNELS, size=n_events, p=_CHANNEL_WEIGHTS)

    now = pd.Timestamp.utcnow().floor("s")
    offsets_sec = rng.integers(0, horizon_days * 86_400, size=n_events)
    timestamps = [now - pd.Timedelta(seconds=int(s)) for s in offsets_sec]

    # amount: non-null only for purchases; log-normal spend distribution
    amounts: list[float | None] = []
    for et in event_types:
        if et == "purchase":
            amounts.append(float(round(rng.lognormal(mean=3.2, sigma=0.8), 2)))
        else:
            amounts.append(None)

    # offer_id: non-null for offer_view and redeem
    offer_ids: list[str | None] = []
    for et in event_types:
        if et in ("offer_view", "redeem"):
            offer_ids.append(rng.choice(offer_pool))
        else:
            offer_ids.append(None)

    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "event_type": event_types,
            "timestamp": timestamps,
            "amount": amounts,
            "offer_id": offer_ids,
            "channel": channels,
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def main() -> Path:
    settings = get_settings()
    out_path = Path(settings.raw_events_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("generating_events", n_events=_N_EVENTS, seed=_SEED)
    df = generate_events()
    df.to_parquet(out_path, index=False)
    logger.info("events_written", path=str(out_path), rows=len(df))
    return out_path


if __name__ == "__main__":
    main()
