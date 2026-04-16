"""Bootstrap the DuckDB feature store from synthetic event data.

Run this once before training the propensity model:

    python seed_feature_store.py

Steps:
    1. Generate 50,000 synthetic loyalty events → data/raw/events.parquet
    2. Compute per-customer RFM features
    3. Write features to DuckDB feature store (data/feature_store.duckdb)
"""

from __future__ import annotations

import time
from pathlib import Path

from data_pipeline.features import compute_features
from data_pipeline.generate import generate_events
from feature_store.store import FeatureStore
from shared.config import get_settings
from shared.logger import get_logger

logger = get_logger(__name__)


def main(version: str | None = None) -> str:
    settings = get_settings()

    # ── 1. Generate raw events ────────────────────────────────────────────
    raw_path = Path(settings.raw_events_path)
    if raw_path.exists():
        logger.info("events_already_exist", path=str(raw_path))
        import pandas as pd
        events = pd.read_parquet(raw_path)
    else:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("generating_events")
        events = generate_events()
        events.to_parquet(raw_path, index=False)
        logger.info("events_written", path=str(raw_path), rows=len(events))

    # ── 2. Compute features ───────────────────────────────────────────────
    logger.info("computing_features", n_events=len(events))
    features = compute_features(events)
    logger.info("features_computed", n_customers=len(features))

    # ── 3. Write to feature store ─────────────────────────────────────────
    ver = version or time.strftime("v%Y%m%d")
    with FeatureStore() as store:
        store.write_version(features, ver)
        versions = store.list_versions()

    logger.info("feature_store_seeded", version=ver, versions=versions)
    print(f"\nFeature store seeded successfully.")
    print(f"  Version : {ver}")
    print(f"  Rows    : {len(features)}")
    print(f"  Path    : {settings.duckdb_path}")
    return ver


if __name__ == "__main__":
    main()
