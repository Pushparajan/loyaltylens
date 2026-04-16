"""DuckDB-backed feature store with versioning and data validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from shared.config import get_settings
from shared.logger import get_logger

logger = get_logger(__name__)

_NULL_THRESHOLD = 0.05  # raise if null fraction exceeds 5% in any column
_FEATURE_COLS = [
    "recency_days",
    "frequency_30d",
    "monetary_90d",
    "offer_redemption_rate",
    "channel_preference",
    "engagement_score",
]


class FeatureValidationError(ValueError):
    """Raised when a feature DataFrame fails quality checks."""


class FeatureStore:
    """Versioned feature store backed by a local DuckDB database.

    The underlying table schema is::

        features (
            customer_id       TEXT,
            version           TEXT,
            recency_days      INTEGER,
            frequency_30d     INTEGER,
            monetary_90d      DOUBLE,
            offer_redemption_rate DOUBLE,
            channel_preference    TEXT,
            engagement_score  DOUBLE,
            written_at        TIMESTAMP DEFAULT now()
        )
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        path = str(db_path or get_settings().duckdb_path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(path)
        self._init_schema()
        logger.info("feature_store_opened", path=path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_version(self, df: pd.DataFrame, version: str) -> None:
        """Validate *df* and persist it under *version*.

        Raises
        ------
        FeatureValidationError
            If null fraction > 5% in any column or ``engagement_score``
            contains values outside [0, 1].
        """
        self._validate(df)
        insert_df = df[["customer_id"] + _FEATURE_COLS].copy()
        insert_df["version"] = version
        insert_df = insert_df[["customer_id", "version"] + _FEATURE_COLS]
        self._conn.execute("DELETE FROM features WHERE version = ?", [version])
        self._conn.execute("INSERT INTO features SELECT *, now() FROM insert_df")
        logger.info("features_written", version=version, rows=len(df))

    def read_latest(self, customer_id: str) -> dict[str, Any] | None:
        """Return the most-recently written feature row for *customer_id*, or None."""
        row = self._conn.execute(
            """
            SELECT * EXCLUDE (written_at)
            FROM features
            WHERE customer_id = ?
            ORDER BY written_at DESC
            LIMIT 1
            """,
            [customer_id],
        ).fetchdf()
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    def list_versions(self) -> list[str]:
        """Return all distinct version strings, newest written first."""
        rows = self._conn.execute(
            "SELECT DISTINCT version, MAX(written_at) AS ts "
            "FROM features GROUP BY version ORDER BY ts DESC"
        ).fetchall()
        return [r[0] for r in rows]

    def get_feature_stats(self, version: str) -> pd.DataFrame:
        """Return DuckDB SUMMARIZE stats for all numeric features in *version*."""
        return self._conn.execute(
            """
            SUMMARIZE
            SELECT recency_days, frequency_30d, monetary_90d,
                   offer_redemption_rate, engagement_score
            FROM features
            WHERE version = ?
            """,
            [version],
        ).fetchdf()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "FeatureStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS features (
                customer_id           TEXT        NOT NULL,
                version               TEXT        NOT NULL,
                recency_days          INTEGER,
                frequency_30d         INTEGER,
                monetary_90d          DOUBLE,
                offer_redemption_rate DOUBLE,
                channel_preference    TEXT,
                engagement_score      DOUBLE,
                written_at            TIMESTAMP   DEFAULT now()
            )
            """
        )

    def _validate(self, df: pd.DataFrame) -> None:
        required = {"customer_id"} | set(_FEATURE_COLS)
        missing = required - set(df.columns)
        if missing:
            raise FeatureValidationError(f"DataFrame missing columns: {missing}")

        for col in _FEATURE_COLS:
            null_frac = df[col].isna().mean()
            if null_frac > _NULL_THRESHOLD:
                raise FeatureValidationError(
                    f"Column '{col}' has {null_frac:.1%} nulls (threshold {_NULL_THRESHOLD:.0%})"
                )

        bad_scores = df["engagement_score"].dropna()
        if ((bad_scores < 0.0) | (bad_scores > 1.0)).any():
            raise FeatureValidationError("engagement_score contains values outside [0, 1]")
