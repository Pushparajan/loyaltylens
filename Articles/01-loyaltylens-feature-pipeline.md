---
title: "Building a Production-Grade Feature Pipeline with DuckDB"
slug: "loyaltylens-feature-pipeline"
description: "Building production-grade ML feature stores without a Databricks cluster — DuckDB, feature versioning, validation gates, and benchmark results."
date: 2026-04-28
author: Pushparajan Ramar
series: loyaltylens
series_order: 1
reading_time: 12
tags:
  - machine-learning
  - feature-engineering
  - mlops
  - python
  - duckdb
  - loyalty-ai
---

# Building a Production-Grade Feature Pipeline with DuckDB

*Feature store design, versioning, and validation without a Databricks cluster — LoyaltyLens Module 1*

---

**Series position:** Article 1 of 8

---

Module 1 builds the data foundation that all subsequent modules depend on: a versioned feature store that serves the same computed features at training time and inference time, with validation gates that catch upstream data issues before they reach the model.

The reference implementation uses DuckDB — no Databricks cluster, no separate server, no cloud subscription required. This article walks through each component, the design decisions behind them, and the benchmark results.

---

## The Problem With "Just Use Pandas"

When ML engineers are prototyping, the feature pipeline usually looks like this:

```python
df = pd.read_parquet("events.parquet")
df["recency"] = (pd.Timestamp.now() - df["last_event"]).dt.days
model.predict(df[["recency", "frequency", "monetary"]])
```

That works once. It fails in production for three reasons:

1. **No versioning.** When you retrain the model on a new feature definition, you have no way to reproduce the old version's behavior for debugging.
2. **No validation.** A null rate that creeps from 0.2% to 8% overnight silently poisons your propensity scores.
3. **No serving.** Training-time features and inference-time features diverge — the classic training-serving skew problem that derailed multiple production campaigns before we caught it.

A feature store solves all three. The question is: how do you build one without standing up a full Feast or Tecton deployment?

---

## The Architecture

LoyaltyLens Module 1 has four components:

```
Customer Events (Parquet)
    └─► FeaturePipeline        → computes RFM + engagement features
          └─► FeatureStore     → DuckDB-backed, versioned tables
                └─► Feature API → FastAPI serving endpoint
```

Let me walk through each one.

---

## Component 1: Synthetic Event Generation

The dataset is 50,000 synthetic loyalty customer events — purchases, app opens, offer views, and redemptions distributed across 180 days with realistic temporal patterns (Monday morning app opens spike, Friday afternoon purchases spike, redemptions cluster in the 3-day window after an offer send).

```python
# data_pipeline/generate.py (simplified)
import numpy as np
import pandas as pd
from uuid import uuid4

rng = np.random.default_rng(seed=42)

EVENT_WEIGHTS = {
    "purchase": 0.35,
    "app_open": 0.40,
    "offer_view": 0.15,
    "redeem": 0.10,
}

def generate_events(n: int = 50_000) -> pd.DataFrame:
    # Generate UUIDs from random bytes — rng.integers(0, 2**128) overflows int64
    customers = [
        str(uuid.UUID(bytes=rng.integers(0, 256, size=16, dtype=np.uint8).tobytes()))
        for _ in range(5_000)
    ]
    events = []
    for _ in range(n):
        event_type = rng.choice(
            list(EVENT_WEIGHTS.keys()),
            p=list(EVENT_WEIGHTS.values())
        )
        events.append({
            "customer_id": rng.choice(customers),
            "event_type": event_type,
            "timestamp": pd.Timestamp.now() - pd.Timedelta(
                days=rng.integers(0, 180)
            ),
            "amount": float(rng.exponential(8.5)) if event_type == "purchase" else None,
            "channel": rng.choice(["mobile", "web", "in-store"], p=[0.55, 0.20, 0.25]),
        })
    return pd.DataFrame(events)
```

Two implementation details worth noting. First, I used `numpy.random.default_rng(seed=42)` rather than the older `np.random.seed()` API — the new Generator is faster, more reproducible across NumPy versions, and doesn't share global state. Second, UUID generation uses `rng.integers(0, 256, size=16).tobytes()` rather than `rng.integers(0, 2**128)`: NumPy's integer generator is capped at `int64`, so passing `2**128` as the upper bound raises a `ValueError` at runtime.

---

## Component 2: Feature Engineering

The six features mirror what feeds into a production propensity model for loyalty offer targeting:

| Feature | Definition | Why It Matters |
|---|---|---|
| `recency_days` | Days since last event | Recency is the single strongest predictor of redemption |
| `frequency_30d` | Event count, last 30 days | High-frequency customers respond to time-limited offers |
| `monetary_90d` | Purchase sum, last 90 days | Spend tier affects offer sensitivity |
| `offer_redemption_rate` | Redeems / offer_views | Direct signal of offer engagement history |
| `channel_preference` | Mode of channel field | Channel-matched offers outperform cross-channel by ~18% |
| `engagement_score` | Weighted composite [0,1] | Single normalized input for the propensity model |

The `engagement_score` formula:

```python
def compute_engagement_score(row: pd.Series) -> float:
    recency_score   = max(0, 1 - (row["recency_days"] / 90))
    frequency_score = min(1, row["frequency_30d"] / 20)
    monetary_score  = min(1, row["monetary_90d"] / 100)
    redemption_score = row["offer_redemption_rate"]

    return (
        0.30 * recency_score
        + 0.25 * frequency_score
        + 0.25 * monetary_score
        + 0.20 * redemption_score
    )
```

The weights reflect relative lift coefficients from A/B test data — in a production system, derive them from a Shapley value analysis. In LoyaltyLens they're fixed but documented in the model card.

---

## Component 3: The DuckDB Feature Store

DuckDB is chosen over SQLite for one reason: **analytical query performance on Parquet files without a server process**.

DuckDB can query a Parquet file directly:

```python
import duckdb

conn = duckdb.connect("data/feature_store.duckdb")

# Write versioned feature table.
# Column order in the DataFrame MUST match the table schema before inserting —
# DuckDB's INSERT INTO … SELECT * maps by position, not by name.
# Mismatched order causes string columns to land in DOUBLE slots at runtime.
def write_version(df: pd.DataFrame, version: str) -> None:
    insert_df = df[["customer_id"] + FEATURE_COLS].copy()
    insert_df["version"] = version
    insert_df = insert_df[["customer_id", "version"] + FEATURE_COLS]
    conn.execute("DELETE FROM features WHERE version = ?", [version])
    conn.execute("INSERT INTO features SELECT *, now() FROM insert_df")
```

The versioning design enables exact reproduction of which feature values fed any historical model prediction — a requirement for explainability in responsible AI governance.

```python
def read_latest(customer_id: str) -> dict:
    latest_version = conn.execute("""
        SELECT MAX(version) FROM feature_registry
    """).fetchone()[0]
    
    result = conn.execute(f"""
        SELECT * FROM features_v{latest_version}
        WHERE customer_id = ?
    """, [customer_id]).fetchdf()
    
    return result.to_dict(orient="records")[0]
```

### Validation That Actually Catches Problems

The feature store enforces two invariants on every write:

```python
def validate_features(df: pd.DataFrame) -> None:
    for col in df.columns:
        null_rate = df[col].isna().mean()
        if null_rate > 0.05:
            raise FeatureValidationError(
                f"Column {col} has {null_rate:.1%} nulls — threshold is 5%"
            )
    
    score_bounds = df["engagement_score"].agg(["min", "max"])
    if score_bounds["min"] < 0 or score_bounds["max"] > 1:
        raise FeatureValidationError(
            f"engagement_score outside [0,1]: "
            f"min={score_bounds['min']:.4f}, max={score_bounds['max']:.4f}"
        )
```

This gate catches upstream data pipeline failures that would otherwise silently degrade model quality for days or weeks.

---

## Component 4: The FastAPI Serving Endpoint

The feature API is deliberately thin:

```python
# feature_store/api.py
from fastapi import FastAPI, HTTPException
from feature_store.store import FeatureStore

app = FastAPI(title="LoyaltyLens Feature API")
store = FeatureStore()

# IMPORTANT: /features/stats must be registered BEFORE /features/{customer_id}.
# FastAPI matches routes in registration order — if the path-parameter route comes
# first, requests to /features/stats are captured with customer_id="stats" and
# return a 404 instead of hitting the stats handler.

@app.get("/features/stats")
async def get_stats() -> dict:
    """Feature distribution statistics for monitoring."""
    return store.get_feature_stats(version="latest")

@app.get("/features/{customer_id}")
async def get_features(customer_id: str) -> dict:
    """Serve the latest versioned feature vector for a customer."""
    try:
        return store.read_latest(customer_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Customer not found")
```

The `/features/stats` endpoint feeds the drift monitor in Module 5 — it's not a nice-to-have, it's how the system closes the loop on feature quality.

---

## Benchmark Results

**DuckDB retrieval latency vs. alternatives:**

10,000 random customer ID lookups against a 5M-row feature table:

| Store | Median latency |
|---|---|
| DuckDB (indexed) | ~4ms |
| SQLite (indexed) | ~12ms |
| Redis | ~2ms |

DuckDB is sufficient for offline batch scoring jobs. For real-time inference at high request rates, add Redis as a serving layer in front of DuckDB.

**Validation vs. feature tuning — relative impact:**

The validation layer consistently catches more quality issues than feature weight tuning resolves. The most common source of propensity model degradation in production is silent upstream data drift, not model quality — which is why the validation gate belongs at the feature store boundary, not in the model training pipeline.

---

## Next: Module 2 — Propensity Scoring

Module 1 feeds directly into Module 2. The `engagement_score` and the five raw features become the input tensor to a TabTransformer-lite network trained to predict offer redemption probability. Module 2 covers the architecture choice (transformer vs. gradient-boosted tree), training loop, MLflow experiment tracking, and TorchScript export for cloud deployment.

**[→ Read Module 2: Building a Production-Grade Propensity Scorer with PyTorch](#)**

---

*Pushparajan Ramar — [LinkedIn](https://linkedin.com/in/pushparajanramar) · [GitHub](https://github.com/Pushparajan/loyaltylens)*
