---
title: "How I Rebuilt a Loyalty Platform's Feature Pipeline in a Weekend (And What I Learned)"
slug: "loyaltylens-feature-pipeline"
description: "Building production-grade ML feature stores without a Databricks cluster — DuckDB, feature versioning, validation gates, and the two things that surprised me."
date: 2025-09-08
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

# How I Rebuilt a Loyalty Platform's Feature Pipeline in a Weekend (And What I Learned)

*Building production-grade ML feature stores without a Databricks cluster — a hands-on walkthrough of LoyaltyLens Module 1*

---


---

In production, I spend a lot of time thinking about data that never sits still. Every tap on the app, every in-store purchase, every offer that gets ignored — all of it flows through a real-time feature pipeline before it ever reaches a propensity model. The infrastructure behind that is Azure Databricks, CDP behavioral signals, and a feature store that has to serve millions of customers at sub-100ms latency.

For my open-source project **LoyaltyLens**, I wanted to reproduce that architecture in a form anyone can run locally — no Azure subscription, no Databricks cluster, no $40,000 monthly cloud bill. What I ended up building taught me more about feature store design than two years of production work did.

Here's what I built, why I made each decision, and the two things that surprised me.

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

The first challenge was data. I generated 50,000 synthetic loyalty customer events using NumPy — purchases, app opens, offer views, and redemptions distributed across 180 days with realistic temporal patterns (Monday morning app opens spike, Friday afternoon purchases spike, redemptions cluster in the 3-day window after an offer send).

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
    customers = [str(uuid4()) for _ in range(5_000)]
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

One detail worth noting: I used `numpy.random.default_rng(seed=42)` rather than the older `np.random.seed()` API. The new Generator API is faster, more reproducible across NumPy versions, and doesn't share global state — important when you're running parallel feature jobs.

---

## Component 2: Feature Engineering

The six features I compute mirror what actually feeds into a production loyalty AI platform's propensity model:

| Feature | Definition | Why It Matters |
|---|---|---|
| `recency_days` | Days since last event | Recency is the single strongest predictor of redemption |
| `frequency_30d` | Event count, last 30 days | High-frequency customers respond to time-limited offers |
| `monetary_90d` | Purchase sum, last 90 days | Spend tier affects offer sensitivity |
| `offer_redemption_rate` | Redeems / offer_views | Direct signal of offer engagement history |
| `channel_preference` | Mode of channel field | Channel-matched offers outperform cross-channel by ~18% |
| `engagement_score` | Weighted composite [0,1] | Single normalized input for the propensity model |

The `engagement_score` formula took the most iteration:

```python
def compute_engagement_score(row: pd.Series) -> float:
    # Weights derived from production campaign performance analysis
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

The weights are not arbitrary — they reflect the relative lift coefficients from historical A/B test data. In a production system you'd derive them from a Shapley value analysis; in LoyaltyLens they're hard-coded but documented in the model card.

---

## Component 3: The DuckDB Feature Store

This is where the design gets interesting. I chose DuckDB over SQLite for one reason: **analytical query performance on Parquet files without a server process**.

DuckDB can query a Parquet file directly:

```python
import duckdb

conn = duckdb.connect("data/feature_store.duckdb")

# Write versioned feature table
def write_version(df: pd.DataFrame, version: int) -> None:
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS features_v{version} AS
        SELECT * FROM df
    """)
    conn.execute(f"""
        INSERT INTO feature_registry VALUES (
            {version},
            current_timestamp,
            {len(df)},
            '{df.columns.tolist()}'
        )
    """)
```

The versioning design gives me something critical: I can reproduce **exactly** which feature values fed any historical model prediction. This is a requirement for explainability in responsible AI governance — something I enforce rigorously in production given the scale of the loyalty program.

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

In production we've caught two upstream data pipeline failures this way — both would have silently degraded model quality for weeks if we hadn't had this gate.

---

## Component 4: The FastAPI Serving Endpoint

The feature API is deliberately thin:

```python
# feature_store/api.py
from fastapi import FastAPI, HTTPException
from feature_store.store import FeatureStore

app = FastAPI(title="LoyaltyLens Feature API")
store = FeatureStore()

@app.get("/features/{customer_id}")
async def get_features(customer_id: str) -> dict:
    """Serve the latest versioned feature vector for a customer."""
    try:
        return store.read_latest(customer_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Customer not found")

@app.get("/features/stats")
async def get_stats() -> dict:
    """Feature distribution statistics for monitoring."""
    return store.get_feature_stats(version="latest")
```

The `/features/stats` endpoint feeds the drift monitor in Module 5 — it's not a nice-to-have, it's how the system closes the loop on feature quality.

---

## The Two Things That Surprised Me

**Surprise 1: DuckDB is genuinely fast enough for production-scale retrieval.**

I ran a benchmark: 10,000 random customer ID lookups against a 5M-row feature table. DuckDB with an index on `customer_id` returned results in **~4ms median**, compared to ~12ms for SQLite and ~2ms for Redis. For an offline batch scoring job, DuckDB is more than sufficient. For real-time inference at enterprise scale, you'd still want Redis in front, but the gap is smaller than I expected.

**Surprise 2: The validation layer is more important than the features themselves.**

I spent two days tuning the engagement score weights. I spent four hours writing the validation layer. The validation layer has caught more real problems in testing than the feature weights have. This tracks with what I've seen in production: the most common source of propensity model degradation isn't model quality, it's silent upstream data drift.

---

## What's Next

Module 1 feeds directly into Module 2: the propensity scoring model. The `engagement_score` and the five raw features become the input tensor to a TabTransformer-lite network trained to predict offer redemption probability.

In the next post I'll walk through why I chose a transformer architecture over a gradient-boosted tree for tabular data, what the training curves look like, and how to package the model for a SageMaker-compatible inference container.

**[→ Read Module 2: Building a Production-Grade Propensity Scorer with PyTorch](#)**

---

*The full LoyaltyLens codebase is open-source. Follow me on [LinkedIn](https://linkedin.com/in/pushparajanramar) for updates.*
