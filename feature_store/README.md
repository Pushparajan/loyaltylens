# feature_store

Computes, stores, and serves ML features for the propensity model. Hot features are cached in Redis; historical snapshots are persisted in Postgres for training and back-testing.

## Purpose

Provide a single source of truth for all ML features so that training and online scoring use identical feature logic, eliminating training–serving skew.

## Inputs

- Clean transaction and customer rows from `data_pipeline` (via Postgres)
- Feature definitions declared as Python dataclasses (schema registry in `store.py`)
- TTL / refresh configuration from `shared.Settings`

## Outputs

- Feature vectors written to Redis (online store) keyed by `customer_id`
- Historical feature snapshots written to `feature_snapshots` Postgres table
- Point-in-time feature DataFrames returned to `propensity_model` for training

## Key Classes

| Class           | Module      | Responsibility |
| --------------- | ----------- | ----------------------------------------------- |
| `FeatureStore`  | `store.py`  | Registry of feature definitions and metadata    |
| `FeatureWriter` | `writer.py` | Compute features and write to Redis + Postgres  |
| `FeatureReader` | `reader.py` | Fetch online features or build training dataset |
