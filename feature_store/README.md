# feature_store

Stores and serves versioned ML feature vectors for the propensity model, backed by a local DuckDB database.

## Purpose

Provide a single source of truth for all ML features so that training and online scoring use identical feature logic, eliminating training–serving skew.

## Inputs

- Computed feature DataFrames from `data_pipeline.features.compute_features`
- Version label string (e.g. `"v1"`, `"v2"`) supplied at write time
- DB path from `shared.Settings.duckdb_path` (default: `data/feature_store.duckdb`)

## Outputs

- Versioned feature rows written to the `features` DuckDB table
- Per-customer feature vectors served via `GET /features/{customer_id}`
- Summary statistics served via `GET /features/stats`

## Key Classes

| Class          | Module     | Responsibility                                          |
| -------------- | ---------- | ------------------------------------------------------- |
| `FeatureStore` | `store.py` | Write, read, and validate versioned feature DataFrames  |

## Running Locally

### 1. Generate features

From the repo root (with the venv active):

```bash
python -m data_pipeline.generate   # produces data/raw/events.parquet
python -m data_pipeline.features   # produces data/processed/features.parquet
```

### 2. Start the API

```bash
uvicorn feature_store.api:app --reload --port 8002
```

### 3. Query endpoints

```bash
# Get feature vector for a customer
curl http://localhost:8002/features/<customer_id>

# Get summary statistics for the latest version
curl http://localhost:8002/features/stats
```

> **Note:** The `/features/stats` route must be registered in `api.py` **before** `/features/{customer_id}`, otherwise FastAPI matches `stats` as a customer ID.

### 4. Run tests

```bash
python -m pytest tests/test_features.py
```
