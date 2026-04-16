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

| Class | Module | Responsibility |
| --- | --- | --- |
| `FeatureStore` | `store.py` | Write, read, and validate versioned feature DataFrames |

---

## Running Locally

### 1. Environment

Uses the root `.env` — no module-level env file needed. Ensure `.env` exists at the repo root (see root `README.md`).

```python
# The module reads config exactly like this — nothing to configure per-module
from shared.config import get_settings
settings = get_settings()   # reads root .env
```

---

### 2. Generate features

From the repo root (with the root `.venv` active):

```bash
python -m data_pipeline.generate   # produces data/raw/events.parquet
python -m data_pipeline.features   # produces data/processed/features.parquet
```

Or use the combined seed script:

```bash
python seed_feature_store.py       # generate + compute + write to DuckDB in one step
```

---

### 3. Start the API

```powershell
# Windows — use python -m uvicorn and 127.0.0.1 to avoid WinError 10013
python -m uvicorn feature_store.api:app --host 127.0.0.1 --port 8001 --reload

# macOS / Linux
python -m uvicorn feature_store.api:app --host 0.0.0.0 --port 8001 --reload
```

Port `8001` is the default (`PORT_FEATURE_STORE` in `.env`).

---

### 4. Query endpoints

```powershell
# Get feature vector for a customer (PowerShell)
Invoke-RestMethod http://127.0.0.1:8001/features/<customer_id>

# Summary statistics for the latest version
Invoke-RestMethod http://127.0.0.1:8001/features/stats
```

```bash
# macOS / Linux (curl)
curl http://localhost:8001/features/<customer_id>
curl http://localhost:8001/features/stats
```

> **Note:** The `/features/stats` route must be registered in `api.py` **before** `/features/{customer_id}`, otherwise FastAPI matches `stats` as a customer ID.

---

### 5. Run tests

```bash
python -m pytest tests/test_features.py -v
```
