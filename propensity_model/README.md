# propensity_model

Trains and serves a **TabTransformer-lite** (PyTorch) model that scores each customer's likelihood to redeem a loyalty offer. Model artefacts and per-epoch metrics are tracked in MLflow.

## Purpose

Produce calibrated propensity scores in `[0, 1]` for offer-redemption likelihood.  
Scores are consumed by `llm_generator` to prioritise and personalise outreach campaigns.

- **Positive class:** `offer_redemption_rate > 0.30`
- **Architecture:** per-feature Linear embedding → 2-layer TransformerEncoder (d_model=64, nhead=4) → MLP head → sigmoid

## Inputs

- Versioned feature snapshots from `feature_store.FeatureStore` (DuckDB)
- Six numerical features: `recency_days`, `frequency_30d`, `monetary_90d`, `offer_redemption_rate`, `channel_preference` (encoded), `engagement_score`
- Hyperparameters via `TabTransformerConfig` dataclass (no magic numbers)

## Outputs

- Best-checkpoint model saved to `models/propensity_v{version}.pt`
- Metadata JSON at `models/propensity_v{version}_meta.json` (`version`, `trained_at`, `val_auc`, `feature_names`, `threshold`)
- Per-epoch metrics (`train_loss`, `val_loss`, `val_auc`, `val_precision`, `val_recall`) logged to MLflow
- Real-time scores served via FastAPI (`POST /predict`, `GET /model/info`)

## Key Classes

| Class / Function      | Module          | Responsibility                                            |
|-----------------------|-----------------|-----------------------------------------------------------|
| `TabTransformerConfig`| `model.py`      | Single dataclass for all hyperparameters                  |
| `TabTransformerNet`   | `model.py`      | PyTorch nn.Module (embedding → Transformer → MLP)         |
| `PropensityModel`     | `model.py`      | Sklearn-style wrapper: `fit`, `predict_proba`, `save/load`|
| `train()`             | `train.py`      | Full pipeline: feature load → split → train → MLflow      |
| `ModelTrainer`        | `trainer.py`    | Convenience wrapper for programmatic use                  |
| `ModelEvaluator`      | `evaluator.py`  | AUC-ROC, average precision, quality gate                  |
| `PropensityPredictor` | `predictor.py`  | Load checkpoint, `predict(dict)`, `predict_batch(df)`     |
| FastAPI `app`         | `api.py`        | `POST /predict`, `GET /model/info`                        |

---

## Running Locally

### Prerequisites

- Python 3.11+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (includes Docker Compose)
- The **data pipeline** must have run first so the feature store is populated (see `data_pipeline/README.md`)

---

### 1. Start Docker Desktop

Open **Docker Desktop** from the Start menu (Windows) or Applications (macOS) and wait until the system-tray icon tooltip says **"Docker Desktop is running"** (30–60 seconds on first launch).

Verify the daemon is up:

```bash
docker info
```

If you see `ERROR: Cannot connect to the Docker daemon`, Docker Desktop is not ready yet.

---

### 2. Start infrastructure

```bash
docker compose up -d
```

Wait for all services to be healthy:

```bash
docker compose ps
```

All rows should show `healthy` before proceeding:

```text
NAME                     STATUS
loyaltylens_postgres     running (healthy)
loyaltylens_redis        running (healthy)
loyaltylens_weaviate     running (healthy)
```

---

### 3. Create a Python environment

Install [uv](https://docs.astral.sh/uv/) if you don't have it:

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal after installation so `uv` is on PATH.

Create and activate a virtual environment at the **repo root** (`c:\Projects\loyaltylens`).

> **Important:** Each sub-module (`data_pipeline/`, etc.) may have its own `.venv`. Always use the **repo-root** `.venv` for the propensity model — using the wrong venv will give `No module named pytest` or missing imports.

```powershell
# Windows (PowerShell) — run from c:\Projects\loyaltylens
uv venv .venv --python 3.11
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux — run from the repo root
uv venv .venv --python 3.11
source .venv/bin/activate
```

Verify the active Python is the repo-root one before continuing:

```powershell
python -c "import sys; print(sys.executable)"
# Must show: C:\Projects\loyaltylens\.venv\Scripts\python.exe
# If it shows data_pipeline\.venv\... run: deactivate, then re-activate above.
```

---

### 4. Install dependencies

Run from the **repo root** (`c:\Projects\loyaltylens`), not from inside `propensity_model/`:

```bash
uv pip install -e ".[dev]"
```

`torch >=2.0` is declared in `pyproject.toml` and will be installed automatically. If you want a CPU-only build (smaller download):

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[dev]" --no-deps torch
```

---

### 5. Configure environment

Copy or create a `.env` file at the repo root. The defaults in `shared/config.py` work out-of-the-box with the Docker Compose stack:

```dotenv
POSTGRES_URL=postgresql://loyaltylens:loyaltylens@localhost:5432/loyaltylens
REDIS_URL=redis://localhost:6379
DUCKDB_PATH=data/feature_store.duckdb
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=loyaltylens
PROPENSITY_MODEL_VERSION=1
PROPENSITY_MODELS_DIR=models
```

---

### 6. Seed the feature store

The training script reads features from DuckDB. `run_pipeline.py` only loads transactions into Postgres — it does **not** populate the feature store. Use the dedicated seed script instead:

```bash
python seed_feature_store.py
```

This does three things in one shot:

1. Generates 50,000 synthetic loyalty events → `data/raw/events.parquet`
2. Computes per-customer RFM features
3. Writes a versioned snapshot (e.g. `v20260416`) into `data/feature_store.duckdb`

Expected output:

```text
{"event": "generating_events", ...}
{"event": "features_computed", "n_customers": 5000, ...}
{"event": "feature_store_seeded", "version": "v20260416", ...}

Feature store seeded successfully.
  Version : v20260416
  Rows    : 5000
  Path    : data/feature_store.duckdb
```

Verify at least one feature version exists before training:

```powershell
# Windows (PowerShell)
python -c "from feature_store.store import FeatureStore; s = FeatureStore(); print(s.list_versions()); s.close()"
```

```bash
# macOS / Linux
python -c "from feature_store.store import FeatureStore; s = FeatureStore(); print(s.list_versions()); s.close()"
```

Expected: `['v20260416']` (or today's date). If the list is empty, re-run `seed_feature_store.py`.

---

### 7. Start MLflow (optional but recommended)

MLflow logs metrics and artefacts for each training run. Start the local server in a separate terminal:

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Then open [http://localhost:5000](http://localhost:5000) to view experiments.

---

### 8. Train the model

```bash
python -m propensity_model.train
```

This runs the full pipeline:

1. Loads the latest feature snapshot from DuckDB
2. Encodes `channel_preference` (in-store=0, mobile=1, web=2)
3. Creates binary labels (`offer_redemption_rate > 0.30`)
4. Stratified 70 / 15 / 15 train / val / test split
5. Trains for up to 20 epochs with early stopping (patience=3)
6. Logs per-epoch `train_loss`, `val_loss`, `val_auc`, `val_precision`, `val_recall` to MLflow
7. Saves best checkpoint to `models/propensity_v1.pt`
8. Saves metadata to `models/propensity_v1_meta.json`

Expected output (truncated):

```text
{"event": "features_loaded", "rows": 50000, "version": "v20260416"}
{"event": "epoch_complete", "epoch": 0, "train_loss": 0.6821, "val_auc": 0.7134}
{"event": "epoch_complete", "epoch": 1, "train_loss": 0.6103, "val_auc": 0.7891}
...
{"event": "training_complete", "version": "1", "val_auc": 0.8142, "test_auc": 0.8076}
```

To train with custom hyperparameters:

```python
from propensity_model.train import train
from propensity_model.model import TabTransformerConfig

model, meta = train(
    version="2",
    cfg=TabTransformerConfig(lr=5e-4, epochs=30, d_model=128),
)
print(meta)
```

---

### 9. Start the inference API

```bash
uvicorn propensity_model.api:app --host 0.0.0.0 --port 8001 --reload
```

The API loads the model version set by `PROPENSITY_MODEL_VERSION` (default `"1"`) from `PROPENSITY_MODELS_DIR` (default `models/`) on startup.

---

### 10. Verify — smoke test the API

**Windows (PowerShell):**

```powershell
Invoke-RestMethod -Method POST -Uri http://localhost:8001/predict `
  -ContentType "application/json" `
  -Body '{
    "customer_id": "cust_001",
    "recency_days": 5,
    "frequency_30d": 12,
    "monetary_90d": 250.0,
    "offer_redemption_rate": 0.45,
    "channel_preference": "mobile",
    "engagement_score": 0.72
  }' | ConvertTo-Json
```

**macOS / Linux (curl):**

```bash
curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "cust_001",
    "recency_days": 5,
    "frequency_30d": 12,
    "monetary_90d": 250.0,
    "offer_redemption_rate": 0.45,
    "channel_preference": "mobile",
    "engagement_score": 0.72
  }' | python -m json.tool
```

Expected response:

```json
{
  "customer_id": "cust_001",
  "propensity_score": 0.834,
  "label": 1,
  "threshold": 0.5,
  "model_version": "1"
}
```

Check loaded model metadata:

```bash
# Windows (PowerShell)
Invoke-RestMethod http://localhost:8001/model/info | ConvertTo-Json

# macOS / Linux
curl -s http://localhost:8001/model/info | python -m json.tool
```

---

### 11. Run the tests

```bash
# All propensity tests (fast, no MLflow / Docker required)
python -m pytest tests/test_propensity.py tests/test_propensity_model.py -v

# With coverage
python -m python -m pytest tests/test_propensity.py tests/test_propensity_model.py -v --cov=propensity_model --cov-report=term-missing
```

Expected output (all tests should pass):

```text
tests/test_propensity.py::TestTabTransformerNet::test_output_shape          PASSED
tests/test_propensity.py::TestTabTransformerNet::test_output_bounds         PASSED
tests/test_propensity.py::TestTabTransformerNet::test_single_sample         PASSED
tests/test_propensity.py::TestTabTransformerNet::test_deterministic_eval    PASSED
tests/test_propensity.py::TestPropensityModel::test_fit_returns_self        PASSED
tests/test_propensity.py::TestPropensityModel::test_predict_proba_shape     PASSED
tests/test_propensity.py::TestPropensityModel::test_predict_proba_bounds    PASSED
tests/test_propensity.py::TestPropensityModel::test_unfitted_raises         PASSED
tests/test_propensity.py::TestPropensityModel::test_save_load_roundtrip     PASSED
tests/test_propensity.py::TestPropensityModel::test_batch_consistency       PASSED
tests/test_propensity.py::TestPropensityPredictor::test_load_and_predict    PASSED
tests/test_propensity.py::TestPropensityPredictor::test_predict_batch_...   PASSED
tests/test_propensity.py::TestPropensityPredictor::test_unloaded_predict_raises  PASSED
tests/test_propensity.py::TestPropensityPredictor::test_metadata_loaded     PASSED
tests/test_propensity.py::TestAPI::test_predict_status_200                  PASSED
tests/test_propensity.py::TestAPI::test_predict_response_schema             PASSED
tests/test_propensity.py::TestAPI::test_model_info_schema                   PASSED
tests/test_propensity.py::TestAPI::test_predict_missing_field_422           PASSED
```

---

### 12. Inspect a saved model checkpoint

```powershell
# Default — inspects models/propensity_v1.pt
python -m propensity_model.inspect_model

# Specific version
python -m propensity_model.inspect_model --version 2

# Custom models directory
python -m propensity_model.inspect_model --version 1 --models-dir path/to/models
```

Expected output:

```text
============================================================
  Checkpoint: models/propensity_v1.pt
============================================================

[Config]
  n_features                   6
  d_model                      64
  nhead                        4
  num_encoder_layers           2
  dim_feedforward              128
  dropout                      0.1
  mlp_hidden                   64
  lr                           0.001
  epochs                       20
  batch_size                   64
  early_stopping_patience      3
  val_split                    0.15
  test_split                   0.15
  label_threshold              0.3
  decision_threshold           0.5

[Architecture]
  Total parameters             26,369
  Trainable parameters         26,369

[State dict layers]
  feature_embeddings.0.weight                        [64, 1]
  feature_embeddings.0.bias                          [64]
  ...
  mlp.2.weight                                       [1, 64]
  mlp.2.bias                                         [1]

============================================================
  Metadata: models/propensity_v1_meta.json
============================================================

[Meta]
  version                      1
  trained_at                   2026-04-16T20:52:09+00:00
  val_auc                      0.8142
  feature_names                ['recency_days', ...]
  threshold                    0.5
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `No feature versions found in feature store` | DuckDB is empty | Run `python run_pipeline.py` first |
| `FileNotFoundError: models/propensity_v1.pt` | Model not trained | Run `python -m propensity_model.train` |
| `503 Model not loaded` from API | Model file missing or wrong version | Check `PROPENSITY_MODEL_VERSION` in `.env` |
| `MlflowException: ... connection refused` | MLflow server not running | Start with `mlflow ui --port 5000` or set `MLFLOW_TRACKING_URI=sqlite:///mlflow.db` |
| Low val AUC (< 0.60) | Too few features with class imbalance | Adjust `label_threshold` in `TabTransformerConfig` |
