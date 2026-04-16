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
- Metadata JSON at `models/propensity_v{version}_meta.json`
- Per-epoch metrics logged to MLflow
- Real-time scores served via FastAPI (`POST /predict`, `GET /model/info`) on port `8002`

## Key Classes

| Class / Function | Module | Responsibility |
| --- | --- | --- |
| `TabTransformerConfig` | `model.py` | Single dataclass for all hyperparameters |
| `TabTransformerNet` | `model.py` | PyTorch nn.Module (embedding → Transformer → MLP) |
| `PropensityModel` | `model.py` | Sklearn-style wrapper: `fit`, `predict_proba`, `save/load` |
| `train()` | `train.py` | Full pipeline: feature load → split → train → MLflow |
| `ModelTrainer` | `trainer.py` | Convenience wrapper for programmatic use |
| `ModelEvaluator` | `evaluator.py` | AUC-ROC, average precision, quality gate |
| `PropensityPredictor` | `predictor.py` | Load checkpoint, `predict(dict)`, `predict_batch(df)` |
| FastAPI `app` | `api.py` | `POST /predict`, `GET /model/info` |

---

## Running Locally

### Prerequisites

- Python 3.11+
- Docker Desktop running (see root `README.md`)
- Feature store populated — run `python seed_feature_store.py` first

---

### 1. Start infrastructure

```powershell
docker compose up -d postgres weaviate redis
docker compose ps   # wait for (healthy)
```

---

### 2. Python environment

Always use the **repo-root** `.venv`. Sub-module venvs (`propensity_model/.venv`) are legacy — do not use them.

```powershell
# Windows (PowerShell) — from c:\Projects\loyaltylens
uv venv .venv --python 3.11
.venv\Scripts\Activate.ps1

# Verify you are using the root venv
python -c "import sys; print(sys.executable)"
# Must show: C:\Projects\loyaltylens\.venv\Scripts\python.exe
# If it shows propensity_model\.venv\..., run deactivate then re-activate above.
```

```bash
# macOS / Linux
uv venv .venv --python 3.11
source .venv/bin/activate
```

---

### 3. Install dependencies

```powershell
uv sync --dev
uv pip install -e .    # makes shared/ importable without PYTHONPATH
```

CPU-only PyTorch (smaller download):

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

### 4. Environment

One `.env` at the **repo root** — all modules share it. No `propensity_model/.env` needed.

```powershell
Copy-Item .env.example .env   # if not already done
```

Relevant defaults (all work out-of-the-box with Docker Compose):

```dotenv
POSTGRES_URL=postgresql://loyaltylens:loyaltylens@localhost:5432/loyaltylens
DUCKDB_PATH=data/feature_store.duckdb
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=loyaltylens
PROPENSITY_MODEL_VERSION=1
PROPENSITY_MODELS_DIR=models
PORT_PROPENSITY=8002
```

---

### 5. Seed the feature store

```bash
python seed_feature_store.py
```

Expected output:

```text
Feature store seeded successfully.
  Version : v20260416
  Rows    : 5000
  Path    : data/feature_store.duckdb
```

Verify:

```bash
python -c "from feature_store.store import FeatureStore; s = FeatureStore(); print(s.list_versions()); s.close()"
# Expected: ['v20260416']
```

---

### 6. Start MLflow (optional but recommended)

```bash
python -m mlflow ui --host 127.0.0.1 --port 5000
# Open http://localhost:5000
```

---

### 7. Train the model

```bash
python -m propensity_model.train
```

Expected output (truncated):

```text
{"event": "features_loaded", "rows": 50000, "version": "v20260416"}
{"event": "epoch_complete", "epoch": 0, "train_loss": 0.6821, "val_auc": 0.7134}
...
{"event": "training_complete", "version": "1", "val_auc": 0.8142, "test_auc": 0.8076}
```

Custom hyperparameters:

```python
from propensity_model.train import train
from propensity_model.model import TabTransformerConfig

model, meta = train(version="2", cfg=TabTransformerConfig(lr=5e-4, epochs=30, d_model=128))
```

---

### 8. Start the inference API

```powershell
# Windows
python -m uvicorn propensity_model.api:app --host 127.0.0.1 --port 8002 --reload

# macOS / Linux
python -m uvicorn propensity_model.api:app --host 0.0.0.0 --port 8002 --reload
```

---

### 9. Smoke test the API

**Windows (PowerShell):**

```powershell
Invoke-RestMethod -Method POST -Uri http://127.0.0.1:8002/predict `
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

**macOS / Linux:**

```bash
curl -s -X POST http://localhost:8002/predict \
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

---

### 10. Run tests

```bash
python -m pytest tests/test_propensity.py tests/test_propensity_model.py -v

# With coverage
python -m pytest tests/test_propensity.py tests/test_propensity_model.py -v \
  --cov=propensity_model --cov-report=term-missing
```

---

### 11. Inspect a saved checkpoint

```powershell
python -m propensity_model.inspect_model           # default: models/propensity_v1.pt
python -m propensity_model.inspect_model --version 2
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `No feature versions found in feature store` | DuckDB is empty | Run `python seed_feature_store.py` |
| `FileNotFoundError: models/propensity_v1.pt` | Model not trained | Run `python -m propensity_model.train` |
| `503 Model not loaded` from API | Model file missing | Check `PROPENSITY_MODEL_VERSION` in root `.env` |
| `MlflowException: connection refused` | MLflow not running | Run `python -m mlflow ui --port 5000` |
| `No module named 'shared'` | Project not installed as editable | Run `uv pip install -e .` from repo root |
| Low val AUC (< 0.60) | Class imbalance | Adjust `label_threshold` in `TabTransformerConfig` |
