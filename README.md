# LoyaltyLens

ML-powered loyalty analytics platform that combines propensity modelling with RAG-driven personalisation to predict customer churn, recommend retention actions, and generate personalised communications at scale.

## Architecture

```text
loyaltylens/
├── data_pipeline/      # ETL: ingest transactions & customer data → Postgres
├── feature_store/      # Compute & serve ML features via DuckDB + Redis
├── propensity_model/   # Train & score offer-redemption propensity (TabTransformer)
├── rag_retrieval/      # Embed & retrieve offers via pgvector + Weaviate
├── llm_generator/      # Generate personalised offer copy via LLM
├── llmops/             # Track LLM calls, evaluate quality, expose metrics
├── feedback_loop/      # Collect response feedback; retrain signals
├── shared/             # Cross-cutting: config, DB clients, logging, schemas
└── tests/              # Pytest suite for all modules
```

## Infrastructure

All ports are defined once in `shared/config.py` and read from the root `.env`. Change a port in `.env` — it propagates everywhere.

| Service | Image | Default Port | Env Var |
| --- | --- | --- | --- |
| PostgreSQL + pgvector | `pgvector/pgvector:pg16` | 5432 | `POSTGRES_PORT` |
| Weaviate | `semitechnologies/weaviate:1.28.2` | 8080 / 50051 | `WEAVIATE_HTTP_PORT` / `WEAVIATE_GRPC_PORT` |
| Redis | `redis:7.2-alpine` | 6379 | `REDIS_PORT` |
| MLflow UI | — | 5000 | `MLFLOW_PORT` |

## Service API Ports

| Module | Default Port | Env Var |
| --- | --- | --- |
| `feature_store` | 8001 | `PORT_FEATURE_STORE` |
| `propensity_model` | 8002 | `PORT_PROPENSITY` |
| `rag_retrieval` | 8003 | `PORT_RAG_RETRIEVAL` |
| `llm_generator` | 8004 | `PORT_LLM_GENERATOR` |
| `feedback_loop` | 8005 | `PORT_FEEDBACK_LOOP` |
| Prometheus metrics | 8006 | `PORT_METRICS` |

## Quickstart

### 1. Clone and set up environment

```powershell
git clone <repo-url>
cd loyaltylens

# Create root venv (canonical — all modules use this one; skip if it already exists)
uv venv .venv --python 3.11

# Activate — prompt will show (loyaltylens)
& C:\Projects\loyaltylens\.venv\Scripts\Activate.ps1    # Windows
# source .venv/bin/activate                              # macOS / Linux

# Install all dependencies (shared/ installed as editable via hatchling)
uv sync --dev
```

> **Windows execution policy:** If Activate.ps1 is blocked, run once:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### 2. Configure environment

```powershell
Copy-Item .env.example .env   # Windows
# cp .env.example .env        # macOS / Linux
```

Edit `.env` and set at minimum `OPENAI_API_KEY` (for M4/M5) and `HF_TOKEN` (optional, for gated HuggingFace models). All other values work out-of-the-box with the Docker Compose defaults.

> **Single source of truth:** every module reads `shared.config.get_settings()` which loads the root `.env`. Never create module-level `.env` files.

### 3. Start infrastructure

```powershell
docker compose up -d postgres weaviate redis

# Wait for all three to show (healthy)
docker compose ps
```

### 4. Seed data and build embeddings

```powershell
python seed_feature_store.py              # generates events + features
python rag_retrieval/generate_offers.py   # 200 synthetic offers
python rag_retrieval/embeddings.py        # embed into pgvector + Weaviate
```

### 5. Run the test suite

```powershell
python -m pytest tests/ -v
```

### 6. Start services (one terminal each)

```powershell
python -m uvicorn feature_store.api:app      --host 127.0.0.1 --port 8001 --reload
python -m uvicorn propensity_model.api:app   --host 127.0.0.1 --port 8002 --reload
python -m uvicorn rag_retrieval.api:app      --host 127.0.0.1 --port 8003 --reload
python -m uvicorn llm_generator.api:app      --host 127.0.0.1 --port 8004 --reload
python -m uvicorn feedback_loop.api:app      --host 127.0.0.1 --port 8005 --reload
```

> **Windows note:** Use `127.0.0.1` instead of `0.0.0.0` to avoid `WinError 10013` firewall errors. Use `python -m uvicorn` instead of bare `uvicorn` — venv scripts are not on PATH unless the venv is activated.

### Restart Docker from scratch

```powershell
docker compose down -v          # stop containers and wipe all volumes
docker compose up -d postgres weaviate redis
python rag_retrieval/generate_offers.py
python rag_retrieval/embeddings.py
```

## CI/CD

GitHub Actions runs on every push and pull request:

1. **Lint** — `ruff check .`
2. **Type-check** — `mypy .`
3. **Test** — `pytest` with coverage uploaded to Codecov
4. **Eval Gate** — `python llmops/eval_harness/run_eval.py`; pipeline fails if mean score < 0.75

Add a `POSTGRES_PASSWORD` secret to your GitHub repository settings before running CI.

## Module Responsibilities

| Module | Layer | Role in the loyalty loop |
| --- | --- | --- |
| `data_pipeline` | Data Ingestion | Ingest POS transactions and CRM exports into Postgres |
| `feature_store` | Feature Platform | Compute and serve ML features at low latency via DuckDB |
| `propensity_model` | Propensity Engine | TabTransformer-lite that scores offer-redemption likelihood |
| `rag_retrieval` | Contextual Retrieval | Embed and retrieve offers from pgvector + Weaviate |
| `llm_generator` | Offer Generation | Draft individualised retention offers via an LLM |
| `llmops` | Observability & Eval | Track latency, tokens, quality scores; enforce eval gate |
| `feedback_loop` | Reinforcement Signals | Convert redemption events into pseudo-labels for retraining |
| `shared` | Platform SDK | Centralised config, DB clients, and structured logging |
