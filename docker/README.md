# LoyaltyLens — Docker Setup

Everything you need to run the full LoyaltyLens stack in containers.

---

## Files in this directory

| File | Purpose |
| --- | --- |
| `Dockerfile` | Development image — all tools, Node.js, Claude Code, source mounted |
| `Dockerfile.prod` | Production image — slim, API services only |
| `docker-compose.yml` | Full stack orchestration with profiles |
| `init-pgvector.sql` | Auto-runs on first Postgres start to enable the pgvector extension |
| `entrypoint.sh` | Container startup — waits for services, runs DB init, prints summary |
| `Makefile` | Developer shortcuts (run from repo root) |

---

## Prerequisites

- Docker Desktop 4.x or Docker Engine + Compose v2
- 8 GB RAM (16 GB recommended if running local HuggingFace models)
- Root `.env` file — copy `.env.example` to `.env` and fill in `OPENAI_API_KEY`

> **Single `.env` file.** All modules — inside and outside Docker — read the root `.env`. Never create module-level `.env` files. Inside Docker, the container receives env vars from the `--env-file .env` flag or Docker Compose's `env_file` directive.

---

## Quick Start

### Option A — Infrastructure only (run Python locally with uv)

The fastest way to develop. Start the three services; keep everything else on your machine.

```bash
make up-infra
```

Services started:

- PostgreSQL + pgvector → `localhost:5432`
- Weaviate `1.28.2` → `localhost:8080` (HTTP) + `localhost:50051` (gRPC)
- Redis → `localhost:6379`

Then work locally:

```powershell
uv sync --dev
uv pip install -e .
python seed_feature_store.py
python rag_retrieval/generate_offers.py
python rag_retrieval/embeddings.py
python -m pytest tests/
```

---

### Option B — Full dev container

Everything inside Docker. Source code is mounted — edits on your machine reflect instantly.

```bash
# Build the image (first time ~5 min)
make build

# Start infra + drop into the dev container
make up

# Inside the container:
uv run python data_pipeline/generate.py
uv run python propensity_model/train.py
uv run pytest tests/ -v
make all-apis
```

---

### Option C — Production stack (all APIs running)

```bash
make up-prod
```

Starts all six API containers, MLflow, and Streamlit:

| Service | URL | Port Env Var |
| --- | --- | --- |
| Feature Store API | [localhost:8001](http://localhost:8001) | `PORT_FEATURE_STORE` |
| Propensity Model API | [localhost:8002](http://localhost:8002) | `PORT_PROPENSITY` |
| RAG Retrieval API | [localhost:8003](http://localhost:8003) | `PORT_RAG_RETRIEVAL` |
| LLM Generator API | [localhost:8004](http://localhost:8004) | `PORT_LLM_GENERATOR` |
| Feedback Loop API | [localhost:8005](http://localhost:8005) | `PORT_FEEDBACK_LOOP` |
| MLflow UI | [localhost:5000](http://localhost:5000) | `MLFLOW_PORT` |
| Streamlit Dashboard | [localhost:8501](http://localhost:8501) | — |

---

## Common Commands

```bash
make build          # Build the dev image
make up-infra       # Start postgres + weaviate + redis only
make up             # Start full dev environment (interactive)
make up-prod        # Start all production containers
make down           # Stop everything
make down-v         # Stop everything and wipe all data volumes (fresh start)
make logs           # Tail all container logs
make shell          # Open bash in the running dev container

make test           # Run pytest
make lint           # Run ruff
make format         # Run ruff formatter
make typecheck      # Run mypy
make verify         # All three: lint + typecheck + test

make generate-data  # Run M1 data pipeline
make train          # Train M2 propensity model
make index-offers   # Embed and index M3 offers

make mlflow         # Start MLflow locally (no Docker)
make streamlit      # Start Streamlit locally (no Docker)
make all-apis       # Start all 5 APIs locally
```

---

## Restart from scratch

```bash
make down-v                            # wipe all volumes
make up-infra                          # fresh containers
python rag_retrieval/generate_offers.py
python rag_retrieval/embeddings.py
```

---

## Docker Compose Profiles

| Profile | Services started |
| --- | --- |
| `infra` | postgres, weaviate, redis |
| `dev` | infra + app (source-mounted dev container) |
| `prod` | infra + all 5 API containers + mlflow + streamlit |
| `dashboard` | mlflow + streamlit only |

```bash
docker compose -f docker/docker-compose.yml --profile infra up -d
docker compose -f docker/docker-compose.yml --profile dashboard up -d
```

---

## Volume Mounts (dev container)

| Mount | Purpose |
| --- | --- |
| `..:/app` | Source code — live-edits reflect instantly |
| `uv-cache` | uv package cache — avoids re-downloading on rebuild |
| `hf-cache` | HuggingFace model weights — persisted across restarts |
| `mlflow-data` | MLflow experiment runs |
| `feedback-data` | SQLite feedback database |

---

## Environment Variables

All variables come from the root `.env` (created from `.env.example`). All port values default to the values in `shared/config.py` — override in `.env` to move a service.

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | Yes* | — | OpenAI API key for M4 generation + M5 eval judge |
| `HF_TOKEN` | No | — | HuggingFace token (only for gated models) |
| `POSTGRES_URL` | Auto-set | `...@postgres:5432/...` | Set automatically inside Docker |
| `WEAVIATE_URL` | Auto-set | `http://weaviate:8080` | Set automatically inside Docker |
| `REDIS_URL` | Auto-set | `redis://redis:6379` | Set automatically inside Docker |
| `WEAVIATE_GRPC_PORT` | No | `50051` | Required by weaviate-client v4 |
| `PORT_FEATURE_STORE` | No | `8001` | Feature Store API port |
| `PORT_PROPENSITY` | No | `8002` | Propensity Model API port |
| `PORT_RAG_RETRIEVAL` | No | `8003` | RAG Retrieval API port |
| `PORT_LLM_GENERATOR` | No | `8004` | LLM Generator API port |
| `PORT_FEEDBACK_LOOP` | No | `8005` | Feedback Loop API port |
| `PORT_METRICS` | No | `8006` | Prometheus metrics server port |
| `SAGEMAKER_ENDPOINT` | No | — | Only if deploying M2 to AWS SageMaker |

*If `OPENAI_API_KEY` is not set, the HuggingFace backend (`Mistral-7B`) is used automatically. Requires 8 GB RAM.

---

## First-Time Setup Checklist

```bash
# 1. Copy and fill in environment variables
cp .env.example .env
# edit .env — add OPENAI_API_KEY at minimum

# 2. Start infrastructure
make up-infra

# 3. Install Python dependencies (local dev)
uv sync --dev && uv pip install -e .

# 4. Generate data, train model, index offers
make generate-data
make train
make index-offers

# 5. Run tests to verify
make verify

# 6. Start all APIs
make all-apis
# or: make up-prod (Docker)

# 7. Open dashboards
# MLflow:    http://localhost:5000
# Streamlit: http://localhost:8501
```

---

## Troubleshooting

**pgvector extension not found**
`init-pgvector.sql` runs automatically on first container start. If still missing:

```bash
docker exec -it loyaltylens_postgres psql -U loyaltylens -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**Weaviate connection error (`WeaviateStartUpError`)**
The Python client v4 requires Weaviate server **≥ 1.27.0**. The repo ships `1.28.2`. If you see this error, your container may be running an older image — run `make down-v && make up-infra` to pull the correct image.

**`No module named 'shared'`**
Run `uv pip install -e .` from the repo root. This installs the project as an editable package so all modules are importable without setting `PYTHONPATH`.

**HuggingFace model download fails**
Set `HF_TOKEN` in `.env`. Pre-download to the host cache:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

**Port already in use (Windows `WinError 10013`)**
Use `127.0.0.1` instead of `0.0.0.0` when running `python -m uvicorn` locally on Windows. To change which port a service uses, set the corresponding `PORT_*` variable in `.env`.

**Out of memory with local HuggingFace model**
Mistral-7B requires ~5 GB RAM. Set `OPENAI_API_KEY` in `.env` and the generator uses the API backend instead.
