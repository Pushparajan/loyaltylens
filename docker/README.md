# LoyaltyLens — Docker Setup

Everything you need to run the full LoyaltyLens stack in containers.

---

## Files in this directory

| File | Purpose |
|---|---|
| `Dockerfile` | Development image — all tools, Node.js, Claude Code, source mounted |
| `Dockerfile.prod` | Production image — slim, API services only |
| `docker-compose.yml` | Full stack orchestration with profiles |
| `init-pgvector.sql` | Auto-runs on first Postgres start to enable the pgvector extension |
| `entrypoint.sh` | Container startup — waits for services, runs DB init, prints summary |
| `Makefile` | Developer shortcuts (run from repo root) |
| `.dockerignore` | Excludes build artefacts, model weights, data files from the build context |

---

## Prerequisites

- Docker Desktop 4.x or Docker Engine + Compose v2
- 8GB RAM (16GB recommended if running local HuggingFace models)
- Copy `.env.example` to `.env` and fill in `OPENAI_API_KEY`

---

## Quick Start

### Option A — Infrastructure only (run Python locally with uv)

The fastest way to develop. Start the three services, keep everything else on your machine.

```bash
make up-infra
```

Services started:
- PostgreSQL + pgvector → `localhost:5432`
- Weaviate → `localhost:8080`
- Redis → `localhost:6379`

Then work locally:
```bash
uv sync
uv run python data_pipeline/generate.py
uv run pytest tests/
```

---

### Option B — Full dev container

Everything inside Docker. Source code is mounted — edits on your machine reflect instantly.

```bash
# Build the image (first time, ~5 min)
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

| Service | URL |
|---|---|
| Feature Store API | http://localhost:8001 |
| Propensity Model API | http://localhost:8002 |
| RAG Retrieval API | http://localhost:8003 |
| LLM Generator API | http://localhost:8004 |
| Feedback Loop API | http://localhost:8005 |
| MLflow UI | http://localhost:5000 |
| Streamlit Dashboard | http://localhost:8501 |

---

## Common Commands

```bash
make build          # Build the dev image
make up-infra       # Start postgres + weaviate + redis only
make up             # Start full dev environment (interactive)
make up-prod        # Start all production containers
make down           # Stop everything
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

## Docker Compose Profiles

The `docker-compose.yml` uses profiles so you only start what you need.

| Profile | Services started |
|---|---|
| `infra` | postgres, weaviate, redis |
| `dev` | infra + app (source-mounted dev container) |
| `prod` | infra + all 5 API containers + mlflow + streamlit |
| `dashboard` | mlflow + streamlit only |

```bash
# Start only infra
docker-compose -f docker/docker-compose.yml --profile infra up -d

# Start only the dashboard services
docker-compose -f docker/docker-compose.yml --profile dashboard up -d
```

---

## Volume Mounts (dev container)

| Mount | Purpose |
|---|---|
| `..:/app` | Source code — live-edits reflect instantly |
| `uv-cache` | uv package cache — avoids re-downloading on rebuild |
| `hf-cache` | HuggingFace model weights — persisted across restarts |
| `mlflow-data` | MLflow experiment runs |
| `feedback-data` | SQLite feedback database |

---

## Environment Variables

All variables are loaded from `.env` (which you create from `.env.example`).

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes* | — | OpenAI API key for M4 generation + M5 eval judge |
| `HF_TOKEN` | No | — | HuggingFace token (only for gated models) |
| `POSTGRES_URL` | Auto-set | `...@postgres:5432/...` | Set automatically inside Docker |
| `WEAVIATE_URL` | Auto-set | `http://weaviate:8080` | Set automatically inside Docker |
| `REDIS_URL` | Auto-set | `redis://redis:6379` | Set automatically inside Docker |
| `SAGEMAKER_ENDPOINT` | No | — | Only if deploying M2 to AWS SageMaker |
| `GITHUB_TOKEN` | No | — | Only if using the M6 retraining trigger |
| `GITHUB_REPO` | No | — | e.g. `yourusername/loyaltylens` |

*If `OPENAI_API_KEY` is not set, the HuggingFace backend (`Mistral-7B`) is used automatically. Requires 8GB RAM.

---

## First-Time Setup Checklist

```bash
# 1. Copy and fill in environment variables
cp .env.example .env
# edit .env — add OPENAI_API_KEY at minimum

# 2. Start infrastructure
make up-infra

# 3. Install Python dependencies
uv sync

# 4. Generate data and train the model
make generate-data
make train
make index-offers

# 5. Run tests to verify
make verify

# 6. Start all APIs
make all-apis
# or: make up-prod (Docker)

# 7. Open the dashboard
# MLflow:    http://localhost:5000
# Streamlit: http://localhost:8501
```

---

## Troubleshooting

**pgvector extension not found**
The `init-pgvector.sql` script runs automatically on first container start. If you're seeing errors, connect manually and run:
```bash
docker exec -it loyaltylens-postgres psql -U loyaltylens -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**HuggingFace model download fails**
Set `HF_TOKEN` in `.env` — some models require authentication. Or pre-download to the host cache:
```bash
uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```
The `hf-cache` volume persists this across container restarts.

**Port already in use**
```bash
# Check what's using a port
lsof -i :8002
# Stop all containers
make down
```

**Out of memory with local HuggingFace model**
Mistral-7B requires ~5GB RAM. If memory is tight, set `OPENAI_API_KEY` in `.env` and the generator will use the API backend instead.
