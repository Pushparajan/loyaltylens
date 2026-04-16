# LoyaltyLens — Tools & Functionality Reference

A complete reference for every tool, library, service, and framework used across the six LoyaltyLens modules. Organised by category. Each entry covers what it is, what it does in this project, and the `uv` command to install it.

---

## Contents

1. [Environment & Package Management](#1-environment--package-management)
2. [Infrastructure Services](#2-infrastructure-services)
3. [Data Engineering & Feature Store](#3-data-engineering--feature-store)
4. [Machine Learning & Model Training](#4-machine-learning--model-training)
5. [Vector Databases & Embeddings](#5-vector-databases--embeddings)
6. [LLM Orchestration & RAG](#6-llm-orchestration--rag)
7. [LLM Backends](#7-llm-backends)
8. [LLMOps & Evaluation](#8-llmops--evaluation)
9. [API Layer & Serving](#9-api-layer--serving)
10. [Frontend & UI](#10-frontend--ui)
11. [CI/CD & DevOps](#11-cicd--devops)
12. [Observability & Logging](#12-observability--logging)
13. [Code Quality & Testing](#13-code-quality--testing)
14. [Module-to-Tool Map](#14-module-to-tool-map)
15. [Full pyproject.toml Reference](#15-full-pyprojecttoml-reference)

---

## 1. Environment & Package Management

### uv
**What it is:** A Rust-based Python package manager from Astral. Replaces `pip`, `pip-tools`, `pyenv`, `virtualenv`, and `poetry` in a single binary.

**What it does in LoyaltyLens:** Manages the Python version, creates the virtual environment, installs all dependencies from `pyproject.toml`, and produces a reproducible `uv.lock` lockfile. Every `python` command in this project is run via `uv run`.

**Key commands:**
```bash
uv python install 3.11   # Download and install Python 3.11
uv python pin 3.11       # Pin this project to 3.11 (creates .python-version)
uv venv                  # Create .venv/ using pinned Python version
uv sync                  # Install all deps from pyproject.toml + uv.lock
uv add <pkg>             # Add a package and update pyproject.toml
uv add --dev <pkg>       # Add a dev-only dependency
uv remove <pkg>          # Remove a package
uv run python script.py  # Run a script inside the venv (no activate needed)
uv run pytest            # Run any tool through the venv
uv lock --upgrade        # Regenerate lockfile with latest compatible versions
```

**Install:** `curl -LsSf https://astral.sh/uv/install.sh | sh` (macOS/Linux)  
**Docs:** https://docs.astral.sh/uv/

---

### Python 3.11
**What it is:** The programming language runtime for all LoyaltyLens Python modules.

**Why 3.11 specifically:** Match statements (`match event_type:`), `tomllib` (built-in TOML parsing), `dict | dict` union syntax, and significant performance improvements over 3.10.

**Install via uv:** `uv python install 3.11`

---

### Node.js 18+
**What it is:** JavaScript runtime required for the React feedback UI (Module 6) and Claude Code.

**What it does in LoyaltyLens:** Powers the Vite-based React frontend in `feedback_loop/ui/`. Not used for any Python modules.

**Install:** https://nodejs.org — use the LTS release.

---

## 2. Infrastructure Services

All three services are managed via Docker Compose and run locally. Start them with `docker-compose up -d`.

### PostgreSQL + pgvector
**What it is:** PostgreSQL 16 with the pgvector extension pre-installed (image: `pgvector/pgvector:pg16`).

**What it does in LoyaltyLens:** Primary vector store for Module 3 (offer embeddings). Also stores feedback records in Module 6.

**Key connection detail:**
```
postgresql://loyaltylens:loyaltylens@localhost:5432/loyaltylens
```

**One-time setup:**
```bash
docker exec -it loyaltylens-postgres-1 psql -U loyaltylens \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

---

### Weaviate
**What it is:** An open-source vector database with a REST/GraphQL API (image: `semitechnologies/weaviate:1.24.1`).

**What it does in LoyaltyLens:** Secondary vector store for Module 3. Benchmarked against pgvector on latency and operational complexity.

**Key connection detail:** `http://localhost:8080`

---

### Redis
**What it is:** An in-memory key-value store (image: `redis:7-alpine`).

**What it does in LoyaltyLens:** Caching layer available for feature serving and session storage. Referenced in `shared/config.py` but not heavily used in the base implementation — included because production feature stores typically front-end their DB with Redis for sub-millisecond latency.

**Key connection detail:** `redis://localhost:6379`

---

## 3. Data Engineering & Feature Store

### DuckDB
**What it is:** An embedded analytical database that runs in-process — no server required. Designed for fast SQL analytics on Parquet, CSV, and JSON files.

**What it does in LoyaltyLens (M1):** Backs the versioned feature store. Stores feature tables as `features_v1`, `features_v2`, etc. Queried by customer ID at serving time.

**Why not SQLite:** DuckDB queries Parquet files natively, supports window functions and vectorised execution, and benchmarked at ~4ms median lookup on a 5M-row table — close enough to Redis for batch ML workloads.

**Install:** `uv add duckdb`  
**Docs:** https://duckdb.org

---

### NumPy
**What it is:** Fundamental library for numerical computation in Python.

**What it does in LoyaltyLens (M1):** Generates synthetic loyalty event data using `numpy.random.default_rng(seed=42)` for reproducibility.

**Install:** `uv add numpy`

---

### Pandas
**What it is:** DataFrame library for tabular data manipulation.

**What it does in LoyaltyLens (M1):** Feature computation pipeline — reads Parquet events, computes RFM features, normalises the engagement score.

**Install:** `uv add pandas`

---

### PyArrow
**What it is:** Python bindings for Apache Arrow — the columnar memory format underlying Parquet.

**What it does in LoyaltyLens:** Enables fast Parquet read/write for the event and feature datasets.

**Install:** `uv add pyarrow`

---

## 4. Machine Learning & Model Training

### PyTorch
**What it is:** Meta's open-source deep learning framework. Neural networks are defined as Python classes (`nn.Module`), using dynamic computation graphs.

**What it does in LoyaltyLens (M2):** Defines and trains the `PropensityModel` — a TabTransformer-lite architecture with a `TransformerEncoder` backbone and MLP head. Also used for CLIP inference in M4.

**Architecture used:**
```
Linear(6→64) + LayerNorm
  └─► TransformerEncoder (2 layers, d_model=64, nhead=4)
        └─► MLP: Linear(64→32) → ReLU → Dropout → Linear(32→1) → Sigmoid
```

**Install:** `uv add torch`  
**Docs:** https://pytorch.org

---

### scikit-learn
**What it is:** Classic ML library providing utilities for preprocessing, metrics, and model selection.

**What it does in LoyaltyLens (M2):** `train_test_split` (stratified), `roc_auc_score`, `precision_score`, `recall_score` for model evaluation.

**Install:** `uv add scikit-learn`

---

### MLflow
**What it is:** Open-source platform for tracking ML experiments — logs parameters, metrics, and model artifacts across training runs.

**What it does in LoyaltyLens (M2):** Logs every training run: hyperparameters, loss curves, AUC-ROC, precision, recall per epoch. The Streamlit dashboard in M5 reads from MLflow to display the active model version and val AUC.

**Run the UI:** `uv run mlflow ui --port 5000`  
**Install:** `uv add mlflow`  
**Docs:** https://mlflow.org

---

### ONNX + onnxruntime
**What it is:** Open Neural Network Exchange — a format for exporting trained models so they can be run outside their training framework.

**What it does in LoyaltyLens (M2, stretch):** Exports the trained PropensityModel to `propensity.onnx` for deployment on AWS SageMaker. The ONNX export cuts the SageMaker container image from ~4GB to ~380MB.

**Install:** `uv add onnx onnxruntime`

---

## 5. Vector Databases & Embeddings

### pgvector (Python client)
**What it is:** Python client for the pgvector PostgreSQL extension. Adds the `vector` type and similarity search operators to psycopg2/asyncpg.

**What it does in LoyaltyLens (M3):** Stores 384-dimensional offer embeddings in Postgres. Enables cosine similarity search via the `<=>` operator and IVFFlat indexing.

**Key SQL:**
```sql
CREATE TABLE offer_embeddings (
  id UUID PRIMARY KEY,
  title TEXT,
  embedding vector(384)
);
CREATE INDEX ON offer_embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 20);
```

**Install:** `uv add pgvector psycopg2-binary`  
**Docs:** https://github.com/pgvector/pgvector-python

---

### Weaviate Python Client
**What it is:** Official Python client for the Weaviate vector database.

**What it does in LoyaltyLens (M3):** Stores offer embeddings in Weaviate as a secondary vector store, benchmarked against pgvector. Used in the `LlamaOfferRetriever` path.

**Install:** `uv add weaviate-client`  
**Docs:** https://weaviate.io/developers/weaviate/client-libraries/python

---

### sentence-transformers (HuggingFace)
**What it is:** A library of transformer models fine-tuned to produce semantically meaningful sentence embeddings.

**Model used:** `all-MiniLM-L6-v2` — a 22M-parameter model producing 384-dimensional embeddings. Runs at ~50ms per batch on CPU.

**What it does in LoyaltyLens (M3):** Embeds all 200 offer descriptions into vectors for storage in pgvector and Weaviate. Also used at query time to embed customer context strings.

**Install:** `uv add sentence-transformers`  
**Docs:** https://www.sbert.net

---

### CLIP (openai/clip-vit-base-patch32)
**What it is:** Contrastive Language-Image Pretraining — a multimodal model from OpenAI that embeds images and text into a shared vector space. A ViT-B/32 backbone processes images; a transformer processes text.

**What it does in LoyaltyLens (M4):** Powers the `BrandImageScorer`. Computes cosine similarity between generated offer copy (text embedding) and brand reference images (image embedding). Flags copy with similarity < 0.35 for human review.

**Install:** `uv add transformers` (CLIP is included in the HuggingFace `transformers` library)

---

## 6. LLM Orchestration & RAG

### LangChain
**What it is:** Open-source Python library for building LLM-powered applications. Provides abstractions for chains, retrievers, memory, and agents.

**What it does in LoyaltyLens (M3):** Primary RAG retrieval chain. `PGVector` retriever connects to the offer embedding table; `similarity_search_with_score` retrieves top-k candidates; results are re-ranked by a composite score.

**Key classes used:**
- `langchain_community.vectorstores.PGVector`
- `langchain_community.embeddings.HuggingFaceEmbeddings`

**Install:** `uv add langchain langchain-community`  
**Docs:** https://python.langchain.com

---

### LlamaIndex
**What it is:** Open-source Python library focused on data indexing and retrieval for LLM applications. Provides `VectorStoreIndex`, query engines, and document loaders.

**What it does in LoyaltyLens (M3):** Alternative RAG retrieval path, benchmarked against LangChain. `VectorStoreIndex` over the Weaviate store; `as_retriever()` with similarity top-k.

**Key classes used:**
- `llama_index.core.VectorStoreIndex`
- `llama_index.core.Document`
- `llama_index.vector_stores.postgres.PGVectorStore`

**Install:** `uv add llama-index llama-index-vector-stores-postgres`  
**Docs:** https://docs.llamaindex.ai

---

## 7. LLM Backends

### OpenAI Python SDK
**What it is:** Official Python client for the OpenAI API.

**What it does in LoyaltyLens (M4):** Powers the `OpenAIBackend` — sends structured prompts to `gpt-4o-mini` and returns offer copy JSON. Also powers the LLM-as-judge evaluator in M5.

**Default model:** `gpt-4o-mini` (~$0.03 per full eval run)

**Install:** `uv add openai`  
**Docs:** https://platform.openai.com/docs

---

### HuggingFace Transformers
**What it is:** The core HuggingFace library for downloading, loading, and running pre-trained transformer models locally.

**What it does in LoyaltyLens (M4):** Powers the `HuggingFaceBackend` using `mistralai/Mistral-7B-Instruct-v0.2` via the `text-generation` pipeline. Zero API cost; requires 8GB RAM.

**Install:** `uv add transformers accelerate`  
**Docs:** https://huggingface.co/docs/transformers

---

### HuggingFace Hub
**What it is:** Python client for downloading models and datasets from the HuggingFace model hub.

**What it does in LoyaltyLens:** Downloads `all-MiniLM-L6-v2`, `clip-vit-base-patch32`, and `Mistral-7B-Instruct-v0.2` on first use.

**Install:** `uv add huggingface-hub`

---

## 8. LLMOps & Evaluation

### sacrebleu
**What it is:** A standardised implementation of the BLEU metric for text evaluation, from the ACL conference.

**What it does in LoyaltyLens (M5):** Computes sentence-level BLEU scores in the `OfferCopyEvaluator` eval harness. One component of the aggregate quality score.

**Install:** `uv add sacrebleu`

---

### rouge-score
**What it is:** Google's Python implementation of the ROUGE family of text evaluation metrics.

**What it does in LoyaltyLens (M5):** Computes ROUGE-L (longest common subsequence) scores in the eval harness. Measures recall-oriented text overlap between generated and reference copy.

**Install:** `uv add rouge-score`

---

### PyYAML
**What it is:** Python YAML parser and emitter.

**What it does in LoyaltyLens (M4, M5):** Reads the versioned prompt registry YAML files (`prompts/system_v*.yaml`). Also reads eval criteria embedded in each prompt file.

**Install:** `uv add pyyaml`

---

### Click
**What it is:** Python library for building command-line interfaces with decorators.

**What it does in LoyaltyLens (M5):** Powers the prompt versioning CLI — `llmops prompt list`, `llmops prompt diff v1 v2`, `llmops prompt activate`, `llmops prompt rollback`.

**Install:** `uv add click`  
**Docs:** https://click.palletsprojects.com

---

### Streamlit
**What it is:** Python library for building interactive data dashboards without writing frontend code.

**What it does in LoyaltyLens (M5):** The LLMOps dashboard — displays active model version, eval score trend (line chart), drift PSI with colour-coded status badge, and prompt version history table.

**Run:** `uv run streamlit run llmops/dashboard/app.py`  
**Install:** `uv add streamlit`  
**Docs:** https://docs.streamlit.io

---

## 9. API Layer & Serving

### FastAPI
**What it is:** A modern Python web framework for building APIs. Automatic OpenAPI docs, Pydantic validation, async support.

**What it does in LoyaltyLens:** Every module exposes a FastAPI service:
- M1: `GET /features/{customer_id}`, `GET /features/stats`
- M2: `POST /predict`, `GET /model/info`
- M3: `POST /retrieve`, `GET /offers/stats`
- M4: `POST /generate`, `POST /generate/random`
- M6: `POST /feedback`, `GET /feedback/stats`, `GET /feedback/export`

**Run any service:** `uv run uvicorn feature_store.api:app --port 8001 --reload`  
**Install:** `uv add fastapi uvicorn`  
**Docs:** https://fastapi.tiangolo.com

---

### Uvicorn
**What it is:** A lightning-fast ASGI server — the standard way to serve FastAPI applications.

**Install:** `uv add uvicorn`

---

### Pydantic v2
**What it is:** Data validation library using Python type annotations. The validation layer for all FastAPI request/response models.

**What it does in LoyaltyLens:** Defines and validates `PropensityResult`, `OfferCopy`, `FeedbackRecord`, `DriftReport`, and all API request/response schemas.

**Install:** Included with FastAPI — `uv add fastapi` installs pydantic v2 automatically.  
**Docs:** https://docs.pydantic.dev

---

## 10. Frontend & UI

### React 18
**What it is:** Meta's JavaScript UI library for building component-based web interfaces.

**What it does in LoyaltyLens (M6):** The feedback collection UI — renders generated offer copy (headline, body, CTA), thumbs up/down controls, star rating, submission handler, and a stats dashboard tab.

**Toolchain:** Vite (build), Tailwind CSS (styling), TypeScript.

**Bootstrap:** `npm create vite@latest ui -- --template react-ts`

---

### Vite
**What it is:** A fast build tool and dev server for modern JavaScript/TypeScript projects.

**What it does in LoyaltyLens (M6):** Builds and hot-reloads the feedback UI during development.

**Run:** `cd feedback_loop/ui && npm run dev`

---

### Tailwind CSS
**What it is:** A utility-first CSS framework — styling via class names rather than custom CSS.

**What it does in LoyaltyLens (M6):** All UI layout and visual styling in the feedback app.

---

## 11. CI/CD & DevOps

### GitHub Actions
**What it is:** GitHub's built-in CI/CD platform. Workflows are YAML files in `.github/workflows/`.

**What it does in LoyaltyLens (M5):** Runs the full ML pipeline on every push:
1. `lint` — ruff
2. `type-check` — mypy
3. `unit-tests` — pytest (with Postgres service container)
4. `eval-gate` — runs `run_eval.py`, exits 1 if aggregate score < 0.75
5. `drift-check` — runs `run_drift.py`, posts PSI as PR comment
6. `deploy` — (main branch only) calls `shared/deploy.py`

**Docs:** https://docs.github.com/en/actions

---

### Docker Compose
**What it is:** Tool for defining and running multi-container applications via a `docker-compose.yml` file.

**What it does in LoyaltyLens:** Manages the three infrastructure services (PostgreSQL+pgvector, Weaviate, Redis) as a single unit.

**Key commands:**
```bash
docker-compose up -d      # Start all services in background
docker-compose ps         # Check status
docker-compose down       # Stop and remove containers
docker-compose logs -f    # Tail all service logs
```

---

## 12. Observability & Logging

### structlog
**What it is:** A Python logging library that produces structured (key-value) log output. Far easier to query in production log systems (Datadog, CloudWatch, Grafana Loki) than plain text.

**What it does in LoyaltyLens:** Used in every module as the sole logging interface. All log lines are JSON-structured with automatic context fields (module, function, timestamp).

**Install:** `uv add structlog`  
**Docs:** https://www.structlog.org

---

## 13. Code Quality & Testing

### pytest
**What it is:** The standard Python testing framework.

**What it does in LoyaltyLens:** Unit and integration tests across all six modules. Key test files:
- `tests/test_features.py` — schema validation, null checks, DuckDB round-trip
- `tests/test_propensity.py` — model forward pass, prediction bounds, API contract
- `tests/test_rag.py` — embedding shape, pgvector round-trip, retrieval count
- `tests/test_generator.py` — prompt rendering, JSON parse, CLIP score bounds
- `tests/test_llmops.py` — PSI calculation, eval score bounds, prompt CLI
- `tests/test_feedback.py` — DB insert/read, aggregator stats, trigger logic
- `tests/test_pipeline.py` — end-to-end pipeline integration test

**Run all tests:** `uv run pytest tests/ -v`  
**Install:** `uv add --dev pytest pytest-asyncio`

---

### ruff
**What it is:** An extremely fast Python linter and formatter written in Rust. Replaces flake8, isort, and black in a single tool.

**What it does in LoyaltyLens:** Enforced by the `lint` step in GitHub Actions CI. Configured in `pyproject.toml`.

**Run:** `uv run ruff check .` (lint) / `uv run ruff format .` (format)  
**Install:** `uv add --dev ruff`  
**Docs:** https://docs.astral.sh/ruff/

---

### mypy
**What it is:** Static type checker for Python. Catches type errors at analysis time rather than runtime.

**What it does in LoyaltyLens:** Enforced by the `type-check` step in GitHub Actions CI. All modules use type annotations on public functions.

**Run:** `uv run mypy loyaltylens/ --ignore-missing-imports`  
**Install:** `uv add --dev mypy`

---

## 14. Module-to-Tool Map

| Module | Tools & Libraries |
|---|---|
| **M1 — Feature Pipeline** | Python 3.11, uv, NumPy, Pandas, PyArrow, DuckDB, FastAPI, Uvicorn, Pydantic, structlog |
| **M2 — Propensity Model** | PyTorch, scikit-learn, MLflow, ONNX, onnxruntime, FastAPI, Pydantic, structlog |
| **M3 — RAG Retrieval** | sentence-transformers, pgvector, psycopg2, Weaviate client, LangChain, LlamaIndex, FastAPI, Pydantic, structlog |
| **M4 — LLM Generator** | OpenAI SDK, HuggingFace Transformers, CLIP, PyYAML, FastAPI, Pydantic, structlog |
| **M5 — LLMOps** | sacrebleu, rouge-score, Click, PyYAML, MLflow, Streamlit, GitHub Actions, structlog |
| **M6 — Feedback Loop** | React 18, Vite, Tailwind CSS, FastAPI, SQLite, Pydantic, structlog |
| **All modules** | uv, Python 3.11, pytest, ruff, mypy, Docker Compose, structlog, Pydantic |

---

## 15. Full pyproject.toml Reference

```toml
[project]
name = "loyaltylens"
version = "0.1.0"
description = "Production-grade loyalty offer intelligence platform"
requires-python = ">=3.11"

dependencies = [
    # Data engineering
    "numpy>=1.26",
    "pandas>=2.1",
    "pyarrow>=14.0",
    "duckdb>=0.10",

    # ML / deep learning
    "torch>=2.2",
    "scikit-learn>=1.4",
    "mlflow>=2.11",
    "onnx>=1.16",
    "onnxruntime>=1.17",

    # Embeddings & vector search
    "sentence-transformers>=2.7",
    "pgvector>=0.2",
    "psycopg2-binary>=2.9",
    "weaviate-client>=4.5",

    # LLM orchestration
    "langchain>=0.2",
    "langchain-community>=0.2",
    "llama-index>=0.10",
    "llama-index-vector-stores-postgres>=0.1",

    # LLM backends
    "openai>=1.30",
    "transformers>=4.40",
    "accelerate>=0.30",
    "huggingface-hub>=0.23",

    # LLMOps / eval
    "sacrebleu>=2.4",
    "rouge-score>=0.1",
    "pyyaml>=6.0",
    "click>=8.1",
    "streamlit>=1.35",

    # API & serving
    "fastapi>=0.111",
    "uvicorn>=0.29",
    "pydantic>=2.7",

    # Observability
    "structlog>=24.1",
]

[dependency-groups]
dev = [
    "pytest>=8.2",
    "pytest-asyncio>=0.23",
    "ruff>=0.4",
    "mypy>=1.10",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]

[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true
strict = false
```

> **Note:** This is the root-level `pyproject.toml`. Each module directory also has its own `requirements.txt` listing only that module's dependencies — useful if you want to run a single module in isolation without installing the full stack.

---

*LoyaltyLens — [pushparajan.tech](https://pushparajan.tech) · [LinkedIn](https://linkedin.com/in/pushparajanramar)*
