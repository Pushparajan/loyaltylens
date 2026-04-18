---
title: "LoyaltyLens Setup Guide and Glossary"
slug: "loyaltylens-setup-glossary"
description: "Complete environment setup, mental model, and plain-English glossary for every acronym in the LoyaltyLens series — RAG, PSI, RLHF, LLMOps, and 33 more."
date: 2026-04-23
author: Pushparajan Ramar
series: loyaltylens
series_order: 0
reading_time: 18
tags:
  - getting-started
  - machine-learning
  - llmops
  - mlops
  - python
  - glossary
  - loyaltylens
---

# LoyaltyLens Setup Guide and Glossary

Complete environment setup, mental model, and reference glossary for the series

---

**Series position:** Article 0 of 8 — Start here

---

This article covers three things before you touch any module code:

1. **Environment setup** — everything installed and verified before you run the first module
2. **Mental model** — how the seven modules fit together and why the architecture is designed the way it is
3. **Glossary** — every acronym and concept used across the series, defined in plain English with a note on where each one appears in the codebase

RAG. PSI. RLHF. BFF. pgvector. LLMOps. TabTransformer. The terms accumulate fast across eight articles. Read this once; refer back to the glossary whenever something doesn't land.

---

## Part 1: Environment Setup

### What You Need Before You Start

| Requirement | Minimum Version | Why |
| --- | --- | --- |
| Python | 3.11+ | Match statements, tomllib, `dict \| dict` syntax |
| uv | 0.4+ | Python package manager — replaces pip, Poetry, and pyenv in one tool |
| Node.js | 18+ | Feedback loop React UI (Vite) |
| Docker Desktop | 4.x | Postgres + pgvector + Weaviate services |
| Git | 2.x | Prompt versioning uses git-tracked YAML |
| 8 GB RAM | — | Running a local HuggingFace model + Weaviate concurrently |
| Claude Code | latest | Primary agentic coding tool for the project |

---

### Step 1: Install uv and Bootstrap the Project

`uv` is a Rust-based Python package manager from Astral that replaces `pip`, `pip-tools`, `pyenv`, `virtualenv`, and `poetry` in a single binary. It installs packages 10–100x faster than pip, manages Python versions itself, and works with the standard `pyproject.toml` format you already know.

#### Install uv

```bash
# macOS / Linux (recommended — installs a standalone binary, no Python required)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternatively, via pip if you already have Python
pip install uv

# Verify
uv --version
# → uv 0.4.x (or later)
```

After install, `uv` is available as a standalone command. It manages its own Python downloads — you do not need a separate `pyenv` or `python.org` installer.

#### Pin your Python version

```bash
# Install Python 3.11 and pin it for this project
uv python install 3.11
uv python pin 3.11
# → Creates .python-version file — commit this to git
```

#### Create project directory and virtual environment

```bash
mkdir loyaltylens && cd loyaltylens
git init

# Always specify --python explicitly — uv will otherwise pick whatever
# CPython version it finds first, which may be 3.12+ and not 3.11
uv venv .venv --python 3.11
# → Creates .venv/ in the project root
# → Output: Using Python 3.11.x interpreter at ...
```

Activate the virtual environment:

```bash
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows (PowerShell) — use absolute path; prompt will show (loyaltylens)
& C:\Projects\loyaltylens\.venv\Scripts\Activate.ps1
```

> **Windows execution policy:** If PowerShell blocks the activation script, run this once:
>
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

After running the Module bootstrap Claude Code prompt (covered in Article 1), your directory will have a `pyproject.toml`. Install everything in one command:

```bash
# Install all project dependencies (production + dev) into the active venv
uv sync --dev

# Add a new package
uv add langchain

# Remove a package
uv remove somepackage
```

> **One venv to rule them all.** The repo root `.venv/` is the single canonical virtual environment. Sub-module folders (`data_pipeline/`, `propensity_model/`) may have their own `.venv/` from earlier experimentation — ignore them. Always activate the root venv and use `uv sync --dev` or `uv add <package>` to manage dependencies.

#### Install dev tools

`pytest`, `ruff`, and `mypy` are listed under `[dependency-groups] dev` in `pyproject.toml`. `uv sync --dev` installs them automatically. To add a new dev dependency:

```bash
uv add --dev pytest-xdist   # example
```

#### Running tests

Use `python -m pytest` rather than bare `pytest` — it always resolves to the active venv's interpreter:

```bash
python -m pytest tests/
```

#### Running FastAPI services

Use `python -m uvicorn` rather than bare `uvicorn` — on Windows, venv scripts are not on `PATH` unless the venv is activated:

```bash
# Reliable on all platforms
python -m uvicorn rag_retrieval.api:app --host 127.0.0.1 --port 8010 --reload --reload-dir rag_retrieval
```

> **Windows port note:** Binding to `0.0.0.0` triggers a Windows firewall permission error (`WinError 10013`). Use `127.0.0.1` for local development. To expose the service on the LAN, run PowerShell as Administrator and allow the port through the firewall first.

#### Key uv commands at a glance

```bash
uv sync --dev            # Install all deps (production + dev) from lockfile
uv pip install <pkg>     # Install a package into active venv
uv add <pkg>             # Add package to pyproject.toml and install
uv remove <pkg>          # Remove a package
uv run python script.py  # Run a script through the project venv (no activate needed)
uv python list           # Show all available Python versions
uv python install 3.11   # Download and install a specific Python version
```

#### Why uv instead of Poetry?

Three practical reasons: `uv sync` is 10–100x faster than `poetry install` on a cold cache, `uv` manages Python versions itself (no separate `pyenv`), and it produces a standard `uv.lock` that is fully reproducible across machines and CI environments without extra configuration. The `pyproject.toml` format is identical — if you have an existing Poetry project, `uv` reads it without changes.

---

### Step 2: Start Infrastructure Services

LoyaltyLens uses three infrastructure services managed via Docker Compose. **Docker Desktop must be running** before executing any `docker compose` command — open it from the Start menu (Windows) or Applications (macOS) and wait until the system tray icon says "Docker Desktop is running" (30–60 seconds on first launch). Verify with `docker info`.

```yaml
# docker-compose.yml (at repo root)
services:

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: loyaltylens
      POSTGRES_PASSWORD: loyaltylens
      POSTGRES_DB: loyaltylens
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  weaviate:
    image: semitechnologies/weaviate:1.28.2   # ≥ 1.27.0 required for the v4 Python client
    ports:
      - "8080:8080"
      - "50051:50051"                          # gRPC port — required by weaviate-client v4
    environment:
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: "true"
      DEFAULT_VECTORIZER_MODULE: none

  redis:
    image: redis:7.2-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
  weaviate_data:
  redis_data:
```

> **Weaviate version matters.** The `weaviate-client` Python library v4 requires Weaviate server **≥ 1.27.0** and also needs the gRPC port `50051` exposed. Using `1.25.x` produces a `WeaviateStartUpError` at connection time. The repo ships `1.28.2`.

Start all services:

```powershell
docker compose up -d postgres weaviate redis

# Confirm all three show (healthy)
docker compose ps
```

#### Automatic pgvector setup

The repo ships `docker/init-pgvector.sql`, which Docker runs automatically on first container start via `docker-entrypoint-initdb.d`. It enables the `vector` extension — **no manual `CREATE EXTENSION` step is needed**.

#### Apply the database schema

```powershell
# Windows (PowerShell — the < operator is not supported natively)
Get-Content db/schema.sql | docker exec -i loyaltylens_postgres psql -U loyaltylens -d loyaltylens

# macOS / Linux
docker exec -i loyaltylens_postgres psql -U loyaltylens -d loyaltylens < db/schema.sql
```

#### Restarting Docker from scratch

When you need a completely clean state — empty databases, fresh volumes:

```powershell
# Stop containers and delete all data volumes
docker compose down -v

# Start fresh
docker compose up -d postgres weaviate redis

# Re-run the data and embedding pipelines
python rag_retrieval/generate_offers.py
python rag_retrieval/embeddings.py
```

> Omit `-v` if you just want to restart containers without losing stored data.

---

### Step 3: Environment Variables

**Single source of truth:** one `.env` file at the **repo root**. No module-level `.env` files — all modules import `shared.config.get_settings()`, which reads the root `.env`.

```powershell
# Windows
Copy-Item .env.example .env

# macOS / Linux
cp .env.example .env
```

Fill in your keys. The full `.env.example` template:

```dotenv
# ── External APIs ──────────────────────────────────────────────────────────
OPENAI_API_KEY=sk-...          # Required for LLM generation (M4) and LLM judge (M5)
HF_TOKEN=hf_...                # Optional — needed for gated HuggingFace models
SAGEMAKER_ENDPOINT=            # Leave blank unless deploying to AWS

# ── Infrastructure: connection URLs ────────────────────────────────────────
POSTGRES_URL=postgresql://loyaltylens:loyaltylens@localhost:5432/loyaltylens
WEAVIATE_URL=http://localhost:8080
REDIS_URL=redis://localhost:6379

# ── Infrastructure: individual ports ───────────────────────────────────────
# All modules read these from shared.config.Settings — change once, applies everywhere.
POSTGRES_PORT=5432
WEAVIATE_HTTP_PORT=8080
WEAVIATE_GRPC_PORT=50051
REDIS_PORT=6379
MLFLOW_PORT=5000

# ── Service API ports ───────────────────────────────────────────────────────
PORT_FEATURE_STORE=8001
PORT_PROPENSITY=8002
PORT_RAG_RETRIEVAL=8003
PORT_LLM_GENERATOR=8004
PORT_FEEDBACK_LOOP=8005
PORT_METRICS=8006

# ── MLflow ──────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=loyaltylens

# ── Local data paths ────────────────────────────────────────────────────────
RAW_EVENTS_PATH=data/raw/events.parquet
PROCESSED_FEATURES_PATH=data/processed/features.parquet
DUCKDB_PATH=data/feature_store.duckdb

# ── Propensity model ────────────────────────────────────────────────────────
PROPENSITY_MODEL_VERSION=1
PROPENSITY_MODELS_DIR=models

# ── Pipeline tunables ───────────────────────────────────────────────────────
BATCH_SIZE=512
EVAL_PASS_THRESHOLD=0.75
```

#### Load `.env` in a PowerShell session

PowerShell does not auto-load `.env` files. Run this to export all keys into the current session:

```powershell
Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), 'Process')
    }
}
```

#### Creating a HuggingFace token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and sign in (or create a free account).
2. Click **New token**, set **Role** to **Read**, and click **Generate a token**.
3. Copy the token — it starts with `hf_` and is shown only once.
4. Paste it as `HF_TOKEN=hf_...` in your `.env` file.

The token is optional for `all-MiniLM-L6-v2` (public model). It is required for gated models such as Llama or Mistral.

**API cost note:** The default LLM backend is `gpt-4o-mini`. Running the full eval harness (50 generations + 50 LLM judge calls) costs approximately **$0.03 per run**. End to end from bootstrap to a working dashboard costs under $1 in API calls. For zero API cost, the HuggingFace backend with Mistral-7B-Instruct runs locally — requires the 8 GB RAM headroom.

---

### Step 4: Install MLflow

MLflow tracks experiments for the propensity model (Module 2). It runs as a local server:

```bash
# Already installed via: uv sync --dev
# Start the tracking server
python -m mlflow ui --port 5000
# → Open http://localhost:5000 to see the experiment dashboard
```

---

### Step 5: Verify the Full Stack

Run this sanity check after completing steps 1–4:

```python
python -c "
import duckdb, torch, langchain, weaviate
print('DuckDB:', duckdb.__version__)
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('LangChain:', langchain.__version__)
print('All imports OK')
"
```

If everything passes, your environment is ready. If `torch.cuda.is_available()` returns `False`, that's fine — LoyaltyLens runs on CPU. Training Module 2 takes ~3 minutes on CPU vs. ~20 seconds on GPU.

Run the pipeline smoke test to confirm the full ETL flow works end-to-end:

```bash
python run_pipeline.py
# Pipeline result: {'transactions': 2, 'customers': 2}
```

Verify the rows landed in Postgres:

```powershell
# Windows PowerShell (no psql required — uses the container)
docker exec loyaltylens_postgres psql -U loyaltylens -d loyaltylens `
  -c "SELECT COUNT(*) FROM transactions; SELECT COUNT(*) FROM customers;"

# macOS / Linux (if psql is on PATH)
psql postgresql://loyaltylens:loyaltylens@localhost:5432/loyaltylens \
  -c "SELECT COUNT(*) FROM transactions; SELECT COUNT(*) FROM customers;"
```

---

### Step 6: Install Claude Code

Claude Code is the agentic coding tool used throughout this series. It reads your entire codebase, understands context across files, and generates multi-file implementations from a single natural-language prompt.

```bash
npm install -g @anthropic-ai/claude-code

# Authenticate
claude login

# Verify
claude --version
```

The prompts in each article target Claude Code specifically — they require full project directory visibility and multi-file generation in a single command. Running them in a web chat interface or basic autocomplete tool will not produce equivalent results.

---

### Service Port Reference

All ports are centralised in `shared/config.py` and read from the root `.env`. Change a port once — it propagates everywhere.

| Service | Default Port | Env Var | Command |
| --- | --- | --- | --- |
| PostgreSQL / pgvector | 5432 | `POSTGRES_PORT` | Docker Compose |
| Weaviate HTTP | 8080 | `WEAVIATE_HTTP_PORT` | Docker Compose |
| Weaviate gRPC | 50051 | `WEAVIATE_GRPC_PORT` | Docker Compose |
| Redis | 6379 | `REDIS_PORT` | Docker Compose |
| MLflow UI | 5000 | `MLFLOW_PORT` | `python -m mlflow ui` |
| Feature Store API | 8001 | `PORT_FEATURE_STORE` | `python -m uvicorn feature_store.api:app --port 8001` |
| Propensity API | 8002 | `PORT_PROPENSITY` | `python -m uvicorn propensity_model.api:app --port 8002` |
| RAG Retrieval API | 8003 | `PORT_RAG_RETRIEVAL` | `python -m uvicorn rag_retrieval.api:app --port 8003` |
| LLM Generator API | 8004 | `PORT_LLM_GENERATOR` | `python -m uvicorn llm_generator.api:app --port 8004` |
| Feedback Loop API | 8005 | `PORT_FEEDBACK_LOOP` | `python -m uvicorn feedback_loop.api:app --port 8005` |
| Prometheus Metrics | 8006 | `PORT_METRICS` | Started by `MetricsCollector.start_server()` |

---

## Part 2: The Mental Model

Before the glossary, it helps to understand how all the pieces relate. Here's the one-paragraph version:

> LoyaltyLens ingests customer behavioral events, computes features, uses those features to score how likely a customer is to redeem an offer (propensity), retrieves the most semantically relevant offer from a catalog using vector search, generates personalized copy for that offer using an LLM, ships it through a quality gate that checks prompt version, copy quality, and drift, and captures feedback signals that can retrigger the whole cycle if quality degrades.

Everything in the system exists to serve that sentence. Every module is one clause.

### The Data Flow at a Glance

```text
Raw Events (Parquet)
  │
  ▼
[M1] Feature Pipeline
  │  recency, frequency, monetary, engagement_score
  ▼
[M2] Propensity Model (PyTorch)
  │  propensity_score: 0.0 → 1.0
  ▼
[M3] RAG Retrieval (LangChain / LlamaIndex + pgvector)
  │  top-5 relevant offers from catalog
  ▼
[M4] LLM Copy Generator (HuggingFace / OpenAI)
  │  headline, body, CTA, tone + brand alignment score
  ▼
[M5] LLMOps Pipeline (eval gate, drift monitor, CI/CD)
  │  quality gate: passes or blocks deployment
  ▼
[M6] Feedback Loop (UI + aggregator + retraining trigger)
  │  signals flow back to M2 (retrain) and M5 (prompt rollback)
  ▼
[M7] Integration & Cloud Deployment (shared/pipeline.py + deploy/)
  └─► end-to-end pipeline orchestration; SageMaker / Vertex AI endpoint
```

### Why Each Module Is Separate

Each module is a separate, independently deployable unit. In production systems, these layers are owned by different teams, deploy on different cadences, and have different scaling requirements:

- The feature pipeline runs on a batch schedule (hourly)
- The propensity model retrains weekly, or on drift trigger
- The offer catalog is updated by the marketing team daily
- The LLM copy generator version-gates on prompt approval
- The LLMOps monitors run continuously
- The feedback loop operates asynchronously

This module boundary reflects operational reality, not just organizational convention — it's what makes the architecture useful as a reference rather than a monolithic script.

---

## Part 3: The Glossary

Listed alphabetically. Each entry includes a plain-English definition and a note on where it appears in the LoyaltyLens project.

---

### A

**A/B Testing**
Running two variants of a system (model A vs. model B, prompt v1 vs. prompt v2) on different slices of traffic simultaneously, then comparing outcomes. In LoyaltyLens: the `/retrieve` endpoint accepts a `retriever` parameter, enabling A/B testing between LangChain and LlamaIndex without a code change. In enterprise deployments: used to test offer copy variants before full campaign rollout.

**CDP (Customer Data Platform)**
An enterprise platform that ingests, unifies, and activates customer data across channels. Handles identity resolution (linking the same customer across mobile, web, and in-store touchpoints). Used in production for real-time segmentation and audience activation. LoyaltyLens's DuckDB feature store is the local analogue of a CDP feature pipeline.

**Agentic AI**
AI systems that can take sequences of actions autonomously — calling tools, browsing the web, writing and executing code — to achieve a goal, rather than just answering a single question. Claude Code is an agentic system. In LoyaltyLens: the `LoyaltyLensPipeline.run_for_customer()` method chains five model/API calls autonomously.

**Apollo GraphQL**
A popular GraphQL implementation used as a BFF (see below) layer. In production: Apollo GraphQL orchestrates API calls across the web and native mobile apps. In LoyaltyLens: not implemented directly, but the FastAPI endpoints are designed to be consumed by a BFF layer.

**AUC-ROC (Area Under the Receiver Operating Characteristic Curve)**
A metric for binary classification models ranging from 0.5 (random) to 1.0 (perfect). Measures how well the model separates positive and negative classes across all decision thresholds. In LoyaltyLens (M2): the propensity model targets val AUC > 0.80. In production systems: typical AUC is 0.85–0.88 with richer feature sets and longer history windows.

---

### B

**BFF (Backend for Frontend)**
An API layer that aggregates multiple backend service calls into a single, frontend-optimized response. Prevents the frontend from making 8 separate API calls to render one screen. In production: Apollo GraphQL serves as the BFF, orchestrating feature store, propensity model, offer catalog, and content APIs. In LoyaltyLens: the `LoyaltyLensPipeline` class plays this role for the feedback UI.

**BLEU (Bilingual Evaluation Understudy)**
A metric that measures the overlap between generated text and reference text by comparing n-gram matches. Originally designed for machine translation; used in LoyaltyLens (M5) as one component of the automated eval harness. Ranges from 0 (no overlap) to 1 (identical). Limitation: measures lexical similarity, not semantic quality — a paraphrase of the reference will score poorly even if it's better copy.

**BM25**
A ranking function for keyword-based search — the algorithm behind most traditional search engines. BM25 and dense vector search are complementary: BM25 is good at exact keyword matches ("double points offer"), dense search is good at semantic matches ("reward for purchases"). Combining them is called hybrid retrieval. Mentioned as a stretch goal in LoyaltyLens M3.

---

### C

**CI/CD (Continuous Integration / Continuous Deployment)**
The practice of automatically running tests, quality checks, and deployment steps every time code is pushed. In LoyaltyLens (M5): the GitHub Actions pipeline runs lint → type-check → unit tests → eval gate → drift check → deploy on every push to main. In production: every ML model promotion goes through a CI/CD pipeline with responsible AI compliance checkpoints.

**CLIP (Contrastive Language-Image Pretraining)**
A neural network from OpenAI that learns to align image and text representations in a shared embedding space. Given an image and a text, CLIP can score how well they match. In LoyaltyLens (M4): used as the `BrandImageScorer` to check whether generated offer copy aligns with the loyalty brand imagery — the local analogue of what a generative AI content platform does at enterprise scale.

**CNN (Convolutional Neural Network)**
A deep learning architecture originally designed for image recognition. Uses convolutional filters that detect local patterns (edges, textures, shapes) and compose them hierarchically. In LoyaltyLens (M4): the CLIP model uses a vision transformer (ViT) backbone rather than a classic CNN, but the conceptual lineage is the same — learning visual features from raw pixels.

**Cosine Similarity**
A measure of the angle between two vectors, ranging from -1 (opposite directions) to 1 (identical direction). Used in vector search to find the offer embedding closest to a query embedding. In LoyaltyLens (M3): pgvector's `<=>` operator and Weaviate's `nearVector` both use cosine similarity by default.

---

### D

**Databricks**
A cloud data engineering platform built on Apache Spark. Handles large-scale batch and streaming data processing. In production: used for the feature engineering pipeline and ML training jobs. In LoyaltyLens: DuckDB is the local analogue — same concepts (versioned tables, SQL queries, Parquet I/O), dramatically simpler infrastructure.

**a production loyalty AI platform**
A major loyalty program's internal AI and machine learning platform powering personalization, offer recommendations, inventory prediction, and operational efficiency. The LoyaltyLens project is a clean-room, open-source reproduction of the offer intelligence component — specifically the propensity scoring, RAG retrieval, and LLMOps layers.

**Drift (Model Drift / Data Drift)**
The phenomenon where a model's performance degrades over time because the real-world data it's receiving no longer matches the distribution it was trained on. Two types: *data drift* (input features change) and *concept drift* (the relationship between features and labels changes). In LoyaltyLens (M5): the `PropensityDriftMonitor` uses PSI to detect data drift in propensity score distributions.

**DuckDB**
An embedded analytical database that runs in-process (no separate server). Designed for analytical queries on Parquet, CSV, and JSON files — effectively "SQLite for data analytics." In LoyaltyLens (M1): used as the feature store backend. Chosen for its zero-dependency setup, fast analytical performance, and Parquet-native query capability.

---

### E

**Embedding**
A dense numerical vector representation of a piece of data (text, image, audio) that captures semantic meaning. Similar items have embeddings that are close together in vector space. In LoyaltyLens (M3): offer descriptions are embedded using `all-MiniLM-L6-v2` into 384-dimensional vectors and stored in pgvector and Weaviate.

**Eval Harness**
An automated system that runs a language model on a set of test inputs and scores the outputs against a quality rubric. Prevents prompt or model changes from degrading output quality undetected. In LoyaltyLens (M5): the `OfferCopyEvaluator` runs 50 generations per eval, scores on BLEU + ROUGE + LLM-judge, and exits with code 1 if aggregate score < 0.75 (blocking the CI/CD deploy).

---

### F

**FastAPI**
A modern Python web framework for building APIs. Faster than Flask, with automatic OpenAPI documentation, Pydantic data validation, and async support. Used across all six LoyaltyLens modules as the HTTP serving layer.

**Feature Store**
A system for storing, versioning, and serving ML features — the computed inputs to a machine learning model. Ensures the same feature computation logic is used at training time and inference time, preventing training-serving skew. In LoyaltyLens (M1): implemented with DuckDB. In production: the feature store is backed by Azure Databricks + Redis for real-time serving.

**Fine-tuning**
The process of continuing to train a pre-trained model on a domain-specific dataset to improve its performance for a specific task. Cheaper than training from scratch. In LoyaltyLens: the preference dataset exported in M6 is in the format needed for fine-tuning an LLM via the OpenAI fine-tuning API or HuggingFace PEFT.

---

### G

**GitHub Actions**
GitHub's built-in CI/CD system. Workflows are defined as YAML files in `.github/workflows/`. In LoyaltyLens (M5): the full ML pipeline — lint, test, eval gate, drift check, deploy — runs as a GitHub Actions workflow on every push.

**GPT (Generative Pre-trained Transformer)**
OpenAI's family of large language models. GPT-4o-mini is used as the default LLM backend in LoyaltyLens M4, and as the LLM judge in M5. "Pre-trained" means trained on a massive corpus before fine-tuning; "transformer" refers to the underlying architecture (see Transformer).

---

### H

**HuggingFace**
A platform and open-source library that hosts thousands of pre-trained ML models and provides a consistent API for downloading and running them. In LoyaltyLens: used for the sentence-transformer embedding model (M3), the Mistral-7B-Instruct local LLM backend (M4), and the CLIP model (M4). Think of it as the npm of the ML world.

**Hybrid Retrieval**
Combining keyword-based search (BM25) with semantic vector search and merging the results using reciprocal rank fusion. Gets the best of both approaches: precision on exact terms, recall on semantic meaning. Mentioned as a stretch goal in LoyaltyLens M3. Used in production offer retrieval systems at scale.

---

### I

**IVFFlat (Inverted File Flat)**
A pgvector index type that divides the vector space into `lists` clusters (like Voronoi cells) and searches only the nearest clusters at query time. Makes approximate nearest neighbor search fast on large datasets. In LoyaltyLens (M3): `CREATE INDEX USING ivfflat (embedding vector_cosine_ops) WITH (lists = 20)`.

---

### L

**LangChain**
An open-source Python library for building applications with LLMs. Provides abstractions for chains (sequences of LLM calls), retrievers (vector search integration), agents, and memory. In LoyaltyLens (M3): used to build the primary RAG retrieval chain connecting pgvector to the offer ranking logic.

**LlamaIndex**
An open-source Python library focused on data indexing and retrieval for LLM applications. Provides `VectorStoreIndex`, query engines, and document loaders. Competing abstraction to LangChain. In LoyaltyLens (M3): used as an alternative retrieval path, benchmarked against LangChain on latency and precision@5.

**LLM (Large Language Model)**
A neural network trained on vast amounts of text data, capable of generating, summarizing, translating, and reasoning about text. Examples: GPT-4o, Claude, Mistral-7B. In LoyaltyLens (M4): the LLM generates personalized offer copy from a structured prompt template.

**LLMOps**
The operational discipline of managing large language models in production — including prompt versioning, model evaluation, deployment pipelines, and monitoring. The LLM-specific extension of MLOps. In LoyaltyLens: Module 5 is entirely devoted to LLMOps infrastructure. In production: implemented as the core of the loyalty AI pipeline.

---

### M

**MLflow**
An open-source platform for tracking ML experiments — logging parameters, metrics, and artifacts across training runs. In LoyaltyLens (M2): logs loss curves, AUC-ROC, precision, and recall for every propensity model training run.

**MLOps (Machine Learning Operations)**
The set of practices that bring ML models from experimentation to reliable production operation — covering training pipelines, model versioning, deployment, monitoring, and retraining. The ML extension of DevOps. LoyaltyLens is a full MLOps reference implementation.

**Multimodal**
An AI system that processes and/or generates more than one type of data (text, images, audio, video). In LoyaltyLens (M4): the `BrandImageScorer` combines a text embedding (offer copy) and an image embedding (brand reference image) using CLIP — a multimodal model. In production: generative AI models combine visual brand assets with text generation.

---

### P

**PEFT (Parameter-Efficient Fine-Tuning)**
A family of techniques for fine-tuning large models by only training a small number of additional parameters, rather than updating the full model weights. LoRA is the most popular PEFT method. Dramatically reduces the compute cost of fine-tuning. Relevant to LoyaltyLens as the downstream use of the preference dataset exported in M6.

**pgvector**
A Postgres extension that adds a native `vector` data type and similarity search operators (`<->` for L2, `<=>` for cosine, `<#>` for inner product). Lets you do vector search in the same database as your relational data — no separate vector DB service required. In LoyaltyLens (M3): used as the primary vector store. In production: used alongside Databricks for the offer embedding retrieval layer.

**Pinecone**
A managed vector database service — fully hosted, serverless, with horizontal scaling. No infrastructure to manage. In LoyaltyLens: used as a conceptual reference (Weaviate is the self-hosted analogue). Mentioned in the vector DB decision matrix.

**Precision@K**
A retrieval quality metric: of the top K results returned, what fraction are actually relevant? In LoyaltyLens (M3): measured at K=5 using a category-relevance heuristic. LangChain and LlamaIndex both achieve precision@5 ≈ 0.71–0.73 on the synthetic offer catalog.

**Prompt Engineering**
The practice of designing input text (prompts) to get desired outputs from an LLM. Includes system prompt design, few-shot examples, chain-of-thought instructions, and output format specification. In LoyaltyLens (M4): the `prompts/system_v*.yaml` files are the versioned, engineered prompts. In production: treat every prompt as a production artifact with the same rigor as code.

**Propensity Score**
A probability estimate (0–1) that a specific customer will take a specific action (redeem an offer, make a purchase, churn). The core output of a propensity model. In LoyaltyLens (M2): the `PropensityModel` outputs a propensity score that gates which offers are shown in M3 retrieval.

**PSI (Population Stability Index)**
A statistical measure of how much the distribution of a variable has shifted between two time periods. PSI < 0.1 = stable, 0.1–0.2 = minor shift (investigate), > 0.2 = significant shift (trigger retraining). In LoyaltyLens (M5): the `PropensityDriftMonitor` computes PSI nightly. In production: a PSI alert caught a timezone-offset bug in the feature pipeline that would have degraded campaign quality for weeks.

**PyTorch**
Meta's open-source deep learning framework. Defines neural networks as Python classes, uses dynamic computation graphs, and is the standard framework for research and production model development. In LoyaltyLens (M2): the `PropensityModel` is a `torch.nn.Module`.

---

### R

**RAG (Retrieval-Augmented Generation)**
An architecture that enhances LLM output by first retrieving relevant documents from a knowledge base, then passing those documents to the LLM as context. Prevents hallucination by grounding the LLM in retrieved facts. In LoyaltyLens (M3): customer context + propensity score → vector search → top-5 relevant offers → LLM copy generation in M4.

**RFM (Recency, Frequency, Monetary)**
A classic customer segmentation framework: how recently a customer purchased, how often they purchase, and how much they spend. In LoyaltyLens (M1): `recency_days`, `frequency_30d`, and `monetary_90d` are the three RFM features, combined with offer redemption rate, channel preference, and engagement score.

**RLHF (Reinforcement Learning from Human Feedback)**
A training technique where human raters evaluate model outputs (preferred vs. rejected), a reward model is trained on those ratings, and the base LLM is fine-tuned using reinforcement learning to maximize the reward. Used by OpenAI to train ChatGPT and by Anthropic to train Claude. In LoyaltyLens (M6): a simplified feedback collection system that captures preference signals and exports them in RLHF training format, without the full RL training loop.

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**
A set of metrics for evaluating text summarization and generation by measuring overlap between generated and reference text. ROUGE-L measures the longest common subsequence. In LoyaltyLens (M5): used alongside BLEU as part of the automated eval harness.

---

### S

**SageMaker**
AWS's fully managed ML platform. Handles model training (distributed, GPU-accelerated), model registry, endpoint deployment, and monitoring. In LoyaltyLens (M7): `deploy/sagemaker_deploy.py` exports the propensity model to TorchScript, packages it as `model.tar.gz`, deploys a real-time endpoint on `ml.t2.medium`, and exposes `invoke`/`teardown` CLI actions. The SageMaker PyTorch container requires TorchScript format (`torch.jit.trace`) — plain state-dict checkpoints fail at container startup.

**Sentence Transformers**
A family of transformer models fine-tuned to produce semantically meaningful sentence embeddings. `all-MiniLM-L6-v2` (used in LoyaltyLens M3) is a 22M parameter model producing 384-dim embeddings at ~50ms per batch on CPU. Much faster than using a large LLM for embeddings.

**Sigmoid**
A mathematical function that maps any real number to the range (0, 1). Used as the final activation in binary classification models to output a probability. In LoyaltyLens (M2): the last layer of `PropensityModel` is `nn.Sigmoid()`, squashing the raw logit to a propensity probability.

**Structlog**
A Python logging library that produces structured (key-value) log output — much easier to query and monitor in production log systems (Datadog, CloudWatch) than plain text logs. Used throughout LoyaltyLens for all logging.

---

### T

**TabTransformer**
A transformer architecture adapted for tabular (structured/relational) data. Uses attention mechanisms over feature columns rather than tokens. In LoyaltyLens (M2): `TabTransformer-lite` — a simplified version with a linear projection into a 2-layer TransformerEncoder. Chosen over XGBoost for composability with the multimodal extension and attention-based interpretability.

**Transformer**
The neural network architecture that underpins virtually all modern LLMs. Introduced in the paper "Attention Is All You Need" (Vaswani et al., 2017). The key mechanism is self-attention: each token in a sequence attends to every other token, capturing long-range dependencies. GPT, BERT, Claude, Mistral — all transformers.

---

### V

**Vector Database**
A database optimized for storing and querying high-dimensional vectors (embeddings). Uses approximate nearest neighbor algorithms (IVFFlat, HNSW) to find similar vectors faster than an exact linear scan. In LoyaltyLens (M3): pgvector (Postgres extension) and Weaviate (standalone service) are both implemented and benchmarked.

**uv**
A Rust-based Python package manager from Astral that replaces pip, pip-tools, pyenv, virtualenv, and poetry in a single binary. Key properties: installs packages 10–100x faster than pip, manages Python versions itself (no separate pyenv needed), produces a reproducible `uv.lock` lockfile, and works with standard `pyproject.toml`. In LoyaltyLens: used as the sole package manager — `uv sync --dev` installs all dependencies, `uv add` adds new ones.

---

**Vertex AI**
Google Cloud's fully managed ML platform. Analogous to AWS SageMaker. Includes: training pipelines, model registry, online prediction endpoints, and a managed Vector Search service (HNSW-backed). In LoyaltyLens (M7): `deploy/vertex_deploy.py` deploys the propensity model to a Vertex AI online prediction endpoint and indexes offer embeddings in Vertex AI Vector Search.

---

### W

**Weaviate**
An open-source vector database with a GraphQL/REST API, native support for multi-tenancy, and built-in vectorizer modules. Can be self-hosted or used as a managed service. In LoyaltyLens (M3): used as the secondary vector store alongside pgvector, benchmarked on latency and compared on operational complexity. **Version ≥ 1.27.0 is required** for the v4 Python client.

---

## Quick Reference: Where Each Technology Appears

| Technology | Module | Role |
| --- | --- | --- |
| DuckDB | M1 | Feature store backend |
| NumPy / Pandas | M1 | Event generation + feature computation |
| PyTorch | M2 | PropensityModel training and inference |
| MLflow | M2 | Experiment tracking |
| HuggingFace sentence-transformers | M3 | Offer embedding generation |
| pgvector | M3 | Primary vector store |
| Weaviate | M3 | Secondary vector store (benchmark) |
| LangChain | M3 | Primary RAG retrieval chain |
| LlamaIndex | M3 | Alternative RAG retrieval chain |
| OpenAI API / Mistral-7B | M4 | LLM offer copy generation |
| CLIP (ViT-B/32) | M4 | Brand image alignment scoring |
| YAML prompt registry | M4, M5 | Versioned prompt storage |
| sacrebleu / rouge-score | M5 | Lexical eval metrics |
| PSI monitor | M5 | Propensity drift detection |
| GitHub Actions | M5 | CI/CD for ML pipeline |
| Streamlit | M5 | LLMOps dashboard |
| React + Vite | M6 | Feedback collection UI |
| SQLite | M6 | Feedback persistence |
| Preference dataset (JSONL) | M6 | RLHF training data export |
| shared/pipeline.py | M7 | End-to-end pipeline orchestration (all 6 modules) |
| torch.jit.trace (TorchScript) | M7 | Model export for SageMaker serving container |
| boto3 | M7 | AWS SageMaker + S3 API client |
| google-cloud-aiplatform | M7 | GCP Vertex AI client |
| uv | All | Python package manager (replaces pip + Poetry + pyenv) |
| FastAPI | All | HTTP API serving |
| structlog | All | Structured logging |
| Pydantic | All | Data validation |
| pytest | All | Unit and integration tests |

---

## What to Read Next

Now that the environment is set up and the vocabulary is clear, the series builds module by module:

- **Article 1** — Feature pipeline, DuckDB feature store, validation
- **Article 2** — Propensity model, PyTorch, model card, TorchScript export
- **Article 3** — RAG retrieval, pgvector vs. Weaviate benchmark, LangChain vs. LlamaIndex
- **Article 4** — LLM copy generation, prompt versioning, CLIP brand alignment
- **Article 5** — LLMOps pipeline, drift monitoring, CI/CD eval gate
- **Article 6** — RLHF feedback loop, preference datasets, retraining trigger
- **Article 7** — Integration layer, end-to-end pipeline, SageMaker + Vertex AI cloud deployment
- **Article 8** — Recap: five lessons, what to build next

Each article is self-contained. The code builds on the previous module but each post explains its inputs and outputs clearly enough that you can start anywhere. If you get stuck on terminology, come back here.

→ Start with Article 1: How I Rebuilt a Loyalty Platform's Feature Pipeline in a Weekend

---

*Pushparajan Ramar is an Enterprise Architect Director in enterprise consulting, leading AI strategy and delivery for Fortune 500 enterprises. He is based in the Greater Chicago Area.*

*[LinkedIn](https://linkedin.com/in/pushparajanramar) · [pushparajan.tech](https://pushparajan.tech)*
