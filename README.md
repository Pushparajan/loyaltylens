# LoyaltyLens

ML-powered loyalty analytics platform that combines propensity modelling with RAG-driven personalisation to predict customer churn, recommend retention actions, and generate personalised communications at scale.

## Architecture

```text
loyaltylens/
├── data_pipeline/      # ETL: ingest transactions & customer data → Postgres
├── feature_store/      # Compute & serve ML features via Redis + Postgres
├── propensity_model/   # Train & score churn/upsell propensity (XGBoost)
├── rag_retrieval/      # Index loyalty docs in Weaviate; retrieve context
├── llm_generator/      # Generate personalised offers/comms via LLM
├── llmops/             # Track LLM calls, evaluate quality, expose metrics
├── feedback_loop/      # Collect response feedback; retrain signals
├── shared/             # Cross-cutting: config, DB clients, logging, schemas
└── tests/              # Pytest suite for all modules
```

## Infrastructure

| Service               | Image                            | Port |
| --------------------- | -------------------------------- | ---- |
| PostgreSQL + pgvector | pgvector/pgvector:pg16           | 5432 |
| Weaviate              | semitechnologies/weaviate:1.25.4 | 8080 |
| Redis                 | redis:7.2-alpine                 | 6379 |

## Quickstart

```bash
# 1. Start infrastructure
docker compose up -d

# 2. Install dependencies
poetry install

# 3. Run the full CI suite locally
poetry run ruff check .
poetry run mypy .
poetry run pytest
```

## CI/CD

GitHub Actions runs on every push and pull request:

1. **Lint** — `ruff check .`
2. **Type-check** — `mypy .`
3. **Test** — `pytest` with coverage report uploaded to Codecov
4. **Eval Gate** — `python llmops/eval_harness/run_eval.py`; pipeline fails if mean score < 0.75

Add a `POSTGRES_PASSWORD` secret to your GitHub repository settings before running CI.

## Module Responsibilities

| Module             | Layer                                   | Role in the loyalty loop                                              |
| ------------------ | --------------------------------------- | --------------------------------------------------------------------- |
| `data_pipeline`    | **Data Ingestion**                      | Ingest POS transactions and CRM exports into Postgres                 |
| `feature_store`    | **Feature Platform (Redis + Postgres)** | Compute and serve ML features at low latency                          |
| `propensity_model` | **Churn / Upsell Propensity Engine**    | XGBoost models that score churn risk and upsell likelihood            |
| `rag_retrieval`    | **Contextual Knowledge Retrieval**      | Embed and retrieve loyalty docs from Weaviate to ground LLM calls     |
| `llm_generator`    | **Personalised Offer Generation**       | Draft individualised retention offers via an LLM                      |
| `llmops`           | **Model Observability & Eval**          | Track latency, token cost, and quality scores; enforce the eval gate  |
| `feedback_loop`    | **Reinforcement Signal Pipeline**       | Convert redemption events into pseudo-labels for model retraining     |
| `shared`           | **Platform SDK / Config**               | Centralised config, DB clients, and structured logging for all modules|
