# =============================================================================
# LoyaltyLens — Makefile
# Developer shortcuts for building, running, and managing the stack.
# Usage: make <target>
# =============================================================================

.PHONY: help build build-prod up up-infra up-prod down logs shell \
        test lint format typecheck verify clean nuke \
        generate-data train-propensity index-offers \
        api-feature api-propensity api-rag api-llm api-feedback \
        mlflow streamlit

# Default target
help:
	@echo ""
	@echo "  LoyaltyLens — Developer Commands"
	@echo "  ================================="
	@echo ""
	@echo "  DOCKER"
	@echo "    make build          Build the dev image"
	@echo "    make build-prod     Build the production API image"
	@echo "    make up             Start infra + dev container (interactive)"
	@echo "    make up-infra       Start postgres, weaviate, redis only"
	@echo "    make up-prod        Start all production API containers"
	@echo "    make down           Stop all containers"
	@echo "    make logs           Tail all container logs"
	@echo "    make shell          Open bash in the running dev container"
	@echo ""
	@echo "  CODE QUALITY"
	@echo "    make test           Run all pytest tests"
	@echo "    make lint           Run ruff linter"
	@echo "    make format         Run ruff formatter"
	@echo "    make typecheck      Run mypy type checker"
	@echo "    make verify         lint + typecheck + test"
	@echo ""
	@echo "  DATA & MODELS"
	@echo "    make generate-data  Generate synthetic event data (M1)"
	@echo "    make train          Train the propensity model (M2)"
	@echo "    make index-offers   Embed and index offers in pgvector + Weaviate (M3)"
	@echo ""
	@echo "  SERVICES (run locally without Docker)"
	@echo "    make api-feature    Start Feature Store API on :8001"
	@echo "    make api-propensity Start Propensity Model API on :8002"
	@echo "    make api-rag        Start RAG Retrieval API on :8003"
	@echo "    make api-llm        Start LLM Generator API on :8004"
	@echo "    make api-feedback   Start Feedback Loop API on :8005"
	@echo "    make mlflow         Start MLflow tracking server on :5000"
	@echo "    make streamlit      Start Streamlit LLMOps dashboard on :8501"
	@echo "    make all-apis       Start all APIs concurrently"
	@echo ""
	@echo "  MAINTENANCE"
	@echo "    make clean          Remove __pycache__, .pyc, build artefacts"
	@echo "    make nuke           Remove all containers, volumes, and images"
	@echo ""

# ---------------------------------------------------------------------------
# Docker commands
# ---------------------------------------------------------------------------

COMPOSE = docker-compose -f docker/docker-compose.yml
IMAGE_DEV  = loyaltylens-dev
IMAGE_PROD = loyaltylens-api

build:
	$(COMPOSE) build app

build-prod:
	$(COMPOSE) build feature-api

up:
	$(COMPOSE) --profile dev up

up-infra:
	$(COMPOSE) --profile infra up -d
	@echo ""
	@echo "Infrastructure running:"
	@echo "  PostgreSQL  →  localhost:5432"
	@echo "  Weaviate    →  localhost:8080"
	@echo "  Redis       →  localhost:6379"

up-prod:
	$(COMPOSE) --profile prod up -d
	@echo ""
	@echo "Production stack running. Check status:"
	@echo "  make logs"

down:
	$(COMPOSE) --profile dev --profile infra --profile prod --profile dashboard down

logs:
	$(COMPOSE) --profile dev --profile infra --profile prod logs -f

shell:
	$(COMPOSE) exec app bash

# ---------------------------------------------------------------------------
# Code quality (run inside container or locally with uv)
# ---------------------------------------------------------------------------

test:
	uv run pytest tests/ -v --tb=short

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy loyaltylens/ --ignore-missing-imports

verify: lint typecheck test
	@echo "All checks passed."

# ---------------------------------------------------------------------------
# Data pipeline and model training
# ---------------------------------------------------------------------------

generate-data:
	uv run python data_pipeline/generate.py
	uv run python data_pipeline/features.py
	@echo "Data generated in data/raw/ and data/processed/"

train:
	uv run python propensity_model/train.py
	@echo "Model saved to models/"

index-offers:
	uv run python rag_retrieval/generate_offers.py
	uv run python rag_retrieval/embeddings.py
	@echo "Offers indexed in pgvector and Weaviate"

# Run the full data pipeline in sequence
pipeline: generate-data train index-offers
	@echo "Full pipeline complete."

# ---------------------------------------------------------------------------
# Individual API services (run locally without Docker)
# ---------------------------------------------------------------------------

api-feature:
	uv run uvicorn feature_store.api:app --host 0.0.0.0 --port 8001 --reload

api-propensity:
	uv run uvicorn propensity_model.api:app --host 0.0.0.0 --port 8002 --reload

api-rag:
	uv run uvicorn rag_retrieval.api:app --host 0.0.0.0 --port 8003 --reload

api-llm:
	uv run uvicorn llm_generator.api:app --host 0.0.0.0 --port 8004 --reload

api-feedback:
	uv run uvicorn feedback_loop.api:app --host 0.0.0.0 --port 8005 --reload

mlflow:
	uv run mlflow server --host 0.0.0.0 --port 5000 \
		--backend-store-uri mlruns \
		--default-artifact-root mlruns

streamlit:
	uv run streamlit run llmops/dashboard/app.py \
		--server.port 8501 --server.address 0.0.0.0

all-apis:
	@echo "Starting all APIs concurrently (use Ctrl+C to stop all)..."
	uv run uvicorn feature_store.api:app --port 8001 & \
	uv run uvicorn propensity_model.api:app --port 8002 & \
	uv run uvicorn rag_retrieval.api:app --port 8003 & \
	uv run uvicorn llm_generator.api:app --port 8004 & \
	uv run uvicorn feedback_loop.api:app --port 8005 & \
	wait

# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info
	@echo "Cleaned."

nuke: down
	$(COMPOSE) down -v --rmi local 2>/dev/null || true
	docker volume prune -f
	@echo "All containers, volumes, and local images removed."
