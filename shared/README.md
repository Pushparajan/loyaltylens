# shared

Cross-cutting utilities used by every other module: configuration management, database clients, structured logging, and base Pydantic schemas.

## Purpose

Avoid duplication of boilerplate (DB connections, env var parsing, logger setup) across modules. All inter-module contracts (shared data schemas) are also defined here to prevent circular imports.

## The `.env` contract

**There is one `.env` file in the entire project — at the repo root.**

All modules import `get_settings()` from this package. `Settings` reads `(".env", "../.env")` in order, so it resolves correctly whether Python is invoked from the repo root or a sub-directory. Never create module-level `.env` files.

```python
# Every module does exactly this — nothing else
from shared.config import get_settings

settings = get_settings()
print(settings.postgres_url)
print(settings.port_rag_retrieval)   # 8003
```

## Inputs

- Root `.env` file (parsed by `Settings` via Pydantic `BaseSettings`)
- Environment variables that override `.env` values (CI, Docker, shell)

## Outputs

- `Settings` singleton consumed by all modules
- `DatabaseClient` instances (SQLAlchemy engine + session factory, Redis client)
- `structlog` logger instances with consistent JSON formatting
- Base Pydantic models with common field validators

## Key Classes

| Class | Module | Responsibility |
| --- | --- | --- |
| `Settings` | `config.py` | Parse and validate all env-var configuration |
| `DatabaseClient` | `db.py` | SQLAlchemy engine and Redis connection pool |
| `get_logger` | `logger.py` | Return a configured `structlog` logger |
| `BaseSchema` | `schemas.py` | Pydantic v2 base with `model_config` presets |

## Port fields in Settings

All service ports live in `Settings` — change them once in `.env`, they propagate everywhere:

| Field | Env Var | Default | Description |
| --- | --- | --- | --- |
| `postgres_port` | `POSTGRES_PORT` | 5432 | PostgreSQL |
| `weaviate_http_port` | `WEAVIATE_HTTP_PORT` | 8080 | Weaviate REST/HTTP |
| `weaviate_grpc_port` | `WEAVIATE_GRPC_PORT` | 50051 | Weaviate gRPC (required by v4 client) |
| `redis_port` | `REDIS_PORT` | 6379 | Redis |
| `mlflow_port` | `MLFLOW_PORT` | 5000 | MLflow UI |
| `port_feature_store` | `PORT_FEATURE_STORE` | 8001 | Feature Store API |
| `port_propensity` | `PORT_PROPENSITY` | 8002 | Propensity Model API |
| `port_rag_retrieval` | `PORT_RAG_RETRIEVAL` | 8003 | RAG Retrieval API |
| `port_llm_generator` | `PORT_LLM_GENERATOR` | 8004 | LLM Generator API |
| `port_feedback_loop` | `PORT_FEEDBACK_LOOP` | 8005 | Feedback Loop API |
| `port_metrics` | `PORT_METRICS` | 8006 | Prometheus metrics server |
