# shared

Cross-cutting utilities used by every other module: configuration management, database clients, structured logging, and base Pydantic schemas.

## Purpose

Avoid duplication of boilerplate (DB connections, env var parsing, logger setup) across modules. All inter-module contracts (shared data schemas) are also defined here to prevent circular imports.

## Inputs

- Environment variables / `.env` file (parsed by `Settings` via Pydantic `BaseSettings`)

## Outputs

- `Settings` singleton consumed by all modules
- `DatabaseClient` instances (SQLAlchemy engine + session factory, Redis client)
- `structlog` logger instances with consistent JSON formatting
- Base Pydantic models with common field validators

## Key Classes

| Class            | Module      | Responsibility |
| ---------------- | ----------- | -------------------------------------------- |
| `Settings`       | `config.py` | Parse and validate all env-var configuration |
| `DatabaseClient` | `db.py`     | SQLAlchemy engine and Redis connection pool  |
| `get_logger`     | `logger.py` | Return a configured `structlog` logger       |
| `BaseSchema`     | `schemas.py`| Pydantic v2 base with `model_config` presets |
