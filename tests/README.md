# tests

Pytest suite covering unit and integration tests for all LoyaltyLens modules.

## Purpose

Provide fast feedback on correctness and prevent regressions. Unit tests mock external services; integration tests run against the Docker Compose stack (Postgres, Weaviate, Redis).

## Inputs

- Source modules under test
- Docker Compose services (for integration tests — set `INTEGRATION=1`)
- Pytest fixtures defined in `conftest.py`

## Outputs

- Pass / fail results written to stdout and `junit.xml`
- Coverage report written to `coverage.xml` and `htmlcov/`

## Key Classes / Fixtures

| Name                   | File            | Purpose |
| ---------------------- | --------------- | -------------------------------------------- |
| `postgres_session`     | `conftest.py`   | Provide a transactional Postgres session     |
| `redis_client`         | `conftest.py`   | In-memory Redis client (fakeredis)           |
| `mock_weaviate`        | `conftest.py`   | Weaviate mock for unit tests                 |
| `sample_transactions`  | `conftest.py`   | Seed DataFrame of synthetic transactions     |
