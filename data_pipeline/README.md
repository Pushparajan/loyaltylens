# data_pipeline

Orchestrates ETL flows that ingest raw loyalty transactions and customer records, validate them, and load clean data into PostgreSQL for downstream consumption.

## Purpose

Extract data from source systems (POS, CRM, e-commerce APIs), apply schema validation and deduplication, then persist to the `loyaltylens` Postgres database so the feature store and propensity model always work from a consistent, audit-trailed dataset.

## Inputs

- Raw transaction events (JSON / CSV) from POS / e-commerce webhooks
- Customer profile exports from CRM (CSV / REST API)
- Configuration via `shared.Settings` (DB connection, batch size, schedule)

## Outputs

- Validated rows written to `transactions` and `customers` Postgres tables
- Pipeline run metrics (rows ingested, errors) emitted to `llmops.MetricsCollector`
- Prefect flow run records for audit and replay

## Key Classes

| Class | Module | Responsibility |
| --------------------- | ---------------------- | --------------------------------------- |
| `TransactionIngester` | `ingester.py`          | Parse, validate, and upsert transactions |
| `CustomerDataLoader`  | `loader.py`            | Load and reconcile customer profiles    |
| `PipelineOrchestrator`| `orchestrator.py`      | Compose Prefect flows, handle scheduling |
