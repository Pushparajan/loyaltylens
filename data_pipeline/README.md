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

## Running Locally

### Prerequisites

- Python 3.11+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (includes Docker Compose)

### 1. Start Docker Desktop

Open **Docker Desktop** from the Start menu (Windows) or Applications (macOS) and wait until the system-tray icon tooltip says **"Docker Desktop is running"**. This takes 30–60 seconds on first launch.

You can verify the daemon is up with:

```bash
docker info
```

If you see `ERROR: Cannot connect to the Docker daemon`, Docker Desktop is not ready yet.

### 2. Start infrastructure

```bash
docker compose up -d
```

Wait for all three services to be healthy:

```bash
docker compose ps
```

All rows should show `healthy` in the `STATUS` column before proceeding:

```text
NAME                     STATUS
loyaltylens_postgres     running (healthy)
loyaltylens_redis        running (healthy)
loyaltylens_weaviate     running (healthy)
```

### 2. Create a Python environment

Install [uv](https://docs.astral.sh/uv/) if you don't have it (no existing Python required):

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal after installation so `uv` is on PATH.

Create and activate a virtual environment at the repo root:

```bash
uv venv .venv --python 3.11
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 4. Install dependencies

Run from the **repo root** (`c:\Projects\loyaltylens`), not from inside `data_pipeline/`:

```bash
cd ..   # if you are currently inside data_pipeline/
uv pip install -e ".[dev]"
# or, without the dev extras:
uv pip install -r data_pipeline/requirements.txt
```

### 5. Configure environment

Copy or create a `.env` file at the repo root. The defaults in `shared/config.py` work out-of-the-box with the Docker Compose stack, so no changes are required unless you override ports or credentials:

```dotenv
POSTGRES_URL=postgresql://loyaltylens:loyaltylens@localhost:5432/loyaltylens
REDIS_URL=redis://localhost:6379
WEAVIATE_URL=http://localhost:8080
RAW_EVENTS_PATH=data/raw/events.parquet
PROCESSED_FEATURES_PATH=data/processed/features.parquet
```

### 6. Create database tables

Apply the schema to the running Postgres container.

**Windows (PowerShell)** — `<` redirection is not supported, use `Get-Content`:

```powershell
Get-Content db/schema.sql | docker exec -i loyaltylens_postgres psql -U loyaltylens -d loyaltylens
```

**macOS / Linux:**

```bash
docker exec -i loyaltylens_postgres psql -U loyaltylens -d loyaltylens < db/schema.sql
```

Or if `psql` is on your PATH:

```bash
psql postgresql://loyaltylens:loyaltylens@localhost:5432/loyaltylens -f db/schema.sql
```

### 8. Generate synthetic data

```bash
python -m data_pipeline.generate
```

This writes `50 000` synthetic loyalty events to `data/raw/events.parquet` (path controlled by `RAW_EVENTS_PATH`).

### 9. Run the pipeline

Use the smoke-test script at the repo root to push a small batch of synthetic records through the full ETL flow:

```bash
python run_pipeline.py
```

Expected output:

```text
Pipeline result: {'transactions': 2, 'customers': 2}
```

Or trigger the orchestrator programmatically:

```python
from data_pipeline.orchestrator import PipelineOrchestrator

PipelineOrchestrator().run(
    transaction_records=[...],
    customer_records=[...],
)
```

### 10. Verify

Connect to Postgres and check row counts.

**Windows (PowerShell):**

```powershell
docker exec loyaltylens_postgres psql -U loyaltylens -d loyaltylens -c "SELECT COUNT(*) FROM transactions; SELECT COUNT(*) FROM customers;"
```

**macOS / Linux (or if `psql` is on PATH):**

```bash
psql postgresql://loyaltylens:loyaltylens@localhost:5432/loyaltylens \
  -c "SELECT COUNT(*) FROM transactions; SELECT COUNT(*) FROM customers;"
```
