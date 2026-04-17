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
| --- | --- | --- |
| `TransactionIngester` | `ingester.py` | Parse, validate, and upsert transactions |
| `CustomerDataLoader` | `loader.py` | Load and reconcile customer profiles |
| `PipelineOrchestrator` | `orchestrator.py` | Compose Prefect flows, handle scheduling |

---

## Running Locally

### Prerequisites

- Python 3.11+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (includes Docker Compose)

---

### 1. Start Docker Desktop

Open **Docker Desktop** from the Start menu (Windows) or Applications (macOS) and wait until the system-tray icon says **"Docker Desktop is running"** (30–60 seconds on first launch).

Verify the daemon is up:

```bash
docker info
```

---

### 2. Start infrastructure

```powershell
docker compose up -d postgres weaviate redis

# Wait for all three to show (healthy)
docker compose ps
```

---

### 3. Create the Python environment

Always use the **repo-root** `.venv` — sub-module venvs (`data_pipeline/.venv`, etc.) are legacy and should not be used.

```powershell
# Windows (PowerShell) — from c:\Projects\loyaltylens
uv venv .venv --python 3.11                              # skip if venv already exists
& C:\Projects\loyaltylens\.venv\Scripts\Activate.ps1    # prompt shows (loyaltylens)
```

```bash
# macOS / Linux
uv venv .venv --python 3.11
source .venv/bin/activate
```

> **Windows execution policy:** If the script is blocked, run once:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

---

### 4. Install dependencies

Run from the **repo root** (`c:\Projects\loyaltylens`), not from inside `data_pipeline/`:

```powershell
uv sync --dev
```

---

### 5. Configure environment

One `.env` at the **repo root** — all modules share it. Copy the template:

```powershell
Copy-Item .env.example .env   # Windows
# cp .env.example .env        # macOS / Linux
```

The defaults work out-of-the-box with the Docker Compose stack. The only values you must fill in are `OPENAI_API_KEY` and (optionally) `HF_TOKEN`. All DB URLs and ports are pre-configured.

> **Never create a `data_pipeline/.env`** — `shared.config.Settings` reads the root `.env` automatically.

---

### 6. Create database tables

The `docker/init-pgvector.sql` script enables the `vector` extension automatically on first start. Apply the full schema:

**Windows (PowerShell):**

```powershell
Get-Content db/schema.sql | docker exec -i loyaltylens_postgres psql -U loyaltylens -d loyaltylens
```

**macOS / Linux:**

```bash
docker exec -i loyaltylens_postgres psql -U loyaltylens -d loyaltylens < db/schema.sql
```

---

### 7. Generate synthetic data

```bash
python -m data_pipeline.generate
```

Writes 50,000 synthetic loyalty events to `data/raw/events.parquet`.

---

### 8. Run the pipeline

```bash
python run_pipeline.py
```

Expected output:

```text
Pipeline result: {'transactions': 2, 'customers': 2}
```

Or trigger programmatically:

```python
from data_pipeline.orchestrator import PipelineOrchestrator

PipelineOrchestrator().run(
    transaction_records=[...],
    customer_records=[...],
)
```

---

### 9. Verify

**Windows (PowerShell):**

```powershell
docker exec loyaltylens_postgres psql -U loyaltylens -d loyaltylens `
  -c "SELECT COUNT(*) FROM transactions; SELECT COUNT(*) FROM customers;"
```

**macOS / Linux:**

```bash
psql postgresql://loyaltylens:loyaltylens@localhost:5432/loyaltylens \
  -c "SELECT COUNT(*) FROM transactions; SELECT COUNT(*) FROM customers;"
```

---

### Restart from scratch

```powershell
docker compose down -v
docker compose up -d postgres weaviate redis
python run_pipeline.py
```
