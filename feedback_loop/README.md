# feedback_loop

Collects explicit and implicit feedback on generated offers and communications, processes it into labelled training signals, and triggers model retraining when drift thresholds are exceeded.

## Purpose

Close the ML loop: customer interactions (opens, clicks, redemptions, ignores) are the ground truth for whether the propensity model and LLM generator are performing well. This module captures that signal and feeds it back upstream.

## Inputs

- Interaction events from downstream systems (email platform webhooks, app analytics)
- Explicit ratings / thumbs from customer-facing interfaces
- Current model performance baselines from `llmops.MetricsCollector`

## Outputs

- Labelled `FeedbackRecord` rows written to `feedback` Postgres table
- Retraining trigger events published to `propensity_model.ModelTrainer`
- Aggregated feedback summaries logged to MLflow for trend analysis

## Key Classes

| Class | Module | Responsibility |
| --- | --- | --- |
| `FeedbackCollector` | `collector.py` | Ingest and validate raw interaction events |
| `FeedbackProcessor` | `processor.py` | Aggregate events into labelled training signals |
| `ModelUpdater` | `updater.py` | Evaluate drift and trigger retraining pipeline |

---

## Running Locally

### 1. Environment

One `.env` at the **repo root** — no module-level env file needed.

```dotenv
POSTGRES_URL=postgresql://loyaltylens:loyaltylens@localhost:5432/loyaltylens
REDIS_URL=redis://localhost:6379
PORT_FEEDBACK_LOOP=8005
```

---

### 2. Install dependencies

```powershell
uv sync --dev
uv pip install -e .
```

---

### 3. Start infrastructure

```powershell
docker compose up -d postgres redis
docker compose ps   # wait for (healthy)
```

---

### 4. Start the API

```powershell
# Windows
python -m uvicorn feedback_loop.api:app --host 127.0.0.1 --port 8005 --reload

# macOS / Linux
python -m uvicorn feedback_loop.api:app --host 0.0.0.0 --port 8005 --reload
```

---

### 5. Run tests

```bash
python -m pytest tests/ -k feedback -v
```
