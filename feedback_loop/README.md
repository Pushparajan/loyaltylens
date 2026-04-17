# feedback_loop

Captures explicit feedback on generated offer copy, aggregates it into quality signals, exports RLHF preference datasets, and triggers retraining when quality drops below threshold.

## Purpose

Close the ML loop: campaign manager ratings and thumbs up/down are the ground truth signal for whether the LLM generator is producing good copy. This module captures that signal, aggregates it into statistics, and automatically fires a retraining alert when metrics degrade.

## Module Map

| Path | Responsibility |
| --- | --- |
| `db.py` | SQLite connection factory + schema init (`feedback` table) |
| `api.py` | FastAPI: `POST /feedback`, `GET /feedback/stats`, `GET /feedback/export` |
| `aggregator.py` | `FeedbackAggregator` — `compute_stats()`, `export_preference_dataset()` |
| `trigger.py` | `RetrainingTrigger` — threshold checks, log + GitHub Actions dispatch |
| `ui/` | React + Vite + TypeScript + Tailwind SPA (review + dashboard tabs) |
| `collector.py` | `FeedbackCollector` — raw interaction events to Postgres (legacy) |
| `processor.py` | `FeedbackProcessor` — aggregate events to training signals (legacy) |
| `updater.py` | `ModelUpdater` — pseudo-label builder for propensity model (legacy) |

## Key Classes

| Class | Module | Responsibility |
| --- | --- | --- |
| `FeedbackAggregator` | `aggregator.py` | Compute stats + export JSONL preference pairs |
| `FeedbackStats` | `aggregator.py` | Dataclass: avg_rating, thumbs rates, by_prompt_version, by_model_version |
| `RetrainingTrigger` | `trigger.py` | Threshold monitor + GitHub Actions webhook stub |

## Trigger Thresholds

| Metric | Threshold | Action |
| --- | --- | --- |
| `avg_rating` | < 3.0 | Fire trigger |
| `thumbs_down_rate` | > 40 % | Fire trigger |
| `record_count` | < 10 | Skip (insufficient data) |

## Preference Dataset Format (JSONL)

```json
{"prompt": "Write loyalty offer copy for a customer (offer_id=O001, prompt_version=v1).", "chosen": "Great headline body cta", "rejected": "Weak headline body cta"}
```

Pairs are built by joining thumbs-up and thumbs-down records for the same `offer_id`.

---

## Running Locally

### 1. Environment

One `.env` at the **repo root** — no module-level env file needed.

```dotenv
PORT_FEEDBACK_LOOP=8005
GITHUB_TOKEN=              # local only — CI uses the built-in secrets.GITHUB_TOKEN automatically
GITHUB_REPO=Pushparajan/loyaltylens   # local only — CI uses ${{ github.repository }} automatically
```

---

### 2. Install dependencies

```powershell
& C:\Projects\loyaltylens\.venv\Scripts\Activate.ps1    # prompt shows (loyaltylens)
uv sync --dev
```

---

### 3. Start the API

```powershell
# Windows
python -m uvicorn feedback_loop.api:app --host 127.0.0.1 --port 8005 --reload

# macOS / Linux
python -m uvicorn feedback_loop.api:app --host 0.0.0.0 --port 8005 --reload
```

The SQLite database is created automatically at `feedback_loop/data/feedback.db` on first start.

---

### 4. Post a feedback record

```powershell
Invoke-RestMethod -Method POST -Uri http://127.0.0.1:8005/feedback `
  -ContentType "application/json" `
  -Body '{
    "offer_id": "O001",
    "customer_id": "C001",
    "generated_copy": {"headline": "Double Star Day", "body": "Earn 2x stars today.", "cta": "Redeem now"},
    "rating": 4,
    "thumbs": "up",
    "prompt_version": "v2",
    "model_version": "gpt-4o-mini"
  }'
```

```powershell
Invoke-RestMethod http://127.0.0.1:8005/feedback/stats  | ConvertTo-Json -Depth 5
Invoke-RestMethod http://127.0.0.1:8005/feedback/export | ConvertTo-Json -Depth 5
```

---

### 5. Run the aggregator

```python
from feedback_loop.aggregator import FeedbackAggregator

agg = FeedbackAggregator()
stats = agg.compute_stats(since_days=7)
print(stats.avg_rating, stats.thumbs_up_rate)

# Export RLHF preference pairs
n = agg.export_preference_dataset("data/preferences.jsonl")
print(f"Exported {n} pairs")
```

---

### 6. Run the retraining trigger

```bash
# Cron-friendly one-shot check
python feedback_loop/trigger.py --check

# Custom window
python feedback_loop/trigger.py --check --since-days 14
```

Exit code `1` if the trigger fired (avg_rating < 3.0 or thumbs_down_rate > 40 %).

---

### 7. Start the feedback UI

```bash
cd feedback_loop/ui
npm install
npm run dev
# → http://localhost:5173
```

The UI proxies `/feedback` requests to `http://localhost:8005`. The LLM generator (`http://localhost:8004`) is called for live offers; mock offers are used automatically if it is not running.

**Tabs:**

- **Review** — displays offer (headline, body, CTA, category), thumbs up/down, 1–5 star rating, submit
- **Dashboard** — avg rating, thumbs-up rate, bar chart by prompt version, table by model version

---

### 8. Run tests

```bash
python -m pytest tests/test_feedback.py -v
```

21 tests cover:

- SQLite schema init, insert, CHECK constraint enforcement
- API `POST /feedback` (201), invalid rating (422), stats aggregation, export
- `FeedbackAggregator` stats computation, per-version breakdown, JSONL export format, empty pair handling
- `RetrainingTrigger` boundary conditions: rating=3.0 does NOT trigger, 2.9 does; thumbs_down=0.40 does NOT, 0.41 does; insufficient data skips; fire writes trigger log

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `No such file: feedback_loop/data/feedback.db` | API never started | Run `python -m uvicorn feedback_loop.api:app --port 8005` |
| UI shows mock offers only | LLM generator not running | Start M4 API: `python -m uvicorn llm_generator.api:app --port 8004` |
| `No module named 'shared'` | Wrong venv | `& C:\Projects\loyaltylens\.venv\Scripts\Activate.ps1` then `uv sync --dev` |
| Trigger fires but no GitHub dispatch | `GITHUB_TOKEN` / `GITHUB_REPO` not set | Add to root `.env` — trigger still logs locally without them |
