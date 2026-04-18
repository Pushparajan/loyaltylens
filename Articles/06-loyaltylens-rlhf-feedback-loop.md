---
title: "Building a Feedback Loop for Production LLMs: Signals, Preference Data, and Retraining Triggers"
slug: "loyaltylens-rlhf-feedback-loop"
description: "Closing the production AI loop — feedback capture API, React review UI, preference dataset export, automated retraining triggers, and the practical limits of RLHF without a research team."
date: 2026-05-14
author: Pushparajan Ramar
series: loyaltylens
series_order: 6
reading_time: 14
tags:
  - rlhf
  - reinforcement-learning
  - llmops
  - feedback-loops
  - mlops
  - production-ai
  - react
  - fastapi
---

# Building a Feedback Loop for Production LLMs: Signals, Preference Data, and Retraining Triggers

*Feedback capture API, React review UI, preference dataset export, and automated retraining triggers — LoyaltyLens Module 6*

---

**Series position:** Article 6 of 8

---

Module 6 closes the production AI loop. It implements the feedback collection and retraining trigger infrastructure that turns human review signals into automated model improvement actions.

RLHF (Reinforcement Learning from Human Feedback) in its full form — reward model training, RL fine-tuning loop — requires research team resources. This module implements the practical subset: capture the right signals, export them in the format needed for fine-tuning, and trigger retraining automatically when rolling quality degrades.

---

## Signal Selection

Three signal types are available in a loyalty AI system:

**Offer redemption** — Ground truth (did the customer redeem?), but arrives 3–7 days after send. Too slow for rapid iteration cycles.

**Campaign manager review** — Approve/reject before a campaign goes out. Highest quality, but expensive and doesn't scale.

**In-app engagement** — Opens, taps, clicks. Real-time but noisy.

LoyaltyLens implements thumbs up/down + 1–5 star rating from a review UI — a simplified proxy for campaign manager signal that's fast to collect and directly tied to generated copy quality.

---

## 1. The Feedback API

The SQLite-backed FastAPI service exposes three endpoints:

```python
# feedback_loop/api.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from feedback_loop.db import init_db, get_connection

@asynccontextmanager
async def _lifespan(app):
    init_db(_DB_PATH)   # creates feedback table on first run
    yield

app = FastAPI(lifespan=_lifespan)

class FeedbackRequest(BaseModel):
    offer_id: str
    customer_id: str
    generated_copy: dict[str, str]   # {headline, body, cta, tone}
    rating: int = Field(ge=1, le=5)
    thumbs: Literal["up", "down"]
    prompt_version: str = ""
    model_version: str = ""

@app.post("/feedback", status_code=201)
def submit_feedback(req: FeedbackRequest) -> FeedbackResponse: ...

@app.get("/feedback/stats")
def get_stats() -> StatsResponse: ...

@app.get("/feedback/export")
def export_feedback() -> list[dict]: ...
```

SQLite is the right choice for this module. The feedback database is local, doesn't require concurrent writes from multiple services, and needs no separate server. No Docker, no connection strings, no migration tooling.

The schema uses CHECK constraints to enforce data quality at the database layer:

```sql
CREATE TABLE IF NOT EXISTS feedback (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    offer_id       TEXT    NOT NULL,
    customer_id    TEXT    NOT NULL,
    generated_copy TEXT    NOT NULL,
    rating         INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
    thumbs         TEXT    NOT NULL CHECK(thumbs IN ('up', 'down')),
    prompt_version TEXT    NOT NULL DEFAULT '',
    model_version  TEXT    NOT NULL DEFAULT '',
    created_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);
```

Start it:

```bash
python -m uvicorn feedback_loop.api:app --host 127.0.0.1 --port 8005 --reload
```

---

## 2. The Feedback UI

The UI is a React + Vite + TypeScript + Tailwind SPA in `feedback_loop/ui/`. It's deliberately minimal — its purpose is signal collection, not product design.

```bash
cd feedback_loop/ui
npm install
npm run dev   # → http://localhost:5173
```

**Review tab** — Fetches an offer from the LLM generator API (falls back to mock offers automatically if the generator isn't running), then shows:

- Offer card: headline, body, CTA, category badge
- Thumbs up / thumbs down buttons
- 1–5 star rating
- Submit → loads next offer

**Dashboard tab** — Fetches `/feedback/stats` and renders:

- Summary metrics: record count, avg rating, thumbs-up rate
- SVG bar chart of avg rating by prompt version (no chart library — plain SVG)
- Model version comparison table

```tsx
// feedback_loop/ui/src/App.tsx (submit handler)
async function handleSubmit() {
  await submitFeedback({
    offer_id: offer.offer_id,
    customer_id: offer.customer_id,
    generated_copy: { headline, body, cta, tone },
    rating,
    thumbs,
    prompt_version: String(offer.prompt_version),
    model_version: offer.model_version,
  });
  setSubmitted(true);
}
```

Vite proxies `/feedback` to `http://localhost:8005` — no CORS config needed in development:

```typescript
// feedback_loop/ui/vite.config.ts
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/feedback": "http://localhost:8005" },
  },
});
```

---

## 3. The Feedback Aggregator

```python
# feedback_loop/aggregator.py
@dataclass
class FeedbackStats:
    avg_rating: float
    thumbs_up_rate: float
    thumbs_down_rate: float
    by_prompt_version: dict[str, float]   # version → avg_rating
    by_model_version: dict[str, float]
    by_category: dict[str, float]
    record_count: int

class FeedbackAggregator:
    def compute_stats(self, since_days: int = 7) -> FeedbackStats:
        rows = conn.execute(
            "SELECT rating, thumbs, prompt_version, model_version "
            "FROM feedback WHERE created_at >= datetime('now', ?)",
            (f"-{since_days} days",),
        ).fetchall()
        # ... averages and group-by
```

The per-version breakdown is the most actionable part. When `by_prompt_version["v2"]` drops below `by_prompt_version["v1"]`, the prompt regression is visible before the aggregate metric crosses the trigger threshold.

### The Preference Dataset Builder

```python
def export_preference_dataset(self, output_path) -> int:
    pairs = conn.execute("""
        SELECT up.generated_copy AS chosen_copy,
               dn.generated_copy AS rejected_copy,
               up.offer_id, up.prompt_version
        FROM feedback AS up
        JOIN feedback AS dn
            ON up.offer_id = dn.offer_id
            AND up.thumbs = 'up' AND dn.thumbs = 'down'
        LIMIT 10000
    """).fetchall()

    for row in pairs:
        chosen   = json.loads(row["chosen_copy"])
        rejected = json.loads(row["rejected_copy"])
        records.append({
            "prompt":   f"Write loyalty offer copy (offer_id={row['offer_id']}).",
            "chosen":   f"{chosen['headline']} {chosen['body']} {chosen['cta']}",
            "rejected": f"{rejected['headline']} {rejected['body']} {rejected['cta']}",
        })
    # write JSONL to output_path
```

Output format (one JSON object per line):

```json
{"prompt": "Write loyalty offer copy (offer_id=O001, prompt_version=v2).", "chosen": "Great headline earned body redeem cta", "rejected": "Weak headline body click"}
```

---

## 4. The Retraining Trigger

```python
# feedback_loop/trigger.py
class RetrainingTrigger:
    def should_retrain(self, stats: FeedbackStats) -> tuple[bool, str]:
        if stats.record_count < 10:
            return False, f"insufficient_data (n={stats.record_count})"
        if stats.avg_rating < 3.0:
            return True, f"avg_rating={stats.avg_rating:.2f} < threshold=3.0"
        if stats.thumbs_down_rate > 0.40:
            return True, f"thumbs_down_rate={stats.thumbs_down_rate:.1%} > threshold=40%"
        return False, "metrics_healthy"

    def fire_trigger(self, reason: str, stats: FeedbackStats) -> None:
        # 1. Append to feedback_loop/data/trigger_log.json
        # 2. logger.critical(...)
        # 3. GitHub Actions workflow_dispatch (if GITHUB_TOKEN + GITHUB_REPO in .env)
        self._dispatch_github_workflow(reason)
```

The GitHub dispatch is a no-op stub when `GITHUB_TOKEN` / `GITHUB_REPO` are not set — the trigger still logs locally, making it safe to run in CI without secrets configured.

Run as a daily cron:

```bash
python feedback_loop/trigger.py --check            # exits 1 if triggered
python feedback_loop/trigger.py --check --since-days 14
```

The thresholds are conservative by design. At exactly the threshold value — `avg_rating = 3.0`, `thumbs_down_rate = 0.40` — no trigger fires. The test suite verifies both boundary conditions.

---

## 5. What the Tests Cover

```bash
python -m pytest tests/test_feedback.py -v
21 passed in 1.22s
```

- **DB**: table creation, insert/read, CHECK constraint enforcement (rating=0 rejected, thumbs="sideways" rejected)
- **API**: 201 on valid POST, 422 on rating=6, stats empty → computed, export returns parsed JSON
- **Aggregator**: zero stats on empty DB, per-version breakdowns correct, JSONL pairs distinct, empty file when no pairs exist
- **Trigger**: rating=3.0 does NOT trigger, 2.9 does; thumbs_down=0.40 does NOT, 0.41 does; record_count < 10 always skips; fire() appends to trigger log

---

## Practical Limits

**This is not full RLHF.** True RLHF trains a reward model on the preference data, then fine-tunes the base LLM using reinforcement learning against that reward model. This module implements the feedback *collection* and *trigger* infrastructure — the prerequisite layer that RLHF would build on.

**Signal quality is unweighted.** A thumbs-up from a reviewer who approves everything quickly is worth less than one from a careful reviewer. Production systems weight feedback by annotator reliability.

**Honest feedback is assumed.** In systems with throughput targets, feedback quality degrades over time. Adversarial robustness in signal collection is a separate engineering concern.

---

## What Closes the Loop

```text
Offer generated (M4)
    → Shown in review UI
    → Rating + thumbs captured (M6 API → SQLite)
    → Aggregated nightly (M6 aggregator)
    → Trigger checks thresholds (M6 trigger)
    → If triggered:
        → GitHub Actions workflow_dispatch
        → Retrains propensity model (M2 train.py)
        → Runs eval harness (M5 eval gate)
        → If eval passes: promotes new model version
        → If eval fails: rolls back prompt (M5 CLI)
    → Preference dataset grows (M6 exporter)
    → PSI monitor runs (M5 drift)
    → Dashboard updated (M5 + M6)
```

Every arrow is a code path that runs without human intervention. The human decisions are: setting the thresholds, reviewing the trigger log, and deciding whether to accept a promoted model.

---

## Next: Module 7 — Integration and Cloud Deployment

The six modules now have a complete operational loop. Module 7 wires them into a single `LoyaltyLensPipeline.run_for_customer()` call and deploys the propensity model to a live AWS SageMaker endpoint — converting the local reference implementation into a cloud-deployable system.

---

*Pushparajan Ramar — [LinkedIn](https://linkedin.com/in/pushparajanramar) · [GitHub](https://github.com/Pushparajan/loyaltylens)*
