---
title: "RLHF Without a Research Team: Building Practical Feedback Loops for Production LLMs"
slug: "loyaltylens-rlhf-feedback-loop"
description: "Closing the production AI loop — feedback capture API, React review UI, preference dataset export, automated retraining triggers, and the honest limits of practical RLHF."
date: 2026-06-02
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

## Closing the loop with customer signals, preference datasets, and automated retraining — LoyaltyLens Module 6

---

RLHF — Reinforcement Learning from Human Feedback — is the technique that turned GPT-3 into ChatGPT. In both cases, it involved massive amounts of human annotation, proprietary reward models, and significant computational resources.

When I talk about "RLHF-style feedback loops" in a loyalty program context, I mean something more pragmatic: a system that captures campaign manager signals, uses those signals to measure whether the AI system is getting better or worse, and automatically triggers retraining when they indicate degradation.

This post walks through LoyaltyLens Module 6, which implements that system end-to-end.

---

## The Signal Problem

Before you can build a feedback loop, you need to decide what signal you're collecting and what it actually measures.

In production, we have three signal types:

**Offer redemption** — Did the customer redeem the offer? Ground truth, but arrives 3–7 days after send. Too slow for rapid iteration.

**Campaign manager review** — Approve/reject before a campaign goes out. Highest quality, but expensive and unscalable.

**In-app engagement** — Opens, taps, clicks. Real-time but noisy.

LoyaltyLens implements a simplified version: thumbs up/down + 1–5 star rating from a review UI simulating the campaign manager signal.

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

SQLite was the right choice here. The feedback database is local to the module, doesn't need concurrent writes from multiple services, and is trivially portable. No Docker, no connection strings, no migrations.

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

## The Honest Limits

**This does not implement full RLHF.** True RLHF trains a reward model on the preference data, then uses that reward model in a reinforcement learning loop to fine-tune the base LLM. LoyaltyLens implements the feedback *collection* and *trigger* infrastructure — the foundation RLHF would be built on.

**It does not separate signal from noise.** A thumbs-up from someone who approves everything quickly is worth less than one from someone who reads carefully. In production you'd weight by annotator reliability.

**It assumes honest feedback.** In a system with throughput targets, feedback quality degrades. Adversarial robustness in signal collection is a real engineering problem.

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

## The End of the Series

LoyaltyLens, taken as a whole, is a production-pattern demonstration of every component in a modern loyalty AI system:

- A feature pipeline that enforces data quality at the store boundary
- A propensity model with a model card, eval metrics, and a deployment path
- RAG retrieval benchmarked across frameworks and vector databases
- LLM-generated copy with prompt versioning and brand alignment checking
- An LLMOps pipeline with drift monitoring, automated evaluation, and CI/CD gates
- A feedback loop that turns human signals into automated model improvement triggers

**← Back to Module 5: The LLMOps Pipeline** *(link to be added on publish)*

---

*Pushparajan Ramar is an Enterprise Architect Director specialising in AI, data, and platform architecture for large-scale enterprise programs.*

*[LinkedIn](https://linkedin.com/in/pushparajanramar)*
