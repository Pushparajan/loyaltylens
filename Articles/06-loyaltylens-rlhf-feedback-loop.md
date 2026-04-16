---
title: "RLHF Without a Research Team: Building Practical Feedback Loops for Production LLMs"
slug: "loyaltylens-rlhf-feedback-loop"
description: "Closing the production AI loop — feedback capture UI, preference dataset export, automated retraining triggers, and the honest limits of practical RLHF."
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
---

# RLHF Without a Research Team: Building Practical Feedback Loops for Production LLMs

*Closing the loop with customer signals, preference datasets, and automated retraining — LoyaltyLens Module 6*

---


---

RLHF — Reinforcement Learning from Human Feedback — is the technique that turned GPT-3 into ChatGPT. Anthropic used it to build Claude. OpenAI used it to make their models safe and helpful. In both cases, it involved massive amounts of human annotation, proprietary reward models, and significant computational resources.

When I talk about "RLHF-style feedback loops" in the context of a large-scale loyalty program, I mean something much more pragmatic: a system that captures customer and campaign manager signals, uses those signals to measure whether the AI system is getting better or worse over time, and automatically triggers retraining when the signals indicate degradation.

It's not academic RLHF. It's the same principle applied to a production system with real constraints.

This post is about how I built that system in LoyaltyLens Module 6, and what the equivalent looks like at enterprise scale.

---

## The Signal Problem

Before you can build a feedback loop, you need to decide what signal you're collecting and what it actually measures.

In production, we have three signal types:

**1. Offer redemption** — Did the customer redeem the offer? This is the ground truth signal, but it arrives 3–7 days after the offer send. Too slow for rapid iteration, but excellent for long-horizon model evaluation.

**2. Campaign manager review** — Before a campaign goes out, a campaign manager reviews a sample of generated copies. Their approve/reject decisions are the highest-quality signal we have, but they're expensive (human time) and not available at scale.

**3. In-app engagement** — Did the customer open the notification? Did they tap on the offer? These are noisier signals (opening a notification and ignoring it tells you less than redeeming the offer), but they're available in real time.

For LoyaltyLens, I implemented a simplified version: a thumbs up/down + 1–5 star rating from a demo UI that simulates the campaign manager review signal. In production you'd augment this with the engagement and redemption signals.

---

## The Feedback UI

The UI is a single-page React application built with Vite and Tailwind. It's deliberately minimal — its purpose is to collect signal, not to be a product:

```tsx
// feedback_loop/ui/src/App.tsx
import { useState, useEffect } from "react"

interface OfferCopy {
  headline: string
  body: string
  cta: string
  tone: string
  offer_id: string
  customer_id: string
  prompt_version: number
  model_version: string
}

export default function FeedbackApp() {
  const [copy, setCopy] = useState<OfferCopy | null>(null)
  const [rating, setRating] = useState<number>(0)
  const [thumbs, setThumbs] = useState<"up" | "down" | null>(null)
  const [submitted, setSubmitted] = useState(false)

  useEffect(() => { fetchNextOffer() }, [])

  async function fetchNextOffer() {
    // Calls the M4 /generate endpoint with a random customer
    const res = await fetch("/api/generate/random")
    setCopy(await res.json())
    setRating(0)
    setThumbs(null)
    setSubmitted(false)
  }

  async function submitFeedback() {
    await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        offer_id: copy!.offer_id,
        customer_id: copy!.customer_id,
        generated_copy: JSON.stringify(copy),
        rating,
        thumbs,
        prompt_version: copy!.prompt_version,
        model_version: copy!.model_version,
      }),
    })
    setSubmitted(true)
    setTimeout(fetchNextOffer, 1000)
  }

  return (
    <div className="max-w-2xl mx-auto p-8 font-sans">
      {copy && (
        <div className="bg-white rounded-xl shadow-lg p-6 space-y-4">
          <h1 className="text-2xl font-bold text-green-800">{copy.headline}</h1>
          <p className="text-gray-700 leading-relaxed">{copy.body}</p>
          <button className="bg-green-700 text-white px-6 py-2 rounded-full font-semibold">
            {copy.cta}
          </button>

          <div className="border-t pt-4 space-y-4">
            {/* Star rating */}
            <div className="flex gap-2">
              {[1,2,3,4,5].map(n => (
                <button key={n}
                  onClick={() => setRating(n)}
                  className={`text-2xl ${n <= rating ? "text-yellow-400" : "text-gray-300"}`}>
                  ★
                </button>
              ))}
            </div>

            {/* Thumbs */}
            <div className="flex gap-4">
              <button onClick={() => setThumbs("up")}
                className={`px-4 py-2 rounded ${thumbs === "up" ? "bg-green-100 border-2 border-green-500" : "bg-gray-100"}`}>
                👍 On-brand
              </button>
              <button onClick={() => setThumbs("down")}
                className={`px-4 py-2 rounded ${thumbs === "down" ? "bg-red-100 border-2 border-red-500" : "bg-gray-100"}`}>
                👎 Off-brand
              </button>
            </div>

            <button
              onClick={submitFeedback}
              disabled={!thumbs || rating === 0}
              className="w-full bg-green-700 text-white py-3 rounded-lg disabled:opacity-40">
              Submit & Next
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
```

Simple. Functional. Takes 30 seconds to understand and 5 seconds to use. Both of those things matter if you want campaign managers to actually use it.

---

## The Feedback Persistence Layer

Every submission gets written to SQLite with full provenance:

```python
# feedback_loop/api.py
from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3

app = FastAPI()

class FeedbackRecord(BaseModel):
    offer_id: str
    customer_id: str
    generated_copy: str      # JSON string of OfferCopy
    rating: int              # 1-5
    thumbs: str              # "up" | "down"
    prompt_version: int
    model_version: str

@app.post("/feedback")
async def record_feedback(feedback: FeedbackRecord):
    with sqlite3.connect("feedback_loop/data/feedback.db") as conn:
        conn.execute("""
            INSERT INTO feedback
            (offer_id, customer_id, generated_copy, rating, thumbs,
             prompt_version, model_version, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """, (
            feedback.offer_id, feedback.customer_id,
            feedback.generated_copy, feedback.rating,
            feedback.thumbs, feedback.prompt_version,
            feedback.model_version,
        ))
    return {"status": "recorded"}

@app.get("/feedback/stats")
async def get_stats():
    """Aggregate stats per prompt and model version."""
    with sqlite3.connect("feedback_loop/data/feedback.db") as conn:
        rows = conn.execute("""
            SELECT
                prompt_version,
                model_version,
                COUNT(*) as n,
                AVG(rating) as avg_rating,
                SUM(CASE WHEN thumbs = 'up' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as thumbs_up_rate
            FROM feedback
            WHERE created_at >= datetime('now', '-7 days')
            GROUP BY prompt_version, model_version
            ORDER BY prompt_version DESC
        """).fetchall()
    return {"stats": rows}
```

The `prompt_version` and `model_version` fields on every record are what make this analytically useful. You can answer: "Did the prompt v2 change actually improve campaign manager approval rates?" That's a question that can't be answered without the versioning system from Module 5.

---

## The Feedback Aggregator

```python
# feedback_loop/aggregator.py
from dataclasses import dataclass
import sqlite3
import numpy as np

@dataclass
class FeedbackStats:
    avg_rating: float
    thumbs_up_rate: float
    thumbs_down_rate: float
    record_count: int
    by_prompt_version: dict
    by_model_version: dict
    by_category: dict

class FeedbackAggregator:
    def compute_stats(self, since_days: int = 7) -> FeedbackStats:
        with sqlite3.connect("feedback_loop/data/feedback.db") as conn:
            rows = conn.execute("""
                SELECT rating, thumbs, prompt_version, model_version
                FROM feedback
                WHERE created_at >= datetime('now', ?)
            """, [f"-{since_days} days"]).fetchall()

        if not rows:
            return FeedbackStats(avg_rating=0, thumbs_up_rate=0,
                                 thumbs_down_rate=0, record_count=0,
                                 by_prompt_version={}, by_model_version={},
                                 by_category={})

        ratings    = [r[0] for r in rows]
        thumbs_up  = sum(1 for r in rows if r[1] == "up")

        return FeedbackStats(
            avg_rating=float(np.mean(ratings)),
            thumbs_up_rate=thumbs_up / len(rows),
            thumbs_down_rate=1 - (thumbs_up / len(rows)),
            record_count=len(rows),
            by_prompt_version=self._group_by(rows, key_idx=2),
            by_model_version=self._group_by(rows, key_idx=3),
            by_category={},  # populated when offer category is joined
        )
```

---

## The Preference Dataset Builder

This is where LoyaltyLens's feedback loop connects to the broader RLHF literature.

The OpenAI fine-tuning format for RLHF uses chosen/rejected pairs — examples of preferred and dispreferred outputs for the same input. LoyaltyLens generates these automatically from the feedback data:

```python
def export_preference_dataset(self, output_path: str) -> int:
    """
    Export feedback as JSONL preference pairs for RLHF fine-tuning.
    Format: {"prompt": "...", "chosen": "...", "rejected": "..."}
    """
    with sqlite3.connect("feedback_loop/data/feedback.db") as conn:
        # Match thumbs-up and thumbs-down for the same offer
        rows = conn.execute("""
            SELECT
                f1.generated_copy as chosen_copy,
                f2.generated_copy as rejected_copy,
                f1.offer_id
            FROM feedback f1
            JOIN feedback f2 ON f1.offer_id = f2.offer_id
            WHERE f1.thumbs = 'up' AND f2.thumbs = 'down'
            LIMIT 10000
        """).fetchall()

    records = []
    for chosen_raw, rejected_raw, offer_id in rows:
        chosen   = json.loads(chosen_raw)
        rejected = json.loads(rejected_raw)
        records.append({
            "prompt": self._build_prompt_for_offer(offer_id),
            "chosen": f"{chosen['headline']} {chosen['body']} {chosen['cta']}",
            "rejected": f"{rejected['headline']} {rejected['body']} {rejected['cta']}",
        })

    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    return len(records)
```

This dataset is the asset that enables future fine-tuning. You wouldn't necessarily use it to fine-tune the base model — that's expensive and requires careful infrastructure. But you could use it to fine-tune a smaller reward model, or to train a classifier that replaces the LLM judge in the evaluation harness.

In production, similar feedback data was one of the inputs to reinforcement learning from campaign performance — using campaign outcomes to shape the reward signal for offer selection decisions.

---

## The Retraining Trigger

```python
# feedback_loop/trigger.py
import json
import requests
from feedback_loop.aggregator import FeedbackAggregator

class RetrainingTrigger:
    def __init__(self, config):
        self.config = config
        self.aggregator = FeedbackAggregator()

    def should_retrain(self, stats: FeedbackStats) -> tuple[bool, str]:
        if stats.avg_rating < 3.0:
            return True, f"avg_rating={stats.avg_rating:.2f} < threshold=3.0"
        if stats.thumbs_down_rate > 0.40:
            return True, f"thumbs_down_rate={stats.thumbs_down_rate:.1%} > threshold=40%"
        if stats.record_count < 10:
            return False, "insufficient_data"
        return False, "metrics_healthy"

    def fire_trigger(self, reason: str) -> None:
        # 1. Log the trigger event
        event = {
            "triggered_at": datetime.now().isoformat(),
            "reason": reason,
            "action": "retrain_propensity_model + prompt_rollback_check",
        }
        trigger_log = Path("feedback_loop/data/trigger_log.json")
        history = json.loads(trigger_log.read_text()) if trigger_log.exists() else []
        history.append(event)
        trigger_log.write_text(json.dumps(history, indent=2))

        logger.critical(f"Retraining trigger fired: {reason}")

        # 2. Fire GitHub Actions workflow_dispatch
        if self.config.GITHUB_TOKEN and self.config.GITHUB_REPO:
            requests.post(
                f"https://api.github.com/repos/{self.config.GITHUB_REPO}"
                f"/actions/workflows/retrain.yml/dispatches",
                headers={"Authorization": f"token {self.config.GITHUB_TOKEN}"},
                json={"ref": "main", "inputs": {"reason": reason}},
            )

    def check_and_trigger(self) -> None:
        stats = self.aggregator.compute_stats(since_days=7)
        should, reason = self.should_retrain(stats)
        if should:
            self.fire_trigger(reason)
        else:
            logger.info(f"Trigger check passed: {reason}")
```

```bash
# Run as a daily cron (or GitHub Actions scheduled workflow)
python feedback_loop/trigger.py --check
```

The `should_retrain` method is deliberately conservative. I've seen teams build triggers that fire on every minor dip and end up retraining weekly with no improvement — the noise in the feedback signal is larger than the signal itself. The thresholds (avg < 3.0, thumbs_down > 40%) represent genuine degradation, not normal variance.

---

## What Closes the Loop

The complete LoyaltyLens feedback loop looks like this:

```
Offer generated (M4)
    → Shown to campaign manager in UI
    → Rating + thumbs captured (M6 API)
    → Aggregated nightly (M6 aggregator)
    → Trigger checks thresholds (M6 trigger)
    → If triggered:
        → GitHub Actions workflow_dispatch
        → Retrains propensity model (M2 train.py)
        → Runs eval harness (M5 eval gate)
        → If eval passes: promotes new model version
        → If eval fails: alerts + rolls back prompt
    → PSI monitor runs (M5 drift)
    → Dashboard updated (M5 dashboard)
    → Preference dataset grows (M6 exporter)
```

Every arrow in that diagram is a code path that runs without human intervention. The human decisions are: setting the thresholds, reviewing the trigger log, and deciding whether to accept a promoted model. The automation handles the rest.

---

## The Honest Limits

There are three things this feedback loop does not do, and it's important to be honest about them.

**It does not implement full RLHF.** True RLHF trains a reward model on the preference data, then uses that reward model in a reinforcement learning loop to fine-tune the base LLM. That's computationally expensive and requires the kind of infrastructure that research labs run. What LoyaltyLens implements is the feedback *collection* and *trigger* infrastructure — the foundation that RLHF would be built on.

**It does not separate signal from noise.** The thumbs up/down signal from a campaign manager is a noisy proxy for customer value. A customer who thumbs-up an offer because it looks beautiful but doesn't redeem it is a false positive in the preference dataset. In production you'd weight the feedback signal by downstream redemption data.

**It assumes the feedback is honest.** In a production system with incentive structures (campaign managers might approve more copies to hit throughput targets), the feedback quality degrades. Building adversarial robustness into the signal collection is a real engineering challenge.

These limits don't make the system useless — they define the engineering work that remains to be done as the system scales. That's true of every production AI system I've worked on.

---

## The End of the Series: What This All Adds Up To

LoyaltyLens, taken as a whole, is a production-pattern demonstration of every component in a modern loyalty AI system:

- A feature pipeline that enforces data quality at the store boundary
- A propensity model with a model card, eval metrics, and a deployment path
- RAG retrieval benchmarked across frameworks and vector databases
- LLM-generated copy with prompt versioning and brand alignment checking
- An LLMOps pipeline with drift monitoring, automated evaluation, and CI/CD gates
- A feedback loop that turns human signals into automated model improvement triggers

The codebase is open. The architecture is documented. The prompts are versioned.

If you're building something similar, I'd be glad to talk through the design decisions. The contact link is below.

**[← Back to Module 5: The LLMOps Pipeline](#)** | **[View the LoyaltyLens repository on GitHub](#)**

---

*Pushparajan Ramar is an Enterprise Architect Director specializing in AI, data, and platform architecture for large-scale enterprise programs. He lives in the Greater Chicago Area.*

*[LinkedIn](https://linkedin.com/in/pushparajanramar) · [pushparajan.tech](https://pushparajan.tech)*
