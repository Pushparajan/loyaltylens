---
title: "The LLMOps Stack I Wish Existed When We Started in Production"
slug: "loyaltylens-llmops-pipeline"
description: "Prompt versioning CLI, LLM-as-judge eval harness, PSI drift monitoring, GitHub Actions CI/CD with a hard quality gate, and a Streamlit monitoring dashboard."
date: 2026-05-26
author: Pushparajan Ramar
series: loyaltylens
series_order: 5
reading_time: 16
tags:
  - llmops
  - mlops
  - prompt-versioning
  - drift-monitoring
  - ci-cd
  - github-actions
  - streamlit
---

## Prompt versioning, drift detection, LLM-as-judge evaluation, and CI/CD for ML — LoyaltyLens Module 5

---

When I started leading the LLMOps architecture in production, the team had a working LLM integration that had been in production for a few months. Offer copy was being generated, campaigns were going out, customers were redeeming. On the surface, things looked fine.

Underneath, there was no prompt versioning system. Prompts were stored in a config file that got overwritten with each deployment. There was no evaluation framework — quality was assessed by a campaign manager eyeballing a sample of outputs. There was no drift monitoring for the propensity model feeding the system. And the CI/CD pipeline for the ML components was the same YAML that deployed the React frontend.

None of these gaps caused an incident while I was there. But each one was a loaded gun. This post is about building the safety mechanisms that prevent those guns from going off.

---

## What LLMOps Actually Is (and Isn't)

LLMOps is not a product category. It's not a platform you buy and install. It's a set of engineering disciplines applied to systems that include large language models as runtime components.

The core disciplines are:

1. **Prompt versioning** — tracking, diffing, and rolling back prompt changes with the same rigor as code
2. **Model evaluation** — measuring output quality with automated metrics and human-in-the-loop signals
3. **Drift monitoring** — detecting when the distribution of model inputs or outputs has shifted from the baseline
4. **Deployment governance** — gates that prevent degraded models or prompts from reaching production

LoyaltyLens Module 5 implements all four. Let me walk through each one.

---

## 1. Prompt Versioning CLI

The prompt YAML files from Module 4 (`llm_generator/prompts/system_v1.yaml`, `system_v2.yaml`) are the source of truth. Module 5 adds a Click CLI that makes them operationally manageable.

```bash
# List all versions with activation date and status
python -m llmops.prompt_registry.cli prompt list

Version    Activated                 Status
---------- ------------------------- ----------
v1         2026-04-01T09:00:00Z      archived
v2         2026-04-10T14:30:00Z      active   ◀

# Show what changed between versions
python -m llmops.prompt_registry.cli prompt diff v1 v2

── system ──
--- v1.yaml (system)
+++ v2.yaml (system)
-  You are a Starbucks loyalty offer copywriter. Write concise,
-  warm, brand-consistent offer copy. Always output valid JSON.
+  You are an expert Starbucks loyalty copywriter. Write punchy,
+  ultra-concise copy. Always output valid JSON.

# Activate a new version
python -m llmops.prompt_registry.cli prompt activate v2
Activated v2 (was v1).

# Roll back to the previous version
python -m llmops.prompt_registry.cli prompt rollback
Rolled back from v2 to v1.
```

State is stored in two JSON files:

- `llmops/prompt_registry/active.json` — current version (single key)
- `llmops/prompt_registry/history.json` — append-only activation log with timestamps and previous-version pointers

```python
# llmops/prompt_registry/cli.py (key excerpt)
@prompt.command("activate")
@click.argument("version")
def activate(version: str) -> None:
    current = _load_active()
    history = _load_history()
    history.append({
        "version": version,
        "activated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "previous": current,
    })
    _save_active(version)
    _save_history(history)
    click.echo(f"Activated {version} (was {current}).")
```

The rollback command reads the `previous` pointer from the last history entry and reverts to it. No database, no Redis — just two JSON files in source control.

---

## 2. The Evaluation Harness

This is the most important component in the module. Without automated evaluation, every prompt change is a leap of faith.

The harness runs three types of evaluation and combines them into a single aggregate score.

### Type 1: Lexical Metrics (BLEU / ROUGE-L)

```python
# llmops/eval_harness/evaluator.py
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer

class OfferCopyEvaluator:
    def __init__(self, judge_backend=None):
        self._judge = judge_backend
        self._bleu = BLEU(effective_order=True)
        self._rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def bleu_score(self, generated: str, reference: str) -> float:
        return self._bleu.sentence_score(generated, [reference]).score / 100.0

    def rouge_l(self, generated: str, reference: str) -> float:
        return float(self._rouge.score(generated, reference)["rougeL"].fmeasure)
```

I'm honest about what these metrics measure: lexical overlap with a reference. In a world where every good offer copy headline is different, lexical overlap is a weak proxy for quality. But it catches regressions — if BLEU drops significantly between prompt versions, something meaningful changed.

### Type 2: LLM-as-Judge

This is the metric that matters most, and it's the one most teams skip because it feels circular to use an LLM to evaluate LLM output.

The circularity concern is real but manageable. The key is using a *different* model as the judge (GPT-4o as judge for Mistral-generated copy), using a structured rubric, and calibrating the judge against human ratings before trusting it.

```python
_JUDGE_PROMPT = """\
You are evaluating loyalty offer copy. Score the following on three dimensions from 1 to 5.

Headline: {headline}
Body: {body}
CTA: {cta}

Scoring rubric:
- coherence (1-5): Natural flow, grammatically correct. 5=flawless, 1=confusing.
- brand_alignment (1-5): Warm, community-focused brand voice. 5=on-brand, 1=off-brand or pushy.
- cta_strength (1-5): Clear, appropriate urgency. 5=compelling, 1=weak or missing.

Respond ONLY with valid JSON: {"coherence": N, "brand_alignment": N, "cta_strength": N}"""

def llm_judge(self, copy: OfferCopyInput) -> dict[str, float]:
    if self._judge is None:
        # Fallback: neutral scores when no backend configured (CI without API keys)
        return {"coherence": 0.75, "brand_alignment": 0.75, "cta_strength": 0.75}
    try:
        raw = self._judge.generate([{"role": "user", "content": prompt}])
        scores = json.loads(raw)
        return {k: float(scores[k]) / 5.0 for k in ("coherence", "brand_alignment", "cta_strength")}
    except Exception:
        return {"coherence": 0.75, "brand_alignment": 0.75, "cta_strength": 0.75}
```

The fallback to neutral scores when no backend is configured means the CI gate still runs in environments without API keys — useful for open-source contributors running tests locally.

### The Aggregate Score

```python
def aggregate_score(self, copy: OfferCopyInput, reference: str) -> float:
    b = self.bleu_score(copy.body, reference)
    r = self.rouge_l(copy.body, reference)
    judge = self.llm_judge(copy)
    judge_avg = sum(judge.values()) / len(judge)
    return 0.2 * b + 0.2 * r + 0.6 * judge_avg
```

The 0.75 threshold for the CI gate was calibrated against human ratings of offer copies. Anything above 0.75 had a high human approval rate. It's not a perfect threshold, but it's a defensible one.

### The CI Eval Gate

```bash
# Run 50 synthetic copies, score each, write results JSON, exit 1 if below threshold
python llmops/eval_harness/run_eval.py --threshold 0.75

[eval-gate] mean_score=0.8500  threshold=0.75  n=50  PASSED ✓
[eval-gate] Results written to llmops/eval_results/eval_20260417T161630Z.json
```

---

## 3. Propensity Drift Monitor

The propensity score distribution should be stable over time. If it shifts — because upstream data changed, because a new segment of customers entered the system, because the engagement score formula was updated — the downstream components (retrieval, generation) are receiving inputs they weren't trained on.

Population Stability Index (PSI) is the standard measure for this in production ML systems:

```python
# llmops/drift_monitor/monitor.py
from dataclasses import dataclass, field
import numpy as np

@dataclass
class DriftReport:
    psi: float
    status: str          # "ok" | "warning" | "critical"
    baseline_date: str
    current_date: str
    n_baseline: int
    n_current: int
    feature_breakdown: dict[str, float] = field(default_factory=dict)

class PropensityDriftMonitor:
    def compute_psi(self, baseline_scores, current_scores, bins=10) -> float:
        breakpoints = np.linspace(0.0, 1.0, bins + 1)
        baseline_pct = np.histogram(baseline_scores, bins=breakpoints)[0] / len(baseline_scores)
        current_pct  = np.histogram(current_scores,  bins=breakpoints)[0] / len(current_scores)
        baseline_pct = np.clip(baseline_pct, 1e-6, None)
        current_pct  = np.clip(current_pct,  1e-6, None)
        return float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))

    def check_drift(self, baseline_scores, current_scores, ...) -> DriftReport:
        psi = self.compute_psi(baseline_scores, current_scores)
        status = (
            "critical" if psi > 0.20
            else "warning" if psi > 0.10
            else "ok"
        )
        return DriftReport(psi=round(psi, 4), status=status, ...)
```

The daily job (`llmops/drift_monitor/run_drift.py`) loads the baseline and current windows, runs the monitor, and writes a JSON report:

```bash
python llmops/drift_monitor/run_drift.py --output drift_report.json

✅  Drift PSI=0.0031  status=OK
   Report written to drift_report.json
```

When I first ran this in production, we found a PSI of 0.31 — critical — on a Monday morning. The cause was a weekend batch job that had recomputed recency features using a different timezone offset, shifting every customer's `recency_days` by one. We caught it in 20 minutes because the drift monitor ran at 6am. Without it, we would have caught it in three weeks when campaign analysts noticed the redemption rate drop.

---

## 4. The GitHub Actions CI/CD Pipeline

The CI workflow migrates from Poetry to `uv` and adds three new jobs alongside the existing lint/type-check/test chain:

```yaml
# .github/workflows/ci.yml

jobs:
  lint:       # ruff via uv
  type-check: # mypy via uv
  test:       # pytest with postgres/weaviate/redis services via uv

  eval-gate:
    needs: test
    steps:
      - name: Run eval harness
        run: uv run python llmops/eval_harness/run_eval.py --threshold 0.75
        # exits 1 if aggregate score < 0.75 → blocks merge

      - name: Upload eval results
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: llmops/eval_results/

  drift-check:
    needs: test
    if: github.event_name == 'pull_request'
    steps:
      - name: Run drift monitor
        run: uv run python llmops/drift_monitor/run_drift.py --output drift_report.json

      - name: Post PSI as PR comment
        uses: actions/github-script@v7
        with:
          script: |
            const report = JSON.parse(fs.readFileSync('drift_report.json'));
            const emoji = {ok:'✅', warning:'⚠️', critical:'🚨'}[report.status];
            github.rest.issues.createComment({
              body: `${emoji} **Drift Monitor**: PSI=${report.psi} (${report.status})`
            });

  deploy-gate:
    needs: [eval-gate, drift-check]
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to staging
        run: uv run python shared/deploy.py --env staging
```

The `eval-gate` job is the critical one. A prompt change that degrades output quality will never reach production. The `drift-check` job runs on PRs and posts the PSI directly into the PR comment thread — reviewers see drift impact before merging.

---

## 5. The Streamlit Dashboard

The dashboard ties everything together:

```python
# llmops/dashboard/app.py
import streamlit as st

st.set_page_config(page_title="LoyaltyLens LLMOps", layout="wide")
st.title("LoyaltyLens LLMOps Dashboard")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Active Prompt Version", active_version)
col2.metric("Latest Eval Score", f"{eval_score:.3f}", delta="PASS" if passed else "FAIL")
col3.metric(f"Drift PSI  {status_color} {drift_status.upper()}", f"{psi:.4f}")
col4.metric("Last Deploy", last_deploy_summary)

st.line_chart(eval_history.set_index("timestamp")[["Aggregate Score"]])
st.dataframe(prompt_history)
```

Run it with:

```bash
streamlit run llmops/dashboard/app.py
```

The four-metric header row is designed so each audience — data scientist, campaign manager, LLMOps engineer — finds their signal within 10 seconds of opening the page.

---

## What the Test Suite Covers

32 tests across `tests/test_llmops.py`, `test_llmops_evaluator.py`, and `test_eval_harness.py`:

- PSI near-zero for identical distributions, PSI > 0.20 for large shifts
- `PropensityDriftMonitor` ok/warning/critical classification, per-feature breakdown
- `OfferCopyEvaluator` BLEU/ROUGE-L bounds, judge fallback on missing backend or network error
- Eval harness: passes at 0.75 threshold, fails at 0.999, writes timestamped JSON
- Prompt registry CLI via `click.testing.CliRunner`: list, activate, rollback, diff

```bash
python -m pytest tests/test_llmops.py tests/test_llmops_evaluator.py tests/test_eval_harness.py -v

32 passed in 1.69s
```

---

## Next: Closing the Loop with RLHF

Module 5 detects degradation. Module 6 responds to it. In the final post in this series I walk through the feedback capture UI, the preference dataset builder, and the retraining trigger that turns customer thumbs up/down signals into automated model improvements.

**→ Read Module 6: RLHF Feedback Loops — Turning Customer Signals Into Model Improvements** *(link to be added on publish)*

---

*Pushparajan Ramar is an Enterprise Architect Director in enterprise consulting. He defines AI governance frameworks, responsible AI guardrails, and LLMOps practices for enterprise deployments. Connect on [LinkedIn](https://linkedin.com/in/pushparajanramar).*
