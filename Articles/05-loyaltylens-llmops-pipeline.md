---
title: "The LLMOps Stack I Wish Existed When We Started in Production"
slug: "loyaltylens-llmops-pipeline"
description: "Prompt versioning CLI, LLM-as-judge eval harness, PSI drift monitoring, GitHub Actions CI/CD with a hard quality gate, and a responsible AI audit framework."
date: 2025-10-06
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
  - responsible-ai
---

# The LLMOps Stack I Wish Existed When We Started in Production

*Prompt versioning, drift detection, LLM-as-judge evaluation, and CI/CD for ML — LoyaltyLens Module 5*

---


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

The prompt registry from Module 4 stores prompt YAML files. Module 5 adds a CLI that makes those files operationally manageable:

```
$ llmops prompt list

┌─────────┬─────────────────────┬──────────────┬──────────┐
│ Version │ Created             │ Author       │ Status   │
├─────────┼─────────────────────┼──────────────┼──────────┤
│ v1      │ 2024-01-15 09:12:44 │ P. Ramar     │ archived │
│ v2      │ 2024-02-03 14:30:11 │ P. Ramar     │ active   │
│ v3      │ 2024-02-18 16:45:02 │ P. Ramar     │ staging  │
└─────────┴─────────────────────┴──────────────┴──────────┘

$ llmops prompt diff v1 v2

--- system_v1.yaml (system prompt)
+++ system_v2.yaml (system prompt)
  You are a loyalty program offer copywriter with deep knowledge of
  the brand voice: warm, personal, community-focused, and
- never pushy.
+ never pushy. Never use the word "deal".

  You write copy that feels like a message from a friend, not an advertisement.
```

```python
# llmops/prompt_registry/cli.py
import click
import json
from pathlib import Path
import difflib

REGISTRY_PATH = Path("llmops/prompt_registry")
HISTORY_FILE  = REGISTRY_PATH / "history.json"
ACTIVE_FILE   = REGISTRY_PATH / "active.json"

@click.group()
def cli(): pass

@cli.command()
def list():
    """Show all prompt versions with status."""
    history = json.loads(HISTORY_FILE.read_text())
    active  = json.loads(ACTIVE_FILE.read_text())["version"]
    # ... render table

@cli.command()
@click.argument("v1")
@click.argument("v2")
def diff(v1, v2):
    """Show diff between two prompt versions."""
    p1 = yaml.safe_load((REGISTRY_PATH / f"system_{v1}.yaml").read_text())
    p2 = yaml.safe_load((REGISTRY_PATH / f"system_{v2}.yaml").read_text())
    
    for field in ["system", "user_template"]:
        lines1 = p1[field].splitlines(keepends=True)
        lines2 = p2[field].splitlines(keepends=True)
        diff_lines = list(difflib.unified_diff(
            lines1, lines2,
            fromfile=f"{v1}.yaml ({field})",
            tofile=f"{v2}.yaml ({field})",
        ))
        if diff_lines:
            click.echo(f"\n--- {field} ---")
            click.echo("".join(diff_lines))

@cli.command()
def rollback():
    """Revert active prompt to previous version."""
    history = json.loads(HISTORY_FILE.read_text())
    active  = json.loads(ACTIVE_FILE.read_text())["version"]
    versions = [h["version"] for h in history]
    prev_version = versions[versions.index(active) - 1]
    
    ACTIVE_FILE.write_text(json.dumps({"version": prev_version}))
    click.echo(f"Rolled back from {active} to {prev_version}")
```

The rollback command is the one that saves you at 2am. In production we've executed prompt rollbacks twice — both times after an automated eval score drop, not after a customer complaint.

---

## 2. The Evaluation Harness

This is the most important component in the module. Without automated evaluation, every prompt change is a leap of faith.

The harness runs three types of evaluation:

### Type 1: Lexical Metrics (BLEU / ROUGE-L)

```python
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer

bleu = BLEU(effective_order=True)
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def bleu_score(generated: str, reference: str) -> float:
    return bleu.sentence_score(generated, [reference]).score / 100

def rouge_l_score(generated: str, reference: str) -> float:
    return rouge.score(generated, reference)["rougeL"].fmeasure
```

I'm honest about what these metrics measure: lexical overlap with a reference. In a world where every good offer copy headline is different, lexical overlap is a weak proxy for quality. But it catches regressions — if BLEU drops by more than 15 points between prompt versions, something meaningful changed.

### Type 2: LLM-as-Judge

This is the metric that matters most, and it's the one most teams skip because it feels circular to use an LLM to evaluate LLM output.

The circularity concern is real but manageable. The key is using a *different* model as the judge (GPT-4o as judge for Mistral-generated copy), using a structured rubric, and calibrating the judge against human ratings before trusting it.

```python
JUDGE_PROMPT = """
You are evaluating offer copy for a loyalty program.
Score the following copy on three dimensions from 1 to 5:

Copy to evaluate:
Headline: {headline}
Body: {body}
CTA: {cta}

Scoring rubric:
- coherence (1-5): Does the copy flow naturally? Is it grammatically correct?
  5=flawless, 3=minor issues, 1=confusing or broken
- brand_alignment (1-5): Does it sound like a warm, community-focused brand?
  5=perfectly on-brand, 3=generic, 1=off-brand or pushy
- cta_strength (1-5): Does the CTA create clear, appropriate urgency?
  5=compelling, 3=generic, 1=weak or missing

Respond ONLY with valid JSON: {{"coherence": N, "brand_alignment": N, "cta_strength": N}}
"""

class OfferCopyEvaluator:
    def llm_judge(self, copy: OfferCopy) -> dict[str, float]:
        prompt = JUDGE_PROMPT.format(
            headline=copy.headline,
            body=copy.body,
            cta=copy.cta,
        )
        raw = self.judge_backend.generate(system="", user=prompt)
        scores = json.loads(raw)
        # Normalize to [0, 1]
        return {k: v / 5 for k, v in scores.items()}
```

### The Aggregate Score

```python
def aggregate_score(self, copy: OfferCopy, reference: str) -> float:
    bleu    = self.bleu_score(copy.body, reference)
    rouge   = self.rouge_l_score(copy.body, reference)
    judge   = self.llm_judge(copy)

    return (
        0.20 * bleu
        + 0.20 * rouge
        + 0.20 * judge["coherence"]
        + 0.20 * judge["brand_alignment"]
        + 0.20 * judge["cta_strength"]
    )
```

The 0.75 threshold for the CI gate — fail the build if aggregate score drops below 0.75 — was calibrated against human ratings of 100 offer copies. Anything above 0.75 had >90% human approval rate. It's not a perfect threshold, but it's a defensible one.

---

## 3. Propensity Drift Monitor

The propensity score distribution should be stable over time. If it shifts — because upstream data changed, because a new segment of customers entered the system, because the engagement score formula was updated — the downstream components (retrieval, generation) are receiving inputs they weren't trained on.

Population Stability Index (PSI) is the standard measure for this in production ML systems:

```python
# llmops/drift_monitor/monitor.py
import numpy as np
from dataclasses import dataclass

@dataclass
class DriftReport:
    psi: float
    status: str          # "ok" | "warning" | "critical"
    baseline_date: str
    current_date: str
    n_baseline: int
    n_current: int

class PropensityDriftMonitor:
    def compute_psi(
        self,
        baseline_scores: np.ndarray,
        current_scores: np.ndarray,
        bins: int = 10,
    ) -> float:
        breakpoints = np.linspace(0, 1, bins + 1)
        
        baseline_pct = np.histogram(baseline_scores, bins=breakpoints)[0] / len(baseline_scores)
        current_pct  = np.histogram(current_scores,  bins=breakpoints)[0] / len(current_scores)
        
        # Clip to avoid log(0)
        baseline_pct = np.clip(baseline_pct, 1e-6, None)
        current_pct  = np.clip(current_pct,  1e-6, None)
        
        psi = np.sum(
            (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)
        )
        return float(psi)

    def check_drift(self, threshold_warning=0.1, threshold_critical=0.2) -> DriftReport:
        baseline = self._load_baseline()
        current  = self._load_current_scores()
        
        psi = self.compute_psi(baseline["scores"], current)
        
        status = (
            "critical" if psi > threshold_critical
            else "warning" if psi > threshold_warning
            else "ok"
        )
        
        return DriftReport(
            psi=round(psi, 4),
            status=status,
            baseline_date=baseline["date"],
            current_date=datetime.now().isoformat(),
            n_baseline=len(baseline["scores"]),
            n_current=len(current),
        )
```

When I first ran this in production, we found a PSI of 0.31 — critical — on a Monday morning. The cause was a weekend batch job that had recomputed recency features using a different timezone offset, shifting every customer's `recency_days` by one. The propensity model's input distribution had shifted, which meant the scores were wrong, which meant the campaign sends that morning were misconfigured.

We caught it in 20 minutes because the drift monitor ran at 6am. Without it, we would have caught it in three weeks when campaign analysts noticed the redemption rate drop.

---

## 4. The GitHub Actions CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: LoyaltyLens CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Lint with ruff
        run: ruff check .

  type-check:
    needs: lint
    runs-on: ubuntu-latest
    steps:
      - name: Type check with mypy
        run: mypy loyaltylens/ --ignore-missing-imports

  unit-tests:
    needs: type-check
    runs-on: ubuntu-latest
    services:
      postgres:
        image: pgvector/pgvector:pg16
        env:
          POSTGRES_PASSWORD: test
    steps:
      - name: Run pytest
        run: pytest tests/ -v --tb=short

  eval-gate:
    needs: unit-tests
    runs-on: ubuntu-latest
    steps:
      - name: Run evaluation harness
        run: python llmops/eval_harness/run_eval.py
        # exits with code 1 if aggregate_score < 0.75
      - name: Upload eval results
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: llmops/eval_results/

  drift-check:
    needs: unit-tests
    runs-on: ubuntu-latest
    steps:
      - name: Check propensity drift
        run: python llmops/drift_monitor/run_drift.py --output drift_report.json
      - name: Post drift PSI as PR comment
        uses: actions/github-script@v7
        with:
          script: |
            const report = JSON.parse(require('fs').readFileSync('drift_report.json'));
            const emoji = report.status === 'ok' ? '✅' : report.status === 'warning' ? '⚠️' : '🚨';
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              body: `${emoji} **Drift Monitor**: PSI = ${report.psi} (${report.status})`
            });

  deploy:
    needs: [eval-gate, drift-check]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to staging
        run: python shared/deploy.py --env staging
```

The `eval-gate` job is the critical one. It runs 50 offer copy generations, scores them, and fails the build if the aggregate score drops below the threshold. A prompt change that degrades output quality will never reach production.

---

## 5. The MLflow Dashboard

The Streamlit dashboard ties everything together in one view:

```python
# llmops/dashboard/app.py
import streamlit as st
import mlflow
import pandas as pd

st.set_page_config(page_title="LoyaltyLens LLMOps", layout="wide")
st.title("LoyaltyLens LLMOps Dashboard")

col1, col2, col3 = st.columns(3)

# Active model
runs = mlflow.search_runs(order_by=["metrics.best_val_auc DESC"], max_results=1)
best_run = runs.iloc[0]
col1.metric("Active Model", f"v{best_run['tags.version']}")
col1.metric("Val AUC", f"{best_run['metrics.best_val_auc']:.3f}")

# Latest eval score
latest_eval = load_latest_eval()
col2.metric("Eval Score", f"{latest_eval['aggregate_score']:.3f}",
            delta=f"{latest_eval['delta']:+.3f} vs previous")

# Drift status
drift_report = load_drift_report()
status_color = {"ok": "green", "warning": "orange", "critical": "red"}
col3.metric("Drift PSI", f"{drift_report['psi']:.4f}")
col3.markdown(f":{status_color[drift_report['status']]}[{drift_report['status'].upper()}]")

# Eval score trend
st.subheader("Evaluation Score Trend")
eval_history = load_eval_history()
st.line_chart(eval_history.set_index("timestamp")["aggregate_score"])

# Prompt version history
st.subheader("Prompt Version History")
st.dataframe(load_prompt_history())
```

The dashboard is used by three different audiences: data scientists checking model health, campaign managers checking offer quality, and the LLMOps engineer (me, in most clients' organizations) checking everything. The layout is designed so each audience finds their signal within 10 seconds of opening the page.

---

## The Responsible AI Layer

There's a component of this pipeline I haven't mentioned yet: the responsible AI compliance checkpoint.

Before any model version is deployed, a structured audit runs automatically:

```python
# shared/responsible_ai.py
class ResponsibleAIAudit:
    def run(self, model_version: str) -> AuditReport:
        return AuditReport(
            bias_check=self._check_channel_bias(model_version),
            # Does the model score mobile-first customers differently?
            
            explainability=self._check_feature_attribution(model_version),
            # Are SHAP values available for top predictions?
            
            data_lineage=self._check_feature_lineage(model_version),
            # Can we trace every input feature to its source event?
            
            pii_handling=self._check_pii_fields(model_version),
            # No PII in model inputs beyond anonymized customer_id?
            
            audit_trail=self._check_prediction_logging(model_version),
            # Are all predictions logged with feature values for 90 days?
        )
```

In production, every AI component that touches customer data goes through a version of this audit. The LoyaltyLens implementation is a simplified analogue — but it follows the same structure, because the structure is what matters.

---

## Next: Closing the Loop with RLHF

Module 5 detects degradation. Module 6 responds to it. In the final post in this series I walk through the feedback capture UI, the preference dataset builder, and the retraining trigger that turns customer thumbs up/down signals into automated model improvements.

**[→ Read Module 6: RLHF Feedback Loops — Turning Customer Signals Into Model Improvements](#)**

---

*Pushparajan Ramar is an Enterprise Architect Director in enterprise consulting. He defines AI governance frameworks, responsible AI guardrails, and LLMOps practices for enterprise deployments. Connect on [LinkedIn](https://linkedin.com/in/pushparajanramar).*
