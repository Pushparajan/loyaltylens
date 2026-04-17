# llmops

Provides observability, automated evaluation, prompt versioning, PSI drift monitoring, and Prometheus metrics for every LLM call made by `llm_generator`. Integrates with MLflow for experiment tracking.

## Purpose

Ensure LLM quality does not silently degrade: version prompts like code, log every prompt/completion pair, run automated BLEU/ROUGE/LLM-as-judge evaluations, monitor propensity score drift with PSI, and expose latency / cost / quality metrics for alerting.

## Module Map

| Path | Responsibility |
| --- | --- |
| `prompt_registry/cli.py` | Click CLI: `list`, `diff`, `activate`, `rollback` |
| `prompt_registry/active.json` | Currently active prompt version |
| `prompt_registry/history.json` | Activation history (append-only) |
| `eval_harness/evaluator.py` | `OfferCopyEvaluator` — BLEU, ROUGE-L, LLM-as-judge |
| `eval_harness/run_eval.py` | CI eval gate: score 50 copies, write JSON, exit 1 if < 0.75 |
| `drift_monitor/monitor.py` | `PropensityDriftMonitor` + `DriftReport` dataclass |
| `drift_monitor/run_drift.py` | Daily job: load windows, compute PSI, write report |
| `dashboard/app.py` | Streamlit dashboard: scores, drift, prompt history |
| `evaluator.py` | `EvaluationPipeline` — keyword-relevance scoring (legacy) |
| `metrics.py` | Prometheus `Counter`, `Histogram`, `Gauge` instruments |
| `tracker.py` | `LLMOpsTracker` — persist `LLMCallRecord` events |

## Key Classes

| Class | Module | Responsibility |
| --- | --- | --- |
| `LLMOpsTracker` | `tracker.py` | Receive and persist `LLMCallRecord` events |
| `EvaluationPipeline` | `evaluator.py` | Keyword-relevance batch scoring (legacy) |
| `OfferCopyEvaluator` | `eval_harness/evaluator.py` | BLEU + ROUGE-L + LLM-as-judge aggregate scorer |
| `PropensityDriftMonitor` | `drift_monitor/monitor.py` | PSI computation + `DriftReport` |
| `MetricsCollector` | `metrics.py` | Prometheus metric instruments |

## Aggregate Eval Score Formula

```text
aggregate = 0.20 × BLEU + 0.20 × ROUGE-L + 0.60 × mean(coherence, brand_alignment, cta_strength)
```

LLM-judge dimensions are scored 1–5 by a GPT-4o judge call and normalised to [0, 1]. When no judge backend is configured, dimensions default to 0.75 (neutral pass).

## PSI Thresholds

| PSI | Status |
| --- | --- |
| < 0.10 | `ok` |
| 0.10 – 0.20 | `warning` |
| > 0.20 | `critical` |

---

## Running Locally

### 1. Environment

One `.env` at the **repo root** — no module-level env file needed.

```dotenv
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=loyaltylens
EVAL_PASS_THRESHOLD=0.75
PORT_METRICS=8006
```

---

### 2. Install dependencies

```powershell
& C:\Projects\loyaltylens\.venv\Scripts\Activate.ps1    # prompt shows (loyaltylens)
uv sync --dev
```

---

### 3. Prompt Registry CLI

```bash
# List all versions
python -m llmops.prompt_registry.cli prompt list

# Diff two versions
python -m llmops.prompt_registry.cli prompt diff v1 v2

# Activate a version
python -m llmops.prompt_registry.cli prompt activate v2

# Roll back to the previous version
python -m llmops.prompt_registry.cli prompt rollback
```

Version state is persisted in `llmops/prompt_registry/active.json` and `history.json`.

---

### 4. Run the eval harness

```bash
python llmops/eval_harness/run_eval.py
# With a custom threshold
python llmops/eval_harness/run_eval.py --threshold 0.80
```

Results are written to `llmops/eval_results/eval_<timestamp>.json`. Exit code `1` if mean score < threshold.

---

### 5. Run the drift monitor

```bash
python llmops/drift_monitor/run_drift.py
# Custom output path
python llmops/drift_monitor/run_drift.py --output llmops/drift_results/drift_report.json
```

Exit code `1` on critical drift (PSI > 0.20).

---

### 6. Start MLflow

```powershell
# Windows
python -m mlflow ui --host 127.0.0.1 --port 5000
```

---

### 7. Start Prometheus metrics server

```python
from llmops.metrics import MetricsCollector
MetricsCollector().start_server()   # starts on PORT_METRICS (default 8006)
```

---

### 8. Start the Streamlit dashboard

```bash
streamlit run llmops/dashboard/app.py
```

Dashboard shows:

- Active prompt version
- Latest eval score (pass/fail)
- Drift PSI + status badge
- Eval score trend (last 20 runs)
- Prompt activation history

---

### 9. Run tests

```bash
python -m pytest tests/test_llmops.py tests/test_llmops_evaluator.py tests/test_eval_harness.py -v
```

32 tests cover:

- PSI near-zero for identical distributions, positive for shifted distributions
- `PropensityDriftMonitor` ok/warning/critical classification, feature breakdown
- `OfferCopyEvaluator` BLEU/ROUGE-L bounds, judge fallback on missing backend/error
- Eval harness: passes at 0.75, fails at 0.999, writes JSON results
- Prompt registry CLI: list, activate, rollback, diff via `CliRunner`

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: sacrebleu` | Dependencies not installed | Run `uv sync --dev` from repo root |
| `No such file: llmops/prompt_registry/active.json` | Running CLI from wrong directory | Run from `c:\Projects\loyaltylens` |
| `MlflowException: connection refused` | MLflow not running | Run `python -m mlflow ui --port 5000` |
| `No module named 'shared'` | Wrong venv active | Run `& C:\Projects\loyaltylens\.venv\Scripts\Activate.ps1` |
