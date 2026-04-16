# llmops

Provides observability, automated evaluation, and Prometheus metrics for every LLM call made by `llm_generator`. Integrates with MLflow for experiment tracking.

## Purpose

Ensure LLM quality does not silently degrade: log every prompt/completion pair, run automated relevance and faithfulness evaluations, and expose latency / cost / quality metrics for alerting.

## Inputs

- `LLMCallRecord` events emitted by `llm_generator` (prompt, completion, model, latency, tokens)
- Reference context chunks from `rag_retrieval` (for faithfulness scoring)
- Evaluation rubrics / scorer configuration from `shared.Settings`

## Outputs

- Structured call records written to `llm_call_logs` Postgres table
- MLflow runs with quality metrics per generation batch
- Prometheus metrics exposed on port `8006` (default `PORT_METRICS` in `.env`)

## Key Classes

| Class | Module | Responsibility |
| --- | --- | --- |
| `LLMOpsTracker` | `tracker.py` | Receive and persist `LLMCallRecord` events |
| `EvaluationPipeline` | `evaluator.py` | Score relevance, faithfulness, and helpfulness |
| `MetricsCollector` | `metrics.py` | Maintain and expose Prometheus metric objects |

---

## Running Locally

### 1. Environment

One `.env` at the **repo root** — no module-level env file needed.

```dotenv
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=loyaltylens
EVAL_PASS_THRESHOLD=0.75
PORT_METRICS=8006              # Prometheus metrics server
```

---

### 2. Install dependencies

```powershell
uv sync --dev
uv pip install -e .
```

---

### 3. Start MLflow

```powershell
# Windows
python -m mlflow ui --host 127.0.0.1 --port 5000

# macOS / Linux
python -m mlflow ui --host 0.0.0.0 --port 5000
```

---

### 4. Start Prometheus metrics server

The `MetricsCollector.start_server()` method reads `PORT_METRICS` from settings:

```python
from llmops.metrics import MetricsCollector
MetricsCollector().start_server()   # starts on PORT_METRICS (default 8006)
```

---

### 5. Run the eval harness

```bash
python llmops/eval_harness/run_eval.py
```

Exits with code `1` and blocks deployment if mean score < `EVAL_PASS_THRESHOLD` (default `0.75`).

---

### 6. Run tests

```bash
python -m pytest tests/test_llmops_evaluator.py tests/test_eval_harness.py -v
```
