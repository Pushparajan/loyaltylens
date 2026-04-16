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
- Prometheus metrics exposed on `:9090/metrics` (latency histogram, token counters, quality gauges)

## Key Classes

| Class                | Module        | Responsibility |
| -------------------- | ------------- | ------------------------------------------------ |
| `LLMOpsTracker`      | `tracker.py`  | Receive and persist `LLMCallRecord` events       |
| `EvaluationPipeline` | `evaluator.py`| Score relevance, faithfulness, and helpfulness   |
| `MetricsCollector`   | `metrics.py`  | Maintain and expose Prometheus metric objects    |
