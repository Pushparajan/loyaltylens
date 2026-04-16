"""Prometheus metrics for LLM call volume, latency, and quality."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, start_http_server

from shared.config import get_settings
from shared.logger import get_logger

logger = get_logger(__name__)

llm_call_counter = Counter(
    "loyaltylens_llm_calls_total",
    "Total number of LLM calls made",
    ["model"],
)

llm_latency_histogram = Histogram(
    "loyaltylens_llm_latency_ms",
    "LLM call latency in milliseconds",
    ["model"],
    buckets=[50, 100, 250, 500, 1000, 2500, 5000],
)

llm_score_gauge = Gauge(
    "loyaltylens_llm_last_eval_score",
    "Most recent LLM evaluation score",
)

token_counter = Counter(
    "loyaltylens_llm_tokens_total",
    "Total tokens consumed",
    ["model", "type"],  # type: prompt | completion
)


class MetricsCollector:
    """Record per-call metrics into Prometheus instruments."""

    def record_call(self, model: str, latency_ms: float, score: float | None = None) -> None:
        llm_call_counter.labels(model=model).inc()
        llm_latency_histogram.labels(model=model).observe(latency_ms)
        if score is not None:
            llm_score_gauge.set(score)

    def record_tokens(self, model: str, prompt: int, completion: int) -> None:
        token_counter.labels(model=model, type="prompt").inc(prompt)
        token_counter.labels(model=model, type="completion").inc(completion)

    def start_server(self, port: int | None = None) -> None:
        port = port if port is not None else get_settings().port_metrics
        start_http_server(port)
        logger.info("metrics_server_started", port=port)
