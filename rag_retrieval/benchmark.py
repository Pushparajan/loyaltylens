"""Benchmark LangChain vs LlamaIndex retrievers: latency (p50/p95) and precision@5."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from rag_retrieval.langchain_retriever import LangChainOfferRetriever, OfferResult
from rag_retrieval.llama_retriever import LlamaOfferRetriever
from shared.logger import get_logger

logger = get_logger(__name__)

OUT_PATH = Path(__file__).parent / "benchmark.json"

_QUERIES = [
    ("frequent coffee buyer who loves seasonal drinks", 0.6, "beverage"),
    ("customer interested in food rewards and snacks", 0.5, "food"),
    ("high-value member looking for bonus stars opportunities", 0.7, "bonus_stars"),
    ("occasional visitor wanting merchandise discounts", 0.3, "merchandise"),
    ("seasonal shopper active in holiday promotions", 0.4, "seasonal"),
    ("mobile-first customer exploring app-exclusive offers", 0.55, "beverage"),
    ("loyal member near tier upgrade", 0.65, "bonus_stars"),
    ("health-conscious customer seeking light food options", 0.45, "food"),
    ("gift buyer interested in merchandise bundles", 0.35, "merchandise"),
    ("weekend regular wanting brunch deals", 0.5, "food"),
]

K = 5
WARMUP = 2
RUNS = 10


def _precision_at_k(results: list[OfferResult], expected_category: str) -> float:
    if not results:
        return 0.0
    hits = sum(1 for r in results if r.category == expected_category)
    return hits / len(results)


def _benchmark_retriever(name: str, retriever_cls: type) -> dict:
    retriever = retriever_cls()

    latencies: list[float] = []
    precisions: list[float] = []

    queries_cycle = (_QUERIES * ((WARMUP + RUNS) // len(_QUERIES) + 2))[: WARMUP + RUNS]

    for i, (ctx, propensity, category) in enumerate(queries_cycle):
        t0 = time.perf_counter()
        results = retriever.retrieve(ctx, propensity, k=K)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if i >= WARMUP:
            latencies.append(elapsed_ms)
            precisions.append(_precision_at_k(results, category))

    arr = np.array(latencies)
    return {
        "retriever": name,
        "runs": RUNS,
        "latency_p50_ms": round(float(np.percentile(arr, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(arr, 95)), 2),
        "latency_mean_ms": round(float(arr.mean()), 2),
        "precision_at_5_mean": round(float(np.mean(precisions)), 4),
    }


def run_benchmark() -> dict:
    logger.info("benchmark_start")

    lc_stats = _benchmark_retriever("langchain", LangChainOfferRetriever)
    ll_stats = _benchmark_retriever("llamaindex", LlamaOfferRetriever)

    report = {
        "k": K,
        "warmup_runs": WARMUP,
        "benchmark_runs": RUNS,
        "results": [lc_stats, ll_stats],
        "winner_latency": (
            "langchain"
            if lc_stats["latency_p50_ms"] < ll_stats["latency_p50_ms"]
            else "llamaindex"
        ),
        "winner_precision": (
            "langchain"
            if lc_stats["precision_at_5_mean"] >= ll_stats["precision_at_5_mean"]
            else "llamaindex"
        ),
    }

    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("benchmark_complete", output=str(OUT_PATH))
    return report


if __name__ == "__main__":
    result = run_benchmark()
    print(json.dumps(result, indent=2))
