"""LLM output quality evaluation pipeline."""

from __future__ import annotations

import statistics
from typing import Any

from shared.config import get_settings
from shared.logger import get_logger
from shared.schemas import LLMResponse

logger = get_logger(__name__)


def _relevance_heuristic(response_text: str, expected_keywords: list[str]) -> float:
    """Simple keyword-overlap relevance score in [0, 1]."""
    if not expected_keywords:
        return 1.0
    lower = response_text.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in lower)
    return hits / len(expected_keywords)


class EvaluationPipeline:
    """Score a batch of LLM responses and decide if they pass the quality gate."""

    def __init__(self) -> None:
        self._threshold = get_settings().eval_pass_threshold

    def score_response(
        self,
        response: LLMResponse,
        expected_keywords: list[str] | None = None,
    ) -> float:
        score = _relevance_heuristic(response.content, expected_keywords or [])
        return score

    def evaluate_batch(
        self,
        responses: list[LLMResponse],
        expected_keywords: list[str] | None = None,
    ) -> dict[str, Any]:
        scores = [self.score_response(r, expected_keywords) for r in responses]
        mean_score = statistics.mean(scores) if scores else 0.0
        passed = mean_score >= self._threshold
        result: dict[str, Any] = {
            "mean_score": mean_score,
            "threshold": self._threshold,
            "passed": passed,
            "n": len(scores),
            "scores": scores,
        }
        logger.info("eval_batch_complete", **{k: v for k, v in result.items() if k != "scores"})
        return result

    def passes_gate(
        self,
        responses: list[LLMResponse],
        expected_keywords: list[str] | None = None,
    ) -> bool:
        result = self.evaluate_batch(responses, expected_keywords)
        return bool(result["passed"])
