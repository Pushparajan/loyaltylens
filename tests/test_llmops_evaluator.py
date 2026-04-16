"""Unit tests for llmops.evaluator.EvaluationPipeline."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from llmops.evaluator import EvaluationPipeline
from shared.schemas import LLMResponse


def _make_response(content: str) -> LLMResponse:
    return LLMResponse(
        model="gpt-4o-mini",
        prompt_tokens=100,
        completion_tokens=50,
        content=content,
        latency_ms=200.0,
    )


class TestEvaluationPipeline:
    def test_full_keyword_match_scores_1(self) -> None:
        pipeline = EvaluationPipeline()
        r = _make_response("subject body offer_code all here")
        score = pipeline.score_response(r, ["subject", "body", "offer_code"])
        assert score == 1.0

    def test_no_keyword_match_scores_0(self) -> None:
        pipeline = EvaluationPipeline()
        r = _make_response("nothing relevant")
        score = pipeline.score_response(r, ["subject", "body", "offer_code"])
        assert score == 0.0

    def test_partial_match(self) -> None:
        pipeline = EvaluationPipeline()
        r = _make_response("subject is here but nothing else")
        score = pipeline.score_response(r, ["subject", "body", "offer_code"])
        assert pytest.approx(score, rel=1e-3) == 1 / 3

    def test_empty_keywords_scores_1(self) -> None:
        pipeline = EvaluationPipeline()
        r = _make_response("anything")
        score = pipeline.score_response(r, [])
        assert score == 1.0

    def test_passes_gate_above_threshold(self) -> None:
        with patch("llmops.evaluator.get_settings") as mock_settings:
            mock_settings.return_value.eval_pass_threshold = 0.5
            pipeline = EvaluationPipeline()
        responses = [_make_response("subject body offer_code")] * 3
        assert pipeline.passes_gate(responses, ["subject", "body", "offer_code"])

    def test_fails_gate_below_threshold(self) -> None:
        with patch("llmops.evaluator.get_settings") as mock_settings:
            mock_settings.return_value.eval_pass_threshold = 0.99
            pipeline = EvaluationPipeline()
        responses = [_make_response("only subject")] * 3
        assert not pipeline.passes_gate(responses, ["subject", "body", "offer_code"])

    def test_batch_result_structure(self) -> None:
        pipeline = EvaluationPipeline()
        responses = [_make_response("subject body offer_code")]
        result = pipeline.evaluate_batch(responses, ["subject"])
        assert "mean_score" in result
        assert "passed" in result
        assert result["n"] == 1
