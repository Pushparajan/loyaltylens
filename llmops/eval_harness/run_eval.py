"""CI eval gate: score synthetic LLM responses and exit non-zero if below threshold.

Usage:
    python llmops/eval_harness/run_eval.py [--threshold 0.75]
"""

from __future__ import annotations

import argparse
import sys

from shared.config import get_settings
from shared.schemas import LLMResponse
from llmops.evaluator import EvaluationPipeline


_SYNTHETIC_RESPONSES: list[dict[str, object]] = [
    {
        "model": "gpt-4o-mini",
        "prompt_tokens": 120,
        "completion_tokens": 80,
        "content": (
            '{"subject": "Your exclusive Gold reward is waiting",'
            ' "body": "Hi there! As a valued Gold member, enjoy 20% off your next visit.'
            ' Use code GOLD20 at checkout.", "offer_code": "GOLD20"}'
        ),
        "latency_ms": 340.0,
    },
    {
        "model": "gpt-4o-mini",
        "prompt_tokens": 115,
        "completion_tokens": 75,
        "content": (
            '{"subject": "We miss you — here\'s a treat",'
            ' "body": "It\'s been a while! Come back and enjoy a complimentary item'
            ' with your next purchase. Offer code: COMEBACK10.", "offer_code": "COMEBACK10"}'
        ),
        "latency_ms": 280.0,
    },
    {
        "model": "gpt-4o-mini",
        "prompt_tokens": 130,
        "completion_tokens": 90,
        "content": (
            '{"subject": "Double points this weekend only",'
            ' "body": "Earn double loyalty points on all purchases this Saturday and Sunday.'
            ' No code needed — rewards applied automatically.", "offer_code": "DBLPTS"}'
        ),
        "latency_ms": 310.0,
    },
]

_EXPECTED_KEYWORDS = ["offer_code", "subject", "body"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LoyaltyLens LLM eval gate")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the pass threshold (default: EVAL_PASS_THRESHOLD env var or 0.75)",
    )
    args = parser.parse_args(argv)

    threshold = args.threshold or get_settings().eval_pass_threshold

    responses = [LLMResponse.model_validate(r) for r in _SYNTHETIC_RESPONSES]
    pipeline = EvaluationPipeline()
    result = pipeline.evaluate_batch(responses, expected_keywords=_EXPECTED_KEYWORDS)

    mean_score: float = result["mean_score"]
    passed: bool = mean_score >= threshold

    print(
        f"[eval-gate] mean_score={mean_score:.4f}  threshold={threshold:.2f}  "
        f"n={result['n']}  {'PASSED ✓' if passed else 'FAILED ✗'}"
    )

    if not passed:
        print(
            f"[eval-gate] FAIL: mean score {mean_score:.4f} is below threshold {threshold:.2f}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
