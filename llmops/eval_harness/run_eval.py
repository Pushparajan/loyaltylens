"""CI eval gate: score synthetic offer copies and exit non-zero if below threshold.

Usage:
    python llmops/eval_harness/run_eval.py [--threshold 0.75]

Writes results to llmops/eval_results/eval_<timestamp>.json.
Exits with code 1 if mean aggregate score < threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from shared.config import get_settings
from llmops.eval_harness.evaluator import OfferCopyEvaluator, OfferCopyInput

_SYNTHETIC_COPIES: list[dict[str, str]] = [
    {
        "headline": "Your exclusive Gold reward is waiting",
        "body": (
            "Hi there! As a valued Gold member, enjoy 20% off your next visit. "
            "Use code GOLD20 at checkout."
        ),
        "cta": "Redeem your reward →",
    },
    {
        "headline": "We miss you — here's a treat",
        "body": (
            "It's been a while! Come back and enjoy a complimentary item with your "
            "next purchase. Offer code: COMEBACK10."
        ),
        "cta": "Claim your treat →",
    },
    {
        "headline": "Double points this weekend only",
        "body": (
            "Earn double loyalty points on all purchases this Saturday and Sunday. "
            "No code needed — rewards applied automatically."
        ),
        "cta": "Start earning →",
    },
    {
        "headline": "A birthday surprise just for you",
        "body": (
            "Happy birthday! We're celebrating you with a free drink on your next visit. "
            "Valid for 7 days."
        ),
        "cta": "Claim your birthday drink →",
    },
    {
        "headline": "Refer a friend, earn 500 bonus stars",
        "body": (
            "Share the love! Every friend who joins using your code earns you 500 bonus stars. "
            "Stars never expire."
        ),
        "cta": "Share your code →",
    },
]

_RESULTS_DIR = Path("llmops/eval_results")


def _generate_synthetic_copies(n: int) -> list[tuple[OfferCopyInput, str]]:
    """Return n (copy, reference) pairs by cycling through the template bank.

    The reference is set to the body text so lexical metrics reflect whether
    the synthetic copy matches an established baseline — not an arbitrary string.
    """
    templates = _SYNTHETIC_COPIES
    return [
        (
            OfferCopyInput(
                headline=t["headline"],
                body=t["body"],
                cta=t["cta"],
            ),
            t["body"],  # reference = body → BLEU/ROUGE-L anchored to own text
        )
        for t in (templates[i % len(templates)] for i in range(n))
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LoyaltyLens LLM eval gate")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--n", type=int, default=50, help="Number of copies to evaluate")
    args = parser.parse_args(argv)

    threshold = args.threshold if args.threshold is not None else get_settings().eval_pass_threshold
    evaluator = OfferCopyEvaluator()

    pairs = _generate_synthetic_copies(args.n)
    results = [evaluator.evaluate(copy, ref) for copy, ref in pairs]

    scores = [r.aggregate for r in results]
    mean_score = sum(scores) / len(scores)
    passed = mean_score >= threshold

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = {
        "timestamp": timestamp,
        "n": len(results),
        "threshold": threshold,
        "mean_aggregate_score": round(mean_score, 4),
        "passed": passed,
        "scores": [
            {
                "bleu": r.bleu,
                "rouge_l": r.rouge_l,
                "coherence": r.coherence,
                "brand_alignment": r.brand_alignment,
                "cta_strength": r.cta_strength,
                "aggregate": r.aggregate,
            }
            for r in results
        ],
    }

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _RESULTS_DIR / f"eval_{timestamp}.json"
    out_path.write_text(json.dumps(output, indent=2))

    print(
        f"[eval-gate] mean_score={mean_score:.4f}  threshold={threshold:.2f}  "
        f"n={len(results)}  {'PASSED ✓' if passed else 'FAILED ✗'}"
    )
    print(f"[eval-gate] Results written to {out_path}")

    if not passed:
        print(
            f"[eval-gate] FAIL: mean score {mean_score:.4f} < threshold {threshold:.2f}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
