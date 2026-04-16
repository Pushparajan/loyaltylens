"""Integration smoke-test for the eval harness CLI entry point."""

from __future__ import annotations

from llmops.eval_harness.run_eval import main


def test_eval_harness_passes_default_threshold() -> None:
    exit_code = main(["--threshold", "0.75"])
    assert exit_code == 0, "Eval harness should pass with synthetic responses at threshold 0.75"


def test_eval_harness_fails_very_high_threshold() -> None:
    exit_code = main(["--threshold", "0.999"])
    assert exit_code == 1, "Eval harness should fail when threshold is set impossibly high"
