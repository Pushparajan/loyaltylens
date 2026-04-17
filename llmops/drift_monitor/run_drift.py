"""Daily drift monitoring job for propensity score distributions.

Loads baseline and current score windows from the feature store,
computes PSI, writes a JSON report, and exits non-zero on critical drift.

Usage:
    python llmops/drift_monitor/run_drift.py [--output drift_report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from llmops.drift_monitor.monitor import PropensityDriftMonitor
from shared.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_OUTPUT = Path("llmops/drift_results/drift_report.json")


def _load_scores(n: int, seed: int, mean: float, std: float) -> list[float]:
    """Generate synthetic score window; replace with real feature-store query in production."""
    rng = np.random.default_rng(seed)
    return list(np.clip(rng.normal(mean, std, n), 0.0, 1.0).tolist())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LoyaltyLens propensity drift check")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument(
        "--critical",
        type=float,
        default=0.20,
        help="PSI threshold for critical drift (default: 0.20)",
    )
    parser.add_argument(
        "--warning",
        type=float,
        default=0.10,
        help="PSI threshold for warning drift (default: 0.10)",
    )
    args = parser.parse_args(argv)

    now = datetime.now(timezone.utc)
    baseline_date = (now - timedelta(days=30)).strftime("%Y-%m-%d")
    current_date = now.strftime("%Y-%m-%d")

    # In production: replace with feature_store.load_scores(window="30d") / "7d"
    baseline_scores = _load_scores(n=5000, seed=42, mean=0.55, std=0.18)
    current_scores = _load_scores(n=1000, seed=99, mean=0.55, std=0.18)

    monitor = PropensityDriftMonitor(
        warning_threshold=args.warning,
        critical_threshold=args.critical,
    )
    report = monitor.check_drift(
        baseline_scores=baseline_scores,
        current_scores=current_scores,
        baseline_date=baseline_date,
        current_date=current_date,
    )

    output_data = {
        "psi": report.psi,
        "status": report.status,
        "baseline_date": report.baseline_date,
        "current_date": report.current_date,
        "n_baseline": report.n_baseline,
        "n_current": report.n_current,
        "feature_breakdown": report.feature_breakdown,
        "thresholds": {"warning": args.warning, "critical": args.critical},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_data, indent=2))

    emoji = {"ok": "✅", "warning": "⚠️", "critical": "🚨"}.get(report.status, "❓")
    print(f"{emoji}  Drift PSI={report.psi:.4f}  status={report.status.upper()}")
    print(f"   Report written to {args.output}")

    return 1 if report.status == "critical" else 0


if __name__ == "__main__":
    sys.exit(main())
