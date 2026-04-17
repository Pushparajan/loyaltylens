"""Retraining trigger — monitors feedback quality and fires alerts + webhooks.

Usage (cron-friendly):
    python feedback_loop/trigger.py --check
    python feedback_loop/trigger.py --check --since-days 14
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

from feedback_loop.aggregator import FeedbackAggregator, FeedbackStats
from shared.config import get_settings
from shared.logger import get_logger

logger = get_logger(__name__)

_TRIGGER_LOG = Path("feedback_loop/data/trigger_log.json")

_RATING_THRESHOLD = 3.0
_THUMBS_DOWN_THRESHOLD = 0.40
_MIN_RECORDS = 10


class RetrainingTrigger:
    """Evaluate feedback quality and fire retraining alerts when thresholds are breached."""

    def __init__(self, aggregator: FeedbackAggregator | None = None) -> None:
        self._aggregator = aggregator or FeedbackAggregator()

    def should_retrain(self, stats: FeedbackStats) -> tuple[bool, str]:
        """Return (should_retrain, reason).

        Conservative thresholds: fires only on genuine degradation, not noise.
        - avg_rating < 3.0  (below midpoint)
        - thumbs_down_rate > 0.40  (more than 40 % negative)
        At least MIN_RECORDS required before any trigger fires.
        """
        if stats.record_count < _MIN_RECORDS:
            return False, f"insufficient_data (n={stats.record_count})"

        if stats.avg_rating < _RATING_THRESHOLD:
            return True, (
                f"avg_rating={stats.avg_rating:.2f} < threshold={_RATING_THRESHOLD}"
            )
        if stats.thumbs_down_rate > _THUMBS_DOWN_THRESHOLD:
            return True, (
                f"thumbs_down_rate={stats.thumbs_down_rate:.1%} "
                f"> threshold={_THUMBS_DOWN_THRESHOLD:.0%}"
            )
        return False, "metrics_healthy"

    def fire_trigger(self, reason: str, stats: FeedbackStats) -> None:
        """Log the trigger event and optionally call a GitHub Actions workflow_dispatch."""
        event: dict[str, object] = {
            "triggered_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "avg_rating": stats.avg_rating,
            "thumbs_down_rate": stats.thumbs_down_rate,
            "record_count": stats.record_count,
            "action": "retrain_propensity_model",
        }

        _TRIGGER_LOG.parent.mkdir(parents=True, exist_ok=True)
        history: list[dict[str, object]] = []
        if _TRIGGER_LOG.exists():
            try:
                history = json.loads(_TRIGGER_LOG.read_text())
            except json.JSONDecodeError:
                history = []
        history.append(event)
        _TRIGGER_LOG.write_text(json.dumps(history, indent=2))

        logger.critical("retraining_trigger_fired", **{
            k: v for k, v in event.items() if k not in ("triggered_at", "action")
        })

        self._dispatch_github_workflow(reason)

    def _dispatch_github_workflow(self, reason: str) -> None:
        """Call GitHub Actions workflow_dispatch; silently skips if not configured."""
        settings = get_settings()
        token: str = getattr(settings, "github_token", "")
        repo: str = getattr(settings, "github_repo", "")

        if not token or not repo:
            logger.info("github_dispatch_skipped", reason="GITHUB_TOKEN or GITHUB_REPO not set")
            return

        url = f"https://api.github.com/repos/{repo}/actions/workflows/retrain.yml/dispatches"
        payload = json.dumps({"ref": "main", "inputs": {"reason": reason}}).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10):
                logger.info("github_dispatch_sent", repo=repo)
        except urllib.error.URLError as exc:
            logger.error("github_dispatch_failed", error=str(exc))

    def check_and_trigger(self, since_days: int = 7) -> bool:
        """Run the full check cycle; return True if trigger fired."""
        stats = self._aggregator.compute_stats(since_days=since_days)
        should, reason = self.should_retrain(stats)

        if should:
            self.fire_trigger(reason=reason, stats=stats)
            return True

        logger.info("trigger_check_passed", reason=reason, record_count=stats.record_count)
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LoyaltyLens retraining trigger")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run the trigger check and exit with 1 if trigger fired",
    )
    parser.add_argument("--since-days", type=int, default=7)
    args = parser.parse_args(argv)

    if not args.check:
        parser.print_help()
        return 0

    trigger = RetrainingTrigger()
    fired = trigger.check_and_trigger(since_days=args.since_days)

    if fired:
        print("[trigger] FIRED — retraining initiated")
        return 1

    print("[trigger] PASSED — no retraining needed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
