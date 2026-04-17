"""Aggregate feedback records into quality signals and preference datasets."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from feedback_loop.db import DEFAULT_DB_PATH, get_connection, init_db
from shared.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FeedbackStats:
    avg_rating: float
    thumbs_up_rate: float
    thumbs_down_rate: float
    by_prompt_version: dict[str, float]
    by_model_version: dict[str, float]
    by_category: dict[str, float]
    record_count: int


class FeedbackAggregator:
    """Compute quality statistics and export preference datasets from feedback."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or DEFAULT_DB_PATH
        init_db(self._db_path)

    def compute_stats(self, since_days: int = 7) -> FeedbackStats:
        """Return quality statistics for feedback received in the last *since_days* days."""
        with get_connection(self._db_path) as conn:
            rows = conn.execute(
                """
                SELECT rating, thumbs, prompt_version, model_version
                FROM feedback
                WHERE created_at >= datetime('now', ?)
                """,
                (f"-{since_days} days",),
            ).fetchall()

        if not rows:
            return FeedbackStats(
                avg_rating=0.0,
                thumbs_up_rate=0.0,
                thumbs_down_rate=0.0,
                by_prompt_version={},
                by_model_version={},
                by_category={},
                record_count=0,
            )

        ratings = [r["rating"] for r in rows]
        thumbs_up = sum(1 for r in rows if r["thumbs"] == "up")
        n = len(rows)

        pv: dict[str, list[int]] = defaultdict(list)
        mv: dict[str, list[int]] = defaultdict(list)
        for r in rows:
            pv[r["prompt_version"] or "unknown"].append(r["rating"])
            mv[r["model_version"] or "unknown"].append(r["rating"])

        stats = FeedbackStats(
            avg_rating=round(sum(ratings) / n, 3),
            thumbs_up_rate=round(thumbs_up / n, 3),
            thumbs_down_rate=round((n - thumbs_up) / n, 3),
            by_prompt_version={k: round(sum(v) / len(v), 3) for k, v in pv.items()},
            by_model_version={k: round(sum(v) / len(v), 3) for k, v in mv.items()},
            by_category={},
            record_count=n,
        )
        logger.info(
            "feedback_stats_computed",
            since_days=since_days,
            record_count=n,
            avg_rating=stats.avg_rating,
        )
        return stats

    def export_preference_dataset(self, output_path: str | Path) -> int:
        """Write a JSONL preference dataset for RLHF fine-tuning.

        Each line: {"prompt": "...", "chosen": "...", "rejected": "..."}
        chosen  = copy from a thumbs-up record
        rejected = copy from a thumbs-down record for the same offer_id.
        Records with no matching pair are skipped.
        """
        with get_connection(self._db_path) as conn:
            pairs = conn.execute(
                """
                SELECT
                    up.generated_copy  AS chosen_copy,
                    dn.generated_copy  AS rejected_copy,
                    up.offer_id        AS offer_id,
                    up.prompt_version  AS prompt_version
                FROM feedback AS up
                JOIN feedback AS dn
                    ON up.offer_id = dn.offer_id
                    AND up.thumbs = 'up'
                    AND dn.thumbs = 'down'
                LIMIT 10000
                """
            ).fetchall()

        records: list[dict[str, str]] = []
        for row in pairs:
            try:
                chosen = json.loads(row["chosen_copy"])
                rejected = json.loads(row["rejected_copy"])
            except (json.JSONDecodeError, TypeError):
                continue

            records.append(
                {
                    "prompt": (
                        f"Write loyalty offer copy for a customer "
                        f"(offer_id={row['offer_id']}, "
                        f"prompt_version={row['prompt_version'] or 'default'})."
                    ),
                    "chosen": (
                        f"{chosen.get('headline', '')} "
                        f"{chosen.get('body', '')} "
                        f"{chosen.get('cta', '')}"
                    ).strip(),
                    "rejected": (
                        f"{rejected.get('headline', '')} "
                        f"{rejected.get('body', '')} "
                        f"{rejected.get('cta', '')}"
                    ).strip(),
                }
            )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")

        logger.info("preference_dataset_exported", path=str(out), n_pairs=len(records))
        return len(records)
