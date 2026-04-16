"""Aggregate raw feedback events into retraining signals."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from sqlalchemy import text

from shared.db import DatabaseClient
from shared.logger import get_logger

logger = get_logger(__name__)


class FeedbackProcessor:
    """Aggregate feedback events per customer into positive / negative signal counts."""

    POSITIVE_EVENTS = {"click", "redeem"}
    NEGATIVE_EVENTS = {"ignore", "unsubscribe"}

    def __init__(self, db: DatabaseClient | None = None) -> None:
        self._db = db or DatabaseClient()

    def aggregate(self, since_iso: str | None = None) -> list[dict[str, Any]]:
        """Return per-customer feedback signal aggregates since *since_iso*."""
        where = "WHERE created_at >= :since" if since_iso else ""
        sql = text(
            f"""
            SELECT customer_id, event_type, COUNT(*) AS cnt
            FROM feedback_events
            {where}
            GROUP BY customer_id, event_type
            """
        )
        params: dict[str, Any] = {}
        if since_iso:
            params["since"] = since_iso

        with self._db.session() as session:
            rows = session.execute(sql, params).fetchall()

        signals: dict[str, dict[str, int]] = defaultdict(lambda: {"positive": 0, "negative": 0})
        for row in rows:
            cid = str(row.customer_id)
            if row.event_type in self.POSITIVE_EVENTS:
                signals[cid]["positive"] += row.cnt
            elif row.event_type in self.NEGATIVE_EVENTS:
                signals[cid]["negative"] += row.cnt

        result = [
            {"customer_id": cid, **counts} for cid, counts in signals.items()
        ]
        logger.info("feedback_aggregated", n_customers=len(result))
        return result
