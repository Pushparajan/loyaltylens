"""Collect customer feedback events on generated offers."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import text

from shared.db import DatabaseClient
from shared.logger import get_logger

logger = get_logger(__name__)


class FeedbackCollector:
    """Persist raw feedback events (click, redeem, ignore) to Postgres."""

    VALID_EVENTS = {"click", "redeem", "ignore", "unsubscribe"}

    def __init__(self, db: DatabaseClient | None = None) -> None:
        self._db = db or DatabaseClient()

    def collect(self, event: dict[str, Any]) -> None:
        """
        Persist a single feedback event.

        Expected keys: customer_id, request_id, event_type, metadata (optional).
        """
        event_type = event.get("event_type", "")
        if event_type not in self.VALID_EVENTS:
            raise ValueError(f"Unknown event_type {event_type!r}. Must be one of {self.VALID_EVENTS}")

        sql = text(
            """
            INSERT INTO feedback_events (customer_id, request_id, event_type, metadata, created_at)
            VALUES (:customer_id, :request_id, :event_type, :metadata::jsonb, NOW())
            """
        )
        with self._db.session() as session:
            session.execute(
                sql,
                {
                    "customer_id": str(event["customer_id"]),
                    "request_id": str(event["request_id"]),
                    "event_type": event_type,
                    "metadata": str(event.get("metadata", {})),
                },
            )
        logger.info(
            "feedback_collected",
            customer_id=str(event["customer_id"]),
            event_type=event_type,
        )

    def collect_batch(self, events: list[dict[str, Any]]) -> int:
        for e in events:
            self.collect(e)
        return len(events)
