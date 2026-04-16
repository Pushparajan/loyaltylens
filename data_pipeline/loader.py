"""Load and reconcile customer profiles from CRM exports into Postgres."""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from shared.db import DatabaseClient
from shared.logger import get_logger
from shared.schemas import CustomerProfile

logger = get_logger(__name__)


class CustomerDataLoader:
    """Upsert customer profile records, reconciling tier and spend totals."""

    def __init__(self, db: DatabaseClient | None = None) -> None:
        self._db = db or DatabaseClient()

    def load(self, records: list[dict[str, Any]]) -> int:
        """Validate and upsert customer profiles. Returns the number of rows written."""
        profiles = [CustomerProfile.model_validate(r) for r in records]
        written = self._upsert(profiles)
        logger.info("customers_loaded", count=written)
        return written

    def _upsert(self, profiles: list[CustomerProfile]) -> int:
        sql = text(
            """
            INSERT INTO customers (customer_id, email, tier, total_spend, visit_count, created_at, updated_at)
            VALUES (:customer_id, :email, :tier, :total_spend, :visit_count, :created_at, :updated_at)
            ON CONFLICT (customer_id) DO UPDATE SET
                email        = EXCLUDED.email,
                tier         = EXCLUDED.tier,
                total_spend  = EXCLUDED.total_spend,
                visit_count  = EXCLUDED.visit_count,
                updated_at   = EXCLUDED.updated_at
            """
        )
        rows = [
            {
                "customer_id": str(p.customer_id),
                "email": p.email,
                "tier": p.tier,
                "total_spend": p.total_spend,
                "visit_count": p.visit_count,
                "created_at": p.created_at,
                "updated_at": p.updated_at,
            }
            for p in profiles
        ]
        with self._db.session() as session:
            result = session.execute(sql, rows)
            return result.rowcount
