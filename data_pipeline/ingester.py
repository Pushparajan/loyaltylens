"""Parse, validate, and upsert loyalty transactions into Postgres."""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from shared.config import get_settings
from shared.db import DatabaseClient
from shared.logger import get_logger
from shared.schemas import Transaction

logger = get_logger(__name__)


class TransactionIngester:
    """Validate raw transaction payloads and upsert them to the transactions table."""

    def __init__(self, db: DatabaseClient | None = None) -> None:
        self._db = db or DatabaseClient()
        self._batch_size = get_settings().batch_size

    def ingest(self, raw_records: list[dict[str, Any]]) -> int:
        """Validate and persist a batch of raw transaction dicts. Returns row count written."""
        transactions = [Transaction.model_validate(r) for r in raw_records]
        written = 0
        for chunk in self._chunks(transactions, self._batch_size):
            written += self._upsert(chunk)
        logger.info("transactions_ingested", count=written)
        return written

    def _upsert(self, batch: list[Transaction]) -> int:
        sql = text(
            """
            INSERT INTO transactions (transaction_id, customer_id, amount, currency, store_id, items, created_at)
            VALUES (:transaction_id, :customer_id, :amount, :currency, :store_id, :items::jsonb, :created_at)
            ON CONFLICT (transaction_id) DO NOTHING
            """
        )
        rows = [
            {
                "transaction_id": str(t.transaction_id),
                "customer_id": str(t.customer_id),
                "amount": t.amount,
                "currency": t.currency,
                "store_id": t.store_id,
                "items": str(t.items),
                "created_at": t.created_at,
            }
            for t in batch
        ]
        with self._db.session() as session:
            result = session.execute(sql, rows)
            return result.rowcount

    @staticmethod
    def _chunks(items: list[Transaction], size: int) -> list[list[Transaction]]:
        return [items[i : i + size] for i in range(0, len(items), size)]
