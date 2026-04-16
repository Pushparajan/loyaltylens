"""Prefect-based pipeline orchestration: composes ingestion and loading flows."""

from __future__ import annotations

from typing import Any

from prefect import flow, task

from data_pipeline.ingester import TransactionIngester
from data_pipeline.loader import CustomerDataLoader
from shared.logger import get_logger

logger = get_logger(__name__)


@task(name="ingest-transactions", retries=2, retry_delay_seconds=30)
def ingest_transactions_task(records: list[dict[str, Any]]) -> int:
    return TransactionIngester().ingest(records)


@task(name="load-customers", retries=2, retry_delay_seconds=30)
def load_customers_task(records: list[dict[str, Any]]) -> int:
    return CustomerDataLoader().load(records)


class PipelineOrchestrator:
    """Entry-point for triggering the full ETL flow programmatically."""

    @flow(name="loyaltylens-etl")
    def run(
        self,
        transaction_records: list[dict[str, Any]],
        customer_records: list[dict[str, Any]],
    ) -> dict[str, int]:
        tx_count = ingest_transactions_task(transaction_records)
        cust_count = load_customers_task(customer_records)
        logger.info("pipeline_complete", transactions=tx_count, customers=cust_count)
        return {"transactions": tx_count, "customers": cust_count}
