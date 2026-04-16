"""Quick smoke-test: push a small batch through the full ETL pipeline."""

from __future__ import annotations

import uuid
from datetime import datetime

from data_pipeline.orchestrator import PipelineOrchestrator

_NOW = datetime.utcnow().isoformat()

CUSTOMER_ID_1 = str(uuid.uuid4())
CUSTOMER_ID_2 = str(uuid.uuid4())

transaction_records = [
    {
        "customer_id": CUSTOMER_ID_1,
        "amount": 42.50,
        "currency": "USD",
        "store_id": "store-001",
        "items": [{"sku": "ITEM-A", "qty": 1, "price": 42.50}],
        "created_at": _NOW,
        "updated_at": _NOW,
    },
    {
        "customer_id": CUSTOMER_ID_2,
        "amount": 18.00,
        "currency": "USD",
        "store_id": "store-002",
        "items": [{"sku": "ITEM-B", "qty": 2, "price": 9.00}],
        "created_at": _NOW,
        "updated_at": _NOW,
    },
]

customer_records = [
    {
        "customer_id": CUSTOMER_ID_1,
        "email": "alice@example.com",
        "tier": "gold",
        "total_spend": 420.00,
        "visit_count": 12,
        "created_at": _NOW,
        "updated_at": _NOW,
    },
    {
        "customer_id": CUSTOMER_ID_2,
        "email": "bob@example.com",
        "tier": "standard",
        "total_spend": 85.00,
        "visit_count": 3,
        "created_at": _NOW,
        "updated_at": _NOW,
    },
]

if __name__ == "__main__":
    result = PipelineOrchestrator().run(
        transaction_records=transaction_records,
        customer_records=customer_records,
    )
    print(f"Pipeline result: {result}")
