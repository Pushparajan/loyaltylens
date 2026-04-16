"""Read feature vectors from Redis (hot) with Postgres fallback (cold)."""

from __future__ import annotations

import json
from uuid import UUID

from shared.db import DatabaseClient
from shared.logger import get_logger
from shared.schemas import FeatureVector

logger = get_logger(__name__)


class FeatureReader:
    """Serve features from Redis with automatic Postgres fallback."""

    def __init__(self, db: DatabaseClient) -> None:
        self._db = db

    def get(self, customer_id: UUID) -> FeatureVector | None:
        key = f"features:{customer_id}"
        raw = self._db.redis.get(key)
        if raw:
            return FeatureVector.model_validate_json(raw)
        return self._fetch_postgres(customer_id)

    def get_batch(self, customer_ids: list[UUID]) -> list[FeatureVector]:
        keys = [f"features:{cid}" for cid in customer_ids]
        raws = self._db.redis.mget(keys)
        results: list[FeatureVector] = []
        missing: list[UUID] = []
        for cid, raw in zip(customer_ids, raws):
            if raw:
                results.append(FeatureVector.model_validate_json(raw))
            else:
                missing.append(cid)
        if missing:
            results.extend(self._fetch_postgres_batch(missing))
        return results

    def _fetch_postgres(self, customer_id: UUID) -> FeatureVector | None:
        from sqlalchemy import text

        sql = text(
            "SELECT customer_id, feature_names, values, computed_at "
            "FROM feature_vectors WHERE customer_id = :cid"
        )
        with self._db.session() as session:
            row = session.execute(sql, {"cid": str(customer_id)}).fetchone()
        if row is None:
            return None
        return FeatureVector(
            customer_id=row.customer_id,
            feature_names=json.loads(row.feature_names),
            values=json.loads(row.values),
            computed_at=row.computed_at,
        )

    def _fetch_postgres_batch(self, customer_ids: list[UUID]) -> list[FeatureVector]:
        from sqlalchemy import text

        sql = text(
            "SELECT customer_id, feature_names, values, computed_at "
            "FROM feature_vectors WHERE customer_id = ANY(:ids)"
        )
        with self._db.session() as session:
            rows = session.execute(sql, {"ids": [str(c) for c in customer_ids]}).fetchall()
        return [
            FeatureVector(
                customer_id=r.customer_id,
                feature_names=json.loads(r.feature_names),
                values=json.loads(r.values),
                computed_at=r.computed_at,
            )
            for r in rows
        ]
