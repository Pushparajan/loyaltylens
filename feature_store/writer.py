"""Write computed feature vectors to Redis (hot) and Postgres (cold)."""

from __future__ import annotations

import json

from shared.db import DatabaseClient
from shared.logger import get_logger
from shared.schemas import FeatureVector

logger = get_logger(__name__)

_TTL_SECONDS = 86_400  # 24 h


class FeatureWriter:
    """Persist feature vectors to Redis for fast serving and Postgres for durability."""

    def __init__(self, db: DatabaseClient) -> None:
        self._db = db

    def write(self, vector: FeatureVector) -> None:
        self._write_redis(vector)
        self._write_postgres([vector])
        logger.info("feature_written", customer_id=str(vector.customer_id))

    def write_batch(self, vectors: list[FeatureVector]) -> None:
        pipe = self._db.redis.pipeline()
        for v in vectors:
            key = f"features:{v.customer_id}"
            pipe.set(key, v.model_dump_json(), ex=_TTL_SECONDS)
        pipe.execute()
        self._write_postgres(vectors)
        logger.info("features_batch_written", count=len(vectors))

    def _write_redis(self, vector: FeatureVector) -> None:
        key = f"features:{vector.customer_id}"
        self._db.redis.set(key, vector.model_dump_json(), ex=_TTL_SECONDS)

    def _write_postgres(self, vectors: list[FeatureVector]) -> None:
        from sqlalchemy import text

        sql = text(
            """
            INSERT INTO feature_vectors (customer_id, feature_names, values, computed_at)
            VALUES (:customer_id, :feature_names, :values, :computed_at)
            ON CONFLICT (customer_id) DO UPDATE SET
                feature_names = EXCLUDED.feature_names,
                values        = EXCLUDED.values,
                computed_at   = EXCLUDED.computed_at
            """
        )
        rows = [
            {
                "customer_id": str(v.customer_id),
                "feature_names": json.dumps(v.feature_names),
                "values": json.dumps(v.values),
                "computed_at": v.computed_at,
            }
            for v in vectors
        ]
        with self._db.session() as session:
            session.execute(sql, rows)
