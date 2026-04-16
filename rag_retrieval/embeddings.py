"""Embed offer descriptions with all-MiniLM-L6-v2 and store in pgvector + Weaviate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import psycopg2
import weaviate
from sentence_transformers import SentenceTransformer

from shared.config import get_settings
from shared.logger import get_logger

logger = get_logger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
OFFERS_PATH = Path(__file__).parent / "data" / "offers.json"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS offer_embeddings (
    id      TEXT PRIMARY KEY,
    title   TEXT NOT NULL,
    embedding vector(%d)
);
""" % EMBEDDING_DIM

_UPSERT = """
INSERT INTO offer_embeddings (id, title, embedding)
VALUES (%s, %s, %s)
ON CONFLICT (id) DO UPDATE
    SET title = EXCLUDED.title,
        embedding = EXCLUDED.embedding;
"""


class EmbeddingPipeline:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._model = SentenceTransformer(MODEL_NAME)
        self._pg = psycopg2.connect(self._settings.postgres_url)
        host = self._settings.weaviate_url.split("://")[-1].split(":")[0]
        self._weaviate = weaviate.connect_to_custom(
            http_host=host,
            http_port=self._settings.weaviate_http_port,
            http_secure=self._settings.weaviate_url.startswith("https"),
            grpc_host=host,
            grpc_port=self._settings.weaviate_grpc_port,
            grpc_secure=False,
        )
        self._ensure_pg_schema()
        self._ensure_weaviate_schema()

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    def _ensure_pg_schema(self) -> None:
        with self._pg.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(_CREATE_TABLE)
        self._pg.commit()

    def _ensure_weaviate_schema(self) -> None:
        from weaviate.classes.config import Configure, DataType, Property

        existing = {c.name for c in self._weaviate.collections.list_all().values()}
        if "Offer" not in existing:
            self._weaviate.collections.create(
                name="Offer",
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="offer_id", data_type=DataType.TEXT),
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="description", data_type=DataType.TEXT),
                    Property(name="category", data_type=DataType.TEXT),
                    Property(name="channel", data_type=DataType.TEXT),
                    Property(name="min_propensity_threshold", data_type=DataType.NUMBER),
                    Property(name="discount_pct", data_type=DataType.INT),
                    Property(name="expiry_days", data_type=DataType.INT),
                ],
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_offers(self, offers: list[dict[str, Any]] | None = None) -> None:
        if offers is None:
            offers = json.loads(OFFERS_PATH.read_text(encoding="utf-8"))

        texts = [o["description"] for o in offers]
        vectors = self._model.encode(texts, batch_size=64, show_progress_bar=True)

        self._upsert_pg(offers, vectors)
        self._upsert_weaviate(offers, vectors)
        logger.info("embedded_offers", count=len(offers))

    def update_offer(self, offer_id: str) -> None:
        offers = json.loads(OFFERS_PATH.read_text(encoding="utf-8"))
        match = [o for o in offers if o["id"] == offer_id]
        if not match:
            raise ValueError(f"offer {offer_id!r} not found in offers.json")
        self.embed_offers(match)

    def delete_offer(self, offer_id: str) -> None:
        with self._pg.cursor() as cur:
            cur.execute("DELETE FROM offer_embeddings WHERE id = %s;", (offer_id,))
        self._pg.commit()

        collection = self._weaviate.collections.get("Offer")
        collection.data.delete_many(
            where=collection.query.filter.by_property("offer_id").equal(offer_id)
        )
        logger.info("deleted_offer", offer_id=offer_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _upsert_pg(self, offers: list[dict], vectors: Any) -> None:
        with self._pg.cursor() as cur:
            for offer, vec in zip(offers, vectors):
                cur.execute(_UPSERT, (offer["id"], offer["title"], vec.tolist()))
        self._pg.commit()

    def _upsert_weaviate(self, offers: list[dict], vectors: Any) -> None:
        collection = self._weaviate.collections.get("Offer")
        with collection.batch.dynamic() as batch:
            for offer, vec in zip(offers, vectors):
                batch.add_object(
                    properties={
                        "offer_id": offer["id"],
                        "title": offer["title"],
                        "description": offer["description"],
                        "category": offer["category"],
                        "channel": offer["channel"],
                        "min_propensity_threshold": offer["min_propensity_threshold"],
                        "discount_pct": offer["discount_pct"],
                        "expiry_days": offer["expiry_days"],
                    },
                    vector=vec.tolist(),
                )

    def close(self) -> None:
        self._pg.close()
        self._weaviate.close()

    def __enter__(self) -> "EmbeddingPipeline":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


if __name__ == "__main__":
    with EmbeddingPipeline() as pipeline:
        pipeline.embed_offers()
