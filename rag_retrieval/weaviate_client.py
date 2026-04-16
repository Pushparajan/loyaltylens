"""Thin wrapper around the Weaviate v4 Python client."""

from __future__ import annotations

import weaviate

from shared.config import get_settings
from shared.logger import get_logger

logger = get_logger(__name__)

_CLASS_NAME = "LoyaltyDocument"


class WeaviateClient:
    """Manage connection to Weaviate and expose collection helpers."""

    def __init__(self) -> None:
        settings = get_settings()
        host = settings.weaviate_url.split("://")[-1].split(":")[0]
        self._client = weaviate.connect_to_custom(
            http_host=host,
            http_port=settings.weaviate_http_port,
            http_secure=settings.weaviate_url.startswith("https"),
            grpc_host=host,
            grpc_port=settings.weaviate_grpc_port,
            grpc_secure=False,
        )
        logger.info("weaviate_connected", url=settings.weaviate_url)

    @property
    def client(self) -> weaviate.WeaviateClient:
        return self._client

    def ensure_collection(self) -> None:
        """Create the LoyaltyDocument collection if it does not exist."""
        if not self._client.collections.exists(_CLASS_NAME):
            self._client.collections.create(
                name=_CLASS_NAME,
                vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none(),
            )
            logger.info("weaviate_collection_created", name=_CLASS_NAME)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "WeaviateClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
