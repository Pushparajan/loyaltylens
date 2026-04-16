"""Versioned storage and retrieval of prompt templates backed by Redis."""

from __future__ import annotations

import json

from shared.db import DatabaseClient
from shared.logger import get_logger

logger = get_logger(__name__)

_KEY_PREFIX = "prompt_registry"


class PromptRegistry:
    """Store, version, and retrieve prompt templates in Redis."""

    def __init__(self, db: DatabaseClient | None = None) -> None:
        self._db = db or DatabaseClient()

    def register(self, name: str, template: str, version: str = "latest") -> None:
        key = f"{_KEY_PREFIX}:{name}:{version}"
        self._db.redis.set(key, json.dumps({"name": name, "version": version, "template": template}))
        logger.info("prompt_registered", name=name, version=version)

    def get(self, name: str, version: str = "latest") -> str | None:
        key = f"{_KEY_PREFIX}:{name}:{version}"
        raw = self._db.redis.get(key)
        if raw is None:
            return None
        return json.loads(raw)["template"]

    def list_versions(self, name: str) -> list[str]:
        pattern = f"{_KEY_PREFIX}:{name}:*"
        keys = self._db.redis.keys(pattern)
        return [k.split(":")[-1] for k in keys]
