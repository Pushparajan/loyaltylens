"""shared — cross-cutting utilities: config, DB clients, logging, and base schemas."""

from shared.config import Settings
from shared.db import DatabaseClient
from shared.logger import get_logger
from shared.schemas import BaseSchema

__all__ = ["Settings", "DatabaseClient", "get_logger", "BaseSchema"]
