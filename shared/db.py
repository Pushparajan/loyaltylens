"""Database client: SQLAlchemy engine/session factory and Redis connection pool."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import redis as redis_lib
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from shared.config import get_settings


class DatabaseClient:
    """Wraps a SQLAlchemy engine and a Redis client for the application."""

    def __init__(self) -> None:
        settings = get_settings()
        self._engine: Engine = create_engine(
            settings.postgres_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
        self._session_factory = sessionmaker(bind=self._engine, autocommit=False, autoflush=False)
        self._redis: redis_lib.Redis = redis_lib.from_url(  # type: ignore[type-arg]
            settings.redis_url, decode_responses=True
        )

    # ------------------------------------------------------------------
    # Postgres helpers
    # ------------------------------------------------------------------

    @property
    def engine(self) -> Engine:
        return self._engine

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        db: Session = self._session_factory()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def ping(self) -> bool:
        with self._engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True

    # ------------------------------------------------------------------
    # Redis helpers
    # ------------------------------------------------------------------

    @property
    def redis(self) -> redis_lib.Redis:  # type: ignore[type-arg]
        return self._redis

    def redis_ping(self) -> bool:
        return bool(self._redis.ping())
