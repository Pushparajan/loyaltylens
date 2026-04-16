"""Track every LLM call: persist metadata to Postgres and emit Prometheus metrics."""

from __future__ import annotations

from sqlalchemy import text

from shared.db import DatabaseClient
from shared.logger import get_logger
from shared.schemas import LLMResponse

logger = get_logger(__name__)


class LLMOpsTracker:
    """Persist LLMResponse records and expose counters for Prometheus scraping."""

    def __init__(self, db: DatabaseClient | None = None) -> None:
        self._db = db or DatabaseClient()

    def record(self, response: LLMResponse) -> None:
        """Insert an LLM call record into the llm_calls table."""
        sql = text(
            """
            INSERT INTO llm_calls
                (request_id, model, prompt_tokens, completion_tokens, latency_ms, score, created_at)
            VALUES
                (:request_id, :model, :prompt_tokens, :completion_tokens, :latency_ms, :score, NOW())
            ON CONFLICT (request_id) DO NOTHING
            """
        )
        with self._db.session() as session:
            session.execute(
                sql,
                {
                    "request_id": str(response.request_id),
                    "model": response.model,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "latency_ms": response.latency_ms,
                    "score": response.score,
                },
            )
        logger.info(
            "llm_call_recorded",
            request_id=str(response.request_id),
            model=response.model,
            latency_ms=round(response.latency_ms, 1),
        )
