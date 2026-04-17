"""Feedback Loop FastAPI application.

Endpoints:
    POST /feedback          — persist a rating + thumbs record
    GET  /feedback/stats    — aggregate stats per prompt/model version
    GET  /feedback/export   — last 1000 records as JSON
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Literal

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from feedback_loop.db import DEFAULT_DB_PATH, get_connection, init_db
from shared.config import get_settings
from shared.logger import get_logger

logger = get_logger(__name__)

# initialise schema on startup
_DB_PATH: Path = DEFAULT_DB_PATH


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_db(_DB_PATH)
    logger.info("feedback_db_ready", path=str(_DB_PATH))
    yield


app = FastAPI(title="LoyaltyLens Feedback API", version="1.0.0", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── request / response models ─────────────────────────────────────────────────


class FeedbackRequest(BaseModel):
    offer_id: str
    customer_id: str
    generated_copy: dict[str, str]
    rating: int = Field(ge=1, le=5)
    thumbs: Literal["up", "down"]
    prompt_version: str = ""
    model_version: str = ""


class FeedbackResponse(BaseModel):
    id: int
    offer_id: str
    customer_id: str
    rating: int
    thumbs: str
    prompt_version: str
    model_version: str
    created_at: str


class StatsResponse(BaseModel):
    record_count: int
    avg_rating: float
    thumbs_up_rate: float
    thumbs_down_rate: float
    by_prompt_version: dict[str, float]
    by_model_version: dict[str, float]


# ── routes ────────────────────────────────────────────────────────────────────


@app.post("/feedback", status_code=201)
def submit_feedback(req: FeedbackRequest) -> FeedbackResponse:
    """Persist one feedback record."""
    copy_json = json.dumps(req.generated_copy)
    with get_connection(_DB_PATH) as conn:
        cur = conn.execute(
            """
            INSERT INTO feedback
                (offer_id, customer_id, generated_copy, rating, thumbs,
                 prompt_version, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                req.offer_id,
                req.customer_id,
                copy_json,
                req.rating,
                req.thumbs,
                req.prompt_version,
                req.model_version,
            ),
        )
        conn.commit()
        row_id = cur.lastrowid

    row = conn.execute(
        "SELECT id, offer_id, customer_id, rating, thumbs, prompt_version, model_version, created_at "
        "FROM feedback WHERE id = ?",
        (row_id,),
    ).fetchone()

    logger.info(
        "feedback_received",
        offer_id=req.offer_id,
        rating=req.rating,
        thumbs=req.thumbs,
    )
    return FeedbackResponse(**dict(row))


@app.get("/feedback/stats")
def get_stats() -> StatsResponse:
    """Aggregate stats per prompt_version and model_version."""
    with get_connection(_DB_PATH) as conn:
        rows = conn.execute(
            "SELECT rating, thumbs, prompt_version, model_version FROM feedback"
        ).fetchall()

    if not rows:
        return StatsResponse(
            record_count=0,
            avg_rating=0.0,
            thumbs_up_rate=0.0,
            thumbs_down_rate=0.0,
            by_prompt_version={},
            by_model_version={},
        )

    ratings = [r["rating"] for r in rows]
    thumbs_up = sum(1 for r in rows if r["thumbs"] == "up")
    n = len(rows)

    pv_groups: dict[str, list[int]] = defaultdict(list)
    mv_groups: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        pv_groups[r["prompt_version"] or "unknown"].append(r["rating"])
        mv_groups[r["model_version"] or "unknown"].append(r["rating"])

    return StatsResponse(
        record_count=n,
        avg_rating=round(sum(ratings) / n, 3),
        thumbs_up_rate=round(thumbs_up / n, 3),
        thumbs_down_rate=round((n - thumbs_up) / n, 3),
        by_prompt_version={k: round(sum(v) / len(v), 3) for k, v in pv_groups.items()},
        by_model_version={k: round(sum(v) / len(v), 3) for k, v in mv_groups.items()},
    )


@app.get("/feedback/export")
def export_feedback() -> list[dict[str, object]]:
    """Return the last 1000 feedback records as JSON."""
    with get_connection(_DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT id, offer_id, customer_id, generated_copy, rating, thumbs,
                   prompt_version, model_version, created_at
            FROM feedback
            ORDER BY id DESC
            LIMIT 1000
            """
        ).fetchall()

    records = []
    for r in rows:
        rec = dict(r)
        try:
            rec["generated_copy"] = json.loads(rec["generated_copy"])
        except (json.JSONDecodeError, TypeError):
            pass
        records.append(rec)
    return records


if __name__ == "__main__":
    import uvicorn

    port = get_settings().port_feedback_loop
    uvicorn.run("feedback_loop.api:app", host="127.0.0.1", port=port, reload=True)
