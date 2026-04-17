"""SQLite connection factory and schema initialisation for feedback_loop."""

from __future__ import annotations

import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path("feedback_loop/data/feedback.db")

_DDL = """
CREATE TABLE IF NOT EXISTS feedback (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    offer_id         TEXT    NOT NULL,
    customer_id      TEXT    NOT NULL,
    generated_copy   TEXT    NOT NULL,
    rating           INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
    thumbs           TEXT    NOT NULL CHECK(thumbs IN ('up', 'down')),
    prompt_version   TEXT    NOT NULL DEFAULT '',
    model_version    TEXT    NOT NULL DEFAULT '',
    created_at       TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Return a sqlite3 connection with row_factory set."""
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | None = None) -> Path:
    """Create the feedback table if it does not exist; return the db path."""
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as conn:
        conn.executescript(_DDL)
        conn.commit()
    return path
