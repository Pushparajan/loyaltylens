"""Tests for feedback_loop: DB, aggregator, preference dataset, trigger logic."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from feedback_loop.aggregator import FeedbackAggregator, FeedbackStats
from feedback_loop.db import get_connection, init_db
from feedback_loop.trigger import RetrainingTrigger, _RATING_THRESHOLD, _THUMBS_DOWN_THRESHOLD


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_db(tmp_path: Path) -> Path:
    db = tmp_path / "test_feedback.db"
    init_db(db)
    return db


def _insert(
    db: Path,
    *,
    offer_id: str,
    thumbs: str,
    rating: int,
    pv: str = "v1",
    mv: str = "m1",
    copy: dict[str, str] | None = None,
) -> None:
    default_copy = {"headline": "H", "body": "B", "cta": "C"}
    copy_json = json.dumps(copy or default_copy)
    with get_connection(db) as conn:
        conn.execute(
            "INSERT INTO feedback (offer_id, customer_id, generated_copy, rating, thumbs, prompt_version, model_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (offer_id, "cust_001", copy_json, rating, thumbs, pv, mv),
        )
        conn.commit()


# ── DB init / insert / read ────────────────────────────────────────────────────


class TestFeedbackDB:
    def test_init_creates_table(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        with get_connection(db) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'"
            ).fetchall()
        assert len(rows) == 1

    def test_insert_and_read_back(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        _insert(db, offer_id="O001", thumbs="up", rating=4)
        with get_connection(db) as conn:
            row = conn.execute("SELECT * FROM feedback WHERE offer_id='O001'").fetchone()
        assert row["rating"] == 4
        assert row["thumbs"] == "up"

    def test_rating_constraint_rejects_zero(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        with pytest.raises(Exception):
            with get_connection(db) as conn:
                conn.execute(
                    "INSERT INTO feedback (offer_id, customer_id, generated_copy, rating, thumbs) "
                    "VALUES ('O', 'C', '{}', 0, 'up')"
                )
                conn.commit()

    def test_thumbs_constraint_rejects_invalid(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        with pytest.raises(Exception):
            with get_connection(db) as conn:
                conn.execute(
                    "INSERT INTO feedback (offer_id, customer_id, generated_copy, rating, thumbs) "
                    "VALUES ('O', 'C', '{}', 3, 'sideways')"
                )
                conn.commit()


# ── API endpoints ──────────────────────────────────────────────────────────────


class TestFeedbackAPI:
    @pytest.fixture()
    def client(self, tmp_path: Path):
        import feedback_loop.api as api_module

        original = api_module._DB_PATH
        api_module._DB_PATH = tmp_path / "api_test.db"
        init_db(api_module._DB_PATH)
        yield TestClient(api_module.app)
        api_module._DB_PATH = original

    def test_post_feedback_201(self, client: TestClient) -> None:
        res = client.post(
            "/feedback",
            json={
                "offer_id": "O001",
                "customer_id": "C001",
                "generated_copy": {"headline": "H", "body": "B", "cta": "C"},
                "rating": 4,
                "thumbs": "up",
                "prompt_version": "v1",
                "model_version": "gpt-4o",
            },
        )
        assert res.status_code == 201
        data = res.json()
        assert data["rating"] == 4
        assert data["thumbs"] == "up"

    def test_post_feedback_invalid_rating(self, client: TestClient) -> None:
        res = client.post(
            "/feedback",
            json={
                "offer_id": "O001",
                "customer_id": "C001",
                "generated_copy": {},
                "rating": 6,
                "thumbs": "up",
            },
        )
        assert res.status_code == 422

    def test_stats_empty(self, client: TestClient) -> None:
        res = client.get("/feedback/stats")
        assert res.status_code == 200
        data = res.json()
        assert data["record_count"] == 0

    def test_stats_computed(self, client: TestClient) -> None:
        for r, t in [(5, "up"), (4, "up"), (2, "down")]:
            client.post(
                "/feedback",
                json={
                    "offer_id": "O001",
                    "customer_id": "C001",
                    "generated_copy": {"headline": "H", "body": "B", "cta": "C"},
                    "rating": r,
                    "thumbs": t,
                    "prompt_version": "v1",
                    "model_version": "m1",
                },
            )
        res = client.get("/feedback/stats")
        data = res.json()
        assert data["record_count"] == 3
        assert data["avg_rating"] == pytest.approx(11 / 3, abs=0.01)
        assert data["thumbs_up_rate"] == pytest.approx(2 / 3, abs=0.01)

    def test_export_returns_list(self, client: TestClient) -> None:
        client.post(
            "/feedback",
            json={
                "offer_id": "O001",
                "customer_id": "C001",
                "generated_copy": {"headline": "H", "body": "B", "cta": "C"},
                "rating": 3,
                "thumbs": "up",
            },
        )
        res = client.get("/feedback/export")
        assert res.status_code == 200
        records = res.json()
        assert isinstance(records, list)
        assert len(records) == 1
        assert isinstance(records[0]["generated_copy"], dict)


# ── FeedbackAggregator ────────────────────────────────────────────────────────


class TestFeedbackAggregator:
    def test_empty_db_returns_zero_stats(self, tmp_path: Path) -> None:
        agg = FeedbackAggregator(db_path=tmp_path / "agg.db")
        stats = agg.compute_stats(since_days=7)
        assert stats.record_count == 0
        assert stats.avg_rating == pytest.approx(0.0)

    def test_compute_stats_correct(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        _insert(db, offer_id="O1", thumbs="up", rating=5, pv="v1", mv="m1")
        _insert(db, offer_id="O2", thumbs="up", rating=4, pv="v1", mv="m1")
        _insert(db, offer_id="O3", thumbs="down", rating=2, pv="v2", mv="m2")
        agg = FeedbackAggregator(db_path=db)
        stats = agg.compute_stats(since_days=30)
        assert stats.record_count == 3
        assert stats.avg_rating == pytest.approx(11 / 3, abs=0.01)
        assert stats.thumbs_up_rate == pytest.approx(2 / 3, abs=0.01)
        assert stats.thumbs_down_rate == pytest.approx(1 / 3, abs=0.01)
        assert "v1" in stats.by_prompt_version
        assert "v2" in stats.by_prompt_version
        assert stats.by_prompt_version["v1"] == pytest.approx(4.5, abs=0.01)

    def test_by_model_version_populated(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        _insert(db, offer_id="O1", thumbs="up", rating=5, mv="gpt-4o")
        _insert(db, offer_id="O2", thumbs="down", rating=3, mv="mistral-7b")
        agg = FeedbackAggregator(db_path=db)
        stats = agg.compute_stats(since_days=30)
        assert "gpt-4o" in stats.by_model_version
        assert "mistral-7b" in stats.by_model_version

    def test_export_preference_dataset_jsonl(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        # same offer_id with distinct copy → forms a chosen/rejected pair
        _insert(db, offer_id="O1", thumbs="up", rating=5,
                copy={"headline": "Great offer", "body": "Earn 2x stars today", "cta": "Redeem now"})
        _insert(db, offer_id="O1", thumbs="down", rating=2,
                copy={"headline": "Meh offer", "body": "Some deal", "cta": "Click"})
        agg = FeedbackAggregator(db_path=db)
        out = tmp_path / "prefs.jsonl"
        n = agg.export_preference_dataset(out)
        assert n == 1
        assert out.exists()
        record = json.loads(out.read_text().splitlines()[0])
        assert "prompt" in record
        assert "chosen" in record
        assert "rejected" in record
        assert record["chosen"] != record["rejected"]

    def test_export_no_pairs_writes_empty_file(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        _insert(db, offer_id="O1", thumbs="up", rating=5)   # no matching down
        agg = FeedbackAggregator(db_path=db)
        out = tmp_path / "empty.jsonl"
        n = agg.export_preference_dataset(out)
        assert n == 0
        assert out.read_text() == ""


# ── RetrainingTrigger ─────────────────────────────────────────────────────────


def _stats(**kwargs) -> FeedbackStats:
    defaults: dict[str, object] = {
        "avg_rating": 4.0,
        "thumbs_up_rate": 0.8,
        "thumbs_down_rate": 0.2,
        "by_prompt_version": {},
        "by_model_version": {},
        "by_category": {},
        "record_count": 50,
    }
    defaults.update(kwargs)
    return FeedbackStats(**defaults)  # type: ignore[arg-type]


class TestRetrainingTrigger:
    def test_healthy_metrics_do_not_trigger(self) -> None:
        t = RetrainingTrigger()
        should, reason = t.should_retrain(_stats(avg_rating=3.5, thumbs_down_rate=0.25))
        assert not should
        assert "healthy" in reason

    def test_rating_at_threshold_does_not_trigger(self) -> None:
        t = RetrainingTrigger()
        should, _ = t.should_retrain(_stats(avg_rating=_RATING_THRESHOLD))
        assert not should

    def test_rating_below_threshold_triggers(self) -> None:
        t = RetrainingTrigger()
        should, reason = t.should_retrain(_stats(avg_rating=2.9))
        assert should
        assert "avg_rating" in reason

    def test_thumbs_down_at_threshold_does_not_trigger(self) -> None:
        t = RetrainingTrigger()
        should, _ = t.should_retrain(_stats(thumbs_down_rate=_THUMBS_DOWN_THRESHOLD))
        assert not should

    def test_thumbs_down_above_threshold_triggers(self) -> None:
        t = RetrainingTrigger()
        should, reason = t.should_retrain(_stats(thumbs_down_rate=0.41))
        assert should
        assert "thumbs_down_rate" in reason

    def test_insufficient_data_never_triggers(self) -> None:
        t = RetrainingTrigger()
        should, reason = t.should_retrain(_stats(avg_rating=1.0, record_count=5))
        assert not should
        assert "insufficient_data" in reason

    def test_fire_trigger_writes_log(self, tmp_path: Path) -> None:
        import feedback_loop.trigger as trigger_module

        original = trigger_module._TRIGGER_LOG
        log_path = tmp_path / "trigger_log.json"
        trigger_module._TRIGGER_LOG = log_path

        t = RetrainingTrigger()
        t.fire_trigger(reason="test_reason", stats=_stats(avg_rating=2.5))

        trigger_module._TRIGGER_LOG = original
        assert log_path.exists()
        events = json.loads(log_path.read_text())
        assert len(events) == 1
        assert events[0]["reason"] == "test_reason"
