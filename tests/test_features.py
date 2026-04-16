"""Tests for the data generator, feature pipeline, DuckDB store, and API."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from data_pipeline.features import compute_features
from data_pipeline.generate import generate_events
from feature_store.store import FeatureStore, FeatureValidationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def events() -> pd.DataFrame:
    return generate_events(n_events=2_000, n_customers=200, seed=42)


@pytest.fixture(scope="module")
def features(events: pd.DataFrame) -> pd.DataFrame:
    return compute_features(events)


@pytest.fixture()
def store(tmp_path: Path) -> FeatureStore:
    return FeatureStore(db_path=tmp_path / "test.duckdb")


@pytest.fixture()
def populated_store(store: FeatureStore, features: pd.DataFrame) -> FeatureStore:
    store.write_version(features, version="v1")
    return store


@pytest.fixture()
def api_client(populated_store: FeatureStore) -> TestClient:
    from feature_store import api as api_module

    api_module._store = populated_store
    from feature_store.api import app

    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Schema validation — generate_events output
# ---------------------------------------------------------------------------


class TestGenerateEvents:
    REQUIRED_COLS = {"customer_id", "event_type", "timestamp", "amount", "offer_id", "channel"}

    def test_row_count(self, events: pd.DataFrame) -> None:
        assert len(events) == 2_000

    def test_required_columns_present(self, events: pd.DataFrame) -> None:
        assert self.REQUIRED_COLS <= set(events.columns)

    def test_event_types_valid(self, events: pd.DataFrame) -> None:
        valid = {"purchase", "app_open", "offer_view", "redeem"}
        assert set(events["event_type"].unique()) <= valid

    def test_channels_valid(self, events: pd.DataFrame) -> None:
        valid = {"mobile", "web", "in-store"}
        assert set(events["channel"].unique()) <= valid

    def test_amount_null_only_for_non_purchases(self, events: pd.DataFrame) -> None:
        purchases = events[events["event_type"] == "purchase"]
        assert purchases["amount"].notna().all()
        non_purchases = events[events["event_type"] != "purchase"]
        assert non_purchases["amount"].isna().all()

    def test_offer_id_null_only_for_non_offer_events(self, events: pd.DataFrame) -> None:
        offer_events = events[events["event_type"].isin(["offer_view", "redeem"])]
        assert offer_events["offer_id"].notna().all()
        other = events[~events["event_type"].isin(["offer_view", "redeem"])]
        assert other["offer_id"].isna().all()

    def test_reproducible_with_same_seed(self) -> None:
        df1 = generate_events(n_events=100, n_customers=20, seed=99)
        df2 = generate_events(n_events=100, n_customers=20, seed=99)
        assert df1["customer_id"].tolist() == df2["customer_id"].tolist()

    def test_different_seeds_differ(self) -> None:
        df1 = generate_events(n_events=100, n_customers=20, seed=1)
        df2 = generate_events(n_events=100, n_customers=20, seed=2)
        assert df1["customer_id"].tolist() != df2["customer_id"].tolist()


# ---------------------------------------------------------------------------
# 2. Null checks — compute_features output
# ---------------------------------------------------------------------------


class TestComputeFeatures:
    FEATURE_COLS = [
        "recency_days",
        "frequency_30d",
        "monetary_90d",
        "offer_redemption_rate",
        "channel_preference",
        "engagement_score",
    ]

    def test_output_has_required_columns(self, features: pd.DataFrame) -> None:
        assert {"customer_id"} | set(self.FEATURE_COLS) <= set(features.columns)

    def test_no_nulls_in_numeric_features(self, features: pd.DataFrame) -> None:
        numeric = ["recency_days", "frequency_30d", "monetary_90d", "offer_redemption_rate"]
        for col in numeric:
            assert features[col].notna().all(), f"Unexpected nulls in {col}"

    def test_no_nulls_in_channel_preference(self, features: pd.DataFrame) -> None:
        assert features["channel_preference"].notna().all()

    def test_one_row_per_customer(self, features: pd.DataFrame) -> None:
        assert features["customer_id"].nunique() == len(features)

    def test_recency_days_non_negative(self, features: pd.DataFrame) -> None:
        assert (features["recency_days"] >= 0).all()

    def test_frequency_30d_non_negative(self, features: pd.DataFrame) -> None:
        assert (features["frequency_30d"] >= 0).all()

    def test_monetary_90d_non_negative(self, features: pd.DataFrame) -> None:
        assert (features["monetary_90d"] >= 0.0).all()

    def test_redemption_rate_in_unit_interval(self, features: pd.DataFrame) -> None:
        assert features["offer_redemption_rate"].between(0.0, 1.0).all()

    def test_channel_preference_valid(self, features: pd.DataFrame) -> None:
        valid = {"mobile", "web", "in-store"}
        assert set(features["channel_preference"].unique()) <= valid


# ---------------------------------------------------------------------------
# 3. engagement_score bounds
# ---------------------------------------------------------------------------


class TestEngagementScore:
    def test_engagement_score_in_unit_interval(self, features: pd.DataFrame) -> None:
        score = features["engagement_score"]
        assert (score >= 0.0).all(), "engagement_score below 0"
        assert (score <= 1.0).all(), "engagement_score above 1"

    def test_engagement_score_no_nulls(self, features: pd.DataFrame) -> None:
        assert features["engagement_score"].notna().all()

    def test_engagement_score_has_variance(self, features: pd.DataFrame) -> None:
        assert features["engagement_score"].std() > 0.0

    def test_validation_error_on_out_of_bounds_score(
        self, store: FeatureStore, features: pd.DataFrame
    ) -> None:
        bad = features.copy()
        bad.loc[bad.index[0], "engagement_score"] = 1.5
        with pytest.raises(FeatureValidationError, match="engagement_score"):
            store.write_version(bad, version="bad")

    def test_validation_error_on_negative_score(
        self, store: FeatureStore, features: pd.DataFrame
    ) -> None:
        bad = features.copy()
        bad.loc[bad.index[0], "engagement_score"] = -0.1
        with pytest.raises(FeatureValidationError, match="engagement_score"):
            store.write_version(bad, version="bad")


# ---------------------------------------------------------------------------
# 4. DuckDB round-trip
# ---------------------------------------------------------------------------


class TestFeatureStoreDuckDB:
    def test_write_then_list_versions(
        self, store: FeatureStore, features: pd.DataFrame
    ) -> None:
        store.write_version(features, "v1")
        assert "v1" in store.list_versions()

    def test_read_latest_returns_correct_customer(
        self, populated_store: FeatureStore, features: pd.DataFrame
    ) -> None:
        cid = features["customer_id"].iloc[0]
        row = populated_store.read_latest(cid)
        assert row is not None
        assert row["customer_id"] == cid

    def test_read_latest_unknown_customer_returns_none(
        self, populated_store: FeatureStore
    ) -> None:
        assert populated_store.read_latest("no-such-uuid") is None

    def test_write_version_idempotent(
        self, store: FeatureStore, features: pd.DataFrame
    ) -> None:
        store.write_version(features, "v1")
        store.write_version(features, "v1")
        assert store.list_versions().count("v1") == 1

    def test_multiple_versions(
        self, store: FeatureStore, features: pd.DataFrame
    ) -> None:
        store.write_version(features, "v1")
        store.write_version(features, "v2")
        versions = store.list_versions()
        assert "v1" in versions and "v2" in versions

    def test_get_feature_stats_returns_dataframe(
        self, populated_store: FeatureStore
    ) -> None:
        stats = populated_store.get_feature_stats("v1")
        assert not stats.empty

    def test_validation_error_on_excess_nulls(
        self, store: FeatureStore, features: pd.DataFrame
    ) -> None:
        bad = features.copy()
        bad.loc[:, "recency_days"] = np.nan
        with pytest.raises(FeatureValidationError, match="recency_days"):
            store.write_version(bad, version="bad")

    def test_validation_error_on_missing_column(
        self, store: FeatureStore, features: pd.DataFrame
    ) -> None:
        bad = features.drop(columns=["engagement_score"])
        with pytest.raises(FeatureValidationError, match="missing columns"):
            store.write_version(bad, version="bad")


# ---------------------------------------------------------------------------
# 5. API response shape
# ---------------------------------------------------------------------------


class TestFeatureAPI:
    def test_get_features_200_for_known_customer(
        self, api_client: TestClient, features: pd.DataFrame
    ) -> None:
        cid = features["customer_id"].iloc[0]
        resp = api_client.get(f"/features/{cid}")
        assert resp.status_code == 200

    def test_get_features_response_has_all_fields(
        self, api_client: TestClient, features: pd.DataFrame
    ) -> None:
        cid = features["customer_id"].iloc[0]
        body = api_client.get(f"/features/{cid}").json()
        expected = {
            "customer_id", "version", "recency_days", "frequency_30d",
            "monetary_90d", "offer_redemption_rate", "channel_preference",
            "engagement_score",
        }
        assert expected <= set(body.keys())

    def test_get_features_customer_id_matches(
        self, api_client: TestClient, features: pd.DataFrame
    ) -> None:
        cid = features["customer_id"].iloc[0]
        body = api_client.get(f"/features/{cid}").json()
        assert body["customer_id"] == cid

    def test_get_features_404_for_unknown_customer(
        self, api_client: TestClient
    ) -> None:
        resp = api_client.get("/features/unknown-customer-id")
        assert resp.status_code == 404

    def test_get_stats_200(self, api_client: TestClient) -> None:
        resp = api_client.get("/features/stats")
        assert resp.status_code == 200

    def test_get_stats_response_shape(self, api_client: TestClient) -> None:
        body = api_client.get("/features/stats").json()
        assert "version" in body
        assert "stats" in body
        assert isinstance(body["stats"], list)
        assert len(body["stats"]) > 0

    def test_get_stats_version_matches_written(
        self, api_client: TestClient
    ) -> None:
        body = api_client.get("/features/stats").json()
        assert body["version"] == "v1"
