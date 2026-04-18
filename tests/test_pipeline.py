"""Integration tests for LoyaltyLensPipeline — uses mocked module dependencies."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from shared.pipeline import LoyaltyLensPipeline, PipelineResult, _fallback_offer


# ── fixtures / helpers ────────────────────────────────────────────────────────

def _make_feature_dict(customer_id: str) -> dict[str, Any]:
    return {
        "customer_id": customer_id,
        "recency_days": 5,
        "frequency_30d": 3,
        "monetary_90d": 120.0,
        "offer_redemption_rate": 0.4,
        "channel_preference": "email",
        "engagement_score": 0.7,
    }


@dataclass
class _FakePropensityResult:
    customer_id: str
    propensity_score: float = 0.82
    label: int = 1
    threshold: float = 0.5
    model_version: str = "xgb-v1"


@dataclass
class _FakeOfferResult:
    offer_id: str = "O001"
    title: str = "Double Stars Day"
    description: str = "Earn 2x stars on any purchase."
    category: str = "rewards"
    score: float = 0.91


@dataclass
class _FakeCopy:
    headline: str = "Double your stars today!"
    body: str = "Earn 2x stars on any purchase this weekend."
    cta: str = "Redeem now"
    tone: str = "friendly"
    model_version: str = "gpt-4o-mini"
    prompt_version: int = 2
    latency_ms: float = 420.0
    token_count: int = 55


def _build_pipeline_with_mocks(customer_id: str) -> tuple[LoyaltyLensPipeline, dict]:
    pipeline = LoyaltyLensPipeline()

    feature_store = MagicMock()
    feature_store.read_latest.return_value = _make_feature_dict(customer_id)

    predictor = MagicMock()
    predictor.predict.return_value = _FakePropensityResult(customer_id=customer_id)

    retriever = MagicMock()
    retriever.retrieve.return_value = [_FakeOfferResult()]

    generator = MagicMock()
    generator.generate.return_value = _FakeCopy()

    pipeline._feature_store = feature_store
    pipeline._predictor = predictor
    pipeline._retriever = retriever
    pipeline._generator = generator

    mocks = {
        "feature_store": feature_store,
        "predictor": predictor,
        "retriever": retriever,
        "generator": generator,
    }
    return pipeline, mocks


# ── unit tests ────────────────────────────────────────────────────────────────

class TestPipelineResult:
    def test_result_fields_non_null(self) -> None:
        cid = str(uuid.uuid4())
        pipeline, _ = _build_pipeline_with_mocks(cid)
        result = pipeline.run_for_customer(cid)

        assert result.customer_id == cid
        assert result.feature_dict
        assert 0.0 <= result.propensity_score <= 1.0
        assert result.propensity_label in (0, 1)
        assert isinstance(result.top_offers, list)
        assert isinstance(result.offer_copy, dict)
        assert isinstance(result.latency_breakdown, dict)
        assert result.model_version
        assert result.prompt_version
        assert result.pipeline_run_id
        assert result.timestamp

    def test_result_is_pipeline_result_instance(self) -> None:
        cid = str(uuid.uuid4())
        pipeline, _ = _build_pipeline_with_mocks(cid)
        result = pipeline.run_for_customer(cid)
        assert isinstance(result, PipelineResult)

    def test_latency_breakdown_has_four_steps(self) -> None:
        cid = str(uuid.uuid4())
        pipeline, _ = _build_pipeline_with_mocks(cid)
        result = pipeline.run_for_customer(cid)
        assert set(result.latency_breakdown.keys()) == {
            "feature_store_ms",
            "propensity_ms",
            "rag_retrieval_ms",
            "llm_generator_ms",
        }

    def test_offer_copy_has_required_keys(self) -> None:
        cid = str(uuid.uuid4())
        pipeline, _ = _build_pipeline_with_mocks(cid)
        result = pipeline.run_for_customer(cid)
        assert {"headline", "body", "cta", "tone", "model_version"}.issubset(
            result.offer_copy.keys()
        )

    def test_top_offers_have_required_keys(self) -> None:
        cid = str(uuid.uuid4())
        pipeline, _ = _build_pipeline_with_mocks(cid)
        result = pipeline.run_for_customer(cid)
        assert result.top_offers
        for o in result.top_offers:
            assert {"offer_id", "title", "description", "category", "score"}.issubset(o.keys())

    def test_pipeline_run_id_is_uuid(self) -> None:
        cid = str(uuid.uuid4())
        pipeline, _ = _build_pipeline_with_mocks(cid)
        result = pipeline.run_for_customer(cid)
        uuid.UUID(result.pipeline_run_id)  # raises if invalid

    def test_pipeline_run_id_unique_per_call(self) -> None:
        cid = str(uuid.uuid4())
        pipeline, _ = _build_pipeline_with_mocks(cid)
        r1 = pipeline.run_for_customer(cid)
        r2 = pipeline.run_for_customer(cid)
        assert r1.pipeline_run_id != r2.pipeline_run_id

    def test_missing_customer_raises(self) -> None:
        cid = str(uuid.uuid4())
        pipeline, mocks = _build_pipeline_with_mocks(cid)
        mocks["feature_store"].read_latest.return_value = None
        with pytest.raises(ValueError, match="No features found"):
            pipeline.run_for_customer(cid)


class TestPipelineTenCustomers:
    """Run pipeline on 10 synthetic customer IDs and assert schema conformance."""

    def test_ten_customers_all_valid(self) -> None:
        customer_ids = [str(uuid.uuid4()) for _ in range(10)]
        results: list[PipelineResult] = []

        for cid in customer_ids:
            pipeline, _ = _build_pipeline_with_mocks(cid)
            results.append(pipeline.run_for_customer(cid))

        assert len(results) == 10

        for r in results:
            assert r.customer_id
            assert r.feature_dict
            assert isinstance(r.propensity_score, float)
            assert r.propensity_label in (0, 1)
            assert isinstance(r.top_offers, list)
            assert isinstance(r.offer_copy, dict)
            assert len(r.latency_breakdown) == 4
            assert all(v >= 0 for v in r.latency_breakdown.values())
            assert r.model_version
            assert r.pipeline_run_id
            assert r.timestamp

    def test_ten_customers_unique_run_ids(self) -> None:
        customer_ids = [str(uuid.uuid4()) for _ in range(10)]
        run_ids = set()
        for cid in customer_ids:
            pipeline, _ = _build_pipeline_with_mocks(cid)
            result = pipeline.run_for_customer(cid)
            run_ids.add(result.pipeline_run_id)
        assert len(run_ids) == 10


class TestFallbackOffer:
    def test_fallback_offer_has_required_keys(self) -> None:
        o = _fallback_offer()
        assert {"offer_id", "title", "description", "category", "score"}.issubset(o.keys())
