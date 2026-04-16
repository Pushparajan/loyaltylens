"""Unit tests for the shared module."""

from __future__ import annotations

from uuid import uuid4

import pytest

from shared.config import Settings, get_settings
from shared.schemas import (
    BaseSchema,
    CustomerProfile,
    FeatureVector,
    LLMResponse,
    Transaction,
)


def test_settings_defaults() -> None:
    s = Settings()
    assert s.batch_size == 512
    assert s.eval_pass_threshold == 0.75


def test_get_settings_cached() -> None:
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_transaction_schema_defaults() -> None:
    t = Transaction(customer_id=uuid4(), amount=12.50, store_id="store-1")
    assert t.currency == "USD"
    assert t.items == []


def test_customer_profile_defaults() -> None:
    c = CustomerProfile(email="test@example.com")
    assert c.tier == "standard"
    assert c.total_spend == 0.0
    assert c.churn_score is None


def test_feature_vector_round_trip() -> None:
    fv = FeatureVector(
        customer_id=uuid4(),
        feature_names=["f1", "f2"],
        values=[0.1, 0.9],
    )
    json_str = fv.model_dump_json()
    restored = FeatureVector.model_validate_json(json_str)
    assert restored.feature_names == fv.feature_names
    assert restored.values == fv.values


def test_llm_response_defaults() -> None:
    resp = LLMResponse(
        model="gpt-4o-mini",
        prompt_tokens=100,
        completion_tokens=50,
        content="{}",
        latency_ms=200.0,
    )
    assert resp.score is None
    assert resp.request_id is not None


class TestBaseSchema:
    def test_strips_whitespace(self) -> None:
        class MySchema(BaseSchema):
            name: str

        s = MySchema(name="  hello  ")
        assert s.name == "hello"
