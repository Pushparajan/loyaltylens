"""Shared Pydantic v2 base model and common domain schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class BaseSchema(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
        str_strip_whitespace=True,
    )


class TimestampedSchema(BaseSchema):
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Transaction(TimestampedSchema):
    transaction_id: UUID = Field(default_factory=uuid4)
    customer_id: UUID
    amount: float
    currency: str = "USD"
    store_id: str
    items: list[dict[str, Any]] = Field(default_factory=list)


class CustomerProfile(TimestampedSchema):
    customer_id: UUID = Field(default_factory=uuid4)
    email: str
    tier: str = "standard"
    total_spend: float = 0.0
    visit_count: int = 0
    churn_score: float | None = None


class FeatureVector(BaseSchema):
    customer_id: UUID
    feature_names: list[str]
    values: list[float]
    computed_at: datetime = Field(default_factory=datetime.utcnow)


class LLMResponse(BaseSchema):
    request_id: UUID = Field(default_factory=uuid4)
    model: str
    prompt_tokens: int
    completion_tokens: int
    content: str
    latency_ms: float
    score: float | None = None
