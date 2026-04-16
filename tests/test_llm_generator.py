"""Unit tests for llm_generator: PromptBuilder and ResponseParser."""

from __future__ import annotations

from uuid import uuid4

import pytest

from llm_generator.prompt_builder import PromptBuilder
from llm_generator.response_parser import OfferResponse, ResponseParser
from shared.schemas import CustomerProfile, FeatureVector


class TestPromptBuilder:
    def _make_customer(self, **kwargs: object) -> CustomerProfile:
        return CustomerProfile(email="test@example.com", **kwargs)  # type: ignore[arg-type]

    def test_returns_two_messages(self) -> None:
        builder = PromptBuilder()
        msgs = builder.build(self._make_customer(), None, [])
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_includes_tier_in_user_message(self) -> None:
        customer = self._make_customer(tier="gold")
        builder = PromptBuilder()
        msgs = builder.build(customer, None, [])
        assert "gold" in msgs[1]["content"].lower()

    def test_includes_context_docs(self) -> None:
        builder = PromptBuilder()
        docs = [{"text": "Earn 2x points on Mondays.", "metadata": ""}]
        msgs = builder.build(self._make_customer(), None, docs)
        assert "Earn 2x points" in msgs[1]["content"]

    def test_includes_feature_summary(self) -> None:
        fv = FeatureVector(
            customer_id=uuid4(),
            feature_names=["recency", "frequency"],
            values=[0.2, 0.8],
        )
        builder = PromptBuilder()
        msgs = builder.build(self._make_customer(), fv, [])
        assert "recency" in msgs[1]["content"]


class TestResponseParser:
    def test_parses_valid_json(self) -> None:
        raw = '{"subject": "Hi", "body": "Get 20% off", "offer_code": "SAVE20"}'
        parser = ResponseParser()
        offer = parser.parse(raw)
        assert isinstance(offer, OfferResponse)
        assert offer.offer_code == "SAVE20"

    def test_parses_json_embedded_in_prose(self) -> None:
        raw = 'Here is the offer: {"subject": "S", "body": "B", "offer_code": "X1"} - enjoy!'
        parser = ResponseParser()
        offer = parser.parse(raw)
        assert offer.offer_code == "X1"

    def test_raises_on_missing_json(self) -> None:
        parser = ResponseParser()
        with pytest.raises(ValueError, match="No JSON object found"):
            parser.parse("No JSON here at all")

    def test_raises_on_invalid_schema(self) -> None:
        from pydantic import ValidationError

        parser = ResponseParser()
        with pytest.raises((ValidationError, ValueError)):
            parser.parse('{"wrong_key": "value"}')
