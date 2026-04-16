"""Orchestrate LLM calls for personalised offer and communication generation."""

from __future__ import annotations

import time
from uuid import UUID

from openai import OpenAI

from llm_generator.prompt_builder import PromptBuilder
from llm_generator.response_parser import OfferResponse, ResponseParser
from shared.config import get_settings
from shared.logger import get_logger
from shared.schemas import CustomerProfile, FeatureVector, LLMResponse

logger = get_logger(__name__)

_DEFAULT_MODEL = "gpt-4o-mini"


class LLMGenerator:
    """Generate personalised retention offers via an OpenAI-compatible LLM."""

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        settings = get_settings()
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = model
        self._builder = PromptBuilder()
        self._parser = ResponseParser()

    def generate(
        self,
        customer: CustomerProfile,
        features: FeatureVector | None,
        context_docs: list[dict[str, str]],
    ) -> tuple[OfferResponse, LLMResponse]:
        """Generate an offer for *customer*. Returns the parsed offer and raw LLM metadata."""
        messages = self._builder.build(customer, features, context_docs)
        t0 = time.monotonic()
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        latency_ms = (time.monotonic() - t0) * 1000
        choice = completion.choices[0]
        raw_text = choice.message.content or ""

        llm_meta = LLMResponse(
            model=self._model,
            prompt_tokens=completion.usage.prompt_tokens if completion.usage else 0,
            completion_tokens=completion.usage.completion_tokens if completion.usage else 0,
            content=raw_text,
            latency_ms=latency_ms,
        )
        offer = self._parser.parse(raw_text)
        logger.info(
            "offer_generated",
            customer_id=str(customer.customer_id),
            offer_code=offer.offer_code,
            latency_ms=round(latency_ms, 1),
        )
        return offer, llm_meta

    def generate_for_customer_id(
        self,
        customer_id: UUID,
        context_docs: list[dict[str, str]] | None = None,
    ) -> tuple[OfferResponse, LLMResponse]:
        """Convenience wrapper that builds a minimal CustomerProfile from an ID."""
        profile = CustomerProfile(customer_id=customer_id, email="unknown@example.com")
        return self.generate(profile, None, context_docs or [])
