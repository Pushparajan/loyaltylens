"""Orchestrate LLM calls for personalised offer and communication generation."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml
from openai import OpenAI

from llm_generator.backends import LLMBackend
from llm_generator.prompt_builder import PromptBuilder
from llm_generator.response_parser import OfferResponse, ResponseParser
from shared.config import get_settings
from shared.logger import get_logger
from shared.schemas import CustomerProfile, FeatureVector, LLMResponse

logger = get_logger(__name__)

# ── OfferCopy dataclass + OfferCopyGenerator ──────────────────────────────────

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_VAR_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


@dataclass
class OfferCopy:
    headline: str
    body: str
    cta: str
    tone: str
    model_version: str
    prompt_version: int
    latency_ms: float
    token_count: int


def _load_prompt(version: int) -> dict[str, str]:
    path = _PROMPTS_DIR / f"system_v{version}.yaml"
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)  # type: ignore[no-any-return]


def _render(template: str, context: dict[str, Any]) -> str:
    """Substitute only known {key} and {key:fmt} placeholders; leave others untouched."""

    def replacer(m: re.Match) -> str:
        key, spec = m.group(1), m.group(2) or ""
        if key not in context:
            return m.group(0)
        return format(context[key], spec) if spec else str(context[key])

    return _VAR_RE.sub(replacer, template)


def _parse_copy_json(raw: str) -> dict[str, Any]:
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in LLM response: {raw!r}")
    payload: dict[str, Any] = json.loads(raw[start:end])
    missing = {"headline", "body", "cta", "tone"} - payload.keys()
    if missing:
        raise ValueError(f"LLM response missing fields: {missing}")
    return payload


class OfferCopyGenerator:
    """Generate brand-consistent Starbucks offer copy via a pluggable LLM backend."""

    def __init__(self, backend: LLMBackend, prompt_version: int = 1) -> None:
        self._backend = backend
        self._prompt_version = prompt_version
        self._prompt = _load_prompt(prompt_version)

    def generate(
        self,
        customer_context: dict[str, Any],
        offer: dict[str, Any],
    ) -> OfferCopy:
        """Render the prompt, call the backend, parse JSON; retry once on parse failure."""
        context = {**customer_context, **offer}
        user_msg = _render(self._prompt["user_template"], context).strip()
        messages = [
            {"role": "system", "content": self._prompt["system"].strip()},
            {"role": "user", "content": user_msg},
        ]

        t0 = time.monotonic()
        raw = self._backend.generate(messages)
        latency_ms = (time.monotonic() - t0) * 1000

        try:
            data = _parse_copy_json(raw)
        except ValueError:
            logger.warning("offer_copy_parse_retry", prompt_version=self._prompt_version)
            raw = self._backend.generate(messages)
            data = _parse_copy_json(raw)

        copy = OfferCopy(
            headline=data["headline"],
            body=data["body"],
            cta=data["cta"],
            tone=data["tone"],
            model_version=self._backend.model_name,
            prompt_version=self._prompt_version,
            latency_ms=round(latency_ms, 1),
            token_count=len(raw.split()),
        )
        logger.info(
            "offer_copy_generated",
            model=self._backend.model_name,
            prompt_version=self._prompt_version,
            latency_ms=copy.latency_ms,
        )
        return copy

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
