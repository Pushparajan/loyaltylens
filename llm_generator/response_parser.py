"""Parse and validate structured JSON responses from the LLM."""

from __future__ import annotations

import json

from pydantic import BaseModel

from shared.logger import get_logger

logger = get_logger(__name__)


class OfferResponse(BaseModel):
    subject: str
    body: str
    offer_code: str


class ResponseParser:
    """Extract a structured OfferResponse from a raw LLM completion string."""

    def parse(self, raw: str) -> OfferResponse:
        """Parse JSON from *raw*, returning a validated OfferResponse."""
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON object found in LLM response: {raw!r}")
        payload = json.loads(raw[start:end])
        offer = OfferResponse.model_validate(payload)
        logger.info("offer_parsed", offer_code=offer.offer_code)
        return offer
