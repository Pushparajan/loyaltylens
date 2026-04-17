"""FastAPI service for offer copy generation with Flux AI brand image generation."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llm_generator.backends import OpenAIBackend
from llm_generator.generator import OfferCopy, OfferCopyGenerator
from llm_generator.multimodal import BRAND_IMAGES_DIR, BrandImageGenerator
from shared.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="LLM Generator", version="1.0.0")


# ── Request / response schemas ────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    customer_id: str
    offer_id: str
    prompt_version: int = 1
    generate_image: bool = False


class GenerateResponse(BaseModel):
    headline: str
    body: str
    cta: str
    tone: str
    model_version: str
    prompt_version: int
    latency_ms: float
    token_count: int
    generated_image_path: str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_generator(prompt_version: int) -> OfferCopyGenerator:
    return OfferCopyGenerator(backend=OpenAIBackend(), prompt_version=prompt_version)


def _generate_brand_image(copy: OfferCopy, offer_id: str) -> str | None:
    """Generate a Flux image from offer copy and persist it; return the saved path."""
    try:
        gen = BrandImageGenerator()
        prompt = f"{copy.headline}. {copy.body}"
        output_path = BRAND_IMAGES_DIR / f"{offer_id}_generated.png"
        gen.generate_to_path(prompt, output_path)
        return str(output_path)
    except Exception as exc:
        logger.warning("image_generation_failed", offer_id=offer_id, error=str(exc))
        return None


# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.post("/generate", responses={500: {"description": "LLM backend error or image generation failure"}})
async def generate(req: GenerateRequest) -> GenerateResponse:
    """Generate personalised offer copy; optionally produce a Flux AI brand image.

    Pass ``generate_image=true`` to trigger FLUX.1-schnell image generation.
    The generated PNG is saved to ``data/brand_images/{offer_id}_generated.png``
    and its path is returned in ``generated_image_path``.
    """
    customer_context = {
        "tier": "gold",
        "engagement_score": 0.75,
        "channel": "push",
    }
    offer = {
        "offer_title": f"Offer {req.offer_id}",
        "offer_description": "Earn bonus stars on your next handcrafted beverage.",
    }

    try:
        gen = _get_generator(req.prompt_version)
        copy = gen.generate(customer_context, offer)
    except Exception as exc:
        logger.error("generate_failed", customer_id=req.customer_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))

    image_path = _generate_brand_image(copy, req.offer_id) if req.generate_image else None

    return GenerateResponse(
        headline=copy.headline,
        body=copy.body,
        cta=copy.cta,
        tone=copy.tone,
        model_version=copy.model_version,
        prompt_version=copy.prompt_version,
        latency_ms=copy.latency_ms,
        token_count=copy.token_count,
        generated_image_path=image_path,
    )
