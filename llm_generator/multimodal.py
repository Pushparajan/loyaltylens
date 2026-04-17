"""Flux AI brand image generator using HuggingFace free Inference API."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import InferenceClient
from PIL import Image

from shared.config import get_settings
from shared.logger import get_logger

logger = get_logger(__name__)

_FLUX_MODEL = "black-forest-labs/FLUX.1-schnell"
BRAND_IMAGES_DIR = Path("data/brand_images")


class BrandImageGenerator:
    """Generate brand images from offer copy text using FLUX.1-schnell.

    Uses the HuggingFace free Inference API — no GPU required.
    Set HF_TOKEN in .env for higher rate limits.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._client = InferenceClient(
            model=_FLUX_MODEL,
            token=settings.hf_token or None,
        )
        logger.info("flux_backend_ready", model=_FLUX_MODEL)

    def generate(self, copy_text: str) -> Image.Image:
        """Generate a brand image from *copy_text*. Returns a PIL Image."""
        image: Image.Image = self._client.text_to_image(copy_text)
        logger.info("image_generated", prompt=copy_text[:80])
        return image

    def generate_to_path(self, copy_text: str, output_path: str | Path) -> Path:
        """Generate an image from *copy_text* and save it to *output_path*."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.generate(copy_text).save(path)
        logger.info("image_saved", path=str(path))
        return path
