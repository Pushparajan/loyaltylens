"""llm_generator — generate personalised offers and communications via LLM."""

from llm_generator.backends import HuggingFaceBackend, LLMBackend, OpenAIBackend
from llm_generator.generator import LLMGenerator, OfferCopy, OfferCopyGenerator
from llm_generator.multimodal import BrandImageGenerator
from llm_generator.prompt_builder import PromptBuilder
from llm_generator.response_parser import ResponseParser

__all__ = [
    "LLMGenerator",
    "LLMBackend",
    "HuggingFaceBackend",
    "OpenAIBackend",
    "OfferCopy",
    "OfferCopyGenerator",
    "BrandImageGenerator",
    "PromptBuilder",
    "ResponseParser",
]
