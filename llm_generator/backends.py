"""Abstract LLM backend interface plus HuggingFace and OpenAI implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from shared.config import get_settings
from shared.logger import get_logger

logger = get_logger(__name__)

_HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class LLMBackend(ABC):
    """Common interface for all LLM backends."""

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def generate(self, messages: list[dict[str, str]]) -> str:
        """Send *messages* to the model and return the raw text response."""


class HuggingFaceBackend(LLMBackend):
    """Local text-generation backend using Mistral-7B-Instruct via transformers pipeline."""

    def __init__(self) -> None:
        from transformers import pipeline  # type: ignore[import-untyped]

        settings = get_settings()
        self._pipe = pipeline(
            "text-generation",
            model=_HF_MODEL,
            token=settings.hf_token or None,
            max_new_tokens=256,
        )
        logger.info("hf_backend_loaded", model=_HF_MODEL)

    @property
    def model_name(self) -> str:
        return _HF_MODEL

    def generate(self, messages: list[dict[str, str]]) -> str:
        prompt = _format_mistral_instruct(messages)
        result = self._pipe(prompt, return_full_text=False)
        text: str = result[0]["generated_text"]
        logger.info("hf_generate_done", model=_HF_MODEL, approx_tokens=len(text.split()))
        return text


class OpenAIBackend(LLMBackend):
    """OpenAI chat-completions backend. Fails gracefully when API key is absent."""

    def __init__(self, model: str = _DEFAULT_OPENAI_MODEL) -> None:
        self._model = model
        settings = get_settings()
        if not settings.openai_api_key:
            logger.warning("openai_api_key_missing", hint="set OPENAI_API_KEY in .env")
            self._client = None
        else:
            from openai import OpenAI

            self._client = OpenAI(api_key=settings.openai_api_key)

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, messages: list[dict[str, str]]) -> str:
        if self._client is None:
            raise ValueError(
                "OPENAI_API_KEY is not set — add it to .env or use HuggingFaceBackend."
            )
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        return completion.choices[0].message.content or ""


# ── helpers ───────────────────────────────────────────────────────────────────

def _format_mistral_instruct(messages: list[dict[str, str]]) -> str:
    """Render an OpenAI-style messages list as a Mistral [INST] prompt."""
    parts: list[str] = []
    system_prefix = ""
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if role == "system":
            system_prefix = content + "\n"
        elif role == "user":
            parts.append(f"<s>[INST] {system_prefix}{content} [/INST]")
            system_prefix = ""
        elif role == "assistant":
            parts.append(f" {content} </s>")
    return "".join(parts)
