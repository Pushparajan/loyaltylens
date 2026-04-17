"""Tests for prompt registry, OfferCopyGenerator, BrandImageScorer, and /generate API."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from llm_generator.backends import LLMBackend
from llm_generator.generator import OfferCopy, OfferCopyGenerator

# ── Shared fixtures ───────────────────────────────────────────────────────────

_CONTEXT = {"tier": "gold", "engagement_score": 0.82, "channel": "push"}
_OFFER = {
    "offer_title": "Double Star Day",
    "offer_description": "Earn 2x stars on any handcrafted beverage.",
}
_VALID_JSON = json.dumps(
    {
        "headline": "Earn 2× Stars Today",
        "body": "Treat yourself to any handcrafted beverage and earn double stars.",
        "cta": "Order Now",
        "tone": "friendly",
    }
)


class _StubBackend(LLMBackend):
    """Deterministic backend for unit tests — returns a fixed response string."""

    def __init__(self, response: str = _VALID_JSON) -> None:
        self._response = response
        self._calls: list[list[dict[str, str]]] = []

    @property
    def model_name(self) -> str:
        return "stub-v0"

    def generate(self, messages: list[dict[str, str]]) -> str:
        self._calls.append(messages)
        return self._response


class _IterBackend(LLMBackend):
    """Returns responses from an iterator — used to test retry logic."""

    def __init__(self, *responses: str) -> None:
        self._iter = iter(responses)

    @property
    def model_name(self) -> str:
        return "iter-v0"

    def generate(self, messages: list[dict[str, str]]) -> str:
        return next(self._iter)


# ── Prompt rendering ──────────────────────────────────────────────────────────


class TestPromptRendering:
    def test_v1_system_message_present(self) -> None:
        backend = _StubBackend()
        OfferCopyGenerator(backend, prompt_version=1).generate(_CONTEXT, _OFFER)
        assert backend._calls[0][0]["role"] == "system"
        assert "copywriter" in backend._calls[0][0]["content"].lower()

    def test_v1_renders_tier_in_user_message(self) -> None:
        backend = _StubBackend()
        OfferCopyGenerator(backend, prompt_version=1).generate(_CONTEXT, _OFFER)
        assert "gold" in backend._calls[0][1]["content"]

    def test_v1_renders_engagement_score_formatted(self) -> None:
        backend = _StubBackend()
        OfferCopyGenerator(backend, prompt_version=1).generate(_CONTEXT, _OFFER)
        assert "0.82" in backend._calls[0][1]["content"]

    def test_v1_renders_offer_title(self) -> None:
        backend = _StubBackend()
        OfferCopyGenerator(backend, prompt_version=1).generate(_CONTEXT, _OFFER)
        assert "Double Star Day" in backend._calls[0][1]["content"]

    def test_v2_prompt_loads_and_generates(self) -> None:
        copy = OfferCopyGenerator(_StubBackend(), prompt_version=2).generate(_CONTEXT, _OFFER)
        assert isinstance(copy, OfferCopy)
        assert copy.prompt_version == 2


# ── JSON parsing and OfferCopy fields ────────────────────────────────────────


class TestOfferCopyGenerator:
    def test_valid_json_parsed(self) -> None:
        copy = OfferCopyGenerator(_StubBackend(), prompt_version=1).generate(_CONTEXT, _OFFER)
        assert copy.headline == "Earn 2× Stars Today"
        assert copy.cta == "Order Now"

    def test_all_fields_present(self) -> None:
        copy = OfferCopyGenerator(_StubBackend(), prompt_version=1).generate(_CONTEXT, _OFFER)
        assert copy.headline
        assert copy.body
        assert copy.cta
        assert copy.tone in {"friendly", "urgent", "exclusive"}
        assert copy.model_version == "stub-v0"
        assert copy.prompt_version == 1
        assert copy.latency_ms >= 0.0
        assert isinstance(copy.token_count, int) and copy.token_count > 0

    def test_json_embedded_in_prose(self) -> None:
        raw = f"Here is your copy: {_VALID_JSON} — enjoy!"
        copy = OfferCopyGenerator(_StubBackend(raw), prompt_version=1).generate(_CONTEXT, _OFFER)
        assert copy.cta == "Order Now"

    def test_retry_on_first_parse_failure(self) -> None:
        backend = _IterBackend("not json at all", _VALID_JSON)
        copy = OfferCopyGenerator(backend, prompt_version=1).generate(_CONTEXT, _OFFER)
        assert copy.headline == "Earn 2× Stars Today"

    def test_raises_after_two_parse_failures(self) -> None:
        backend = _StubBackend("definitely not json")
        with pytest.raises(ValueError):
            OfferCopyGenerator(backend, prompt_version=1).generate(_CONTEXT, _OFFER)

    def test_missing_fields_raise(self) -> None:
        bad = json.dumps({"headline": "H", "body": "B"})  # missing cta + tone
        with pytest.raises(ValueError, match="missing fields"):
            OfferCopyGenerator(_StubBackend(bad), prompt_version=1).generate(_CONTEXT, _OFFER)


# ── Flux AI brand image generator ────────────────────────────────────────────


def _stub_flux_client(image: PILImage.Image | None = None):
    """Return a mock InferenceClient whose text_to_image returns a PIL Image."""
    mock = MagicMock()
    mock.return_value.text_to_image.return_value = (
        image if image is not None else PILImage.new("RGB", (128, 128), color=(34, 139, 34))
    )
    return mock


class TestBrandImageGenerator:
    def _generator(self, mock_client_cls):
        from llm_generator.multimodal import BrandImageGenerator

        with patch("llm_generator.multimodal.InferenceClient", mock_client_cls):
            gen = BrandImageGenerator()
        gen._client = mock_client_cls.return_value
        return gen

    def test_generate_returns_pil_image(self) -> None:
        mock_client = _stub_flux_client()
        gen = self._generator(mock_client)
        result = gen.generate("Earn 2x stars on any handcrafted beverage.")
        assert isinstance(result, PILImage.Image)

    def test_generate_calls_text_to_image_with_prompt(self) -> None:
        mock_client = _stub_flux_client()
        gen = self._generator(mock_client)
        prompt = "Double Star Day — earn bonus points today."
        gen.generate(prompt)
        mock_client.return_value.text_to_image.assert_called_once_with(prompt)

    def test_generate_to_path_saves_file(self, tmp_path: Path) -> None:
        mock_client = _stub_flux_client()
        gen = self._generator(mock_client)
        out = gen.generate_to_path("Earn stars now!", tmp_path / "brand.png")
        assert out.exists()
        assert out.suffix == ".png"

    def test_generate_to_path_creates_parent_dirs(self, tmp_path: Path) -> None:
        mock_client = _stub_flux_client()
        gen = self._generator(mock_client)
        nested = tmp_path / "a" / "b" / "brand.png"
        gen.generate_to_path("Stars await.", nested)
        assert nested.exists()


# ── API endpoint contract ─────────────────────────────────────────────────────


def _mock_copy(**overrides) -> OfferCopy:
    defaults = {
        "headline": "Stars Await",
        "body": "Double your points this weekend on any order.",
        "cta": "Claim Now",
        "tone": "urgent",
        "model_version": "gpt-4o-mini",
        "prompt_version": 1,
        "latency_ms": 280.5,
        "token_count": 48,
    }
    return OfferCopy(**{**defaults, **overrides})


class TestGenerateAPI:
    @pytest.fixture()
    def client(self):
        from llm_generator.api import app

        return TestClient(app)

    def test_returns_200(self, client: TestClient) -> None:
        with patch("llm_generator.api._get_generator") as mock_gen:
            mock_gen.return_value.generate.return_value = _mock_copy()
            resp = client.post("/generate", json={"customer_id": "C001", "offer_id": "O001"})
        assert resp.status_code == 200

    def test_response_contains_required_fields(self, client: TestClient) -> None:
        with patch("llm_generator.api._get_generator") as mock_gen:
            mock_gen.return_value.generate.return_value = _mock_copy()
            resp = client.post("/generate", json={"customer_id": "C001", "offer_id": "O001"})
        data = resp.json()
        for field in ("headline", "body", "cta", "tone", "model_version",
                      "prompt_version", "latency_ms", "token_count"):
            assert field in data, f"missing field: {field}"

    def test_generated_image_path_none_when_not_requested(self, client: TestClient) -> None:
        with patch("llm_generator.api._get_generator") as mock_gen:
            mock_gen.return_value.generate.return_value = _mock_copy()
            resp = client.post("/generate", json={"customer_id": "C001", "offer_id": "O001"})
        assert resp.json()["generated_image_path"] is None

    def test_prompt_version_forwarded(self, client: TestClient) -> None:
        with patch("llm_generator.api._get_generator") as mock_gen:
            mock_gen.return_value.generate.return_value = _mock_copy(prompt_version=2)
            resp = client.post(
                "/generate",
                json={"customer_id": "C001", "offer_id": "O001", "prompt_version": 2},
            )
        assert resp.json()["prompt_version"] == 2
        mock_gen.assert_called_once_with(2)

    def test_500_on_backend_error(self, client: TestClient) -> None:
        with patch("llm_generator.api._get_generator") as mock_gen:
            mock_gen.return_value.generate.side_effect = ValueError("OPENAI_API_KEY is not set")
            resp = client.post("/generate", json={"customer_id": "C001", "offer_id": "O001"})
        assert resp.status_code == 500
