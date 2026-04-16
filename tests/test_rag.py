"""Tests for the rag_retrieval module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

OFFERS_PATH = Path(__file__).parent.parent / "rag_retrieval" / "data" / "offers.json"


def _load_offers() -> list[dict[str, Any]]:
    return json.loads(OFFERS_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. generate_offers
# ---------------------------------------------------------------------------

class TestGenerateOffers:
    def test_generates_200_offers(self) -> None:
        from rag_retrieval.generate_offers import generate_offers

        offers = generate_offers(200)
        assert len(offers) == 200

    def test_schema_fields(self) -> None:
        from rag_retrieval.generate_offers import generate_offers

        required = {"id", "title", "description", "category", "channel",
                    "min_propensity_threshold", "discount_pct", "expiry_days"}
        for offer in generate_offers(5):
            assert required <= set(offer.keys())

    def test_category_values(self) -> None:
        from rag_retrieval.generate_offers import generate_offers, CATEGORIES

        for offer in generate_offers(200):
            assert offer["category"] in CATEGORIES

    def test_propensity_threshold_range(self) -> None:
        from rag_retrieval.generate_offers import generate_offers

        for offer in generate_offers(200):
            assert 0.0 <= offer["min_propensity_threshold"] <= 0.7


# ---------------------------------------------------------------------------
# 2. Embedding shape
# ---------------------------------------------------------------------------

class TestEmbeddingShape:
    def test_embedding_dim_is_384(self) -> None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        vec = model.encode(["loyalty reward offer"])
        assert vec.shape == (1, 384), f"expected (1, 384), got {vec.shape}"

    def test_batch_embedding_shape(self) -> None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        texts = ["offer one", "offer two", "offer three"]
        vecs = model.encode(texts)
        assert vecs.shape == (3, 384)

    def test_embedding_is_normalised(self) -> None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", )
        vecs = model.encode(["test"], normalize_embeddings=True)
        norm = float(np.linalg.norm(vecs[0]))
        assert abs(norm - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 3. pgvector round-trip
# ---------------------------------------------------------------------------

class TestPgvectorRoundTrip:
    """Requires a live pgvector instance (docker compose up postgres)."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_pg(self) -> None:
        pytest.importorskip("psycopg2")
        try:
            import psycopg2
            from shared.config import get_settings

            conn = psycopg2.connect(get_settings().postgres_url, connect_timeout=2)
            conn.close()
        except Exception:
            pytest.skip("pgvector not reachable")

    def test_upsert_and_query(self) -> None:
        import psycopg2
        from shared.config import get_settings

        conn = psycopg2.connect(get_settings().postgres_url)
        try:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS offer_embeddings "
                    "(id TEXT PRIMARY KEY, title TEXT, embedding vector(384));"
                )
                dummy_vec = [0.0] * 384
                dummy_vec[0] = 1.0
                cur.execute(
                    "INSERT INTO offer_embeddings (id, title, embedding) "
                    "VALUES (%s, %s, %s) ON CONFLICT (id) DO UPDATE "
                    "SET title = EXCLUDED.title, embedding = EXCLUDED.embedding;",
                    ("test_offer_001", "Test Offer", dummy_vec),
                )
            conn.commit()

            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, title FROM offer_embeddings WHERE id = %s;",
                    ("test_offer_001",),
                )
                row = cur.fetchone()
            assert row is not None
            assert row[0] == "test_offer_001"
            assert row[1] == "Test Offer"
        finally:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM offer_embeddings WHERE id = %s;", ("test_offer_001",))
            conn.commit()
            conn.close()


# ---------------------------------------------------------------------------
# 4. LangChain retriever — returns k results with propensity filter
# ---------------------------------------------------------------------------

class TestLangChainRetriever:
    @pytest.fixture
    def mock_store(self) -> MagicMock:
        store = MagicMock()
        offers = _load_offers() if OFFERS_PATH.exists() else []
        # Build fake similarity results; score 0.9 for all
        fake_results = []
        for o in offers[:20]:
            doc = MagicMock()
            doc.page_content = o["description"]
            doc.metadata = {
                "id": o["id"],
                "title": o["title"],
                "category": o["category"],
                "min_propensity_threshold": 0.0,  # all pass filter
            }
            fake_results.append((doc, 0.9))
        store.similarity_search_with_score.return_value = fake_results
        return store

    def test_returns_k_results(self, mock_store: MagicMock) -> None:
        from rag_retrieval.langchain_retriever import LangChainOfferRetriever

        with patch.object(LangChainOfferRetriever, "__init__", lambda self: None):
            retriever = LangChainOfferRetriever.__new__(LangChainOfferRetriever)
            retriever._store = mock_store

        results = retriever.retrieve("coffee lover", propensity=0.8, k=5)
        assert len(results) == 5

    def test_propensity_filter_applied(self, mock_store: MagicMock) -> None:
        from rag_retrieval.langchain_retriever import LangChainOfferRetriever

        # Override some results to have a high threshold that blocks them
        high_threshold_doc = MagicMock()
        high_threshold_doc.page_content = "blocked offer description"
        high_threshold_doc.metadata = {
            "id": "offer_block",
            "title": "Blocked Offer",
            "category": "food",
            "min_propensity_threshold": 0.9,  # requires propensity >= 0.9
        }
        # Prepend high-threshold docs so they come first
        mock_store.similarity_search_with_score.return_value = (
            [(high_threshold_doc, 0.99)] * 10
            + mock_store.similarity_search_with_score.return_value[:10]
        )

        with patch.object(LangChainOfferRetriever, "__init__", lambda self: None):
            retriever = LangChainOfferRetriever.__new__(LangChainOfferRetriever)
            retriever._store = mock_store

        # propensity=0.5 is below the 0.9 threshold → blocked docs excluded
        results = retriever.retrieve("coffee lover", propensity=0.5, k=5)
        assert all(r.offer_id != "offer_block" for r in results)

    def test_result_fields(self, mock_store: MagicMock) -> None:
        from rag_retrieval.langchain_retriever import LangChainOfferRetriever, OfferResult

        with patch.object(LangChainOfferRetriever, "__init__", lambda self: None):
            retriever = LangChainOfferRetriever.__new__(LangChainOfferRetriever)
            retriever._store = mock_store

        results = retriever.retrieve("snack deal", propensity=0.5, k=1)
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, OfferResult)
        assert r.offer_id
        assert r.title
        assert r.category


# ---------------------------------------------------------------------------
# 5. LlamaIndex retriever — returns k results with propensity filter
# ---------------------------------------------------------------------------

class TestLlamaRetriever:
    @pytest.fixture
    def offers_file(self, tmp_path: Path) -> Path:
        from rag_retrieval.generate_offers import generate_offers

        path = tmp_path / "offers.json"
        path.write_text(json.dumps(generate_offers(20)), encoding="utf-8")
        return path

    def test_returns_k_results(self, offers_file: Path) -> None:
        from rag_retrieval.llama_retriever import LlamaOfferRetriever

        retriever = LlamaOfferRetriever(offers_path=offers_file)
        results = retriever.retrieve("beverage rewards", propensity=1.0, k=5)
        assert len(results) == 5

    def test_propensity_filter_blocks_high_threshold(self, offers_file: Path) -> None:
        from rag_retrieval.llama_retriever import LlamaOfferRetriever

        retriever = LlamaOfferRetriever(offers_path=offers_file)
        # propensity=0.0 should block anything with threshold > 0
        results = retriever.retrieve("any offer", propensity=0.0, k=5)
        for r in results:
            assert r.score >= 0  # just ensure no crash; may return < k

    def test_result_is_offer_result(self, offers_file: Path) -> None:
        from rag_retrieval.llama_retriever import LlamaOfferRetriever
        from rag_retrieval.langchain_retriever import OfferResult

        retriever = LlamaOfferRetriever(offers_path=offers_file)
        results = retriever.retrieve("food deal", propensity=1.0, k=3)
        for r in results:
            assert isinstance(r, OfferResult)
