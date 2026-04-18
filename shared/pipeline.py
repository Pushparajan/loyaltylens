"""End-to-end LoyaltyLens pipeline: features → propensity → RAG → copy generation."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    customer_id: str
    feature_dict: dict[str, Any]
    propensity_score: float
    propensity_label: int
    top_offers: list[dict[str, Any]]
    offer_copy: dict[str, Any]
    latency_breakdown: dict[str, float]
    model_version: str
    prompt_version: str
    pipeline_run_id: str
    timestamp: str


class LoyaltyLensPipeline:
    """Wire all 6 modules into a single customer-level inference pipeline.

    Heavy deps (PGVector, OpenAI, XGBoost) are imported lazily so the class
    can be instantiated in unit tests without those packages available.
    """

    def __init__(self) -> None:
        self._feature_store: Any = None
        self._predictor: Any = None
        self._retriever: Any = None
        self._generator: Any = None

    # ── lazy initialisation ───────────────────────────────────────────────────

    def _get_feature_store(self) -> Any:
        if self._feature_store is None:
            from feature_store.store import FeatureStore
            self._feature_store = FeatureStore()
        return self._feature_store

    def _get_predictor(self) -> Any:
        if self._predictor is None:
            from propensity_model.predictor import PropensityPredictor
            from shared.config import get_settings
            settings = get_settings()
            self._predictor = PropensityPredictor().load(
                version=settings.propensity_model_version,
                models_dir=settings.propensity_models_dir,
            )
        return self._predictor

    def _get_retriever(self) -> Any:
        if self._retriever is None:
            self._retriever = _WeaviateOfferRetriever()
        return self._retriever

    def _get_generator(self) -> Any:
        if self._generator is None:
            from llm_generator.backends import OpenAIBackend
            from llm_generator.generator import OfferCopyGenerator
            from shared.config import get_settings
            settings = get_settings()
            if not settings.openai_api_key:
                self._generator = _StubGenerator()
            else:
                self._generator = OfferCopyGenerator(backend=OpenAIBackend())
        return self._generator

    # ── public API ────────────────────────────────────────────────────────────

    def run_for_customer(self, customer_id: str) -> PipelineResult:
        run_id = str(uuid.uuid4())
        latency: dict[str, float] = {}

        log = logger.bind(customer_id=customer_id, pipeline_run_id=run_id)
        log.info("pipeline_start")

        # Step 1 — features
        t0 = time.perf_counter()
        feature_dict = self._get_feature_store().read_latest(customer_id)
        latency["feature_store_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if feature_dict is None:
            raise ValueError(f"No features found for customer_id={customer_id!r}")

        # Step 2 — propensity
        t0 = time.perf_counter()
        result = self._get_predictor().predict(feature_dict)
        latency["propensity_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # Step 3 — RAG retrieval
        t0 = time.perf_counter()
        customer_context = _build_context_string(customer_id, feature_dict)
        raw_offers = self._get_retriever().retrieve(
            customer_context, result.propensity_score, k=5
        )
        latency["rag_retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        top_offers = [o if isinstance(o, dict) else _offer_to_dict(o) for o in raw_offers]
        best_offer = top_offers[0] if top_offers else _fallback_offer()

        # Step 4 — copy generation
        t0 = time.perf_counter()
        customer_ctx = {
            "customer_id": customer_id,
            "tier": feature_dict.get("tier", "Gold"),
            "channel": feature_dict.get("channel_preference", "mobile"),
            **feature_dict,
        }
        offer_ctx = {
            "offer_title": best_offer.get("title", ""),
            "offer_description": best_offer.get("description", ""),
            **best_offer,
        }
        copy = self._get_generator().generate(customer_ctx, offer_ctx)
        latency["llm_generator_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        log.info(
            "pipeline_complete",
            propensity_score=result.propensity_score,
            offers_retrieved=len(top_offers),
            **dict(latency),
        )

        return PipelineResult(
            customer_id=customer_id,
            feature_dict=feature_dict,
            propensity_score=result.propensity_score,
            propensity_label=result.label,
            top_offers=top_offers,
            offer_copy=_copy_to_dict(copy),
            latency_breakdown=latency,
            model_version=result.model_version,
            prompt_version=str(getattr(copy, "prompt_version", "unknown")),
            pipeline_run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_context_string(customer_id: str, features: dict[str, Any]) -> str:
    parts = [f"customer_id={customer_id}"]
    for k in ("recency_days", "frequency_30d", "monetary_90d", "engagement_score"):
        if k in features:
            parts.append(f"{k}={features[k]}")
    return " ".join(parts)


def _offer_to_dict(offer: Any) -> dict[str, Any]:
    return {
        "offer_id": getattr(offer, "offer_id", ""),
        "title": getattr(offer, "title", ""),
        "description": getattr(offer, "description", ""),
        "category": getattr(offer, "category", ""),
        "score": getattr(offer, "score", 0.0),
    }


def _fallback_offer() -> dict[str, Any]:
    return {
        "offer_id": "fallback",
        "title": "Loyalty Reward",
        "description": "Earn bonus points on your next visit.",
        "category": "general",
        "score": 0.0,
    }


def _copy_to_dict(copy: Any) -> dict[str, Any]:
    return {
        "headline": getattr(copy, "headline", ""),
        "body": getattr(copy, "body", ""),
        "cta": getattr(copy, "cta", ""),
        "tone": getattr(copy, "tone", ""),
        "model_version": getattr(copy, "model_version", "stub"),
        "prompt_version": str(getattr(copy, "prompt_version", "unknown")),
        "latency_ms": getattr(copy, "latency_ms", 0.0),
        "token_count": getattr(copy, "token_count", 0),
    }


_OFFERS_PATH = Path(__file__).parent.parent / "rag_retrieval" / "data" / "offers.json"


class _WeaviateOfferRetriever:
    """Weaviate-backed offer retriever — indexes offers.json on first run."""

    def __init__(self) -> None:
        from rag_retrieval.indexer import DocumentIndexer
        from rag_retrieval.retriever import VectorRetriever
        from rag_retrieval.weaviate_client import WeaviateClient

        self._weaviate = WeaviateClient()
        self._indexer = DocumentIndexer(weaviate=self._weaviate)
        self._retriever = VectorRetriever(weaviate=self._weaviate)
        self._offers: dict[str, dict[str, Any]] = {}
        self._ensure_indexed()

    def _ensure_indexed(self) -> None:
        from rag_retrieval.weaviate_client import _CLASS_NAME

        col = self._weaviate.client.collections.get(_CLASS_NAME)
        result = col.query.fetch_objects(limit=1)
        if result.objects:
            self._load_offers_map()
            return

        if not _OFFERS_PATH.exists():
            logger.warning("offers_file_not_found", path=str(_OFFERS_PATH))
            return

        offers: list[dict[str, Any]] = json.loads(_OFFERS_PATH.read_text(encoding="utf-8"))
        self._offers = {o["id"]: o for o in offers}
        docs = [
            {
                "text": f"{o['title']} {o['description']}",
                "metadata": json.dumps({
                    "offer_id": o["id"],
                    "title": o["title"],
                    "description": o["description"],
                    "category": o.get("category", ""),
                    "min_propensity_threshold": o.get("min_propensity_threshold", 0.0),
                }),
            }
            for o in offers
        ]
        n = self._indexer.index(docs)
        logger.info("offers_indexed", count=n)

    def _load_offers_map(self) -> None:
        if not _OFFERS_PATH.exists():
            return
        offers: list[dict[str, Any]] = json.loads(_OFFERS_PATH.read_text(encoding="utf-8"))
        self._offers = {o["id"]: o for o in offers}

    def retrieve(self, customer_context: str, propensity_score: float, k: int = 5) -> list[dict[str, Any]]:
        raw = self._retriever.retrieve(customer_context)
        all_parsed = []
        for i, doc in enumerate(raw):
            try:
                meta: dict[str, Any] = json.loads(doc.get("metadata", "{}"))
            except (json.JSONDecodeError, TypeError):
                meta = {}
            all_parsed.append((i, meta, doc))

        # prefer offers within propensity threshold; fall back to full list if none qualify
        filtered = [
            (i, meta, doc) for i, meta, doc in all_parsed
            if propensity_score >= float(meta.get("min_propensity_threshold", 0.0))
        ]
        candidates = filtered or all_parsed

        results = []
        for i, meta, doc in candidates[:k]:
            results.append({
                "offer_id": meta.get("offer_id", f"doc_{i}"),
                "title": meta.get("title", doc.get("text", "")[:60]),
                "description": meta.get("description", doc.get("text", "")),
                "category": meta.get("category", ""),
                "score": 1.0 / (1.0 + i),
            })

        return results or [_fallback_offer()]

    def close(self) -> None:
        self._weaviate.close()


class _StubGenerator:
    """No-op generator used when OPENAI_API_KEY is not configured."""

    def generate(self, _customer_context: dict[str, Any], offer: dict[str, Any]) -> Any:
        from dataclasses import dataclass as _dc

        @_dc
        class _StubCopy:
            headline: str = f"Special offer: {offer.get('title', 'Reward')}"
            body: str = offer.get("description", "Enjoy your loyalty reward.")
            cta: str = "Redeem now"
            tone: str = "friendly"
            model_version: str = "stub"
            prompt_version: int = 0
            latency_ms: float = 0.0
            token_count: int = 0

        return _StubCopy()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_result(r: PipelineResult) -> None:
    width = 60
    print("=" * width)
    print(f"  Pipeline run  {r.pipeline_run_id[:8]}…")
    print(f"  {r.timestamp}")
    print("=" * width)
    print(f"  Customer        {r.customer_id}")
    print(f"  Propensity      {r.propensity_score:.4f}  (label={r.propensity_label})")
    print(f"  Model version   {r.model_version}")
    print(f"  Prompt version  {r.prompt_version}")
    print()
    print("  Top offer")
    if r.top_offers:
        o = r.top_offers[0]
        print(f"    {o['offer_id']:12s}  {o['title']}")
        print(f"    Category: {o['category']}  Score: {o['score']:.3f}")
    print()
    print("  Generated copy")
    print(f"    Headline  {r.offer_copy['headline']}")
    print(f"    Body      {r.offer_copy['body']}")
    print(f"    CTA       {r.offer_copy['cta']}")
    print()
    print("  Latency breakdown")
    for step, ms in r.latency_breakdown.items():
        print(f"    {step:<28s} {ms:>8.1f} ms")
    total = sum(r.latency_breakdown.values())
    print(f"    {'total':<28s} {total:>8.1f} ms")
    print("=" * width)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LoyaltyLens pipeline for one customer.")
    parser.add_argument("--customer-id", required=True, help="Customer UUID or ID string")
    args = parser.parse_args()

    pipeline = LoyaltyLensPipeline()
    try:
        result = pipeline.run_for_customer(args.customer_id)
        _print_result(result)
    finally:
        if pipeline._retriever is not None and hasattr(pipeline._retriever, "close"):
            pipeline._retriever.close()
