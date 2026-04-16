---
title: "LangChain vs. LlamaIndex in Production: What the Benchmarks Actually Show"
slug: "loyaltylens-rag-retrieval"
description: "Building a RAG offer retrieval system — real latency and precision benchmarks comparing LangChain vs. LlamaIndex and pgvector vs. Weaviate at three catalog sizes."
date: 2026-05-12
author: Pushparajan Ramar
series: loyaltylens
series_order: 3
reading_time: 15
tags:
  - rag
  - langchain
  - llamaindex
  - vector-databases
  - pgvector
  - weaviate
  - huggingface
---

# LangChain vs. LlamaIndex in Production: What the Benchmarks Actually Show

*Building a RAG offer retrieval system with pgvector and Weaviate — LoyaltyLens Module 3*

---


---

Every RAG tutorial on the internet follows the same pattern: load some PDFs, chunk them, embed with OpenAI, store in Chroma, query with LangChain, marvel at the results. It's a fine starting point. It tells you almost nothing about what you need to know before you put RAG in production.

In production, the offer intelligence system I architected has to retrieve the right offer for the right customer from a catalog of hundreds of options, in real time, with sub-200ms end-to-end latency. The vector retrieval step has a budget of roughly 30ms. That constraint changes every design decision.

For LoyaltyLens Module 3, I built a dual retrieval system — LangChain and LlamaIndex running in parallel against pgvector and Weaviate — and benchmarked them honestly. The results surprised me enough that I changed the default configuration of the production system I was running.

Here's what I found.

---

## Why RAG for Offer Retrieval?

Naively, offer retrieval looks like a filtering problem: given a customer propensity score and a catalog of offers, filter by `min_propensity_threshold <= customer_score` and return the top offer sorted by discount percentage. You don't need a vector database for that.

The problem is that filtering ignores *semantic fit*. A customer who primarily purchases cold drinks in summer afternoons should not receive the same offer as a customer who buys hot drinks on winter mornings, even if both have propensity scores of 0.65 and neither has redeemed in 30 days. The propensity score tells you *likelihood to redeem*; semantic retrieval tells you *which offer* maximizes that likelihood.

This is exactly the distinction between propensity scoring and offer intelligence that a production loyalty AI platform handles. The propensity model produces a probability; the retrieval system turns that probability into a specific, contextually relevant offer.

---

## The Offer Catalog

I generated 200 synthetic offers with realistic structure:

```json
{
  "id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "title": "Double Points on Your Favourite Category",
  "description": "Earn 2x bonus points on every purchase in your top category this week. Available on all items. Redeemable in-store and on the app. A reward for your loyalty.",
  "category": "beverage",
  "channel": "mobile",
  "min_propensity_threshold": 0.35,
  "discount_pct": 0,
  "expiry_days": 7
}
```

The `description` field is what gets embedded. It's deliberately written to carry semantic signal: category, channel preference, time-of-day context, and occasion. This is a practice I'd recommend for any production offer catalog — the quality of your retrieval is bounded by the quality of your offer descriptions.

---

## Embedding Pipeline

I used `sentence-transformers/all-MiniLM-L6-v2` — a 22M parameter model that produces 384-dimensional embeddings. It's not the most powerful embedding model available, but it runs in under 50ms per batch on a standard CPU, which matters for a latency-constrained system.

```python
# rag_retrieval/embeddings.py
from sentence_transformers import SentenceTransformer
import psycopg2
import weaviate

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class EmbeddingPipeline:
    def embed_offers(self, offers: list[dict]) -> None:
        texts = [o["description"] for o in offers]
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
        # shape: (200, 384)

        self._write_to_pgvector(offers, embeddings)
        self._write_to_weaviate(offers, embeddings)

    def _write_to_pgvector(self, offers, embeddings):
        with self.pg_conn.cursor() as cur:
            for offer, embedding in zip(offers, embeddings):
                cur.execute("""
                    INSERT INTO offer_embeddings
                    (id, title, category, channel, min_propensity_threshold,
                     discount_pct, expiry_days, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                    SET embedding = EXCLUDED.embedding
                """, (
                    offer["id"], offer["title"], offer["category"],
                    offer["channel"], offer["min_propensity_threshold"],
                    offer["discount_pct"], offer["expiry_days"],
                    embedding.tolist()
                ))
```

Weaviate integration requires the v4 Python client and server **≥ 1.27.0** (the repo ships 1.28.2). The client connects over both HTTP (port 8080) and gRPC (port 50051) — both must be reachable:

```python
# weaviate_client.py
import weaviate
from weaviate.connect import ConnectionParams

client = weaviate.connect_to_custom(
    http_host="localhost", http_port=8080, http_secure=False,
    grpc_host="localhost", grpc_port=50051, grpc_secure=False,
)
```

One thing worth noting: I index the pgvector table using `ivfflat`:

```sql
CREATE INDEX ON offer_embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 20);
```

With 200 offers this index doesn't matter — a linear scan is faster. With 50,000+ offers it becomes essential. I build it anyway because the architecture should work at scale.

---

## LangChain Retrieval

LangChain's PGVector integration is the most mature path for PostgreSQL-backed RAG. One important detail: LangChain manages its own schema (`langchain_pg_collection` / `langchain_pg_embedding` tables) — separate from the custom `offer_embeddings` table written by the embedding pipeline. The retriever uses a dedicated `lc_offer_embeddings` collection and auto-indexes from `offers.json` on first init:

```python
# rag_retrieval/langchain_retriever.py
from langchain_community.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings  # not langchain_community
from langchain_core.documents import Document

_COLLECTION = "lc_offer_embeddings"
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class LangChainOfferRetriever:
    def __init__(self, offers_path: Path = _OFFERS_PATH) -> None:
        self._embeddings = HuggingFaceEmbeddings(model_name=_MODEL_NAME)
        self._store = PGVector(
            connection_string=settings.postgres_url,
            embedding_function=self._embeddings,
            collection_name=_COLLECTION,
            pre_delete_collection=False,
        )
        if self._collection_empty():
            self._index_offers(offers_path)  # auto-index on first run

    def _index_offers(self, offers_path: Path) -> None:
        offers = json.loads(offers_path.read_text())
        docs = [
            Document(
                page_content=o["description"],
                metadata={"id": o["id"], "title": o["title"],
                          "category": o["category"],
                          "min_propensity_threshold": o["min_propensity_threshold"]},
            )
            for o in offers
        ]
        PGVector.from_documents(
            documents=docs, embedding=self._embeddings,
            collection_name=_COLLECTION, connection_string=self._conn,
            pre_delete_collection=True,
        )

    def retrieve(self, customer_context: str, propensity: float, k: int = 5):
        candidates = self._store.similarity_search_with_score(
            customer_context, k=k * 4,  # oversample, then filter in Python
        )
        results = []
        for doc, score in candidates:
            if propensity < float(doc.metadata.get("min_propensity_threshold", 0)):
                continue
            results.append(OfferResult(..., score=float(score)))
            if len(results) == k:
                break
        return results
```

Two things worth noting. First, the import: `HuggingFaceEmbeddings` moved to `langchain_huggingface` in recent versions — the `langchain_community` path raises a deprecation warning and will be removed. Second, the propensity filter runs in Python after oversampling (k×4 candidates), not as a database-side metadata filter. This is intentional — LangChain's metadata filter syntax for pgvector is inconsistent across versions, while a Python-side filter is predictable and easy to test.

---

## LlamaIndex Retrieval

LlamaIndex takes a different philosophy — it treats your data as a set of documents and builds a higher-level query engine on top of vector search:

```python
# rag_retrieval/llama_retriever.py
from llama_index.core import VectorStoreIndex, Document
from llama_index.vector_stores.postgres import PGVectorStore

vector_store = PGVectorStore.from_params(
    database=config.DB_NAME,
    host=config.DB_HOST,
    table_name="offer_embeddings_llama",
    embed_dim=384,
)

documents = [
    Document(
        text=offer["description"],
        metadata={
            "id": offer["id"],
            "title": offer["title"],
            "category": offer["category"],
            "min_propensity": offer["min_propensity_threshold"],
        }
    )
    for offer in offers
]

index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)

class LlamaOfferRetriever:
    def retrieve(self, customer_context: str, propensity_score: float, k: int = 5):
        retriever = index.as_retriever(similarity_top_k=k * 3)
        nodes = retriever.retrieve(customer_context)
        # Filter and rank
        filtered = [
            n for n in nodes
            if n.metadata["min_propensity"] <= propensity_score
        ]
        return filtered[:k]
```

LlamaIndex's abstraction is cleaner for building compound query pipelines — if you wanted to add a reranking step, a metadata filter chain, or a hybrid BM25 + dense retrieval path, LlamaIndex's router abstraction handles it more gracefully than LangChain's chain syntax.

---

## The Benchmark Results

I ran 1,000 retrieval queries against both systems with a simulated offer catalog of 200, 2,000, and 20,000 offers (the last two using duplicated and perturbed data).

### Latency (ms, p50 / p95)

| System | 200 offers | 2,000 offers | 20,000 offers |
|---|---|---|---|
| LangChain + pgvector | 8ms / 14ms | 12ms / 22ms | 31ms / 58ms |
| LlamaIndex + pgvector | 11ms / 19ms | 18ms / 31ms | 47ms / 89ms |
| LangChain + Weaviate | 22ms / 38ms | 24ms / 41ms | 28ms / 47ms |
| LlamaIndex + Weaviate | 26ms / 44ms | 27ms / 46ms | 30ms / 52ms |

**Key finding:** pgvector wins at small-to-medium catalog sizes. Weaviate's latency is more stable — it doesn't degrade as catalog size grows — but it starts slower due to network overhead to the separate service.

### Precision@5 (category relevance heuristic)

| System | Precision@5 |
|---|---|
| LangChain + pgvector | 0.71 |
| LlamaIndex + pgvector | 0.73 |
| LangChain + Weaviate | 0.70 |
| LlamaIndex + Weaviate | 0.72 |

Precision@5 differences are within noise margin. The retrieval quality is nearly identical — which makes sense, since all four configurations are using the same embedding model and cosine similarity.

**My recommendation:** For a greenfield system already running Postgres, **pgvector is the right first choice**. It eliminates a service dependency, simplifies your infrastructure, and performs well up to tens of millions of vectors. Switch to a dedicated vector database (Pinecone in production, Weaviate in self-hosted) when you're above ~50M vectors or need advanced filtering capabilities that SQL can't express efficiently.

This aligns with how I advise clients in enterprise consulting: don't add infrastructure complexity until you have a concrete scale requirement that forces it.

---

## The FastAPI Retrieval Endpoint

```python
# rag_retrieval/api.py
class RetrieveRequest(BaseModel):
    customer_id: str
    propensity_score: float
    context_text: str
    retriever: str = "langchain"  # "langchain" | "llamaindex"

class RetrieveResponse(BaseModel):
    offers: list[dict]
    retriever_used: str
    latency_ms: float

@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_offers(req: RetrieveRequest) -> RetrieveResponse:
    start = time.perf_counter()
    retriever = _get_retriever(req.retriever)
    offers = retriever.retrieve(req.context_text, req.propensity_score, k=5)
    latency_ms = (time.perf_counter() - start) * 1000
    return RetrieveResponse(
        offers=[dataclasses.asdict(o) for o in offers],
        retriever_used=req.retriever,
        latency_ms=round(latency_ms, 2),
    )
```

The `retriever` parameter lets the LLMOps pipeline A/B test retriever configurations without a code deploy.

To start the API locally:

```powershell
# Install the project as an editable package first (required for shared imports)
uv pip install -e .

# Windows — use 127.0.0.1 (0.0.0.0 triggers WinError 10013)
python -m uvicorn rag_retrieval.api:app --host 127.0.0.1 --port 8003 --reload --reload-dir rag_retrieval

# Test it
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8003/retrieve `
    -ContentType 'application/json' `
    -Body '{"customer_id":"C001","propensity_score":0.65,"context_text":"frequent coffee buyer"}'
```

---

## What This Looks Like in production Scale

The production offer catalog is not 200 offers. It's thousands of offers spanning seasonal specials, partner promotions, personalized bonus points mechanics, and product pairings — many of which are targeted at specific customer micro-segments.

The architecture I described above scales to that. The specific changes at enterprise scale:

- **Embedding model upgrade:** We use a domain-adapted model fine-tuned on loyalty program product descriptions and offer language. `all-MiniLM-L6-v2` is a general-purpose model; a domain-adapted model improves semantic retrieval by measurably reducing irrelevant retrievals.
- **Hybrid retrieval:** BM25 keyword search and dense vector search are run in parallel; results are fused using reciprocal rank fusion. This handles cases where exact product names matter more than semantic similarity.
- **Real-time context enrichment:** The `context_text` in LoyaltyLens is a manually constructed string. At the loyalty platform it's built dynamically from the customer's last 10 events, current location, time of day, and weather data — all assembled in the BFF layer before the retrieval call.

---

## Next: Generating Personalized Offer Copy with an LLM

The retrieval step gives us the *right* offer. The next step is generating the *right message*. In Module 4 I walk through prompt engineering for offer copywriting, building a versioned prompt registry, and the CLIP-based brand image alignment scorer that mirrors what a generative AI content platform does in a loyalty content supply chain.

**[→ Read Module 4: LLM Offer Copy Generation and Brand Alignment Scoring](#)**

---

*Pushparajan Ramar is an Enterprise Architect Director in enterprise consulting. He leads AI, data, and platform architecture for global enterprise programs. Connect on [LinkedIn](https://linkedin.com/in/pushparajanramar).*
