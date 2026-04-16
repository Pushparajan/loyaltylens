# rag_retrieval

Indexes loyalty programme documents (T&Cs, offer catalogues, FAQ) in Weaviate and retrieves semantically relevant context chunks for the LLM generator.

## Purpose

Ground LLM responses in factual, up-to-date loyalty programme content by providing a vector-search retrieval layer. Reduces hallucinations and ensures offer recommendations reference real programme rules.

## Inputs

- Documents to index: plain text / PDF / HTML supplied as `Document` objects
- Query strings from `llm_generator` (customer context + intent)
- Embedding model configuration from `shared.Settings`

## Outputs

- Indexed vector embeddings stored in Weaviate `LoyaltyDoc` collection
- Retrieved context chunks (`list[Document]`) returned to `llm_generator`

## Key Classes

| Class             | Module              | Responsibility |
| ----------------- | ------------------- | -------------------------------------------- |
| `WeaviateClient`  | `weaviate_client.py`| Manage Weaviate connection and schema setup  |
| `DocumentIndexer` | `indexer.py`        | Chunk, embed, and upsert documents           |
| `VectorRetriever` | `retriever.py`      | Semantic search and MMR-based re-ranking     |
