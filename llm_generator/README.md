# llm_generator

Generates personalised retention offers and communications by combining customer propensity scores, retrieved programme context, and a Claude / OpenAI LLM.

## Purpose

Translate data signals (scores, features, retrieved docs) into human-readable, brand-consistent messages: email subject lines, push notification copy, offer descriptions, and agent responses.

## Inputs

- Customer propensity scores and feature summary from `feature_store`
- RAG context chunks from `rag_retrieval.VectorRetriever`
- Prompt templates and LLM configuration from `shared.Settings`

## Outputs

- Structured `GeneratedOffer` objects (headline, body, CTA, offer code)
- Raw LLM completions logged to `llmops.LLMOpsTracker` for evaluation
- Generated content persisted to `generated_offers` Postgres table

## Key Classes

| Class            | Module             | Responsibility |
| ---------------- | ------------------ | ------------------------------------------- |
| `LLMGenerator`   | `generator.py`     | Orchestrate prompt → LLM call → parse cycle |
| `PromptBuilder`  | `prompt_builder.py`| Assemble system + user prompt from context  |
| `ResponseParser` | `response_parser.py`| Validate and extract structured output     |
