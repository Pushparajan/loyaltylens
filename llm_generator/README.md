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

| Class | Module | Responsibility |
| --- | --- | --- |
| `LLMGenerator` | `generator.py` | Orchestrate prompt → LLM call → parse cycle |
| `PromptBuilder` | `prompt_builder.py` | Assemble system + user prompt from context |
| `ResponseParser` | `response_parser.py` | Validate and extract structured output |

---

## Running Locally

### 1. Environment

One `.env` at the **repo root** — no module-level env file needed.

```powershell
Copy-Item .env.example .env   # if not already done
```

Required keys for this module:

```dotenv
OPENAI_API_KEY=sk-...         # required for OpenAI backend
# or leave blank to use the HuggingFace backend (needs 8 GB RAM)
HF_TOKEN=hf_...               # optional — only for gated HuggingFace models
PORT_LLM_GENERATOR=8004       # default
```

---

### 2. Install dependencies

From the repo root with the root `.venv` active:

```powershell
uv sync --dev
uv pip install -e .
```

---

### 3. Prerequisites

Ensure upstream services are running:

```powershell
docker compose up -d postgres weaviate redis
python seed_feature_store.py              # feature store populated
python rag_retrieval/generate_offers.py   # offers.json generated
python rag_retrieval/embeddings.py        # offers indexed in pgvector + Weaviate
```

---

### 4. Start the API

```powershell
# Windows
python -m uvicorn llm_generator.api:app --host 127.0.0.1 --port 8004 --reload

# macOS / Linux
python -m uvicorn llm_generator.api:app --host 0.0.0.0 --port 8004 --reload
```

---

### 5. Run tests

```bash
python -m pytest tests/test_llm_generator.py -v
```
