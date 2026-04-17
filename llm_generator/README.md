# llm_generator

Generates personalised Starbucks retention offers by combining customer propensity scores, retrieved programme context, and a pluggable LLM backend (OpenAI or HuggingFace). Optionally generates a matching brand image via Flux AI (FLUX.1-schnell) on the HuggingFace free Inference API.

## Purpose

Translate data signals (scores, features, retrieved docs) into human-readable, brand-consistent messages: email subject lines, push notification copy, offer descriptions, and agent responses.

## Module Map

| File | Responsibility |
| --- | --- |
| `prompts/system_v1.yaml` | Warm, concise Starbucks copywriter prompt (version 1) |
| `prompts/system_v2.yaml` | Punchier, ultra-concise tone variant (version 2) |
| `backends.py` | `LLMBackend` ABC, `OpenAIBackend`, `HuggingFaceBackend` (Mistral-7B-Instruct) |
| `generator.py` | `OfferCopyGenerator` — prompt render → LLM call → JSON parse; `LLMGenerator` (legacy) |
| `multimodal.py` | `BrandImageGenerator` — FLUX.1-schnell image generation from offer copy text |
| `prompt_builder.py` | Static prompt assembly for `LLMGenerator` |
| `response_parser.py` | JSON extraction and Pydantic validation for `LLMGenerator` |
| `api.py` | FastAPI: `POST /generate` — returns `OfferCopy` JSON + optional Flux AI brand image |

## Inputs

- Customer context: `tier`, `engagement_score`, `channel`
- Offer: `offer_title`, `offer_description`
- Prompt registry: `prompts/system_v{1,2}.yaml`

## Outputs

- `OfferCopy` dataclass: `headline`, `body`, `cta`, `tone`, `model_version`, `prompt_version`, `latency_ms`, `token_count`
- `generated_image_path` — path to PNG saved under `data/brand_images/` (only when `generate_image: true`)
- FastAPI on port `8004`

---

## Local Setup

### 1. Environment

One `.env` at the **repo root** — no module-level env file needed.

```powershell
Copy-Item .env.example .env   # if not already done
```

Required / recommended keys:

```dotenv
OPENAI_API_KEY=sk-...         # required for OpenAIBackend
HF_TOKEN=hf_...               # recommended — higher Flux API rate limits; required for gated HF models
PORT_LLM_GENERATOR=8004       # default
```

---

### 2. Install dependencies

```powershell
uv venv --python 3.11                                    # skip if venv already exists
& C:\Projects\loyaltylens\.venv\Scripts\Activate.ps1    # prompt shows (loyaltylens)
uv sync --dev
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

Generate offer copy only:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8004/generate `
    -ContentType 'application/json' `
    -Body '{"customer_id":"C001","offer_id":"O001","prompt_version":1}'
```

Generate offer copy **and** a Flux AI brand image:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8004/generate `
    -ContentType 'application/json' `
    -Body '{"customer_id":"C001","offer_id":"O001","prompt_version":1,"generate_image":true}'
```

The generated PNG is saved to `data/brand_images/O001_generated.png`; its path is returned in `generated_image_path`.

---

### 5. Run tests

Activate the **repo-root** `.venv` — not `data_pipeline/.venv` or any sub-module venv:

```powershell
# From C:\Projects\loyaltylens
& C:\Projects\loyaltylens\.venv\Scripts\Activate.ps1   # prompt shows (loyaltylens)
python -m pytest tests/test_generator.py -v
```

20 tests cover:

- Prompt rendering — tier, engagement score, offer title present in rendered message; v2 prompt loads
- `OfferCopyGenerator` — JSON parse, all `OfferCopy` fields validated, retry-once on bad JSON, missing-field error
- `BrandImageGenerator` — `generate` returns PIL Image, `generate_to_path` saves PNG and creates parent dirs
- API `/generate` — 200 response, all required fields present, `generated_image_path` null when `generate_image=false`, prompt version forwarded, 500 on backend error

---

### 6. Using the backends directly

```python
from llm_generator.backends import OpenAIBackend, HuggingFaceBackend
from llm_generator.generator import OfferCopyGenerator

# OpenAI (requires OPENAI_API_KEY in .env)
gen = OfferCopyGenerator(OpenAIBackend(), prompt_version=1)

# Local HuggingFace — requires ~14 GB GPU RAM for Mistral-7B
gen = OfferCopyGenerator(HuggingFaceBackend(), prompt_version=2)

copy = gen.generate(
    customer_context={"tier": "gold", "engagement_score": 0.82, "channel": "push"},
    offer={"offer_title": "Double Star Day", "offer_description": "Earn 2x stars."},
)
print(copy.headline, copy.cta)
```

---

### 7. Flux AI brand image generation

`BrandImageGenerator` uses the HuggingFace free Inference API with `FLUX.1-schnell` — no GPU required. Add `HF_TOKEN` to `.env` for higher rate limits.

```python
from llm_generator.multimodal import BrandImageGenerator

gen = BrandImageGenerator()

# Returns a PIL Image
image = gen.generate("Earn 2× Stars Today. Treat yourself this weekend.")

# Generate and save to disk
gen.generate_to_path(
    "Double Star Day — 2x points on every order.",
    "data/brand_images/O001_generated.png",
)
```

To swap in a fine-tuned brand model, change the `_FLUX_MODEL` constant in `multimodal.py` — the interface is unchanged.

---

## Key Classes

| Class | Module | Responsibility |
| --- | --- | --- |
| `LLMBackend` | `backends.py` | Abstract interface: `model_name`, `generate(messages)` |
| `OpenAIBackend` | `backends.py` | OpenAI chat completions; graceful error when key missing |
| `HuggingFaceBackend` | `backends.py` | Local Mistral-7B-Instruct via `transformers.pipeline` |
| `OfferCopy` | `generator.py` | Dataclass: headline, body, cta, tone, latency_ms, token_count |
| `OfferCopyGenerator` | `generator.py` | Prompt render → LLM call → JSON parse with retry |
| `BrandImageGenerator` | `multimodal.py` | FLUX.1-schnell image generation via HuggingFace Inference API |
| `LLMGenerator` | `generator.py` | Legacy orchestrator (OpenAI only, `OfferResponse` output) |
| `PromptBuilder` | `prompt_builder.py` | Static prompt assembly for `LLMGenerator` |
| `ResponseParser` | `response_parser.py` | JSON extraction + Pydantic validation for `LLMGenerator` |

## Port Reference

| Service | Port |
| --- | --- |
| LLM Generator API | 8004 |
