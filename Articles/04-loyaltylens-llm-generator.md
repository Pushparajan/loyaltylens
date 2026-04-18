---
title: "LLM Offer Copy Generation: Prompt Registry, Dual Backends, and Brand Image Generation"
slug: "loyaltylens-llm-generator"
description: "Building the LLM offer copy generator — versioned YAML prompt registry, dual LLM backends, JSON parse retry logic, and FLUX.1 brand image generation via HuggingFace."
date: 2026-05-07
author: Pushparajan Ramar
series: loyaltylens
series_order: 4
reading_time: 13
tags:
  - llms
  - prompt-engineering
  - multimodal
  - generative-ai
  - huggingface
  - flux-ai
  - image-generation
---

# LLM Offer Copy Generation: Prompt Registry, Dual Backends, and Brand Image Generation

*Versioned YAML prompts, OpenAI and HuggingFace backends, JSON retry logic, and FLUX.1 brand image generation — LoyaltyLens Module 4*

---

**Series position:** Article 4 of 8

---

Module 4 takes the offer retrieved in Module 3 and generates personalized copy for it. The module has three layers:

1. **Copy generation** — a versioned YAML prompt registry, two LLM backends (OpenAI and HuggingFace/Mistral-7B), and a structured `OfferCopy` generator with JSON parse retry logic
2. **Brand alignment** — FLUX.1-schnell image generation from the offer copy via HuggingFace Inference API
3. **Prompt governance** — machine-readable `eval_criteria` embedded in each prompt YAML, consumed by the eval harness in Module 5

Treating prompts as production artifacts — versioned, testable, and auditable — is the central design principle of this module.

---

## The Three Layers of Offer Copy Generation

The end-to-end offer copy pipeline has three layers:

1. **Retrieval** (Module 3): *Which* offer is most relevant for this customer?
2. **Copy generation** (this module): *What message* best presents that offer to this customer?
3. **Brand alignment** (this module): Does the generated copy *look and feel* like an on-brand loyalty message?

Most tutorials cover the middle layer only. The brand alignment layer — Flux AI image generation from the offer copy itself — is what makes this project meaningfully different from a generic LLM demo.

---

## The Prompt Registry

Every prompt lives in a YAML file under version control. The registry is built before any inference code:

```yaml
# llm_generator/prompts/system_v1.yaml
version: 1
created_at: "2024-01-15"
author: "Pushparajan Ramar"

system: |
  You are a loyalty program offer copywriter with deep knowledge of
  the brand voice: warm, personal, community-focused, and
  never pushy. You write copy that feels like a message from a friend,
  not an advertisement.

  Always output valid JSON and nothing else. No preamble, no explanation.

user_template: |
  Write personalized offer copy for this customer:
  - Loyalty tier: {tier}
  - Engagement score: {engagement_score:.2f} (0=disengaged, 1=highly engaged)
  - Preferred channel: {channel}
  - Days since last visit: {recency_days}

  Offer to promote:
  - Title: {offer_title}
  - Description: {offer_description}
  - Category: {offer_category}

  Output JSON with exactly these fields:
  {{
    "headline": "max 8 words, creates urgency or delight",
    "body": "max 40 words, warm and personal, mention the customer benefit",
    "cta": "max 5 words, action-oriented",
    "tone": "friendly | urgent | exclusive"
  }}

eval_criteria:
  - headline_max_words: 8
  - body_max_words: 40
  - cta_max_words: 5
  - required_fields: [headline, body, cta, tone]
  - valid_tones: [friendly, urgent, exclusive]
```

The `eval_criteria` block is not documentation — it's machine-readable. The evaluation harness in Module 5 reads it to validate every generated output automatically.

Version 2 tightened the tone:

```yaml
# llm_generator/prompts/system_v2.yaml
version: 2
# ...changes from v1:
# - Added instruction to reference specific product category in body
# - Changed tone options to [warm | celebratory | exclusive]
# - Added constraint: never use the word "deal"
```

That last constraint — "never use the word 'deal'" — is a brand guideline: the word signals transactional intent rather than community membership. Capturing brand constraints as versioned, auditable prompt rules is what makes generative AI deployable at enterprise scale.

---

## The Generator Implementation

```python
# llm_generator/generator.py
import json
import yaml
from dataclasses import dataclass
from pathlib import Path

@dataclass
class OfferCopy:
    headline: str
    body: str
    cta: str
    tone: str
    model_version: str
    prompt_version: int
    latency_ms: float
    token_count: int | None = None

class OfferCopyGenerator:
    def __init__(self, backend: LLMBackend, prompt_version: int = 1):
        prompt_path = Path(f"llm_generator/prompts/system_v{prompt_version}.yaml")
        self.prompt_cfg = yaml.safe_load(prompt_path.read_text())
        self.backend = backend
        self.prompt_version = prompt_version

    def generate(self, customer_context: dict, offer: dict) -> OfferCopy:
        user_message = self.prompt_cfg["user_template"].format(
            tier=customer_context.get("tier", "Member"),
            engagement_score=customer_context["engagement_score"],
            channel=customer_context["channel_preference"],
            recency_days=int(customer_context["recency_days"]),
            offer_title=offer["title"],
            offer_description=offer["description"],
            offer_category=offer["category"],
        )

        start = time.perf_counter()
        raw = self.backend.generate(
            system=self.prompt_cfg["system"],
            user=user_message,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        copy = self._parse_with_retry(raw, max_retries=1)
        self._validate(copy)

        return OfferCopy(
            **copy,
            model_version=self.backend.version,
            prompt_version=self.prompt_version,
            latency_ms=round(latency_ms, 2),
        )

    def _parse_with_retry(self, raw: str, max_retries: int) -> dict:
        for attempt in range(max_retries + 1):
            try:
                # Strip markdown code fences if present
                clean = raw.strip().removeprefix("```json").removesuffix("```").strip()
                return json.loads(clean)
            except json.JSONDecodeError:
                if attempt == max_retries:
                    raise CopyGenerationError(f"JSON parse failed after {max_retries+1} attempts")
                # Retry with explicit JSON instruction
                raw = self.backend.generate(
                    system=self.prompt_cfg["system"],
                    user=f"Your previous response was not valid JSON. Output ONLY the JSON object, no other text.\n\n{raw}"
                )
```

The retry logic is not optional. Even well-prompted frontier models produce non-JSON output in roughly 2–4% of calls. At a million daily offer generations, 2% is 20,000 failed renders — a visible degradation in customer experience.

---

## The Two LLM Backends

LoyaltyLens supports two backends with a common interface:

```python
# llm_generator/backends.py
from abc import ABC, abstractmethod

class LLMBackend(ABC):
    @abstractmethod
    def generate(self, system: str, user: str) -> str: ...

class OpenAIBackend(LLMBackend):
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        self.version = model

    def generate(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.version,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.7,
            max_tokens=300,
        )
        return response.choices[0].message.content

class HuggingFaceBackend(LLMBackend):
    def __init__(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        from transformers import pipeline
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            max_new_tokens=300,
            temperature=0.7,
            device_map="auto",
        )
        self.version = model_id.split("/")[-1]

    def generate(self, system: str, user: str) -> str:
        prompt = f"[INST] {system}\n\n{user} [/INST]"
        output = self.pipe(prompt)[0]["generated_text"]
        # Strip the input prompt from the output
        return output[len(prompt):].strip()
```

The abstract base class is what makes this testable. In unit tests, inject a `MockBackend` that returns predetermined JSON — no API calls, no GPU, deterministic results. If an ML component can't be tested without a live model call, the CI pipeline will be slow, flaky, and expensive.

---

## The FLUX.1 Brand Image Generator

Once the LLM produces offer copy, the same copy serves as a text prompt to generate a matching campaign image. The `/generate` endpoint accepts an optional `generate_image: true` flag; when set, it feeds the headline and body into FLUX.1-schnell via the HuggingFace Inference API and saves the result alongside the copy.

```python
# llm_generator/multimodal.py
from huggingface_hub import InferenceClient
from PIL import Image
from pathlib import Path

class BrandImageGenerator:
    """Generate brand images from offer copy using FLUX.1-schnell (free HF Inference API)."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = InferenceClient(
            model="black-forest-labs/FLUX.1-schnell",
            token=settings.hf_token or None,  # optional — higher rate limits with token
        )

    def generate(self, copy_text: str) -> Image.Image:
        return self._client.text_to_image(copy_text)

    def generate_to_path(self, copy_text: str, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.generate(copy_text).save(path)
        return path
```

FLUX.1-schnell is four-step diffusion with no classifier-free guidance — under two seconds on the HuggingFace serverless GPU tier. The interface is designed to swap in a fine-tuned brand model in production without changing calling code.

When `generate_image=true`, the endpoint concatenates headline and body into a single prompt and calls the generator:

```python
# llm_generator/api.py — relevant excerpt
@app.post("/generate")
async def generate(req: GenerateRequest) -> GenerateResponse:
    copy = _get_generator(req.prompt_version).generate(customer_context, offer)
    image_path = None
    if req.generate_image:
        gen = BrandImageGenerator()
        prompt = f"{copy.headline}. {copy.body}"
        image_path = str(gen.generate_to_path(prompt, BRAND_IMAGES_DIR / f"{req.offer_id}_generated.png"))
    return GenerateResponse(..., generated_image_path=image_path)
```

In production you would replace the free Inference API with a fine-tuned Flux variant conditioned on brand-approved visual training data. The interface stays identical — swap the model ID and token. That's the point of the abstraction.

---

## Sample Output

Full generation pipeline output for a high-engagement mobile-first customer with a double-points offer:

```json
{
  "headline": "Your reward moment is here",
  "body": "You've earned this: 2x points on every purchase in your top category this week. A small thank-you for showing up consistently.",
  "cta": "Redeem in-app now",
  "tone": "friendly",
  "model_version": "gpt-4o-mini",
  "prompt_version": 2,
  "latency_ms": 847.3,
  "generated_image_path": "data/brand_images/O042_generated.png"
}
```

The body references the customer's engagement score implicitly ("you've been showing up for it") — personalization without exposing the underlying metric. This is what separates prompt v2 from v1: consistent nuance, not occasional nuance.

---

## What the Latency Numbers Look Like

| Backend | Median latency | p95 latency | Token cost |
|---|---|---|---|
| OpenAI gpt-4o-mini | 847ms | 1,240ms | ~$0.0002/call |
| Mistral-7B local (GPU) | 1,100ms | 1,850ms | $0 (infra cost) |
| Mistral-7B local (CPU) | 4,200ms | 6,100ms | $0 (infra cost) |

For a real-time offer push use case at scale, these latencies are fine — offer generation happens asynchronously, not in the request path. For a live in-store interaction where you need a rendered offer in under 2 seconds, you'd pre-generate offers for high-propensity customers and cache them.

---

## Next: Module 5 — LLMOps Pipeline

Module 4 generates copy. Module 5 monitors whether it's getting better or worse over time: a prompt versioning CLI, an LLM-as-judge evaluation harness with a hard quality gate, a PSI-based propensity drift monitor, and a GitHub Actions CI/CD pipeline that ties all of it together.

**[→ Read Module 5: Building an LLMOps Pipeline with Prompt Versioning, Drift Monitoring, and CI/CD](#)**

---

*Pushparajan Ramar — [LinkedIn](https://linkedin.com/in/pushparajanramar) · [GitHub](https://github.com/Pushparajan/loyaltylens)*
