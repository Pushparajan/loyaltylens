---
title: "Prompt Engineering Is Software Engineering: Versioning, Testing, and Multimodal Extension"
slug: "loyaltylens-llm-generator"
description: "Building the LLM offer copy generator — versioned YAML prompt registry, dual LLM backends, JSON parse retry logic, and a CLIP brand alignment scorer."
date: 2026-05-19
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
  - clip
---

# Prompt Engineering Is Software Engineering: Versioning, Testing, and Multimodal Extension

*Building the LLM offer copy generator and CLIP brand-image scorer — LoyaltyLens Module 4*

---


---

In 2023 I helped architect a global loyalty program's integration of a generative AI content platform into their content supply chain. The goal was to automate the generation of brand-consistent visual assets — the imagery that accompanies offer push notifications, email campaigns, and in-app banners. The platform is remarkable technology, but the hardest part of the project had nothing to do with generative AI.

The hardest part was treating prompts as production artifacts.

When you're generating content for a globally recognized loyalty brand, the difference between a prompt that produces on-brand output and one that produces subtly off-brand output is not obvious until it's in front of a customer. Catching that difference at the prompt level — before it reaches a creative review, before it ships — requires the same discipline as catching a software regression before it reaches production.

That discipline is what Module 4 of LoyaltyLens is about.

---

## The Three Layers of Offer Copy Generation

The end-to-end offer copy pipeline has three layers:

1. **Retrieval** (Module 3): *Which* offer is most relevant for this customer?
2. **Copy generation** (this module): *What message* best presents that offer to this customer?
3. **Brand alignment** (this module): Does the generated copy *look and feel* like an on-brand loyalty message?

Most tutorials cover the middle layer only. The brand alignment layer — the CLIP-based image-text relevance scorer — is the generative image-text alignment scorer that makes this project meaningfully different from a generic LLM demo.

---

## The Prompt Registry

Before writing a single inference call, I built the prompt registry. Every prompt lives in a YAML file under version control:

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

That last constraint — "never use the word 'deal'" — came from a real brand feedback session in production. The word tests poorly with the loyalty customer base; it signals transactional intent rather than community membership. Capturing that as a prompt constraint, versioned and auditable, is exactly the kind of governance that makes generative AI deployable at enterprise scale.

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

The retry logic is not optional. In my testing, even well-prompted frontier models produce non-JSON output in roughly 2–4% of calls. At the scale of a million daily offer generations, 2% is 20,000 failed renders — a visible customer experience degradation.

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

The abstract base class is what makes this testable. In unit tests, I inject a `MockBackend` that returns predetermined JSON — no API calls, no GPU, deterministic test results. This is a practice that I've had to push teams on repeatedly: if your ML component can't be tested without a live model call, your CI pipeline will be slow, flaky, and expensive.

---

## The CLIP Brand Alignment Scorer

This is the component that most directly mirrors what we built in production with a generative AI content platform.

A generative AI content platform produces brand-consistent visual assets by conditioning on brand-specific training data. The alignment check — does this generated image actually match the brand? — is a multimodal similarity computation between the image and a text description of what an on-brand visual should look like.

In LoyaltyLens I implement a simplified version: CLIP computes the cosine similarity between the embedding of a generated offer copy text and the embedding of a brand reference image:

```python
# llm_generator/multimodal.py
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class BrandImageScorer:
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model.eval()

    def score_relevance(self, image_path: str, copy_text: str) -> float:
        """
        Returns cosine similarity between image embedding and text embedding.
        Higher = more aligned. Range: approximately [-1, 1], practically [0, 1].
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            text=[copy_text],
            images=image,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        img_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        similarity = (img_emb * txt_emb).sum(dim=-1).item()

        # Scale to [0, 1] for interpretability
        return (similarity + 1) / 2

    def batch_score(
        self, image_paths: list[str], copy_texts: list[str]
    ) -> list[float]:
        scores = []
        for img_path, text in zip(image_paths, copy_texts):
            scores.append(self.score_relevance(img_path, text))
        return scores
```

In practice, I load four brand reference images representing different loyalty brand visual contexts (in-store environment, product close-ups, seasonal decoration, app interface). The scorer returns a vector of four similarity scores — and a generated copy is flagged for human review if *any* score falls below 0.35.

```python
# Usage in the generate endpoint
brand_scores = scorer.batch_score(
    image_paths=config.BRAND_REFERENCE_IMAGES,
    copy_texts=[offer_copy.body],
)
min_brand_score = min(brand_scores)
brand_aligned = min_brand_score >= config.BRAND_ALIGNMENT_THRESHOLD  # 0.35
```

In production, this kind of automated brand alignment check runs before content enters the campaign execution system. It doesn't replace creative review — it prevents obviously off-brand content from ever reaching a reviewer's queue.

---

## A Sample Output

Here's what the full generation pipeline produces for a high-engagement mobile-first customer with a pending double-points offer:

```json
{
  "headline": "Your reward moment is here",
  "body": "You've earned this: 2x points on every purchase in your top category this week. A small thank-you for showing up consistently.",
  "cta": "Redeem in-app now",
  "tone": "friendly",
  "model_version": "gpt-4o-mini",
  "prompt_version": 2,
  "latency_ms": 847.3,
  "brand_image_score": 0.71
}
```

The phrase "you've been showing up for it" is interesting — it references the customer's engagement score implicitly, creating personalization without exposing the underlying metric. That's the kind of nuance that prompt v2 produces reliably, and prompt v1 produced occasionally.

---

## What the Latency Numbers Look Like

| Backend | Median latency | p95 latency | Token cost |
|---|---|---|---|
| OpenAI gpt-4o-mini | 847ms | 1,240ms | ~$0.0002/call |
| Mistral-7B local (GPU) | 1,100ms | 1,850ms | $0 (infra cost) |
| Mistral-7B local (CPU) | 4,200ms | 6,100ms | $0 (infra cost) |

For a real-time offer push use case at scale, these latencies are fine — offer generation happens asynchronously, not in the request path. For a live in-store interaction where you need a rendered offer in under 2 seconds, you'd pre-generate offers for high-propensity customers and cache them.

---

## Next: The LLMOps Pipeline

Module 4 generates offer copy. Module 5 asks: is the copy getting better or worse over time? In the next post I walk through building a prompt versioning CLI, an LLM-as-judge evaluation harness, a propensity drift monitor with Population Stability Index, and the GitHub Actions pipeline that ties all of it together with a hard quality gate.

**[→ Read Module 5: Building an LLMOps Pipeline with Prompt Versioning, Drift Monitoring, and CI/CD](#)**

---

*Pushparajan Ramar is an Enterprise Architect Director in enterprise consulting, where he leads AI and data platform strategy for Fortune 500 clients. Connect on [LinkedIn](https://linkedin.com/in/pushparajanramar).*
