---
title: "LoyaltyLens: What We Built, What We Learned, and What to Build Next"
slug: "loyaltylens-recap-next-steps"
description: "A complete series recap — module-by-module decisions, five honest lessons from building a production AI system, and a prioritised roadmap for what to build next."
date: 2025-10-20
author: Pushparajan Ramar
series: loyaltylens
series_order: 7
reading_time: 10
tags:
  - machine-learning
  - llmops
  - mlops
  - rag
  - pytorch
  - production-ai
  - loyaltylens
---

# LoyaltyLens: What We Built, What We Learned, and What to Build Next

*A complete recap of the seven-part series on production-grade loyalty AI — and the honest lessons from building it*

---

**Series position:** Article 7 of 7 — Recap & next steps

---

Seven articles. Six modules. One end-to-end system.

If you've read this series from the beginning, you've built — or at least read the detailed design of — a production-pattern loyalty offer intelligence platform: from raw event ingestion to feature engineering, from propensity scoring to RAG retrieval, from LLM copy generation to LLMOps pipelines and RLHF feedback loops.

This final post does three things: recaps what was built and why each decision was made, surfaces the honest lessons that only became clear at the end, and maps out what to build next if you want to take LoyaltyLens into real production.

---

## What We Built, Module by Module

### Module 1 — Feature Pipeline & Feature Store

**What:** A synthetic event generator producing 50,000 loyalty customer records, a feature engineering job computing six RFM-derived features, a DuckDB-backed versioned feature store, and a FastAPI serving endpoint.

**The decision that mattered:** Using DuckDB instead of SQLite or Postgres for the feature store. DuckDB queries Parquet files natively, supports versioned tables with minimal setup, and runs in-process — no server to manage. The benchmark showed ~4ms median lookup latency on a 5M-row table, which is within striking distance of Redis for batch workloads.

**What this mirrors in production:** Real-time feature pipelines, behavioral signal computation, feature versioning for model reproducibility, training-serving skew prevention.

---

### Module 2 — Propensity Scoring Model

**What:** A TabTransformer-lite model in PyTorch trained to predict offer redemption probability, with MLflow experiment tracking, early stopping, a model card, a FastAPI inference endpoint, and an ONNX export for SageMaker deployment.

**The decision that mattered:** Choosing a transformer architecture over XGBoost for tabular data. The AUC difference is likely negligible at six features. The architectural choice was about forward compatibility — the transformer backbone extends naturally to multimodal inputs and sequence data, while a gradient-boosted tree is a dead end.

**What this mirrors in production:** Propensity scoring systems, deep learning for tabular data, responsible AI documentation, cloud-native inference deployment.

---

### Module 3 — RAG Offer Intelligence

**What:** 200 synthetic offers embedded with `all-MiniLM-L6-v2`, stored in both pgvector and Weaviate, with dual retrieval pipelines via LangChain and LlamaIndex, a latency and precision@5 benchmark, and a FastAPI retrieval endpoint.

**The decision that mattered:** Building both retrieval paths and benchmarking them, rather than picking one upfront. The benchmark showed pgvector outperforming Weaviate below ~20,000 vectors on latency (31ms vs. 47ms p50 at scale), while Weaviate's latency is more stable as catalog size grows. Neither framework (LangChain vs. LlamaIndex) showed meaningful precision difference. The practical recommendation: pgvector first, dedicated vector DB when you have a concrete scale requirement.

**What this mirrors in production:** Semantic offer retrieval, hybrid retrieval systems, vector DB selection, RAG pipeline design.

---

### Module 4 — LLM Offer Copy Generator

**What:** A versioned YAML prompt registry, two LLM backends (OpenAI and HuggingFace/Mistral-7B), a structured OfferCopy generator with JSON parse retries, a CLIP-based brand image alignment scorer, and a complete generation endpoint.

**The decision that mattered:** Treating the prompt registry as a first-class software artifact. Every prompt lives in a versioned YAML file, commit-tracked, with machine-readable eval criteria embedded alongside the prompt text. This is the foundation that makes the LLMOps pipeline possible — you can't version what isn't versioned.

**What this mirrors in production:** LLM content generation pipelines, prompt governance, multimodal brand alignment, generative AI in marketing workflows.

---

### Module 5 — LLMOps Pipeline

**What:** A prompt versioning CLI (list, diff, activate, rollback), an automated evaluation harness (BLEU + ROUGE + LLM-as-judge with a 0.75 quality gate), a PSI-based propensity drift monitor, a GitHub Actions CI/CD pipeline, a Streamlit dashboard, and a responsible AI audit stub.

**The decision that mattered:** Making the eval gate a hard CI/CD failure. Not a warning. Not a slack notification. A build failure. When quality degrades, the deployment stops. This forces the team to treat prompt changes with the same seriousness as code changes — because the consequence is the same.

**What this mirrors in production:** LLMOps practices, drift detection, responsible AI governance, ML-specific CI/CD, model observability.

---

### Module 6 — RLHF Feedback Loop

**What:** A React feedback UI (thumbs up/down + star rating), a feedback persistence layer in SQLite, a nightly aggregation job, a preference dataset exporter in OpenAI fine-tuning JSONL format, and a retraining trigger that fires a GitHub Actions `workflow_dispatch` when 7-day rolling quality drops below threshold.

**The decision that mattered:** Exporting preference data in fine-tuning format from day one, even though the fine-tuning step wasn't implemented. The preference dataset is an asset that compounds over time. Starting to collect it before you're ready to use it is always the right call — you can't retroactively generate training signal.

**What this mirrors in production:** Human feedback systems, RLHF data collection, automated retraining triggers, closing the production AI loop.

---

## The Five Honest Lessons

### 1. The plumbing is harder than the model

The propensity model took about four hours to design, implement, and get training. The feature store, feature validation, serving endpoint, drift monitor, and feedback loop — the plumbing around the model — took the rest of the project. This ratio holds in production. If your team is spending most of its time on model architecture, something is wrong. The model is rarely the bottleneck.

### 2. Versioning everything is not optional

Prompts, feature definitions, model artifacts, training datasets — all of them need version numbers, creation timestamps, and the ability to diff between versions. The PSI drift story (a timezone offset shifting every customer's `recency_days` by one) wasn't caught because of good model monitoring. It was caught because we had a baseline distribution to compare against. Without versioning, there is no baseline.

### 3. The eval gate is a forcing function for quality standards

Before adding the automated eval harness, prompt changes were reviewed by whoever happened to be available. After adding the eval gate, prompt changes required passing a quantitative quality bar before deployment. The bar didn't change — the accountability did. Automated evaluation doesn't replace human judgment; it makes the standard explicit and enforceable.

### 4. PSI is the most underused metric in production ML

Population Stability Index is a standard tool in credit risk modeling that almost nobody in the ML/AI community uses outside of that domain. It's fast to compute, interpretable (below 0.1 = stable, above 0.2 = retrain), and catches data distribution shifts before they manifest as prediction quality problems. It belongs in every production ML monitoring stack.

### 5. Collect feedback before you know how to use it

The preference dataset exporter in Module 6 writes JSONL in a format ready for fine-tuning. Fine-tuning isn't implemented in LoyaltyLens. That's fine. The point is that the feedback collection infrastructure is in place. When the team is ready to fine-tune — or to train a reward model, or to run a DPO training run — the data is already there. Start collecting before you have a use case. You'll have one eventually.

---

## What to Build Next

If LoyaltyLens is running end-to-end locally, here are the natural extensions in priority order:

### Priority 1 — Deploy to a real cloud endpoint

Export the propensity model to ONNX, wrap it in a SageMaker PyTorch serving container, and deploy to a `ml.t2.medium` real-time endpoint. This is the step that converts the project from a local demo to a portfolio artifact that demonstrates cloud ML deployment. Vertex AI is an equally valid target if you're GCP-certified.

```bash
# The two commands that take LoyaltyLens to cloud inference
python propensity_model/export_onnx.py
python shared/deploy.py --target sagemaker --env staging
```

### Priority 2 — Add a time-based train/val split

Replace the random 70/15/15 split in `train.py` with a temporal split: train on events from months 1–8, validate on month 9, test on month 10. This is the single change that most improves the realism of the model evaluation. Random splits leak future information into training; temporal splits don't.

### Priority 3 — Domain-adapt the embedding model

Fine-tune `all-MiniLM-L6-v2` on your offer descriptions using contrastive learning (positive pairs: offers in the same category that were both redeemed by the same customer; negative pairs: offers that were never co-redeemed). A domain-adapted embedding model reliably outperforms a general-purpose one on domain-specific retrieval — typically 5–12 precision@5 improvement.

### Priority 4 — Implement hybrid retrieval

Add BM25 keyword search alongside the dense vector retrieval in Module 3, and fuse the results using reciprocal rank fusion. This handles the cases where exact product names matter more than semantic similarity — a common pattern in large offer catalogs where customers search for specific reward types.

### Priority 5 — Write the blog post about deploying it

The most underrated portfolio move for any engineer is publishing a specific, honest account of building something — including what didn't work. If you've followed this series and built LoyaltyLens, you have a story to tell. Write it on your own blog, link to the GitHub repo, and share it. That post will do more for your professional presence than the code itself.

---

## The Complete Series

| Article | Topic | Key Concept |
|---|---|---|
| **Article 0** | Setup & Glossary | Environment, acronyms, mental model |
| **Article 1** | Feature Pipeline | DuckDB feature store, RFM features, validation |
| **Article 2** | Propensity Model | PyTorch TabTransformer, MLflow, SageMaker |
| **Article 3** | RAG Retrieval | LangChain vs. LlamaIndex, pgvector vs. Weaviate |
| **Article 4** | LLM Generator | Prompt registry, CLIP brand alignment, multimodal |
| **Article 5** | LLMOps Pipeline | PSI drift, eval gate, CI/CD, responsible AI |
| **Article 6** | RLHF Feedback Loop | Preference data, retraining trigger, closing the loop |
| **Article 7** | Recap & Next Steps | Lessons, priorities, what to build next |

---

## Where to Go From Here

The LoyaltyLens repository is open. The full developer guide (with the Claude Code prompts for every module) is linked in the repo README. If you build on it, improve it, or deploy it — I'd genuinely like to hear about it.

If there are topics in the series you'd like me to go deeper on — the SageMaker deployment path, the domain-adaptation fine-tuning, the DPO fine-tuning loop, or anything else — let me know via LinkedIn or the contact form on this site.

---

*Pushparajan Ramar is an Enterprise Architect  specializing in AI, data, and platform architecture. He writes about production AI systems, ML engineering, and the gap between research and deployment.*

*[LinkedIn](https://linkedin.com/in/pushparajanramar) · [pushparajan.tech](https://pushparajan.tech)*
