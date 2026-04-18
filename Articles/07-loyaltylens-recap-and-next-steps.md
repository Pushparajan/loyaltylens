---
title: "LoyaltyLens Series Recap: Architecture Decisions, Lessons, and What to Build Next"
slug: "loyaltylens-recap-next-steps"
description: "Module-by-module architecture decisions, five key lessons from building a production AI system end-to-end, and a prioritised roadmap for extending LoyaltyLens to production."
date: 2026-05-21
author: Pushparajan Ramar
series: loyaltylens
series_order: 8
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

# LoyaltyLens Series Recap: Architecture Decisions, Lessons, and What to Build Next

*Module-by-module decisions, key lessons, and the production extension roadmap*

---

**Series position:** Article 8 of 8 — Recap & next steps

---

Eight articles. Seven modules. One end-to-end system.

This article does three things: recaps the key architecture decision in each module and why it was made, distills the five lessons that only emerge when the full system runs end-to-end, and maps the prioritised roadmap for extending LoyaltyLens to production scale.

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

### Module 7 — Integration Layer & Cloud Deployment

**What:** A `LoyaltyLensPipeline` class in `shared/pipeline.py` that wires all six modules — feature store, propensity predictor, Weaviate retriever, and LLM generator — into a single `run_for_customer()` call with per-stage latency tracking. A `deploy/` module that exports the PyTorch model to TorchScript, packages it for SageMaker, and deploys a real-time inference endpoint on AWS SageMaker and GCP Vertex AI.

**The decision that mattered:** Using `torch.jit.trace` (not `torch.jit.script`) for the TorchScript export, with `check_trace=False`. The SageMaker PyTorch serving container's default `model_fn` requires TorchScript. Plain state-dict checkpoints fail at container startup with `ModelLoadError`. `TransformerEncoderLayer` takes a different fused-op path across trace invocations — the graph structure differs between two traces even when the values are identical — so the standard sanity check must be disabled. This is the kind of detail that only surfaces when you actually deploy to a real endpoint, not a local container.

**What this mirrors in production:** End-to-end ML pipeline orchestration, cloud-native inference deployment, model serving container constraints, IAM role separation (user credentials vs. execution role).

---

## Five Key Lessons

### 1. The plumbing outweighs the model

The propensity model takes a few hours to implement and train. The feature store, validation layer, serving endpoint, drift monitor, and feedback loop take the rest of the project. This ratio is consistent in production: if the team is spending most of its time on model architecture, the model is not the bottleneck. The operational infrastructure around it is.

### 2. Versioning is a prerequisite, not a feature

Prompts, feature definitions, model artifacts, training datasets — all require version numbers, creation timestamps, and the ability to diff between versions. The PSI drift example (a timezone offset shifting every customer's `recency_days` by one) is only detectable because a baseline distribution exists to compare against. Without versioning, there is no baseline and no detection.

### 3. The eval gate enforces standards that reviews cannot

An automated eval gate converts a qualitative standard ("good copy") into a quantitative threshold (aggregate score ≥ 0.75) that blocks deployment when violated. Human review is inconsistent; a gate is not. Automated evaluation does not replace human judgment — it makes the standard explicit and removes the dependency on whoever happens to be available.

### 4. PSI belongs in every production ML monitoring stack

Population Stability Index is standard practice in credit risk modeling and largely absent from ML/AI monitoring outside that domain. It is fast to compute, directly interpretable (PSI < 0.1 = stable, > 0.2 = retrain), and catches data distribution shifts before they manifest as prediction quality problems. The timezone-offset bug surfaces as PSI = 0.31 at 6am; without the monitor, it surfaces as a redemption rate drop three weeks later.

### 5. Collect preference data before you need it

The preference dataset exporter in Module 6 writes JSONL in the format required for LLM fine-tuning. The fine-tuning step is not implemented in LoyaltyLens — and that is the correct order of operations. The collection infrastructure should be in place before the use case is defined. When the team is ready to fine-tune, run DPO, or train a reward model, the labeled data already exists.

---

## Extension Roadmap

Extensions in priority order for taking LoyaltyLens from reference implementation to production system:

### Priority 1 — ✅ Deploy to a real cloud endpoint

Done. The propensity model is exported to TorchScript and deployed to a `ml.t2.medium` SageMaker real-time endpoint. The live endpoint returns consistent predictions with the local model (`propensity_score: 0.99` for a high-engagement customer). GCP Vertex AI deployment is also implemented in `deploy/vertex_deploy.py`.

```bash
# Deploy (packages TorchScript model, uploads to S3, creates endpoint — ~10 min)
python deploy/sagemaker_deploy.py --action deploy --model-path models/propensity_v1.pt

# Invoke
python deploy/sagemaker_deploy.py --action invoke \
  --payload '{"recency_days": 3, "frequency_30d": 5, "monetary_90d": 120,
              "offer_redemption_rate": 0.4, "channel_preference": "mobile",
              "engagement_score": 0.7}'

# Teardown (always — SageMaker charges for endpoint uptime)
python deploy/sagemaker_deploy.py --action teardown
```

### Priority 2 — Add a time-based train/val split

Replace the random 70/15/15 split in `train.py` with a temporal split: train on events from months 1–8, validate on month 9, test on month 10. This is the single change that most improves the realism of the model evaluation. Random splits leak future information into training; temporal splits don't.

### Priority 3 — Domain-adapt the embedding model

Fine-tune `all-MiniLM-L6-v2` on offer descriptions using contrastive learning: positive pairs are offers co-redeemed by the same customer; negative pairs are offers never co-redeemed. Domain adaptation reliably improves retrieval precision — typically 5–12 precision@5 points on loyalty-specific catalogs.

### Priority 4 — Implement hybrid retrieval

Add BM25 keyword search alongside the dense vector retrieval in Module 3, and fuse the results using reciprocal rank fusion. This handles the cases where exact product names matter more than semantic similarity — a common pattern in large offer catalogs where customers search for specific reward types.

### Priority 5 — Document and share your deployment experience

Building LoyaltyLens and deploying it produces a specific, concrete technical narrative — including what failed and why. Publishing that account, linked to the GitHub repo, creates a more durable professional artifact than the code alone. The production errors (the `ModelLoadError`, the IAM execution role separation, the `check_trace=False` workaround) are exactly the details that make technical writing useful.

---

## The Complete Series

| Article | Topic | Key Concept |
|---|---|---|
| **Article 0** | Setup & Glossary | Environment, acronyms, mental model |
| **Article 1** | Feature Pipeline | DuckDB feature store, RFM features, validation |
| **Article 2** | Propensity Model | PyTorch TabTransformer, MLflow, TorchScript |
| **Article 3** | RAG Retrieval | LangChain vs. LlamaIndex, pgvector vs. Weaviate |
| **Article 4** | LLM Generator | Prompt registry, CLIP brand alignment, multimodal |
| **Article 5** | LLMOps Pipeline | PSI drift, eval gate, CI/CD, responsible AI |
| **Article 6** | RLHF Feedback Loop | Preference data, retraining trigger, closing the loop |
| **Article 7** | Integration & Cloud Deploy | Pipeline orchestration, SageMaker, Vertex AI |
| **Article 8** | Recap & Next Steps | Lessons, priorities, what to build next |

---

## Where to Go From Here

The LoyaltyLens repository is open. The full developer guide with the Claude Code prompts for every module is linked in the repo README.

For deeper coverage of specific topics — the SageMaker deployment path, domain-adaptation fine-tuning, the DPO fine-tuning loop, or hybrid retrieval implementation — reach out via LinkedIn or the contact form.

---

*Pushparajan Ramar — Enterprise Architect Director, AI and data platform architecture.*

*[LinkedIn](https://linkedin.com/in/pushparajanramar) · [GitHub](https://github.com/Pushparajan/loyaltylens)*
