# LoyaltyLens — LinkedIn Promotion Posts

*One post per article. Formatted for LinkedIn: short paragraphs, no markdown headers, mix of hook + insight + link. Recommended cadence: one post per week.*

---

## Post 0 — Setup & Glossary
*Publish: Week 1*

I've been working on production AI systems for few years. The number one friction point for every new team member — regardless of experience — isn't the code. It's the acronyms.

RAG. PSI. RLHF. BFF. LLMOps. TabTransformer. pgvector.

So before releasing a seven-part series on building a production-grade offer intelligence platform from scratch, I wrote the article I wish existed: a complete setup guide + plain-English glossary for every term in the series.

37 terms. 6 setup steps. One mental model that makes all the modules click.

It's Article 0 of the LoyaltyLens series — and it's free on my blog.

👉 [link]

If you're new to ML engineering or ramping up on LLMOps, this is the best place to start.

---

## Post 1 — Feature Pipeline
*Publish: Week 2*

Most ML tutorials skip the feature pipeline entirely. They start with a clean CSV and a model.

In production, the feature pipeline *is* the model. It's where data quality problems hide. It's where training-serving skew starts. And it's the layer that silently degrades your propensity scores for weeks before anyone notices.

I rebuilt a production-style loyalty feature pipeline using Python and DuckDB — no Databricks cluster, no $40K/month cloud bill. Here's what I built, why DuckDB outperformed SQLite at 5M rows, and the two things that surprised me.

Surprise #1: DuckDB is genuinely fast enough for batch ML serving.
Surprise #2: The validation layer catches more real problems than the feature engineering does.

Full walkthrough in Article 1 of the LoyaltyLens series.

👉 [link]

---

## Post 2 — Propensity Model
*Publish: Week 3*

"Why didn't you just use XGBoost?"

It's the first question I get whenever I mention using a transformer for tabular propensity scoring. My answer has three parts — and only the third one is about model performance.

The short version: XGBoost is a dead end for multimodal extension. A transformer backbone isn't.

In Article 2 of LoyaltyLens, I walk through:
→ The full TabTransformer-lite architecture in PyTorch (with code)
→ How to construct binary labels from redemption rate data
→ MLflow experiment tracking for the training loop
→ The model card format — including the bias consideration I actually monitor in production
→ The ONNX export path for SageMaker deployment

Val AUC: 0.81 on synthetic data. Production equivalent: 0.85–0.88 with richer features.

👉 [link]

---

## Post 3 — RAG Retrieval
*Publish: Week 4*

I ran a benchmark nobody seems to publish cleanly: LangChain vs. LlamaIndex, pgvector vs. Weaviate, at 200 / 2,000 / 20,000 vectors.

The results were clear enough that I changed the default architecture recommendation I give to clients.

Key findings:
• pgvector beats Weaviate on latency below ~20K vectors (31ms vs. 47ms p50)
• Weaviate's latency is more stable as catalog size grows
• LangChain and LlamaIndex deliver near-identical precision@5 (0.71 vs. 0.73)
• The oversampling + re-ranking pattern matters more than the framework choice

The practical recommendation: pgvector first. Dedicated vector DB when you have a concrete scale requirement. Don't add infrastructure complexity speculatively.

Full benchmark and code in Article 3 of LoyaltyLens.

👉 [link]

---

## Post 4 — LLM Generator
*Publish: Week 5*

Prompt engineering is software engineering.

That's not a metaphor. It means: version your prompts. Diff them. Review changes before they ship. Roll back when quality drops.

In a production offer generation system running at scale, 2% non-JSON LLM output is tens of thousands of failed renders per day. A prompt change that subtly shifts brand voice is invisible until it reaches a customer.

Article 4 of LoyaltyLens covers:
→ A YAML-based versioned prompt registry (machine-readable eval criteria embedded in the prompt file)
→ Two LLM backends: OpenAI API and local Mistral-7B via HuggingFace
→ JSON parse retries that make the generator production-reliable
→ A CLIP-based brand image alignment scorer — a multimodal check that flags off-brand copy before it hits the review queue

The CLIP scorer is the part I get the most questions about. It's simpler than it sounds.

👉 [link]

---

## Post 5 — LLMOps Pipeline
*Publish: Week 6*

Here's a story about a PSI of 0.31 on a Monday morning.

PSI — Population Stability Index — is a metric from credit risk modeling that almost nobody in the ML community uses outside that domain. It measures how much the input distribution to your model has shifted from its training baseline.

A weekend batch job recomputed recency features using a different timezone offset. Every customer's `recency_days` shifted by one. The propensity model's input distribution shifted. The scores were wrong. The campaign sends that morning were misconfigured.

We caught it in 20 minutes. Without the drift monitor, we'd have caught it in three weeks when campaign analysts noticed the redemption rate drop.

Article 5 of LoyaltyLens covers the full LLMOps stack:
→ Prompt versioning CLI (list, diff, rollback)
→ Automated eval harness with a hard CI/CD quality gate
→ PSI-based drift monitor
→ GitHub Actions ML pipeline
→ Responsible AI audit framework

👉 [link]

---

## Post 6 — RLHF Feedback Loop
*Publish: Week 7*

RLHF doesn't require a research team.

The version that runs in production at scale isn't academic reinforcement learning with proprietary reward models. It's this:

1. Collect human feedback signals (ratings, thumbs up/down, redemption outcomes)
2. Aggregate them by prompt version and model version
3. Detect when rolling quality drops below threshold
4. Trigger retraining automatically

Article 6 of LoyaltyLens builds this end-to-end: a React feedback UI, SQLite persistence, a nightly aggregator, a preference dataset exporter in OpenAI fine-tuning format, and a GitHub Actions retraining trigger.

The most important design decision: export preference data in fine-tuning format from day one, even before you're ready to fine-tune. You can't retroactively generate training signal. Collect it before you need it.

👉 [link]

---

## Post 7 — Series Recap
*Publish: Week 8*

Eight weeks. Seven articles. Six modules. One complete production-pattern AI system built in the open.

Here's what I learned building LoyaltyLens — the honest version:

The plumbing is harder than the model. Feature validation, feature versioning, drift monitoring, feedback collection — these take longer than model architecture and matter more for production reliability.

Versioning everything is not optional. You can't diff what isn't versioned. You can't roll back what has no history. You can't catch drift without a baseline.

The eval gate is a forcing function. Automated quality evaluation doesn't replace human judgment. It makes the standard explicit and enforceable. Those are different things.

PSI is the most underused metric in production ML. If you're not computing Population Stability Index on your model inputs, you're flying blind on data drift.

Collect feedback before you know how to use it. The preference dataset compounds. Start collecting it before you have a use case.

Full recap, honest lessons, and what to build next: Article 7 of LoyaltyLens.

👉 [link]

---

## Series Launch Post (Optional — publish before Week 1)
*Use this to announce the series before releasing Article 0*

I'm publishing a seven-part series on building a production-grade offer intelligence platform from scratch.

Not a tutorial. A real system — with a feature store, a propensity model, RAG retrieval, LLM copy generation, drift monitoring, eval harness, CI/CD pipeline, and an RLHF feedback loop.

Every component is anchored to real production patterns. Every architectural decision is explained. Every benchmark is real.

The stack: Python, PyTorch, DuckDB, LangChain, LlamaIndex, pgvector, Weaviate, HuggingFace, CLIP, FastAPI, MLflow, GitHub Actions, Streamlit.

The tool: Claude Code for agentic multi-file code generation.

Article 0 (setup guide + complete glossary) drops this week.

Follow to get the full series as it releases.

👉 [link to Article 0 when live]
