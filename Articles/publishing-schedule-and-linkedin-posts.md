# LoyaltyLens — Publishing Schedule & LinkedIn Posts with Hashtags

---

## Part 1: Publishing Schedule

### The Data-Backed Answer (2026)

Research from Buffer (4.8M posts analysed, March 2026), Hootsuite (1M+ posts, January 2026), and Sprout Social (updated April 2026) all point to the same window for **tech and thought leadership content**:

> **Tuesday or Wednesday, 4–5 PM in your audience's timezone**

Here's why this specific slot — and not the "classic" 8–9 AM advice you'll see in older guides:

- Buffer's 2026 analysis found a notable shift from 2025: peak engagement windows have moved later into the day, with posts shared during traditional morning hours (6–11 AM) now seeing *lower* engagement compared to afternoon and evening slots. The specific top slots for 2026 are **Tuesday 4–5 PM, Wednesday 4–5 PM, and Thursday 4–5 PM**.

- For tech professionals specifically, Sprout Social's 2026 data shows early afternoon peaks on Wednesdays and Thursdays, aligning with the global nature of tech work and capturing the overlap of cross-continental workdays.

- LinkedIn's algorithm operates in distinct stages: during the first 60 minutes it runs an initial classification check (spam, policy, quality), then enters a 1–2 hour engagement testing phase. Posting at the right time ensures high-impact engagement occurs during that all-important first hour.

**Your audience is primarily US-based technical professionals (Chicago timezone = CT).** The optimal slot for you:

| Priority | Day | Time (CT) | Why |
|---|---|---|---|
| 🥇 Best | **Tuesday** | **4:00–4:30 PM CT** | Highest engagement day + afternoon peak |
| 🥈 Second | **Wednesday** | **4:00–4:30 PM CT** | Strong mid-week + global tech overlap |
| 🥉 Third | **Thursday** | **2:00–3:00 PM CT** | Good for longer-form thought leadership |

**Avoid:** Monday mornings (inbox chaos), Friday afternoons (wind-down), all day Saturday/Sunday (near-zero tech professional engagement).

---

### Blog Article Publishing Schedule

Blog articles should go live **the same day as the LinkedIn post**, ideally **30–60 minutes before** the LinkedIn post goes out. This way, when someone clicks your link, the article is already indexed and loading cleanly.

| Week | Date | Article | Blog Publish (CT) | LinkedIn Post (CT) | Frontmatter Date |
|---|---|---|---|---|---|
| 1 | **Tue Apr 21** | Article 0 — Setup & Glossary | 3:00 PM | 4:00 PM | 2026-04-21 |
| 2 | **Tue Apr 28** | Article 1 — Feature Pipeline | 3:00 PM | 4:00 PM | 2026-04-28 |
| 3 | **Tue May 5** | Article 2 — Propensity Model | 3:00 PM | 4:00 PM | 2026-05-05 |
| 4 | **Tue May 12** | Article 3 — RAG Retrieval | 3:00 PM | 4:00 PM | 2026-05-12 |
| 5 | **Tue May 19** | Article 4 — LLM Generator | 3:00 PM | 4:00 PM | 2026-05-19 |
| 6 | **Tue May 26** | Article 5 — LLMOps Pipeline | 3:00 PM | 4:00 PM | 2026-05-26 |
| 7 | **Tue Jun 2** | Article 6 — RLHF Feedback Loop | 3:00 PM | 4:00 PM | 2026-06-02 |
| 8 | **Tue Jun 9** | Article 7 — Recap & Next Steps | 3:00 PM | 4:00 PM | 2026-06-09 |

**Series runs: April 21 → June 9, 2026 (8 weeks)**

One-week gap between articles gives the LinkedIn algorithm time to settle each post, avoids cannibilising your own reach, and gives readers time to engage and share before the next one lands.

---

### Hashtag Strategy for 2026

Three rules from the latest research:

1. LinkedIn removed the ability to follow hashtags in late 2024 and now uses a 150-billion-parameter AI model to decide who sees content. Hashtags now function as categorisation signals that help the algorithm understand your topic — not as a discovery tool users browse. This means **relevance beats volume** — always.

2. Using more than five hashtags creates severe diminishing returns and can trigger LinkedIn's automated spam filters. The most effective strategy uses the "Broad + Niche + Branded" formula.

3. Mix broad hashtags with niche-specific ones to reach both wide and targeted audiences. Use 3–5 hashtags per post, placed at the end.

**Your formula for every post:**
- 1 broad tech/AI tag (large audience, general signal)
- 1–2 niche tags (specific to the post's topic)
- 1 series tag (branded, builds series identity over 8 weeks)

**Your branded series hashtag: `#LoyaltyLens`** — use it on every post to build a searchable thread.

---

## Part 2: LinkedIn Posts — Updated with Hashtags

---

### Series Launch Post (Optional — 1 week before Article 0)
*Publish: Tue Apr 14 — 4:00 PM CT (optional teaser)*

I'm publishing a seven-part series on building a production-grade offer intelligence platform from scratch.

Not a tutorial. A real system — with a feature store, a propensity model, RAG retrieval, LLM copy generation, drift monitoring, eval harness, CI/CD pipeline, and an RLHF feedback loop.

Every component is anchored to real production patterns. Every architectural decision is explained. Every benchmark is real.

The stack: Python, PyTorch, DuckDB, LangChain, LlamaIndex, pgvector, Weaviate, HuggingFace, CLIP, FastAPI, MLflow, GitHub Actions, Streamlit.

The tool: Claude Code for agentic multi-file code generation.

Article 0 drops next Tuesday — a complete setup guide and plain-English glossary for every term in the series.

Follow to get the full series as it releases.

👉 [link to Article 0 when live]

#GenerativeAI #MLOps #LoyaltyLens

---

### Post 0 — Setup & Glossary
*Publish: Tue Apr 21 — 4:00 PM CT*

I've been working on production AI systems for six years. The number one friction point for every new team member — regardless of experience — isn't the code. It's the acronyms.

RAG. PSI. RLHF. BFF. LLMOps. TabTransformer. pgvector.

So before releasing a seven-part series on building a production-grade offer intelligence platform from scratch, I wrote the article I wish existed: a complete setup guide + plain-English glossary for every term in the series.

37 terms. 6 setup steps. One mental model that makes all the modules click.

Article 0 of the LoyaltyLens series is live now.

👉 [link]

#MachineLearning #LLMOps #LoyaltyLens

---

### Post 1 — Feature Pipeline
*Publish: Tue Apr 28 — 4:00 PM CT*

Most ML tutorials skip the feature pipeline entirely. They start with a clean CSV and a model.

In production, the feature pipeline *is* the model. It's where data quality problems hide. It's where training-serving skew starts. And it's the layer that silently degrades your propensity scores for weeks before anyone notices.

I rebuilt a production-style loyalty feature pipeline using Python and DuckDB — no Databricks cluster, no $40K/month cloud bill. Here's what I built, why DuckDB outperformed SQLite at 5M rows, and the two things that surprised me.

Surprise #1: DuckDB is genuinely fast enough for batch ML serving.
Surprise #2: The validation layer catches more real problems than the feature engineering does.

Full walkthrough in Article 1 of the LoyaltyLens series.

👉 [link]

#MLOps #DataEngineering #LoyaltyLens

---

### Post 2 — Propensity Model
*Publish: Tue May 5 — 4:00 PM CT*

"Why didn't you just use XGBoost?"

It's the first question I get whenever I mention using a transformer for tabular propensity scoring. My answer has three parts — and only the third one is about model performance.

The short version: XGBoost is a dead end for multimodal extension. A transformer backbone isn't.

In Article 2 of LoyaltyLens, I walk through:
→ The full TabTransformer-lite architecture in PyTorch (with code)
→ How to construct binary labels from redemption rate data
→ MLflow experiment tracking for the training loop
→ The model card — including the bias consideration I actually monitor in production
→ The ONNX export path for SageMaker deployment

Val AUC: 0.81 on synthetic data. Production equivalent: 0.85–0.88 with richer features.

👉 [link]

#DeepLearning #PyTorch #LoyaltyLens

---

### Post 3 — RAG Retrieval
*Publish: Tue May 12 — 4:00 PM CT*

I ran a benchmark nobody seems to publish cleanly: LangChain vs. LlamaIndex, pgvector vs. Weaviate, at 200 / 2,000 / 20,000 vectors.

The results were clear enough that I changed the default architecture recommendation I give to enterprise clients.

Key findings:
• pgvector beats Weaviate on latency below ~20K vectors (31ms vs. 47ms p50)
• Weaviate's latency is more stable as catalog size grows
• LangChain and LlamaIndex deliver near-identical precision@5 (0.71 vs. 0.73)
• Oversampling + re-ranking matters more than framework choice

The practical recommendation: pgvector first. Dedicated vector DB when you have a concrete scale requirement. Don't add infrastructure complexity speculatively.

Full benchmark and code in Article 3 of LoyaltyLens.

👉 [link]

#RAG #VectorDatabase #LoyaltyLens

---

### Post 4 — LLM Generator
*Publish: Tue May 19 — 4:00 PM CT*

Prompt engineering is software engineering.

That's not a metaphor. It means: version your prompts. Diff them. Review changes before they ship. Roll back when quality drops.

In a production offer generation system running at scale, 2% non-JSON LLM output is tens of thousands of failed renders per day. A prompt change that subtly shifts brand voice is invisible until it reaches a customer.

Article 4 of LoyaltyLens covers:
→ A YAML-based versioned prompt registry with machine-readable eval criteria
→ Two LLM backends: OpenAI API and local Mistral-7B via HuggingFace
→ JSON parse retries that make the generator production-reliable
→ A CLIP-based brand image alignment scorer — a multimodal check that flags off-brand copy before it hits review

👉 [link]

#LLM #PromptEngineering #LoyaltyLens

---

### Post 5 — LLMOps Pipeline
*Publish: Tue May 26 — 4:00 PM CT*

Here's a story about a PSI of 0.31 on a Monday morning.

PSI — Population Stability Index — is a metric from credit risk modeling that almost nobody in the ML community uses outside that domain. It measures how much the input distribution to your model has shifted from baseline.

A weekend batch job recomputed recency features using a different timezone offset. Every customer's recency days shifted by one. The propensity model's input distribution shifted. The scores were wrong. Campaign sends that morning were misconfigured.

We caught it in 20 minutes. Without the drift monitor, we'd have caught it in three weeks.

Article 5 of LoyaltyLens covers the full LLMOps stack:
→ Prompt versioning CLI (list, diff, rollback)
→ Automated eval harness with a hard CI/CD quality gate
→ PSI-based drift monitor
→ GitHub Actions ML pipeline
→ Responsible AI audit framework

👉 [link]

#LLMOps #MLOps #LoyaltyLens

---

### Post 6 — RLHF Feedback Loop
*Publish: Tue Jun 2 — 4:00 PM CT*

RLHF doesn't require a research team.

The version that runs in production at scale isn't academic reinforcement learning with proprietary reward models. It's this:

1. Collect human feedback signals (ratings, thumbs up/down, redemption outcomes)
2. Aggregate them by prompt version and model version
3. Detect when rolling quality drops below threshold
4. Trigger retraining automatically

Article 6 of LoyaltyLens builds this end-to-end: a React feedback UI, SQLite persistence, a nightly aggregator, a preference dataset exporter in OpenAI fine-tuning format, and a GitHub Actions retraining trigger.

The most important design decision: export preference data in fine-tuning format from day one, even before you're ready to fine-tune. You can't retroactively generate training signal.

👉 [link]

#RLHF #GenerativeAI #LoyaltyLens

---

### Post 7 — Series Recap
*Publish: Tue Jun 9 — 4:00 PM CT*

Eight weeks. Seven articles. Six modules. One complete production-pattern AI system built in the open.

Here's what I learned building LoyaltyLens — the honest version:

The plumbing is harder than the model. Feature validation, drift monitoring, feedback loops — these take longer than model architecture and matter more for production reliability.

Versioning everything is not optional. You can't diff what isn't versioned. You can't roll back what has no history. You can't catch drift without a baseline.

The eval gate is a forcing function. Automated quality evaluation doesn't replace human judgment. It makes the standard explicit and enforceable. Those are different things.

PSI is the most underused metric in production ML. If you're not computing Population Stability Index on your model inputs, you're flying blind on data drift.

Collect feedback before you know how to use it. The preference dataset compounds. Start collecting before you have a use case.

Full recap + what to build next: Article 7 of LoyaltyLens.

👉 [link]

#MachineLearning #AIStrategy #LoyaltyLens

---

## Part 3: Hashtag Reference by Post

| Post | Broad | Niche | Branded |
|---|---|---|---|
| Series Launch | #GenerativeAI | #MLOps | #LoyaltyLens |
| 0 — Glossary | #MachineLearning | #LLMOps | #LoyaltyLens |
| 1 — Feature Pipeline | #MLOps | #DataEngineering | #LoyaltyLens |
| 2 — Propensity Model | #DeepLearning | #PyTorch | #LoyaltyLens |
| 3 — RAG Retrieval | #RAG | #VectorDatabase | #LoyaltyLens |
| 4 — LLM Generator | #LLM | #PromptEngineering | #LoyaltyLens |
| 5 — LLMOps Pipeline | #LLMOps | #MLOps | #LoyaltyLens |
| 6 — RLHF Feedback Loop | #RLHF | #GenerativeAI | #LoyaltyLens |
| 7 — Series Recap | #MachineLearning | #AIStrategy | #LoyaltyLens |

**Why only 3 hashtags per post:** Using more than five hashtags triggers LinkedIn's spam filters and signals low-effort content to your network. The algorithm cannot determine who a post is for if it contains too many unrelated tags — so it shows it to no one. Three focused, relevant hashtags outperform ten generic ones every time.

**Why `#LoyaltyLens` on every post:** It creates a searchable thread across all eight posts. Anyone who finds one article and searches the hashtag discovers the entire series. It also trains the algorithm to associate your profile with a consistent topic cluster over the eight-week run.

---

## Part 4: Blog Timing — pushparajan.tech

**Same logic applies.** If you're using a CMS like Ghost, Webflow, or WordPress, schedule articles to publish at **3:00 PM CT on Tuesday** — one hour before the LinkedIn post. This ensures:

- Google has at least 30 minutes to crawl and index the page
- The URL resolves cleanly when LinkedIn's link preview scraper hits it at 4:00 PM
- Any early organic traffic from LinkedIn doesn't hit a 404

**Tip:** Use the `date` field in the frontmatter (already set in your markdown files) to match the schedule above. If your CMS supports scheduled publishing, load all eight articles now and let them auto-publish — that way you're never scrambling on a Tuesday afternoon.
