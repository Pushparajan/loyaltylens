---
title: "Wiring Six AI Modules Into One Pipeline — and Deploying to the Cloud"
slug: "loyaltylens-integration-deploy"
description: "How I connected LoyaltyLens's six modules into a single orchestration layer, exported the PyTorch model to TorchScript, and deployed a real-time inference endpoint on AWS SageMaker."
date: 2026-05-19
author: Pushparajan Ramar
series: loyaltylens
series_order: 7
reading_time: 16
tags:
  - mlops
  - sagemaker
  - aws
  - pytorch
  - vertex-ai
  - cloud-deployment
  - production-ai
  - loyaltylens
---

# Wiring Six AI Modules Into One Pipeline — and Deploying to the Cloud

*How to build an integration layer that ties feature store, propensity scoring, RAG, and LLM generation together — then ship the model to a real cloud endpoint — LoyaltyLens Module 7*

---

**Series position:** Article 7 of 8

---

Six articles in, LoyaltyLens has a feature store, a propensity model, a RAG retrieval system, an LLM copy generator, a full LLMOps pipeline, and an RLHF feedback loop. Each module has its own FastAPI endpoint, its own test suite, its own configuration. They all work. Independently.

The problem: nobody had wired them together into a single callable function. And none of the model artifacts had ever left the local `models/` directory.

This article covers two things:

1. **The integration layer** — `shared/pipeline.py`, a single class that takes a `customer_id` and returns a fully generated offer — features fetched, propensity scored, offers retrieved, copy written, latency measured end to end.

2. **Cloud deployment** — exporting the PyTorch model to TorchScript, packaging it for SageMaker, deploying a real-time inference endpoint, and the one error that almost derailed the whole thing.

---

## Why Integration Is Harder Than It Looks

The individual modules are easy to test in isolation. Wire them together and you immediately discover three problems that don't show up in unit tests:

**1. Lazy loading is not optional.** If you import all six modules at startup, your `shared/pipeline.py` will take 8–12 seconds to load — Weaviate client, sentence transformer model, and PyTorch model all initialise simultaneously. In a FastAPI context, that's a 10-second cold start on the first request. In a batch job, it's startup overhead you pay once. The right pattern: initialise each subsystem on first use, not at import time.

**2. Retriever shape mismatches.** The RAG module's `retrieve()` method was built to work with LangChain Document objects. The pipeline needs plain dicts. The integration layer's job is to translate between module-specific types so each module doesn't need to know about the others.

**3. The propensity threshold filters everything.** On synthetic data, a newly trained model can output scores as low as 0.0007 for some customers. If your offers all have `min_propensity_threshold: 0.01`, every offer gets filtered out and you get a fallback result. The fix: fall back gracefully to the full unfiltered result set when the threshold eliminates all candidates.

These are the kinds of issues you only see when you run the full pipeline end to end, and they're worth documenting because they're exactly the class of integration bugs that derail production deployments.

---

## The Integration Layer: `shared/pipeline.py`

The design is a single class with lazy initialisation for each subsystem:

```python
# shared/pipeline.py
class LoyaltyLensPipeline:
    def __init__(self) -> None:
        self._feature_store: Any = None
        self._predictor: Any = None
        self._retriever: Any = None
        self._generator: Any = None

    def _get_feature_store(self) -> Any:
        if self._feature_store is None:
            from feature_store.store import FeatureStore
            self._feature_store = FeatureStore()
        return self._feature_store

    def _get_predictor(self) -> Any:
        if self._predictor is None:
            from propensity_model.predictor import PropensityPredictor
            from shared.config import get_settings
            settings = get_settings()
            self._predictor = PropensityPredictor().load(
                version=settings.propensity_model_version,
                models_dir=settings.propensity_models_dir,
            )
        return self._predictor
    # ... same pattern for retriever and generator
```

`PropensityPredictor().load()` is an instance method, not a classmethod — a subtle point that only matters when you call it for the first time in an integration context. The pattern is `PropensityPredictor().load(...)` not `PropensityPredictor.load(...)`.

The main `run_for_customer()` method wires the four stages with per-stage latency tracking:

```python
def run_for_customer(self, customer_id: str) -> PipelineResult:
    run_id = str(uuid.uuid4())
    latency: dict[str, float] = {}

    # Step 1 — features
    t0 = time.perf_counter()
    feature_dict = self._get_feature_store().read_latest(customer_id)
    latency["feature_store_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Step 2 — propensity
    t0 = time.perf_counter()
    result = self._get_predictor().predict(feature_dict)
    latency["propensity_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Step 3 — RAG retrieval
    t0 = time.perf_counter()
    customer_context = _build_context_string(customer_id, feature_dict)
    raw_offers = self._get_retriever().retrieve(
        customer_context, result.propensity_score, k=5
    )
    latency["rag_retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Step 4 — copy generation
    t0 = time.perf_counter()
    copy = self._get_generator().generate(
        customer_ctx={"customer_id": customer_id,
                      "tier": feature_dict.get("tier", "Gold"),
                      "channel": feature_dict.get("channel_preference", "mobile"),
                      **feature_dict},
        offer_ctx={"offer_title": best_offer.get("title", ""),
                   "offer_description": best_offer.get("description", ""),
                   **best_offer},
    )
    latency["llm_generator_ms"] = round((time.perf_counter() - t0) * 1000, 1)
```

Two things worth noting about the context mapping in Step 4:

The LLM generator's prompt template uses `{offer_title}` and `{offer_description}` as variable names. The RAG retriever returns offers with keys `title` and `description`. If you pass the offer dict directly, the template variables don't resolve. The explicit mapping — `"offer_title": best_offer.get("title", "")` — is the fix. This is exactly the kind of silent bug that passes unit tests and fails in integration.

Similarly, the prompt template uses `{channel}` and `{tier}` as customer context variables, but `feature_dict` uses `channel_preference`. The `customer_ctx` dict maps them explicitly.

### Running the Pipeline

```bash
python -m shared.pipeline --customer-id C001
```

```text
============================================================
  Pipeline run  4f3a2b1c…
  2026-04-18T00:14:22.441Z
============================================================
  Customer        C001
  Propensity      0.9934  (label=1)
  Model version   1
  Prompt version  1

  Top offer
    O037          Birthday Bonus Stars
    Category: birthday  Score: 1.000

  Generated copy
    Headline  Celebrate Your Birthday with Bonus Stars!
    Body      Redeem 500 bonus points on your next visit — your
              loyalty tier unlocks this exclusive reward.
    CTA       Claim Your Birthday Stars

  Latency breakdown
    feature_store_ms               4.2 ms
    propensity_ms                  2.1 ms
    rag_retrieval_ms              47.8 ms
    llm_generator_ms             612.0 ms
    total                        666.1 ms
============================================================
```

Four modules, one result, 666ms total — feature store and propensity are fast (sub-5ms each), RAG adds ~50ms, and the LLM is the tail. This is the typical latency profile for an LLM-in-the-loop pipeline: everything else is noise compared to the generation step.

### The Weaviate Retriever

The integration layer ships its own Weaviate retriever — `_WeaviateOfferRetriever` in `shared/pipeline.py` — rather than using the LangChain-based one from Module 3. The reason: the Module 3 retriever returns `langchain_core.Document` objects, and the pipeline needs plain dicts. Rather than introducing a type-conversion layer, the integration retriever talks to Weaviate directly via the v4 Python client and returns the metadata dict from each result object.

It also handles the Weaviate-on-first-run case: if the collection is empty, it auto-indexes all 200 offers from `rag_retrieval/data/offers.json` before the first query. This means the pipeline is self-bootstrapping — no separate indexing step required.

---

## Cloud Deployment: The Design

The `deploy/` module has three concerns:

1. **Configuration** — `deploy/config.py` — a `DeploySettings` class that reads from `.env` files and validates the `DEPLOY_TARGET` value (`local`, `sagemaker`, or `vertex`).
2. **Storage** — `deploy/cloud_storage.py` — a `upload_artifact()` function that routes to S3 or GCS based on `DEPLOY_TARGET`.
3. **Deployment** — `deploy/sagemaker_deploy.py` and `deploy/vertex_deploy.py` — platform-specific deployers.

The `DeploySettings` design reflects a key principle: one env var controls everything. You don't change code to switch from local to cloud — you change `DEPLOY_TARGET`.

```python
# deploy/config.py
class DeploySettings(BaseSettings):
    deploy_target: str = "local"

    @field_validator("deploy_target")
    @classmethod
    def validate_target(cls, v: str) -> str:
        allowed = {"local", "sagemaker", "vertex"}
        if v not in allowed:
            raise ValueError(f"DEPLOY_TARGET must be one of {allowed}, got {v!r}")
        return v
```

### `deploy/.env.sagemaker`

```dotenv
DEPLOY_TARGET=sagemaker
AWS_REGION=us-east-1
SAGEMAKER_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerRole
SAGEMAKER_INSTANCE_TYPE=ml.t2.medium
SAGEMAKER_ENDPOINT_NAME=loyaltylens-propensity
S3_BUCKET=loyaltylens-artifacts
```

---

## The SageMaker Deployment Path (and the Error That Broke It)

### Step 1: IAM Setup

SageMaker needs two IAM identities:

- **Your user** — needs `AmazonSageMakerFullAccess` and `AmazonS3FullAccess` to call the SageMaker and S3 APIs.
- **SageMaker execution role** — a separate IAM role that SageMaker assumes when running the endpoint container. Create it via **IAM → Roles → Create role → AWS service → SageMaker**, attach `AmazonSageMakerFullAccess`, name it `SageMakerRole`. Copy the ARN into your `SAGEMAKER_ROLE_ARN` env var.

The execution role is easy to miss — first attempt at deploying returned:

```
ValidationException: Could not assume provided execution role.
```

The user credentials and the execution role are separate. The user calls the SageMaker API. SageMaker uses the execution role to run the container. Both need the right permissions.

### Step 2: The TorchScript Requirement

The second error was more interesting. The endpoint deployed successfully (~8 minutes), but the first invoke returned:

```
ModelError: Received server error (500) from primary with message:
  ModelLoadError: Failed to load propensity_v1.pt.
  Please ensure model is saved using torchscript.
```

The SageMaker PyTorch serving container's default `model_fn` tries to load the model file with `torch.jit.load`. Our checkpoint (`propensity_v1.pt`) is a plain state dict — a Python dict with `state_dict` and `config` keys. Not TorchScript. The default handler crashes during initialisation before the custom `inference.py` entry point is even invoked.

The fix: export the model to TorchScript before packaging.

```python
# deploy/sagemaker_deploy.py
def export_to_torchscript(self, checkpoint_path: str, output_path: str = "model.pt") -> str:
    import torch
    from propensity_model.model import TabTransformerNet

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    net = TabTransformerNet(cfg)
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()

    dummy = torch.zeros(1, cfg.n_features)
    # check_trace=False: TransformerEncoderLayer takes a different fused-op path
    # across invocations — graph differs between traces, values are identical.
    scripted = torch.jit.trace(net, dummy, check_trace=False)
    torch.jit.save(scripted, output_path)
    return output_path
```

One subtlety: `torch.jit.script` would be the first instinct, but `TabTransformerNet.forward()` uses `enumerate(self.feature_embeddings)` over a `ModuleList`. TorchScript handles this, but PyTorch's `TransformerEncoderLayer` uses a fused kernel path (`_transformer_encoder_layer_fwd`) that creates non-deterministic graph structures across trace invocations. `check_trace=False` disables the sanity check that compares two traces — the values are identical, only the graph representation differs. `torch.jit.trace` is the right choice here.

### Step 3: Packaging

SageMaker's PyTorch serving container expects:

```
model.tar.gz/
  model.pt            ← TorchScript weights, loaded by model_fn
  code/inference.py   ← entry point, invoked by SAGEMAKER_PROGRAM
```

```python
def package_model(self, model_path: str, output_dir: str = ".") -> str:
    archive = Path(output_dir) / "model.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(model_path, arcname="model.pt")
        tar.add(str(inference_src), arcname="code/inference.py")
    return str(archive)
```

The `arcname="model.pt"` is important — regardless of the source filename, the file must land at `model.pt` in the archive root. The SageMaker container extracts the archive and calls `model_fn(model_dir)` with `model_dir` pointing to the extraction directory.

### Step 4: The Inference Script

The `inference.py` entry point handles the full request lifecycle:

```python
# SageMaker entry point — written to propensity_model/inference.py at deploy time
_CHANNEL_MAP = {"in-store": 0.0, "mobile": 1.0, "web": 2.0}

def model_fn(model_dir):
    return torch.jit.load(str(Path(model_dir) / "model.pt"), map_location="cpu")

def input_fn(request_body, content_type="application/json"):
    data = json.loads(request_body)
    channel_val = _CHANNEL_MAP.get(
        str(data.get("channel_preference", "web")).lower(), 2.0
    )
    return torch.tensor([[
        float(data.get("recency_days", 0)),
        float(data.get("frequency_30d", 0)),
        float(data.get("monetary_90d", 0)),
        float(data.get("offer_redemption_rate", 0)),
        channel_val,
        float(data.get("engagement_score", 0)),
    ]], dtype=torch.float32)

def predict_fn(input_data, model):
    with torch.no_grad():
        score = float(model(input_data).squeeze().item())
    return {"propensity_score": score,
            "model_version": os.environ.get("MODEL_VERSION", "1")}

def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction), accept
```

### Step 5: Deploy and Invoke

```bash
# Deploy (packages, exports to TorchScript, uploads to S3, creates endpoint — ~10 min)
python deploy/sagemaker_deploy.py --action deploy --model-path models/propensity_v1.pt

# Invoke
python deploy/sagemaker_deploy.py --action invoke \
  --payload '{"recency_days": 3, "frequency_30d": 5, "monetary_90d": 120,
              "offer_redemption_rate": 0.4, "channel_preference": "mobile",
              "engagement_score": 0.7}'
```

```json
{
  "propensity_score": 0.9899324178695679,
  "model_version": "1"
}
```

Score `0.99` — consistent with the local model output. The SageMaker endpoint is serving the same predictions as the local PyTorch model.

### Step 6: Teardown

**Always teardown endpoints when not in use.** SageMaker charges for endpoint uptime regardless of traffic.

```bash
python deploy/sagemaker_deploy.py --action teardown
```

`ml.t2.medium` is ~$0.056/hour. A forgotten endpoint running for a week costs ~$9.40. Easily avoidable.

---

## GCP Vertex AI

The `deploy/vertex_deploy.py` module mirrors the SageMaker implementation for GCP. Two deployers:

**`VertexModelDeployer`** — uploads the model artifact to GCS, creates a Vertex AI Model resource, deploys to an online prediction endpoint.

**`VertexVectorSearchDeployer`** — indexes offer embeddings in Vertex AI Vector Search (the managed HNSW service), deploys the index to an endpoint for low-latency retrieval.

```bash
# Deploy model
python deploy/vertex_deploy.py --action deploy-model \
  --config deploy/.env.vertex \
  --gcs-uri gs://loyaltylens-artifacts/models/propensity_v1.pt

# Predict
python deploy/vertex_deploy.py --action predict \
  --config deploy/.env.vertex \
  --payload '{"recency_days": 3, "frequency_30d": 5, "monetary_90d": 120,
              "offer_redemption_rate": 0.4, "engagement_score": 0.7}'
```

Full setup instructions — including service account creation, ADC authentication, and GCS bucket setup — are in `deploy/README.md`.

---

## The Raw boto3 Decision

The `sagemaker` Python SDK (`pip install sagemaker`) ships a high-level `PyTorchModel` class that handles packaging, role validation, and endpoint creation in one call. I chose raw boto3 instead.

Reason: the version of the `sagemaker` package available via `uv pip install sagemaker` is a minimal distribution that doesn't include `sagemaker.pytorch.PyTorchModel`. Rather than debugging package version constraints, I used the raw APIs directly — `sm.create_model`, `sm.create_endpoint_config`, `sm.create_endpoint`, `sm.get_waiter("endpoint_in_service")`. This is more verbose but has zero hidden dependencies and makes the SageMaker API surface explicit in the code.

In production, the high-level SDK is the right choice for teams who work with SageMaker daily. For a demo codebase that needs to install cleanly across environments, the boto3 path is more reliable.

---

## Test Coverage

The deploy module ships with 14 unit tests in `tests/test_deploy.py`:

- **Config validation:** default target is `local`, invalid targets raise `ValidationError`
- **Cloud storage routing:** `upload_artifact()` routes to S3 or GCS based on `DEPLOY_TARGET`, raises `ValueError` for missing buckets
- **SageMaker:** `invoke_endpoint()` parses the response correctly, `teardown()` calls all three delete APIs
- **Vertex AI:** `predict()` extracts the propensity score from the predictions list, handles empty predictions gracefully

All cloud calls are mocked — no AWS or GCP credentials required to run the test suite.

```bash
python -m pytest tests/test_deploy.py tests/test_pipeline.py -v
```

---

## What This Looks Like at Production Scale

The `LoyaltyLensPipeline` class in this module is a clean-room version of what a real-time offer intelligence service looks like in production. In an enterprise context the same pattern runs across:

- Millions of customers per day, not hundreds
- Feature stores backed by Redis + Databricks instead of DuckDB
- SageMaker multi-model endpoints shared across offer scoring models for different segments
- RAG retrieval against catalogs with 50,000+ offers, not 200
- LLM generation behind an internal API gateway with rate limiting and audit logging
- The entire pipeline wired into a campaign management workflow, not a CLI

The architecture is the same. The scale multipliers are infrastructure and data problems, not design problems. That's the point of building the reference implementation at the right level of abstraction.

---

## Next: Recap and What to Build Next

The final article in the series steps back from the code and covers the five honest lessons from building LoyaltyLens end to end — and the prioritised roadmap for taking it from portfolio project to production system.

**[→ Read Article 8: LoyaltyLens Recap and What to Build Next](#)**

---

*Pushparajan Ramar is an Enterprise Architect Director specializing in AI, data, and platform architecture. Connect on [LinkedIn](https://linkedin.com/in/pushparajanramar).*
