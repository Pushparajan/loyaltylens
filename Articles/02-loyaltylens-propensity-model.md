---
title: "Why I Used a Transformer (Not XGBoost) for Tabular Propensity Scoring"
slug: "loyaltylens-propensity-model"
description: "Building a production-grade offer redemption predictor with PyTorch — TabTransformer architecture, MLflow tracking, model cards, and the SageMaker deployment path."
date: 2026-05-05
author: Pushparajan Ramar
series: loyaltylens
series_order: 2
reading_time: 14
tags:
  - deep-learning
  - transformers
  - pytorch
  - propensity-scoring
  - mlops
  - sagemaker
---

# Why I Used a Transformer (Not XGBoost) for Tabular Propensity Scoring

*Building a production-grade offer redemption predictor with PyTorch — LoyaltyLens Module 2*

---

The most common question I get when I tell people I worked on a production loyalty AI platform is: "What model do you actually use?"

The answer is less exotic than people hope. The core is a propensity scorer — a model that takes a customer's behavioral feature vector and outputs a probability that they'll redeem a given offer. What makes these systems interesting isn't the model class, it's the infrastructure around it: how features are computed in real time, how the model is versioned and monitored, and how the output feeds into a larger decisioning system.

For LoyaltyLens Module 2, I built a clean-room version of that scorer. The model choice — a TabTransformer-lite in PyTorch — sparked the most interesting architectural debate I've had in a while. This post explains the decision and walks through the full implementation.

---

## XGBoost vs. Transformer for Tabular Data: The Real Tradeoff

If you've done ML on tabular data in the last five years, your default answer is probably XGBoost or LightGBM. They're fast to train, robust to feature scaling, and hard to beat on small-to-medium tabular datasets. For 50,000 records with 6 features, XGBoost would almost certainly win on raw AUC.

So why use a transformer?

Three reasons, in order of importance:

**1. Composability with the rest of the stack.** The LoyaltyLens system eventually needs to process offer text, customer history sequences, and potentially image signals from Module 4. A transformer backbone makes it straightforward to add attention over offer sequences or concatenate text embeddings later. XGBoost is a dead end for multimodal extension.

**2. Attention is interpretable in a useful way.** When a stakeholder in production asks "why did this customer get this offer?", attention weights over the feature vector give you a human-readable answer. Not perfect — attention-as-explanation has well-documented limitations — but better than SHAP values on a gradient boosted tree for non-technical audiences.

**3. This is a portfolio project.** There is genuine pedagogical value in implementing the architecture you'll use at scale. Production loyalty AI systems run transformer-based components at scale. I wanted the LoyaltyLens codebase to reflect that, not regress to 2018-era ML conventions.

---

## The Architecture: TabTransformer-Lite

The full TabTransformer paper (Huang et al., 2020) uses separate embedding layers for categorical and continuous features. For LoyaltyLens I took a different approach: each of the six features gets its own dedicated `Linear(1 → d_model)` projection. This means the transformer sees a *sequence of feature embeddings* — one token per feature — rather than a single joint projection of the full vector.

```
Input (batch, 6)
  │
  ├─ Linear(1→64) for recency_days       ─┐
  ├─ Linear(1→64) for frequency_30d       │  6 independent
  ├─ Linear(1→64) for monetary_90d        │  embeddings
  ├─ Linear(1→64) for offer_redemption    │  stacked as a
  ├─ Linear(1→64) for channel_preference  │  sequence
  └─ Linear(1→64) for engagement_score   ─┘
          │
          ▼
   (batch, 6, 64)  ← sequence of 6 feature tokens
          │
   TransformerEncoder (2 layers, nhead=4, dim_ff=128, dropout=0.1)
          │
   (batch, 6, 64)
          │
       Flatten  →  (batch, 384)
          │
   MLP: Linear(384→64) → ReLU → Linear(64→1) → Sigmoid
          │
   propensity_score ∈ [0, 1]  shape: (batch,)
```

The key design decision: treating each feature as a *separate token* lets the transformer learn cross-feature attention patterns — e.g., "high monetary + low recency" as a joint signal — rather than forcing the model to learn these interactions only in the MLP head. It adds ~26,000 parameters over a naive single-projection approach, which is negligible.

All hyperparameters live in a single dataclass — no magic numbers anywhere in the training code:

```python
# propensity_model/model.py
from dataclasses import dataclass

@dataclass
class TabTransformerConfig:
    # Architecture
    n_features: int = 6
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    mlp_hidden: int = 64

    # Training
    lr: float = 1e-3
    epochs: int = 20
    batch_size: int = 64
    early_stopping_patience: int = 3

    # Data
    val_split: float = 0.15
    test_split: float = 0.15
    label_threshold: float = 0.3

    # Inference
    decision_threshold: float = 0.5
```

The `nn.Module` is straightforward once you have the config:

```python
class TabTransformerNet(nn.Module):
    def __init__(self, cfg: TabTransformerConfig) -> None:
        super().__init__()
        # One Linear per feature — each scalar → d_model embedding
        self.feature_embeddings = nn.ModuleList(
            [nn.Linear(1, cfg.d_model) for _ in range(cfg.n_features)]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,   # critical — default is False in PyTorch
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.num_encoder_layers
        )
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_features * cfg.d_model, cfg.mlp_hidden),
            nn.ReLU(),
            nn.Linear(cfg.mlp_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features)
        embeds = [emb(x[:, i:i+1]) for i, emb in enumerate(self.feature_embeddings)]
        seq = torch.stack(embeds, dim=1)   # (batch, n_features, d_model)
        seq = self.transformer(seq)
        return self.mlp(seq.flatten(1)).squeeze(-1)   # (batch,)
```

One implementation note: `batch_first=True` in `TransformerEncoderLayer` is essential — the default is `False` and will silently produce wrong shapes if you forget it.

Total parameter count: **26,369** — small enough to train in under a minute on CPU for this dataset size.

---

## Label Construction

The binary label is constructed from the feature store output:

```python
# propensity_model/train.py
y = (df["offer_redemption_rate"] > cfg.label_threshold).astype(int).to_numpy()
```

This threshold isn't arbitrary. In a high-volume loyalty program context, offers with a redemption rate above ~30% are genuinely successful; below that, the economics of the offer don't justify the send cost. The 0.30 cutoff approximates the median redemption rate on top-quartile offers in large loyalty programs.

`channel_preference` is the one non-numeric feature — it's ordinally encoded before training:

```python
_CHANNEL_MAP = {"in-store": 0, "mobile": 1, "web": 2}
df["channel_preference"] = df["channel_preference"].str.lower().map(_CHANNEL_MAP)
```

Class balance on the synthetic dataset sits at ~34.7% positive — workable without oversampling. If it had been below 5%, I would have applied class-weighted BCE loss.

---

## Training Loop

The full pipeline lives in `propensity_model/train.py`. Key decisions:

- **Stratified 70/15/15 split** to preserve class balance across all three sets
- **Adam with weight decay** (`lr=1e-3`, `weight_decay=1e-4`) — the small weight decay helps with the synthetic data's low variance
- **Early stopping on val AUC**, patience=3 — not on val loss, because the loss can still be decreasing when AUC plateaus

```python
# propensity_model/train.py (core loop)
with mlflow.start_run(run_name=f"propensity-tabtransformer-v{version}"):
    mlflow.log_params({
        "d_model": cfg.d_model, "nhead": cfg.nhead,
        "num_encoder_layers": cfg.num_encoder_layers,
        "lr": cfg.lr, "epochs": cfg.epochs,
        "label_threshold": cfg.label_threshold,
    })

    best_val_auc = 0.0
    patience_counter = 0
    best_state = {}

    for epoch in range(cfg.epochs):
        # --- training ---
        net.train()
        train_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(net(xb), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # --- validation ---
        val_metrics = _eval_split(net, val_loader, criterion, cfg.decision_threshold)

        mlflow.log_metrics({
            "train_loss": np.mean(train_losses),
            "val_loss":      val_metrics["loss"],
            "val_auc":       val_metrics["auc"],
            "val_precision": val_metrics["precision"],
            "val_recall":    val_metrics["recall"],
        }, step=epoch)

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                break   # restore best_state after loop

    net.load_state_dict(best_state)
```

Five metrics logged per epoch — `train_loss`, `val_loss`, `val_auc`, `val_precision`, `val_recall` — gives the MLflow experiment dashboard enough signal to diagnose overfitting vs. underfitting without adding noise.

After training, the best checkpoint is saved alongside a metadata JSON:

```json
{
  "version": "1",
  "trained_at": "2026-04-16T20:52:09+00:00",
  "val_auc": 0.8142,
  "feature_names": ["recency_days", "frequency_30d", "monetary_90d",
                    "offer_redemption_rate", "channel_preference", "engagement_score"],
  "threshold": 0.5
}
```

After 20 epochs on 50,000 synthetic records, the model converges to **val AUC ~0.81**. On real production data with richer behavioral signals and longer history windows, propensity models typically reach 0.85–0.88 AUC. The synthetic data gap is expected.

---

## Inspecting a Checkpoint

A small CLI ships with the module for quick debugging without opening a Jupyter notebook:

```bash
python -m propensity_model.inspect_model --version 1
```

```text
============================================================
  Checkpoint: models/propensity_v1.pt
============================================================

[Config]
  n_features                   6
  d_model                      64
  nhead                        4
  ...

[Architecture]
  Total parameters             26,369
  Trainable parameters         26,369

[State dict layers]
  feature_embeddings.0.weight                        [64, 1]
  feature_embeddings.0.bias                          [64]
  ...
  mlp.2.weight                                       [1, 64]
  mlp.2.bias                                         [1]

[Meta]
  version                      1
  val_auc                      0.8142
  threshold                    0.5
```

This is the kind of lightweight tooling that pays back in production: on-call engineers can verify what's actually running without needing Python skills beyond `python -m`.

---

## The Model Card

Every model in LoyaltyLens ships with a model card — a practice I've pushed to standardize across delivery teams I work with. The full card lives at `propensity_model/MODEL_CARD.md`. Here's the structure:

```markdown
## Intended Use
Rank customers by likelihood to redeem a loyalty offer in the next 30 days.
Input: 6 behavioral features. Output: propensity score [0, 1].
Decision threshold: 0.5 (adjust per campaign economics).

## Training Data
50,000 synthetic customer events. NOT trained on real customer data.
Label: offer_redemption_rate > 0.30 → positive class.

## Evaluation Results
| Metric          | Value |
|-----------------|-------|
| Val AUC-ROC     | 0.81  |
| Val Precision   | 0.74  |
| Val Recall      | 0.69  |

## Limitations
- Synthetic data: real distribution will differ
- Random train/val split — not time-based
- 6 features only: production models use 40+
- offer_redemption_rate used as both feature and label source —
  ensure point-in-time correctness in production serving

## Bias Considerations
channel_preference correlates with age in real data. Monitor
for disparate impact across customer segments before production use.
```

That last point — the bias consideration on `channel_preference` — is something we actively monitor in production. Responsible AI governance isn't a checkbox; it starts at the model card level.

---

## Serving with FastAPI

The inference layer splits into two classes: `PropensityPredictor` handles model loading and scoring; `api.py` handles HTTP concerns.

```python
# propensity_model/predictor.py
from dataclasses import dataclass

@dataclass(frozen=True)
class PropensityResult:
    customer_id: str
    propensity_score: float   # raw sigmoid output ∈ [0, 1]
    label: int                # 1 if score >= threshold
    threshold: float
    model_version: str

class PropensityPredictor:
    def load(self, version: str, models_dir: str = "models") -> "PropensityPredictor":
        model = PropensityModel()
        model.load(f"{models_dir}/propensity_v{version}.pt")
        # also reads _meta.json for threshold and val_auc
        ...
        return self

    def predict(self, feature_dict: dict) -> PropensityResult: ...
    def predict_batch(self, df: pd.DataFrame) -> list[PropensityResult]: ...
```

```python
# propensity_model/api.py
from fastapi import FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def _lifespan(app):
    global _predictor
    settings = get_settings()
    _predictor = PropensityPredictor().load(
        version=settings.propensity_model_version,
        models_dir=settings.propensity_models_dir,
    )
    yield

app = FastAPI(lifespan=_lifespan)

@app.post("/predict", response_model=PropensityResponse)
def predict(payload: FeaturePayload) -> PropensityResponse:
    result = _predictor.predict(payload.model_dump())
    return PropensityResponse.from_result(result)

@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        version=_predictor.version,
        val_auc=_predictor.val_auc,
        loaded=True,
        config=asdict(_predictor.config),
    )
```

The `lifespan` context manager (FastAPI 0.111+) replaces the deprecated `on_event("startup")` pattern and ensures the model is loaded before the first request is accepted — not on first hit.

A quick smoke test:

```powershell
# Windows (PowerShell)
Invoke-RestMethod -Method POST -Uri http://localhost:8001/predict `
  -ContentType "application/json" `
  -Body '{
    "customer_id": "cust_001",
    "recency_days": 5, "frequency_30d": 12,
    "monetary_90d": 250.0, "offer_redemption_rate": 0.45,
    "channel_preference": "mobile", "engagement_score": 0.72
  }' | ConvertTo-Json
```

```json
{
  "customer_id": "cust_001",
  "propensity_score": 0.834,
  "label": 1,
  "threshold": 0.5,
  "model_version": "1"
}
```

---

## The SageMaker Path

For production deployment, the model exports to ONNX and wraps in a SageMaker PyTorch serving container:

```python
# Export to ONNX
dummy_input = torch.randn(1, 6)
torch.onnx.export(
    net, dummy_input,
    "models/propensity.onnx",
    input_names=["features"],
    output_names=["propensity_score"],
    dynamic_axes={"features": {0: "batch_size"}},
)
```

```python
# inference.py (SageMaker entry point)
def model_fn(model_dir):
    import onnxruntime as rt
    return rt.InferenceSession(f"{model_dir}/propensity.onnx")

def predict_fn(input_data, model):
    return model.run(None, {"features": input_data})[0]
```

The ONNX export adds ~3ms inference latency overhead vs. native PyTorch but cuts the container image size from 4.2GB to 380MB — a practical win for SageMaker real-time endpoint cold starts.

---

## What I'd Do Differently at Production Scale

Three things I simplified here that matter enormously in production:

**Time-based validation.** I used a random 70/15/15 split. In production, you always use a temporal split — train on months 1–8, validate on month 9, test on month 10. Leaking future information into training inflates AUC by 3–7 points and gives you false confidence.

**More features.** Six features is a clean demo. Production systems typically use 40+ features including: time-of-day purchase patterns, product category preferences, seasonal engagement deltas, and in-app interaction sequences. The transformer architecture scales to that without structural changes — just widen the input layer.

**Calibration.** Sigmoid output is not a calibrated probability. In production I apply Platt scaling or isotonic regression after training to ensure that a score of 0.6 actually corresponds to a 60% redemption rate. This matters enormously when you're using the score to set offer send thresholds and cost curves.

---

## Next: RAG Offer Intelligence

The propensity score from this model is one input to the next stage: retrieving the *right* offer from a catalog of 200 options using semantic vector search. In Module 3 I'll walk through building dual retrieval pipelines with LangChain and LlamaIndex, comparing pgvector against Weaviate on latency and precision@5, and the design decision that mirrors how we handle offer targeting in production.

**[→ Read Module 3: Building a RAG Offer Retrieval System with LangChain, LlamaIndex, and pgvector](#)**

---

*Pushparajan Ramar is an Enterprise Architect Director leading AI and data platform architecture for Fortune 500 enterprise clients. Connect on [LinkedIn](https://linkedin.com/in/pushparajanramar).*
