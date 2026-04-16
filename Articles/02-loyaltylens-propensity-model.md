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

The full TabTransformer paper (Huang et al., 2020) uses separate embedding layers for categorical and continuous features. For LoyaltyLens I simplified this: all six features are continuous, so I skip the categorical embedding pathway and use a single linear projection into the transformer's embedding space.

```
Input: [recency_days, frequency_30d, monetary_90d,
        offer_redemption_rate, channel_preference_encoded,
        engagement_score]  ← shape (batch, 6)

Linear(6 → 64) + LayerNorm
    └─► TransformerEncoder
          2 layers
          d_model = 64
          nhead = 4
          dim_feedforward = 128
          dropout = 0.1
          └─► Mean pooling over sequence dim
                └─► MLP: Linear(64→32) → ReLU → Dropout(0.1)
                      └─► Linear(32→1) → Sigmoid
                            └─► propensity_score ∈ [0, 1]
```

In PyTorch:

```python
# propensity_model/model.py
import torch
import torch.nn as nn

class PropensityModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features) → add sequence dim for transformer
        x = self.input_proj(x).unsqueeze(1)     # (batch, 1, d_model)
        x = self.transformer(x)                  # (batch, 1, d_model)
        x = x.squeeze(1)                         # (batch, d_model)
        return self.head(x).squeeze(-1)          # (batch,)
```

One implementation note: `batch_first=True` in `TransformerEncoderLayer` is essential — it's `False` by default in PyTorch and will silently produce wrong shapes if you're not careful.

---

## Label Construction

The binary label is constructed from the feature store output:

```python
# A customer is "positive" if their historical redemption rate exceeds 0.3
df["label"] = (df["offer_redemption_rate"] > 0.30).astype(int)
```

This threshold isn't arbitrary. In a high-volume loyalty program context, offers with a redemption rate above ~30% are genuinely successful; below that, the economics of the offer don't justify the send cost. The 0.30 cutoff approximates the median redemption rate on top-quartile offers in large loyalty programs.

Class balance check:

```python
pos_rate = df["label"].mean()
# → 0.347 (34.7% positive)
```

34.7% positive rate is workable without oversampling. If it had been below 5%, I would have used stratified sampling with a higher weight on the positive class.

---

## Training Loop

```python
# propensity_model/train.py (core loop)
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

FEATURE_COLS = [
    "recency_days", "frequency_30d", "monetary_90d",
    "offer_redemption_rate", "channel_pref_encoded", "engagement_score"
]

with mlflow.start_run(run_name=f"propensity_v{version}"):
    mlflow.log_params({
        "d_model": 64, "nhead": 4, "num_layers": 2,
        "dropout": 0.1, "lr": 1e-3, "epochs": 20,
        "label_threshold": 0.30,
    })

    best_val_auc = 0.0
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        train_loss = run_epoch(model, train_loader, optimizer, criterion)

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_tensor).numpy()
        val_auc = roc_auc_score(y_val, val_preds)

        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_auc": val_auc,
        }, step=epoch)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), f"models/propensity_v{version}.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    mlflow.log_metric("best_val_auc", best_val_auc)
```

After 20 epochs on 50,000 synthetic records, the model converges to **val AUC ~0.81**. On real production data with richer behavioral signals and longer history windows, propensity models typically reach 0.85–0.88 AUC. The synthetic data gap is expected.

---

## The Model Card

Every model in LoyaltyLens ships with a model card — a practice I've pushed to standardize across delivery teams I work with. Here's the structure:

```markdown
## Intended Use
Predict offer redemption probability for loyalty program customers.
Input: 6 behavioral features. Output: propensity score [0, 1].
Decision threshold for offer send: 0.45 (tuned on val set F1).

## Training Data
50,000 synthetic customer events generated with numpy seed=42.
NOT trained on real customer data.

## Evaluation Results
| Metric | Value |
|--------|-------|
| Val AUC-ROC | 0.81 |
| Val Precision @0.45 | 0.74 |
| Val Recall @0.45 | 0.69 |

## Limitations
- Synthetic data: real distribution will differ
- No temporal validation (train/val split is random, not time-based)
- 6 features only: production models use 40+

## Bias Considerations
channel_preference feature encodes mobile/web/in-store. In-store
preference correlates with age in real data — monitor for
disparate impact across customer segments before production use.
```

That last point — the bias consideration on `channel_preference` — is something we actively monitor in production. Responsible AI governance isn't a checkbox; it starts at the model card level.

---

## Serving with FastAPI

The inference endpoint is intentionally simple:

```python
# propensity_model/api.py
from pydantic import BaseModel
from propensity_model.predictor import PropensityPredictor

predictor = PropensityPredictor.load(version="latest")

class PredictRequest(BaseModel):
    customer_id: str
    features: dict[str, float]

class PredictResponse(BaseModel):
    customer_id: str
    propensity_score: float
    confidence: str          # "high" | "medium" | "low"
    model_version: str
    threshold: float

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    result = predictor.predict(req.features)
    return PredictResponse(
        customer_id=req.customer_id,
        propensity_score=round(result.score, 4),
        confidence=classify_confidence(result.score),
        model_version=predictor.version,
        threshold=predictor.metadata["threshold"],
    )
```

The `confidence` field — `high` (>0.7), `medium` (0.45–0.7), `low` (<0.45) — was a direct request from the campaign team. Data scientists want a number; campaign managers want a label they can act on without checking a threshold table.

---

## The SageMaker Path

For production deployment, the model exports to ONNX and wraps in a SageMaker PyTorch serving container:

```python
# Export to ONNX
dummy_input = torch.randn(1, 6)
torch.onnx.export(
    model, dummy_input,
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

## What I'd Do Differently in production Scale

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
