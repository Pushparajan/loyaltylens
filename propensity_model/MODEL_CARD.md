# Model Card: LoyaltyLens Propensity Model

## Model Details

| Field | Value |
|---|---|
| **Model name** | `propensity_v{version}` |
| **Architecture** | TabTransformer-lite (PyTorch) |
| **Task** | Binary classification — offer redemption propensity |
| **Output** | Scalar probability ∈ [0, 1] |
| **Framework** | PyTorch ≥ 2.0 |
| **Author** | LoyaltyLens ML Team |
| **License** | Internal / proprietary |

### Architecture Summary

Each of the 6 input features is independently projected from ℝ¹ to ℝ⁶⁴ via a dedicated
`nn.Linear` layer, forming a sequence of length 6. A 2-layer `TransformerEncoder`
(d_model=64, nhead=4, dim_feedforward=128) attends across features. The output sequence is
flattened and passed through an MLP (Linear → ReLU → Linear → Sigmoid) to produce a scalar
propensity score.

```
Input (B, 6)
  └─ Per-feature Linear(1→64) ×6  →  (B, 6, 64)
       └─ TransformerEncoder ×2   →  (B, 6, 64)
            └─ Flatten            →  (B, 384)
                 └─ MLP → Sigmoid →  (B,)
```

---

## Intended Use

**Primary use case:** Rank customers by their likelihood to redeem a loyalty offer in the next
30-day window, enabling targeted campaign prioritisation.

**In-scope:**
- Scoring existing loyalty-programme customers with at least one recorded event.
- Batch scoring pipelines (nightly / weekly campaigns).
- Real-time single-customer scoring via the `/predict` API endpoint.

**Out-of-scope:**
- New customers with no event history (cold-start scenario).
- Predicting purchase *amount* or *category* (this model outputs a binary propensity only).
- Decisions with legal or financial consequences without human review.

---

## Training Data

| Attribute | Detail |
|---|---|
| **Source** | Internal DuckDB feature store (versioned snapshots) |
| **Granularity** | One row per customer per feature-store version |
| **Time window** | Features computed over 30-day and 90-day rolling windows |
| **Label** | `offer_redemption_rate > 0.30` → positive class (1), else negative (0) |
| **Split** | 70% train / 15% validation / 15% test, stratified by label |

### Features

| Feature | Type | Description |
|---|---|---|
| `recency_days` | int | Days since most recent event |
| `frequency_30d` | int | Total event count in last 30 days |
| `monetary_90d` | float | Sum of purchase amounts in last 90 days (£) |
| `offer_redemption_rate` | float | redeems / offer_views; 0.0 when no offers seen |
| `channel_preference` | categorical (encoded) | Modal channel: in-store=0, mobile=1, web=2 |
| `engagement_score` | float ∈ [0, 1] | Weighted composite engagement index |

---

## Training Procedure

| Hyperparameter | Default |
|---|---|
| Optimiser | Adam (lr=1e-3, weight_decay=1e-4) |
| Loss | Binary Cross-Entropy |
| Epochs | 20 (with early stopping, patience=3) |
| Batch size | 64 |
| d_model | 64 |
| nhead | 4 |
| Encoder layers | 2 |
| dim_feedforward | 128 |
| Dropout | 0.1 |

Early stopping monitors validation AUC-ROC; the checkpoint with the highest validation AUC is
saved.

---

## Evaluation Results

Metrics are logged per epoch to MLflow and reported on the held-out test split at the end of
each training run.

| Metric | Description |
|---|---|
| **AUC-ROC** | Primary ranking metric; target ≥ 0.75 |
| **Precision** | At decision threshold (default 0.5) |
| **Recall** | At decision threshold (default 0.5) |
| **BCE Loss** | Training and validation loss per epoch |

> **Note:** Actual numbers depend on the feature-store snapshot used for training.
> Check the MLflow experiment `loyaltylens` for the run corresponding to a specific version.

---

## Limitations

1. **Class imbalance:** If fewer than 30% of customers redeem offers, the model may exhibit
   low recall on the positive class. Consider adjusting the decision threshold per campaign.
2. **Feature drift:** The model assumes stable feature distributions. Significant changes in
   offer mechanics or customer behaviour may degrade performance over time.
3. **Cold-start:** Customers with `frequency_30d = 0` and `monetary_90d = 0` are effectively
   new; propensity scores for these customers are unreliable.
4. **Channel encoding:** `channel_preference` is ordinally encoded; the model implicitly
   treats in-store < mobile < web as ordered. This is an approximation.
5. **Temporal leakage:** `offer_redemption_rate` is used both as a feature and to derive the
   label. When serving real-time predictions, this feature must be computed from *past* events
   only (not the current campaign being scored).

---

## Bias Considerations

- **Recency bias:** The feature set favours recently active customers. Customers who are
  seasonally inactive (e.g., travel loyalty programmes) may be systematically under-scored.
- **Monetary proxy:** `monetary_90d` may correlate with income, which could introduce
  socioeconomic bias. Scores should not be the sole basis for excluding customers from
  loyalty benefits.
- **Channel access:** Customers without mobile or web access will have `channel_preference`
  skewed toward in-store (encoded as 0), which may disadvantage groups with lower digital
  access.

Fairness audits segmented by demographic proxies (where available and lawful) are recommended
before deploying new versions in regulated markets.

---

## Usage

```python
from propensity_model.predictor import PropensityPredictor

predictor = PropensityPredictor().load(version="1")

result = predictor.predict({
    "customer_id": "cust_001",
    "recency_days": 5,
    "frequency_30d": 12,
    "monetary_90d": 250.0,
    "offer_redemption_rate": 0.45,
    "channel_preference": "mobile",
    "engagement_score": 0.72,
})

print(result.propensity_score)  # e.g. 0.83
print(result.label)             # 1
```

---

## Version History

| Version | Notes |
|---|---|
| v1 | Initial TabTransformer-lite release, replacing XGBoost baseline |
