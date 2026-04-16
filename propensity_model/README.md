# propensity_model

Trains and serves XGBoost models that score each customer's churn and upsell propensity. Model artefacts and metrics are tracked in MLflow.

## Purpose

Produce calibrated probability scores for two tasks:

1. **Churn propensity** — likelihood a customer becomes inactive in the next 30 days
2. **Upsell propensity** — likelihood a customer upgrades their loyalty tier

Scores are consumed by `llm_generator` to prioritise and personalise outreach.

## Inputs

- Point-in-time feature DataFrames from `feature_store.FeatureReader`
- Ground-truth labels from `transactions` (churn flag, tier change events)
- Hyperparameter config from `shared.Settings` or MLflow experiment config

## Outputs

- Trained model artefacts logged to MLflow Model Registry
- Propensity scores written back to `feature_store` (Redis + Postgres)
- Evaluation report (AUC-ROC, Brier score, calibration curve) logged to MLflow

## Key Classes

| Class             | Module         | Responsibility |
| ----------------- | -------------- | ----------------------------------------------- |
| `PropensityModel` | `model.py`     | XGBoost wrapper with calibration and prediction |
| `ModelTrainer`    | `trainer.py`   | Feature prep, cross-validation, MLflow logging  |
| `ModelEvaluator`  | `evaluator.py` | Compute and log evaluation metrics              |
