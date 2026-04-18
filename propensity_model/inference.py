"""SageMaker entry point — TorchScript model, custom input/output handlers."""
import json
import os
from pathlib import Path
import torch

_CHANNEL_MAP = {"in-store": 0.0, "mobile": 1.0, "web": 2.0}


def model_fn(model_dir):
    return torch.jit.load(str(Path(model_dir) / "model.pt"), map_location="cpu")


def input_fn(request_body, content_type="application/json"):
    data = json.loads(request_body)
    channel_val = _CHANNEL_MAP.get(str(data.get("channel_preference", "web")).lower(), 2.0)
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
    return {"propensity_score": score, "model_version": os.environ.get("MODEL_VERSION", "1")}


def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction), accept
