"""propensity_model — TabTransformer propensity scoring for loyalty customers."""

from propensity_model.evaluator import ModelEvaluator
from propensity_model.model import PropensityModel, TabTransformerConfig, TabTransformerNet
from propensity_model.predictor import PropensityPredictor, PropensityResult
from propensity_model.trainer import ModelTrainer

__all__ = [
    "PropensityModel",
    "TabTransformerConfig",
    "TabTransformerNet",
    "ModelTrainer",
    "ModelEvaluator",
    "PropensityPredictor",
    "PropensityResult",
]
