"""Evaluation helpers: how this project measures whether a model is any good.

Week 4 adds :mod:`src.evaluation.metrics` with :func:`evaluate_model` and
:func:`cross_validated_accuracy`. Later weeks extend that module rather than
replacing it.

Week 8 completes the picture in three ways: :func:`confusion_frame` and the new
per-class keys of :func:`evaluate_model` say *where* the errors are,
:mod:`src.evaluation.tuning` searches hyperparameters with cross-validation
inside it, and :mod:`src.evaluation.explainability` explains a model's reliance
on each feature (:func:`permutation_feature_importance`) and a single
prediction (:func:`explain_prediction`).
"""

from src.evaluation.explainability import (
    DEFAULT_BACKGROUND_SAMPLES,
    DEFAULT_N_REPEATS,
    EXPLAINER_BACKEND,
    SHAP_AVAILABLE,
    explain_prediction,
    permutation_feature_importance,
)
from src.evaluation.metrics import (
    DEFAULT_CV_FOLDS,
    build_cv,
    confusion_frame,
    cross_validated_accuracy,
    evaluate_model,
)
from src.evaluation.tuning import (
    DEFAULT_N_ITER,
    DEFAULT_SEARCH_SCORING,
    SEARCH_STRATEGIES,
    tune_model,
)

__all__ = [
    "DEFAULT_BACKGROUND_SAMPLES",
    "DEFAULT_CV_FOLDS",
    "DEFAULT_N_ITER",
    "DEFAULT_N_REPEATS",
    "DEFAULT_SEARCH_SCORING",
    "EXPLAINER_BACKEND",
    "SEARCH_STRATEGIES",
    "SHAP_AVAILABLE",
    "build_cv",
    "confusion_frame",
    "cross_validated_accuracy",
    "evaluate_model",
    "explain_prediction",
    "permutation_feature_importance",
    "tune_model",
]
