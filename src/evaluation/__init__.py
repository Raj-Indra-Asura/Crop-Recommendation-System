"""Evaluation helpers: how this project measures whether a model is any good.

Week 4 adds :mod:`src.evaluation.metrics` with :func:`evaluate_model` and
:func:`cross_validated_accuracy`. Later weeks extend that module rather than
replacing it.
"""

from src.evaluation.metrics import (
    DEFAULT_CV_FOLDS,
    build_cv,
    cross_validated_accuracy,
    evaluate_model,
)

__all__ = [
    "DEFAULT_CV_FOLDS",
    "build_cv",
    "cross_validated_accuracy",
    "evaluate_model",
]
