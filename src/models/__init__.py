"""Model factories: the estimators this project trains and compares.

Week 4 adds :mod:`src.models.baseline`, whose :func:`get_baseline_model`
returns the deliberately unintelligent :class:`~sklearn.dummy.DummyClassifier`
that every later model has to beat.
"""

from src.models.baseline import (
    BASELINE_STRATEGIES,
    DEFAULT_BASELINE_STRATEGY,
    get_baseline_model,
)

__all__ = [
    "BASELINE_STRATEGIES",
    "DEFAULT_BASELINE_STRATEGY",
    "get_baseline_model",
]
