"""Model factories: the estimators this project trains and compares.

Week 4 adds :mod:`src.models.baseline`, whose :func:`get_baseline_model`
returns the deliberately unintelligent :class:`~sklearn.dummy.DummyClassifier`
that every later model has to beat.

Week 5 adds :mod:`src.models.classical_models` with the first three real
classifiers — :func:`get_logistic_regression`, :func:`get_knn` and
:func:`get_naive_bayes` — all returned unfitted, so the same
``fit``/``predict`` loop and the same cross-validation folds apply to every one
of them.
"""

from src.models.baseline import (
    BASELINE_STRATEGIES,
    DEFAULT_BASELINE_STRATEGY,
    get_baseline_model,
)
from src.models.classical_models import (
    CLASSICAL_MODEL_FACTORIES,
    DEFAULT_K_NEIGHBORS,
    DEFAULT_KNN_WEIGHTS,
    DEFAULT_LOGISTIC_C,
    DEFAULT_LOGISTIC_MAX_ITER,
    DEFAULT_VAR_SMOOTHING,
    KNN_WEIGHT_OPTIONS,
    get_knn,
    get_logistic_regression,
    get_naive_bayes,
)

__all__ = [
    "BASELINE_STRATEGIES",
    "CLASSICAL_MODEL_FACTORIES",
    "DEFAULT_BASELINE_STRATEGY",
    "DEFAULT_KNN_WEIGHTS",
    "DEFAULT_K_NEIGHBORS",
    "DEFAULT_LOGISTIC_C",
    "DEFAULT_LOGISTIC_MAX_ITER",
    "DEFAULT_VAR_SMOOTHING",
    "KNN_WEIGHT_OPTIONS",
    "get_baseline_model",
    "get_knn",
    "get_logistic_regression",
    "get_naive_bayes",
]
