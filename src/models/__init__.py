"""Model factories: the estimators this project trains and compares.

Week 4 adds :mod:`src.models.baseline`, whose :func:`get_baseline_model`
returns the deliberately unintelligent :class:`~sklearn.dummy.DummyClassifier`
that every later model has to beat.

Week 5 adds :mod:`src.models.classical_models` with the first three real
classifiers — :func:`get_logistic_regression`, :func:`get_knn` and
:func:`get_naive_bayes` — all returned unfitted, so the same
``fit``/``predict`` loop and the same cross-validation folds apply to every one
of them.

Week 6 extends the same module with :func:`get_svm` — a margin-based model whose
kernel decides whether its boundary is flat or curved — and
:func:`get_decision_tree`, the rule-based model whose ``max_depth`` makes the
bias-variance tradeoff visible.
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
    DEFAULT_SVM_C,
    DEFAULT_SVM_GAMMA,
    DEFAULT_SVM_KERNEL,
    DEFAULT_TREE_CRITERION,
    DEFAULT_TREE_MAX_DEPTH,
    DEFAULT_TREE_MIN_SAMPLES_LEAF,
    DEFAULT_VAR_SMOOTHING,
    KNN_WEIGHT_OPTIONS,
    SVM_KERNEL_OPTIONS,
    TREE_CRITERION_OPTIONS,
    get_decision_tree,
    get_knn,
    get_logistic_regression,
    get_naive_bayes,
    get_svm,
)

__all__ = [
    "BASELINE_STRATEGIES",
    "CLASSICAL_MODEL_FACTORIES",
    "DEFAULT_BASELINE_STRATEGY",
    "DEFAULT_KNN_WEIGHTS",
    "DEFAULT_K_NEIGHBORS",
    "DEFAULT_LOGISTIC_C",
    "DEFAULT_LOGISTIC_MAX_ITER",
    "DEFAULT_SVM_C",
    "DEFAULT_SVM_GAMMA",
    "DEFAULT_SVM_KERNEL",
    "DEFAULT_TREE_CRITERION",
    "DEFAULT_TREE_MAX_DEPTH",
    "DEFAULT_TREE_MIN_SAMPLES_LEAF",
    "DEFAULT_VAR_SMOOTHING",
    "KNN_WEIGHT_OPTIONS",
    "SVM_KERNEL_OPTIONS",
    "TREE_CRITERION_OPTIONS",
    "get_baseline_model",
    "get_decision_tree",
    "get_knn",
    "get_logistic_regression",
    "get_naive_bayes",
    "get_svm",
]
