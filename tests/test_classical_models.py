"""Tests for Week 5's three classical classifiers.

The properties checked here are the ones the week is graded on:

1. **The factories return unfitted, correctly configured estimators.** Each
   defaults to the settings the notebook and the docs quote, rejects nonsense
   arguments, and comes back unfitted so cross-validation can re-fit it per
   fold.
2. **All three share one training loop.** ``fit(X_train, y_train)`` then
   ``predict(X_test)`` works identically for logistic regression, KNN and naive
   Bayes, inside a pipeline as well as on their own.
3. **Each model behaves the way its algorithm says it should.** KNN with
   ``k = 1`` memorises its training data and is sensitive to feature scaling;
   naive Bayes is not; KNN degrades when meaningless features are added, which
   is the curse of dimensionality in miniature.
4. **All three beat the Week 4 baseline**, on synthetic data and on the real
   1,760 training rows, measured on the same cross-validation folds.

Most tests run on a small synthetic frame so they pass with or without the CSV;
those needing the real dataset are marked with ``requires_raw_dataset``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import (
    DEFAULT_RANDOM_STATE,
    EXPECTED_LABEL_COUNT,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_data,
    stratified_split,
)
from src.evaluation import cross_validated_accuracy, evaluate_model
from src.models import (
    CLASSICAL_MODEL_FACTORIES,
    DEFAULT_K_NEIGHBORS,
    DEFAULT_KNN_WEIGHTS,
    DEFAULT_LOGISTIC_C,
    DEFAULT_LOGISTIC_MAX_ITER,
    DEFAULT_VAR_SMOOTHING,
    KNN_WEIGHT_OPTIONS,
    get_baseline_model,
    get_knn,
    get_logistic_regression,
    get_naive_bayes,
)
from src.preprocessing import build_preprocessor
from tests.conftest import requires_raw_dataset

#: Accuracy a constant guess earns on the 22 balanced crops (Week 4's floor).
BALANCED_BASELINE_ACCURACY = 1 / EXPECTED_LABEL_COUNT


@pytest.fixture
def separable_frame() -> pd.DataFrame:
    """Build a deterministic four-class frame whose classes are easy to tell apart.

    Two informative features, well-separated class means and a fixed seed, so
    every model here should score far above the 1/4 a constant guess earns —
    the same shape of problem as the real dataset, at a size that keeps the
    tests fast.
    """
    rng = np.random.default_rng(seed=0)
    rows_per_class = 60
    frames = []
    centres = [(0.0, 0.0), (6.0, 0.0), (0.0, 6.0), (6.0, 6.0)]
    for (first, second), name in zip(centres, ["one", "two", "three", "four"], strict=True):
        frames.append(
            pd.DataFrame(
                {
                    "first": rng.normal(first, 1.0, rows_per_class),
                    "second": rng.normal(second, 1.0, rows_per_class),
                    TARGET_COLUMN: name,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def features_and_labels(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a fixture frame into its feature columns and its label column."""
    feature_names = [name for name in frame.columns if name != TARGET_COLUMN]
    return frame[feature_names], frame[TARGET_COLUMN]


def model_pipeline(model) -> Pipeline:
    """Put the Week 3 preprocessor in front of a model, as the notebook does."""
    return Pipeline([("preprocess", build_preprocessor(["first", "second"])), ("model", model)])


ALL_FACTORIES = [get_logistic_regression, get_knn, get_naive_bayes]


# --------------------------------------------------------------------------
# The factories: types, defaults and validation
# --------------------------------------------------------------------------


def test_get_logistic_regression_returns_a_logistic_regression():
    assert isinstance(get_logistic_regression(), LogisticRegression)


def test_get_knn_returns_a_kneighbors_classifier():
    assert isinstance(get_knn(), KNeighborsClassifier)


def test_get_naive_bayes_returns_a_gaussian_nb():
    assert isinstance(get_naive_bayes(), GaussianNB)


@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_factories_return_unfitted_models(factory):
    """Unfitted is the contract: cross-validation fits a fresh clone per fold."""
    with pytest.raises(NotFittedError):
        factory().predict([[0.0, 0.0]])


def test_logistic_regression_defaults():
    model = get_logistic_regression()

    assert model.C == DEFAULT_LOGISTIC_C == 1.0
    assert model.max_iter == DEFAULT_LOGISTIC_MAX_ITER == 1_000
    assert model.random_state == DEFAULT_RANDOM_STATE


def test_knn_defaults():
    model = get_knn()

    assert model.n_neighbors == DEFAULT_K_NEIGHBORS == 5
    assert model.weights == DEFAULT_KNN_WEIGHTS == "uniform"


def test_naive_bayes_defaults():
    assert get_naive_bayes().var_smoothing == DEFAULT_VAR_SMOOTHING == pytest.approx(1e-9)


def test_factories_pass_their_arguments_through():
    assert get_logistic_regression(C=0.25, max_iter=50).C == 0.25
    assert get_logistic_regression(C=0.25, max_iter=50).max_iter == 50
    assert get_knn(n_neighbors=11, weights="distance").n_neighbors == 11
    assert get_knn(n_neighbors=11, weights="distance").weights == "distance"
    assert get_naive_bayes(var_smoothing=1e-6).var_smoothing == pytest.approx(1e-6)


@pytest.mark.parametrize("bad_c", [0.0, -1.0, -0.5])
def test_logistic_regression_rejects_non_positive_c(bad_c):
    with pytest.raises(ValueError, match="strictly positive"):
        get_logistic_regression(C=bad_c)


@pytest.mark.parametrize("bad_max_iter", [0, -10])
def test_logistic_regression_rejects_a_non_positive_iteration_budget(bad_max_iter):
    with pytest.raises(ValueError, match="at least 1"):
        get_logistic_regression(max_iter=bad_max_iter)


@pytest.mark.parametrize("bad_k", [0, -1, -5])
def test_knn_rejects_fewer_than_one_neighbour(bad_k):
    with pytest.raises(ValueError, match="at least 1"):
        get_knn(n_neighbors=bad_k)


@pytest.mark.parametrize("bad_weights", ["closest", "UNIFORM", ""])
def test_knn_rejects_an_unsupported_weighting(bad_weights):
    with pytest.raises(ValueError, match="Unsupported KNN weighting"):
        get_knn(weights=bad_weights)


@pytest.mark.parametrize("weights", KNN_WEIGHT_OPTIONS)
def test_knn_accepts_every_supported_weighting(weights, separable_frame):
    X, y = features_and_labels(separable_frame)
    model = get_knn(weights=weights).fit(X, y)

    assert len(model.predict(X)) == len(y)


def test_naive_bayes_rejects_negative_smoothing():
    with pytest.raises(ValueError, match="must not be negative"):
        get_naive_bayes(var_smoothing=-1e-9)


def test_factory_registry_lists_the_three_week_five_models():
    assert list(CLASSICAL_MODEL_FACTORIES) == ["logistic_regression", "knn", "naive_bayes"]
    assert [factory() for factory in CLASSICAL_MODEL_FACTORIES.values()]


# --------------------------------------------------------------------------
# The shared training loop: fit(X_train, y_train) -> predict(X_test)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_every_model_supports_the_same_fit_predict_loop(factory, separable_frame):
    X, y = features_and_labels(separable_frame)
    model = factory()

    fitted = model.fit(X, y)
    predictions = fitted.predict(X)

    assert fitted is model
    assert len(predictions) == len(y)
    assert set(np.unique(predictions)) <= set(y.unique())


@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_every_model_works_behind_the_week_three_preprocessor(factory, separable_frame):
    X, y = features_and_labels(separable_frame)
    pipeline = model_pipeline(factory()).fit(X, y)

    assert evaluate_model(pipeline, X, y)["accuracy"] > 0.9


@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_every_model_is_reproducible(factory, separable_frame):
    X, y = features_and_labels(separable_frame)

    first = factory().fit(X, y).predict(X)
    second = factory().fit(X, y).predict(X)

    assert np.array_equal(first, second)


@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_cross_validation_leaves_the_estimator_unfitted(factory, separable_frame):
    X, y = features_and_labels(separable_frame)
    model = factory()

    cross_validated_accuracy(model, X, y)

    with pytest.raises(NotFittedError):
        model.predict(X)


@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_every_model_uses_the_features(factory, separable_frame):
    """Unlike the baseline, scrambling the features must change the answers."""
    X, y = features_and_labels(separable_frame)
    model = factory().fit(X, y)

    scrambled = X.sample(frac=1.0, random_state=0).reset_index(drop=True)
    assert not np.array_equal(model.predict(X), model.predict(scrambled))


# --------------------------------------------------------------------------
# Algorithm-specific behaviour
# --------------------------------------------------------------------------


def test_one_nearest_neighbour_memorises_its_training_data(separable_frame):
    """k=1 scores a perfect — and completely uninformative — training accuracy."""
    X, y = features_and_labels(separable_frame)
    model = get_knn(n_neighbors=1).fit(X, y)

    assert evaluate_model(model, X, y)["accuracy"] == pytest.approx(1.0)


def test_a_large_k_smooths_the_boundary_towards_ignorance(separable_frame):
    """As k approaches the training size, KNN converges on the majority guess."""
    X, y = features_and_labels(separable_frame)

    sharp = cross_validated_accuracy(get_knn(n_neighbors=5), X, y)["mean"]
    blunt = cross_validated_accuracy(get_knn(n_neighbors=len(X) - len(X) // 5 - 1), X, y)["mean"]

    assert sharp > blunt
    assert blunt < 0.6


def test_knn_is_sensitive_to_feature_scaling(separable_frame):
    """Changing a column's units changes which neighbours count as near."""
    X, y = features_and_labels(separable_frame)
    stretched = X.assign(second=X["second"] * 1_000)

    on_raw = get_knn().fit(X, y).predict(X)
    on_stretched = get_knn().fit(stretched, y).predict(stretched)

    assert not np.array_equal(on_raw, on_stretched)


def test_naive_bayes_is_insensitive_to_standardisation(separable_frame):
    """Standardising moves every class's mean and variance for a column alike.

    The invariance is exact only up to ``var_smoothing``, which scikit-learn
    scales by the largest feature variance in the data — so a wild rescale can
    still nudge a few rows, while standardisation cannot.
    """
    X, y = features_and_labels(separable_frame)
    standardised = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    on_raw = get_naive_bayes().fit(X, y).predict(X)
    on_standardised = get_naive_bayes().fit(standardised, y).predict(standardised)

    assert np.array_equal(on_raw, on_standardised)


def test_knn_degrades_when_meaningless_features_are_added(separable_frame):
    """The curse of dimensionality: noise columns dilute the distance measure."""
    X, y = features_and_labels(separable_frame)
    rng = np.random.default_rng(seed=1)
    noise = pd.DataFrame(
        rng.normal(0.0, 3.0, (len(X), 100)),
        columns=[f"noise_{i}" for i in range(100)],
    )
    noisy = pd.concat([X, noise], axis=1)

    informative = cross_validated_accuracy(get_knn(), X, y)["mean"]
    diluted = cross_validated_accuracy(get_knn(), noisy, y)["mean"]

    assert diluted < informative - 0.3


def test_naive_bayes_stores_one_mean_and_variance_per_feature_per_class(separable_frame):
    """The independence assumption is visible in what the fitted model holds."""
    X, y = features_and_labels(separable_frame)
    model = get_naive_bayes().fit(X, y)

    assert model.theta_.shape == (y.nunique(), X.shape[1])
    assert model.var_.shape == (y.nunique(), X.shape[1])


def test_logistic_regression_learns_one_weight_per_feature_per_class(separable_frame):
    """A linear model is exactly a coefficient matrix and an intercept vector."""
    X, y = features_and_labels(separable_frame)
    model = get_logistic_regression().fit(X, y)

    assert model.coef_.shape == (y.nunique(), X.shape[1])
    assert model.intercept_.shape == (y.nunique(),)


def test_probabilities_sum_to_one_for_every_model(separable_frame):
    X, y = features_and_labels(separable_frame)

    for factory in ALL_FACTORIES:
        probabilities = factory().fit(X, y).predict_proba(X)
        assert probabilities.shape == (len(X), y.nunique())
        assert np.allclose(probabilities.sum(axis=1), 1.0)


# --------------------------------------------------------------------------
# The comparison: all three against the Week 4 baseline, on the same folds
# --------------------------------------------------------------------------


@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_every_model_beats_the_baseline_on_synthetic_data(factory, separable_frame):
    X, y = features_and_labels(separable_frame)

    baseline = cross_validated_accuracy(get_baseline_model(), X, y)["mean"]
    model = cross_validated_accuracy(model_pipeline(factory()), X, y)["mean"]

    assert baseline == pytest.approx(0.25, abs=0.01)
    assert model > baseline + 0.5


def test_comparing_models_uses_identical_folds(separable_frame):
    """A fair comparison means the same seed, the same splitter, the same rows."""
    X, y = features_and_labels(separable_frame)

    first = cross_validated_accuracy(model_pipeline(get_knn()), X, y)
    again = cross_validated_accuracy(model_pipeline(get_knn()), X, y)

    assert np.array_equal(first["scores"], again["scores"])
    assert first["n_splits"] == again["n_splits"]


# --------------------------------------------------------------------------
# The real dataset: the Week 5 result the notebook reports
# --------------------------------------------------------------------------


@requires_raw_dataset
@pytest.mark.parametrize("name", list(CLASSICAL_MODEL_FACTORIES))
def test_every_model_beats_the_real_baseline_by_a_wide_margin(name):
    crops = load_data()
    train, _ = stratified_split(crops)
    X, y = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]
    pipeline = Pipeline(
        [("preprocess", build_preprocessor()), ("model", CLASSICAL_MODEL_FACTORIES[name]())]
    )

    result = cross_validated_accuracy(pipeline, X, y)

    assert result["mean"] > BALANCED_BASELINE_ACCURACY
    assert result["mean"] > 0.90


@requires_raw_dataset
def test_naive_bayes_is_the_best_of_the_three_this_week():
    """The Week 5 headline: the simplest model wins, at ~99.5%."""
    crops = load_data()
    train, _ = stratified_split(crops)
    X, y = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]

    means = {
        name: cross_validated_accuracy(
            Pipeline([("preprocess", build_preprocessor()), ("model", factory())]), X, y
        )["mean"]
        for name, factory in CLASSICAL_MODEL_FACTORIES.items()
    }

    assert max(means, key=means.get) == "naive_bayes"
    assert means["naive_bayes"] == pytest.approx(0.995, abs=0.01)


@requires_raw_dataset
def test_the_real_models_are_reproducible():
    crops = load_data()
    train, _ = stratified_split(crops)
    X, y = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]

    first = cross_validated_accuracy(_real_pipeline(get_naive_bayes()), X, y)
    second = cross_validated_accuracy(_real_pipeline(get_naive_bayes()), X, y)

    assert np.array_equal(first["scores"], second["scores"])


def _real_pipeline(model) -> Pipeline:
    """Week 3 preprocessor plus a model, over the project's seven features."""
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])
