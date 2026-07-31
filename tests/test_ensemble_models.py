"""Tests for the Week 7 ensembles: random forest and gradient boosting.

The properties checked here are the ones the week is graded on:

1. **The factories return unfitted, correctly configured estimators**, defaulting
   to the settings the notebook and the docs quote and rejecting nonsense
   arguments — the same contract every earlier factory keeps.
2. **The boosting factory works with or without XGBoost.** Which implementation
   is returned depends on the environment, so the tests assert the *contract*
   (an unfitted classifier that trains, predicts and reports feature
   importances) rather than a class, and exercise the scikit-learn fallback
   explicitly by disabling the import flag.
3. **Bagging really does reduce variance.** A forest of unlimited trees is more
   stable across resamples of the training data than a single unlimited tree,
   which is the entire argument for averaging them.
4. **Feature randomness really does decorrelate the trees.** A forest whose
   trees all see every feature (``max_features=None``) is plain bagging; the
   default ``"sqrt"`` forest builds visibly different trees.
5. **Boosting really is sequential error-correction.** More rounds monotonically
   improve the training fit, and a single-round ensemble is much weaker than a
   hundred-round one — the opposite of a forest, where one tree is already a
   whole model.
6. **``feature_importances_`` means what Week 7 says it means**: one
   non-negative number per feature, summing to 1, larger for a feature that
   separates the classes than for a column of pure noise.
7. **The ensembles beat the tree they are made of, and every other Week 5/6
   single model except Gaussian naive Bayes**, on the real 1,760 training rows
   and on the same cross-validation folds. The naive Bayes comparison is
   asserted as a *tie inside the fold spread*, because that is what the numbers
   support and Week 7's honest headline says so.

Most tests run on a small synthetic frame so they pass with or without the CSV;
those needing the real dataset are marked with ``requires_raw_dataset``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

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
    DEFAULT_BOOSTING_LEARNING_RATE,
    DEFAULT_BOOSTING_MAX_DEPTH,
    DEFAULT_BOOSTING_N_ESTIMATORS,
    DEFAULT_FOREST_MAX_DEPTH,
    DEFAULT_FOREST_MAX_FEATURES,
    DEFAULT_FOREST_N_ESTIMATORS,
    ENSEMBLE_MODEL_FACTORIES,
    FOREST_MAX_FEATURES_OPTIONS,
    GRADIENT_BOOSTING_BACKEND,
    XGBOOST_AVAILABLE,
    ensemble_models,
    get_decision_tree,
    get_gradient_boosting,
    get_knn,
    get_logistic_regression,
    get_naive_bayes,
    get_random_forest,
    get_svm,
)
from src.preprocessing import build_preprocessor
from tests.conftest import requires_raw_dataset

#: Accuracy a constant guess earns on the 22 balanced crops (Week 4's floor).
BALANCED_BASELINE_ACCURACY = 1 / EXPECTED_LABEL_COUNT

#: Boosting is the slow model in this project, so tests that only need "it
#: trains and predicts" use a handful of rounds instead of the default 100.
FAST_ROUNDS = 10


@pytest.fixture
def separable_frame() -> pd.DataFrame:
    """Build a deterministic four-class frame whose classes are easy to tell apart.

    Two informative features, well-separated class means and a fixed seed — the
    same shape of problem as the real dataset, at a size that keeps the tests
    fast.
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


@pytest.fixture
def noisy_frame() -> pd.DataFrame:
    """Build a frame with one informative feature and one column of pure noise.

    ``feature_importances_`` should rank the informative column far above the
    noise; if it does not, the number is not measuring what Week 7 claims.
    """
    rng = np.random.default_rng(seed=7)
    rows_per_class = 80
    signal = np.concatenate(
        [rng.normal(0.0, 0.5, rows_per_class), rng.normal(5.0, 0.5, rows_per_class)]
    )
    return pd.DataFrame(
        {
            "signal": signal,
            "noise": rng.normal(0.0, 1.0, 2 * rows_per_class),
            TARGET_COLUMN: ["low"] * rows_per_class + ["high"] * rows_per_class,
        }
    )


@pytest.fixture
def overlapping_frame() -> pd.DataFrame:
    """Build a four-class frame whose classes overlap, so a tree has real choices.

    ``separable_frame`` is too easy for the variance argument: every resample of
    it produces almost the same tree, so there is no instability for bagging to
    remove. Moving the class centres closer together and widening their spread
    leaves genuinely ambiguous rows near the boundaries, which is where a single
    tree's answer flips from resample to resample.
    """
    rng = np.random.default_rng(seed=0)
    rows_per_class = 80
    frames = []
    centres = [(0.0, 0.0), (3.0, 0.0), (0.0, 3.0), (3.0, 3.0)]
    for (first, second), name in zip(centres, ["one", "two", "three", "four"], strict=True):
        frames.append(
            pd.DataFrame(
                {
                    "first": rng.normal(first, 1.6, rows_per_class),
                    "second": rng.normal(second, 1.6, rows_per_class),
                    TARGET_COLUMN: name,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def features_and_labels(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a fixture frame into its feature columns and its label column."""
    feature_names = [name for name in frame.columns if name != TARGET_COLUMN]
    return frame[feature_names], frame[TARGET_COLUMN]


def model_pipeline(model, features: list[str]) -> Pipeline:
    """Put the Week 3 preprocessor in front of a model, as the notebook does."""
    return Pipeline([("preprocess", build_preprocessor(features)), ("model", model)])


def fast_boosting(**overrides):
    """A gradient booster with few rounds, for tests that do not measure accuracy."""
    return get_gradient_boosting(n_estimators=FAST_ROUNDS, **overrides)


ALL_FACTORIES = [get_random_forest, fast_boosting]


# --------------------------------------------------------------------------
# The factories: types, defaults and validation
# --------------------------------------------------------------------------


def test_get_random_forest_returns_a_random_forest():
    assert isinstance(get_random_forest(), RandomForestClassifier)


def test_get_gradient_boosting_returns_a_classifier_either_way():
    """Which class comes back depends on the environment; the contract does not."""
    model = get_gradient_boosting()

    if XGBOOST_AVAILABLE:
        assert isinstance(model, ensemble_models.XGBoostStringLabelClassifier)
    else:
        assert isinstance(model, GradientBoostingClassifier)
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_gradient_boosting_backend_matches_the_availability_flag():
    assert GRADIENT_BOOSTING_BACKEND == ("xgboost" if XGBOOST_AVAILABLE else "sklearn")
    assert GRADIENT_BOOSTING_BACKEND in ("xgboost", "sklearn")


def test_gradient_boosting_falls_back_to_sklearn_without_xgboost(monkeypatch):
    """The fallback is the whole point: no environment is blocked on an install."""
    monkeypatch.setattr(ensemble_models, "XGBOOST_AVAILABLE", False)

    model = ensemble_models.get_gradient_boosting()

    assert isinstance(model, GradientBoostingClassifier)
    assert model.n_estimators == DEFAULT_BOOSTING_N_ESTIMATORS


@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_factories_return_unfitted_models(factory):
    """Unfitted is the contract: cross-validation fits a fresh clone per fold."""
    with pytest.raises((NotFittedError, AttributeError)):
        factory().predict([[0.0, 0.0]])


def test_random_forest_defaults():
    model = get_random_forest()

    assert model.n_estimators == DEFAULT_FOREST_N_ESTIMATORS == 100
    assert model.max_depth is DEFAULT_FOREST_MAX_DEPTH is None
    assert model.max_features == DEFAULT_FOREST_MAX_FEATURES == "sqrt"
    assert model.random_state == DEFAULT_RANDOM_STATE


def test_gradient_boosting_defaults():
    model = get_gradient_boosting()

    assert model.n_estimators == DEFAULT_BOOSTING_N_ESTIMATORS == 100
    assert model.learning_rate == DEFAULT_BOOSTING_LEARNING_RATE == pytest.approx(0.1)
    assert model.max_depth == DEFAULT_BOOSTING_MAX_DEPTH == 3
    assert model.random_state == DEFAULT_RANDOM_STATE


def test_factories_pass_their_arguments_through():
    forest = get_random_forest(n_estimators=7, max_depth=4, max_features=2)
    boosting = get_gradient_boosting(n_estimators=7, learning_rate=0.5, max_depth=1)

    assert (forest.n_estimators, forest.max_depth, forest.max_features) == (7, 4, 2)
    assert (boosting.n_estimators, boosting.max_depth) == (7, 1)
    assert boosting.learning_rate == pytest.approx(0.5)


@pytest.mark.parametrize("bad_count", [0, -1, -100])
def test_random_forest_rejects_fewer_than_one_tree(bad_count):
    with pytest.raises(ValueError, match="at least 1"):
        get_random_forest(n_estimators=bad_count)


@pytest.mark.parametrize("bad_depth", [0, -3])
def test_random_forest_rejects_a_non_positive_depth(bad_depth):
    with pytest.raises(ValueError, match="at least 1"):
        get_random_forest(max_depth=bad_depth)


@pytest.mark.parametrize("bad_max_features", ["all", "SQRT", ""])
def test_random_forest_rejects_an_unknown_max_features_string(bad_max_features):
    with pytest.raises(ValueError, match="Unsupported `max_features`"):
        get_random_forest(max_features=bad_max_features)


@pytest.mark.parametrize("bad_max_features", [0, -1, -0.5])
def test_random_forest_rejects_a_non_positive_max_features_number(bad_max_features):
    with pytest.raises(ValueError, match="must be positive"):
        get_random_forest(max_features=bad_max_features)


@pytest.mark.parametrize("max_features", FOREST_MAX_FEATURES_OPTIONS)
def test_random_forest_accepts_every_supported_max_features(max_features, separable_frame):
    X, y = features_and_labels(separable_frame)

    model = get_random_forest(n_estimators=5, max_features=max_features).fit(X, y)

    assert len(model.predict(X)) == len(y)


def test_random_forest_accepts_none_max_features(separable_frame):
    """``None`` means "every feature at every split", i.e. plain bagged trees."""
    X, y = features_and_labels(separable_frame)

    model = get_random_forest(n_estimators=5, max_features=None).fit(X, y)

    assert len(model.predict(X)) == len(y)


@pytest.mark.parametrize("bad_count", [0, -1])
def test_gradient_boosting_rejects_fewer_than_one_round(bad_count):
    with pytest.raises(ValueError, match="at least 1"):
        get_gradient_boosting(n_estimators=bad_count)


@pytest.mark.parametrize("bad_rate", [0.0, -0.1])
def test_gradient_boosting_rejects_a_non_positive_learning_rate(bad_rate):
    with pytest.raises(ValueError, match="strictly positive"):
        get_gradient_boosting(learning_rate=bad_rate)


@pytest.mark.parametrize("bad_depth", [0, -2])
def test_gradient_boosting_rejects_a_non_positive_depth(bad_depth):
    with pytest.raises(ValueError, match="at least 1"):
        get_gradient_boosting(max_depth=bad_depth)


def test_ensemble_factory_registry_lists_both_models():
    assert list(ENSEMBLE_MODEL_FACTORIES) == ["random_forest", "gradient_boosting"]
    assert ENSEMBLE_MODEL_FACTORIES["random_forest"] is get_random_forest
    assert ENSEMBLE_MODEL_FACTORIES["gradient_boosting"] is get_gradient_boosting


# --------------------------------------------------------------------------
# The shared training loop
# --------------------------------------------------------------------------


@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_ensembles_train_and_predict_like_every_other_model(factory, separable_frame):
    X, y = features_and_labels(separable_frame)

    model = factory().fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert set(predictions) <= set(y)


@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_ensembles_work_inside_the_week3_pipeline(factory, separable_frame):
    X, y = features_and_labels(separable_frame)

    pipeline = model_pipeline(factory(), ["first", "second"]).fit(X, y)

    assert evaluate_model(pipeline, X, y)["accuracy"] > 0.9


@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_ensembles_can_be_cloned_unfitted(factory):
    """``clone`` is what cross-validation calls before every fold."""
    model = factory()

    copy = clone(model)

    assert copy.get_params() == model.get_params()


@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_ensembles_report_probabilities_that_sum_to_one(factory, separable_frame):
    X, y = features_and_labels(separable_frame)

    probabilities = factory().fit(X, y).predict_proba(X)

    assert probabilities.shape == (len(y), y.nunique())
    assert np.allclose(probabilities.sum(axis=1), 1.0)


@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_ensembles_keep_the_string_labels(factory, separable_frame):
    """The target here is a crop name, not an integer — including under XGBoost."""
    X, y = features_and_labels(separable_frame)

    model = factory().fit(X, y)

    assert list(model.classes_) == sorted(y.unique())
    assert isinstance(model.predict(X)[0], str)


@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_ensembles_are_reproducible(factory, separable_frame):
    X, y = features_and_labels(separable_frame)

    first = factory().fit(X, y).predict(X)
    second = factory().fit(X, y).predict(X)

    assert np.array_equal(first, second)


# --------------------------------------------------------------------------
# Bagging: why averaging many overfit trees helps
# --------------------------------------------------------------------------


def test_a_forest_is_more_stable_across_resamples_than_a_single_tree(overlapping_frame):
    """The variance argument for bagging, measured rather than asserted.

    Fit a single unlimited tree and a forest of unlimited trees on several
    bootstrap resamples of the same data, and count how often each one changes
    its answer. The forest changes less: that reduction *is* the variance
    reduction bagging buys.
    """
    X, y = features_and_labels(overlapping_frame)
    rng = np.random.default_rng(seed=3)

    def disagreement(factory) -> float:
        predictions = []
        for _ in range(5):
            rows = rng.integers(0, len(X), len(X))
            model = factory().fit(X.iloc[rows], y.iloc[rows])
            predictions.append(model.predict(X))
        stacked = np.vstack(predictions)
        return float(np.mean([len(set(column)) > 1 for column in stacked.T]))

    tree_instability = disagreement(get_decision_tree)
    forest_instability = disagreement(lambda: get_random_forest(n_estimators=30))

    assert forest_instability < tree_instability


def test_a_forest_beats_its_average_member(overlapping_frame):
    """Averaging is worth more than the average member: the point of an ensemble.

    The claim is about the *average* member, not the best one. On any given
    split some individual tree will get lucky and beat the forest; the forest's
    advantage is that it does not depend on which one that was.
    """
    X, y = features_and_labels(overlapping_frame)
    train_rows, test_rows = slice(None, None, 2), slice(1, None, 2)
    X_train, y_train = X.iloc[train_rows], y.iloc[train_rows]
    X_test, y_test = X.iloc[test_rows], y.iloc[test_rows]

    forest = get_random_forest(n_estimators=50).fit(X_train, y_train)
    forest_accuracy = evaluate_model(forest, X_test, y_test)["accuracy"]
    # Each member votes in the forest's own class order, so its integer answers
    # are decoded through `classes_` rather than through the label column.
    member_accuracies = [
        float(np.mean(forest.classes_[tree.predict(X_test).astype(int)] == y_test.to_numpy()))
        for tree in forest.estimators_
    ]

    assert forest_accuracy > float(np.mean(member_accuracies))


def test_feature_randomness_produces_different_trees(separable_frame):
    """``max_features="sqrt"`` is what turns bagged trees into a *random* forest."""
    X, y = features_and_labels(separable_frame)

    random_forest = get_random_forest(n_estimators=20, max_features="sqrt").fit(X, y)
    first_splits = {tree.tree_.feature[0] for tree in random_forest.estimators_}

    assert len(first_splits) > 1


def test_more_trees_never_hurts_much(separable_frame):
    """A forest's ``n_estimators`` is a budget, not a bias-variance dial."""
    X, y = features_and_labels(separable_frame)

    few = cross_validated_accuracy(get_random_forest(n_estimators=5), X, y, n_splits=3)["mean"]
    many = cross_validated_accuracy(get_random_forest(n_estimators=100), X, y, n_splits=3)["mean"]

    assert many >= few - 0.02


# --------------------------------------------------------------------------
# Boosting: sequential error correction
# --------------------------------------------------------------------------


def test_one_boosting_round_is_much_weaker_than_a_hundred(separable_frame):
    """Boosting members are *weak*: the strength is in the chain, not the link."""
    X, y = features_and_labels(separable_frame)

    single = get_gradient_boosting(n_estimators=1, max_depth=1).fit(X, y)
    chained = get_gradient_boosting(n_estimators=100, max_depth=1).fit(X, y)

    assert (
        evaluate_model(chained, X, y)["accuracy"] > evaluate_model(single, X, y)["accuracy"] + 0.2
    )


def test_more_rounds_improve_the_training_fit(separable_frame):
    """Each round is fitted on the error that remains, so the fit cannot get worse."""
    X, y = features_and_labels(separable_frame)

    accuracies = [
        evaluate_model(get_gradient_boosting(n_estimators=rounds, max_depth=1).fit(X, y), X, y)[
            "accuracy"
        ]
        for rounds in (1, 5, 25)
    ]

    assert accuracies == sorted(accuracies)


def test_a_smaller_learning_rate_learns_more_slowly(separable_frame):
    """``learning_rate`` shrinks each round's contribution — the brake on boosting."""
    X, y = features_and_labels(separable_frame)

    timid = get_gradient_boosting(n_estimators=3, learning_rate=0.01, max_depth=1).fit(X, y)
    eager = get_gradient_boosting(n_estimators=3, learning_rate=1.0, max_depth=1).fit(X, y)

    assert evaluate_model(timid, X, y)["accuracy"] < evaluate_model(eager, X, y)["accuracy"]


# --------------------------------------------------------------------------
# feature_importances_
# --------------------------------------------------------------------------


@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_feature_importances_are_one_normalised_number_per_feature(factory, noisy_frame):
    X, y = features_and_labels(noisy_frame)

    importances = factory().fit(X, y).feature_importances_

    assert importances.shape == (X.shape[1],)
    assert (importances >= 0).all()
    assert importances.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_feature_importances_rank_signal_above_noise(factory, noisy_frame):
    X, y = features_and_labels(noisy_frame)

    importances = pd.Series(factory().fit(X, y).feature_importances_, index=X.columns)

    assert importances["signal"] > importances["noise"]


def test_feature_importances_split_credit_between_duplicated_columns(separable_frame):
    """The limitation Week 8's permutation importance and SHAP exist to address.

    Copy an informative feature and the two copies share its importance, so
    neither looks as useful as the single column did — even though the model is
    exactly as accurate. Importance describes the fitted model, not the data.
    """
    X, y = features_and_labels(separable_frame)
    alone = pd.Series(
        get_random_forest(n_estimators=30).fit(X, y).feature_importances_, index=X.columns
    )
    X_duplicated = X.assign(first_copy=X["first"])
    duplicated = pd.Series(
        get_random_forest(n_estimators=30).fit(X_duplicated, y).feature_importances_,
        index=X_duplicated.columns,
    )

    assert duplicated["first"] < alone["first"]
    assert duplicated["first"] + duplicated["first_copy"] > duplicated["first"]


# --------------------------------------------------------------------------
# On the real dataset
# --------------------------------------------------------------------------


@requires_raw_dataset
@pytest.mark.parametrize("name", list(ENSEMBLE_MODEL_FACTORIES))
def test_both_ensembles_beat_the_real_baseline_by_a_wide_margin(name):
    crops = load_data()
    train, _ = stratified_split(crops)
    X, y = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]
    pipeline = Pipeline(
        [("preprocess", build_preprocessor()), ("model", ENSEMBLE_MODEL_FACTORIES[name]())]
    )

    result = cross_validated_accuracy(pipeline, X, y)

    assert result["mean"] > BALANCED_BASELINE_ACCURACY
    # 0.98 rather than a tighter bound because the boosting backend — XGBoost or
    # scikit-learn — differs between environments, and so does its exact score.
    assert result["mean"] > 0.98


@requires_raw_dataset
def test_the_random_forest_beats_the_single_tree_it_is_made_of():
    """The Week 7 headline: many overfit trees, averaged, beat one overfit tree."""
    crops = load_data()
    train, _ = stratified_split(crops)
    X, y = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]

    tree = cross_validated_accuracy(get_decision_tree(), X, y)["mean"]
    forest = cross_validated_accuracy(get_random_forest(), X, y)["mean"]

    assert forest > tree


@requires_raw_dataset
def test_the_ensembles_do_not_beat_naive_bayes_but_tie_with_it():
    """Week 7's honest headline, pinned so the prose cannot drift from the data.

    Gaussian naive Bayes has led since Week 5 and still does — by ~0.2 points,
    which is well inside either model's fold-to-fold spread. "The ensembles are
    the best models in the table" is not a claim this experiment supports; "they
    are level with the leader and ahead of everything else" is.
    """
    crops = load_data()
    train, _ = stratified_split(crops)
    X, y = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]

    naive_bayes = cross_validated_accuracy(get_naive_bayes(), X, y)
    forest = cross_validated_accuracy(get_random_forest(), X, y)

    assert abs(forest["mean"] - naive_bayes["mean"]) < naive_bayes["std"] + forest["std"]


@requires_raw_dataset
@pytest.mark.parametrize("name", list(ENSEMBLE_MODEL_FACTORIES))
def test_the_ensembles_beat_every_single_model_except_naive_bayes(name):
    crops = load_data()
    train, _ = stratified_split(crops)
    X, y = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]

    ensemble = cross_validated_accuracy(ENSEMBLE_MODEL_FACTORIES[name](), X, y)["mean"]

    for factory in (get_decision_tree, get_svm, get_logistic_regression, get_knn):
        assert ensemble > cross_validated_accuracy(factory(), X, y)["mean"]


@requires_raw_dataset
def test_the_forest_importances_cover_the_seven_real_features():
    crops = load_data()
    train, _ = stratified_split(crops)
    X, y = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]

    importances = pd.Series(get_random_forest().fit(X, y).feature_importances_, index=X.columns)

    assert list(importances.index) == list(FEATURE_COLUMNS)
    assert importances.sum() == pytest.approx(1.0)
    assert importances.idxmax() in ("humidity", "rainfall", "K")
