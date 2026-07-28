"""Tests for Week 4's baseline model and evaluation helpers.

Three properties are checked, and they are the ones the week is graded on:

1. **The baseline is genuinely uninformed.** ``most_frequent`` predicts a single
   class whatever the features say, and permuting the feature columns cannot
   change a prediction.
2. **Its accuracy is 1/k on balanced data.** On the real 22-crop dataset the
   cross-validated accuracy lands on 1/22 ≈ 4.55%, the number every later model
   must beat.
3. **The evaluation helpers report what they claim.** ``evaluate_model``'s
   accuracy matches a hand computation and its report names the classes, and
   ``cross_validated_accuracy`` returns one score per fold, deterministically,
   without fitting the estimator it was handed.

Most tests run on a small synthetic frame so they pass with or without the CSV;
those needing the real 2,200 rows are marked with ``requires_raw_dataset``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import StratifiedKFold

from src.data import (
    DEFAULT_RANDOM_STATE,
    EXPECTED_LABEL_COUNT,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_data,
    stratified_split,
)
from src.evaluation import (
    DEFAULT_CV_FOLDS,
    build_cv,
    cross_validated_accuracy,
    evaluate_model,
)
from src.models import BASELINE_STRATEGIES, DEFAULT_BASELINE_STRATEGY, get_baseline_model
from tests.conftest import requires_raw_dataset

#: Accuracy a constant guess earns on the 22 balanced crops.
BALANCED_BASELINE_ACCURACY = 1 / EXPECTED_LABEL_COUNT


@pytest.fixture
def balanced_frame() -> pd.DataFrame:
    """Build a deterministic, perfectly balanced four-class frame.

    Four classes hold 50 rows each, so a constant guess scores exactly 1/4 —
    the same arithmetic as the real dataset's 1/22, at a size that keeps the
    tests fast.
    """
    rng = np.random.default_rng(seed=0)
    rows_per_class = 50
    frames = []
    for offset, name in enumerate(["one", "two", "three", "four"]):
        frames.append(
            pd.DataFrame(
                {
                    "first": rng.normal(5.0 + offset, 1.0, rows_per_class),
                    "second": rng.normal(50.0 + 10.0 * offset, 5.0, rows_per_class),
                    TARGET_COLUMN: name,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def imbalanced_frame() -> pd.DataFrame:
    """Build a 95/5 two-class frame, where accuracy alone is misleading.

    A constant "majority" guess scores 95% here while never once predicting the
    minority class — the case Week 8's precision and recall exist to expose.
    """
    rng = np.random.default_rng(seed=1)
    majority = pd.DataFrame(
        {
            "first": rng.normal(0.0, 1.0, 190),
            "second": rng.normal(0.0, 1.0, 190),
            TARGET_COLUMN: "common",
        }
    )
    minority = pd.DataFrame(
        {
            "first": rng.normal(3.0, 1.0, 10),
            "second": rng.normal(3.0, 1.0, 10),
            TARGET_COLUMN: "rare",
        }
    )
    return pd.concat([majority, minority], ignore_index=True)


def features_and_labels(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a fixture frame into its feature columns and its label column."""
    feature_names = [name for name in frame.columns if name != TARGET_COLUMN]
    return frame[feature_names], frame[TARGET_COLUMN]


# --------------------------------------------------------------------------
# get_baseline_model
# --------------------------------------------------------------------------


def test_get_baseline_model_returns_an_unfitted_dummy_classifier():
    model = get_baseline_model()

    assert isinstance(model, DummyClassifier)
    with pytest.raises(NotFittedError):
        model.predict([[0.0, 0.0]])


def test_get_baseline_model_defaults_to_most_frequent():
    assert DEFAULT_BASELINE_STRATEGY == "most_frequent"
    assert get_baseline_model().strategy == "most_frequent"


@pytest.mark.parametrize("strategy", BASELINE_STRATEGIES)
def test_get_baseline_model_accepts_every_supported_strategy(strategy, balanced_frame):
    X, y = features_and_labels(balanced_frame)
    model = get_baseline_model(strategy)

    assert model.strategy == strategy
    model.fit(X, y)
    assert len(model.predict(X)) == len(y)


@pytest.mark.parametrize("strategy", ["constant", "MOST_FREQUENT", "best", ""])
def test_get_baseline_model_rejects_an_unsupported_strategy(strategy):
    with pytest.raises(ValueError, match="Unsupported baseline strategy"):
        get_baseline_model(strategy)


def test_get_baseline_model_uses_the_project_seed_by_default():
    assert get_baseline_model("stratified").random_state == DEFAULT_RANDOM_STATE


def test_random_baselines_are_reproducible(balanced_frame):
    X, y = features_and_labels(balanced_frame)

    first = get_baseline_model("stratified").fit(X, y).predict(X)
    second = get_baseline_model("stratified").fit(X, y).predict(X)
    other_seed = get_baseline_model("stratified", random_state=7).fit(X, y).predict(X)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, other_seed)


def test_most_frequent_baseline_predicts_a_single_class(imbalanced_frame):
    X, y = features_and_labels(imbalanced_frame)
    model = get_baseline_model("most_frequent").fit(X, y)

    predictions = model.predict(X)
    assert set(np.unique(predictions)) == {"common"}


def test_baseline_ignores_the_features(balanced_frame):
    """Scrambling the features cannot change a prediction: that is the point."""
    X, y = features_and_labels(balanced_frame)
    model = get_baseline_model().fit(X, y)

    scrambled = X.sample(frac=1.0, random_state=0).reset_index(drop=True)
    assert np.array_equal(model.predict(X), model.predict(scrambled))


def test_baseline_scores_one_over_k_on_balanced_data(balanced_frame):
    X, y = features_and_labels(balanced_frame)
    model = get_baseline_model().fit(X, y)

    assert evaluate_model(model, X, y)["accuracy"] == pytest.approx(0.25)


def test_baseline_accuracy_is_misleadingly_high_on_imbalanced_data(imbalanced_frame):
    """95% accuracy while never predicting the minority class — Week 8's motivation."""
    X, y = features_and_labels(imbalanced_frame)
    model = get_baseline_model().fit(X, y)

    result = evaluate_model(model, X, y)
    assert result["accuracy"] == pytest.approx(0.95)
    assert "rare" in result["report"]


# --------------------------------------------------------------------------
# evaluate_model
# --------------------------------------------------------------------------


def test_evaluate_model_returns_accuracy_report_and_sample_count(balanced_frame):
    X, y = features_and_labels(balanced_frame)
    result = evaluate_model(get_baseline_model().fit(X, y), X, y)

    assert set(result) >= {"accuracy", "report", "n_samples"}
    assert isinstance(result["accuracy"], float)
    assert isinstance(result["report"], str)
    assert result["n_samples"] == len(y)


def test_evaluate_model_accuracy_matches_a_hand_computation(balanced_frame):
    X, y = features_and_labels(balanced_frame)
    model = get_baseline_model().fit(X, y)

    expected = float((model.predict(X) == y.to_numpy()).mean())
    assert evaluate_model(model, X, y)["accuracy"] == pytest.approx(expected)


def test_evaluate_model_report_names_every_class(balanced_frame):
    X, y = features_and_labels(balanced_frame)
    report = evaluate_model(get_baseline_model().fit(X, y), X, y)["report"]

    for class_name in y.unique():
        assert class_name in report
    for heading in ["precision", "recall", "f1-score", "support", "accuracy"]:
        assert heading in report


def test_evaluate_model_does_not_fit_the_model(balanced_frame):
    X, y = features_and_labels(balanced_frame)

    with pytest.raises(NotFittedError):
        evaluate_model(get_baseline_model(), X, y)


def test_evaluate_model_rejects_mismatched_lengths(balanced_frame):
    X, y = features_and_labels(balanced_frame)
    model = get_baseline_model().fit(X, y)

    with pytest.raises(ValueError, match="they must match"):
        evaluate_model(model, X, y.iloc[:-1])


def test_evaluate_model_scores_a_perfect_model_at_one(balanced_frame):
    """A memorising model must score 1.0, or the helper is measuring nothing."""
    X, y = features_and_labels(balanced_frame)

    class Memoriser:
        def predict(self, features):
            return y.to_numpy()

    assert evaluate_model(Memoriser(), X, y)["accuracy"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# build_cv and cross_validated_accuracy
# --------------------------------------------------------------------------


def test_build_cv_is_stratified_shuffled_and_seeded():
    cv = build_cv()

    assert isinstance(cv, StratifiedKFold)
    assert cv.get_n_splits() == DEFAULT_CV_FOLDS == 5
    assert cv.shuffle is True
    assert cv.random_state == DEFAULT_RANDOM_STATE


def test_build_cv_rejects_fewer_than_two_folds():
    with pytest.raises(ValueError, match="at least 2"):
        build_cv(n_splits=1)


def test_cv_folds_preserve_the_class_balance(balanced_frame):
    X, y = features_and_labels(balanced_frame)

    for _, validation_index in build_cv().split(X, y):
        counts = y.iloc[validation_index].value_counts()
        assert len(counts) == 4
        assert counts.max() - counts.min() <= 1


def test_cv_folds_are_disjoint_and_cover_every_row(balanced_frame):
    X, y = features_and_labels(balanced_frame)

    validated = []
    for _, validation_index in build_cv().split(X, y):
        validated.extend(validation_index.tolist())

    assert sorted(validated) == list(range(len(X)))


def test_cross_validated_accuracy_returns_one_score_per_fold(balanced_frame):
    X, y = features_and_labels(balanced_frame)
    result = cross_validated_accuracy(get_baseline_model(), X, y)

    assert result["n_splits"] == DEFAULT_CV_FOLDS
    assert result["scores"].shape == (DEFAULT_CV_FOLDS,)
    assert result["mean"] == pytest.approx(result["scores"].mean())
    assert result["std"] == pytest.approx(result["scores"].std())


def test_cross_validated_accuracy_honours_the_fold_count(balanced_frame):
    X, y = features_and_labels(balanced_frame)
    result = cross_validated_accuracy(get_baseline_model(), X, y, n_splits=3)

    assert result["n_splits"] == 3
    assert result["scores"].shape == (3,)


def test_cross_validated_accuracy_is_reproducible(balanced_frame):
    X, y = features_and_labels(balanced_frame)

    first = cross_validated_accuracy(get_baseline_model("stratified"), X, y)
    second = cross_validated_accuracy(get_baseline_model("stratified"), X, y)

    assert np.array_equal(first["scores"], second["scores"])


def test_cross_validated_accuracy_leaves_the_estimator_unfitted(balanced_frame):
    """scikit-learn clones the estimator per fold, so the original is untouched."""
    X, y = features_and_labels(balanced_frame)
    model = get_baseline_model()

    cross_validated_accuracy(model, X, y)

    with pytest.raises(NotFittedError):
        model.predict(X)


def test_cross_validated_accuracy_rejects_mismatched_lengths(balanced_frame):
    X, y = features_and_labels(balanced_frame)

    with pytest.raises(ValueError, match="they must match"):
        cross_validated_accuracy(get_baseline_model(), X, y.iloc[:-1])


def test_cross_validated_accuracy_matches_a_balanced_baseline(balanced_frame):
    X, y = features_and_labels(balanced_frame)
    result = cross_validated_accuracy(get_baseline_model(), X, y)

    assert result["mean"] == pytest.approx(0.25, abs=0.01)


# --------------------------------------------------------------------------
# The real dataset: the number Week 5 has to beat
# --------------------------------------------------------------------------


@requires_raw_dataset
def test_real_baseline_lands_on_one_over_twenty_two():
    crops = load_data()
    train, _ = stratified_split(crops)
    X, y = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]

    result = cross_validated_accuracy(get_baseline_model(), X, y)

    assert result["mean"] == pytest.approx(BALANCED_BASELINE_ACCURACY, abs=0.005)
    assert result["mean"] < 0.06


@requires_raw_dataset
@pytest.mark.parametrize("strategy", BASELINE_STRATEGIES)
def test_no_baseline_strategy_escapes_the_one_over_k_ceiling(strategy):
    crops = load_data()
    train, _ = stratified_split(crops)
    X, y = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]

    result = cross_validated_accuracy(get_baseline_model(strategy), X, y)

    assert result["mean"] < 0.10


@requires_raw_dataset
def test_real_most_frequent_baseline_predicts_exactly_one_crop():
    crops = load_data()
    train, test = stratified_split(crops)
    model = get_baseline_model().fit(train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN])

    predictions = model.predict(test[list(FEATURE_COLUMNS)])
    assert len(set(predictions)) == 1
