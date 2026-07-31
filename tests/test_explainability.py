"""Tests for Week 8's explainability helpers.

These are **smoke and contract tests**. They deliberately do not assert that a
SHAP value matches a hand-computed Shapley value — that is SHAP's own test
suite's job, and re-deriving it here would test the library rather than this
project. What they check is that the helpers run, return the documented shapes,
say which backend produced the numbers, and agree with facts that are true by
construction of the toy data:

1. **:func:`permutation_feature_importance` ranks a signal column above a noise
   column**, returns one row per feature with a mean and a spread, and refuses
   mismatched inputs.
2. **:func:`explain_prediction` runs on both backends.** The fallback is
   exercised explicitly with ``method="permutation"`` so it is tested in every
   environment, installed SHAP or not.
3. **The probability breakdown is a real distribution** over the model's classes,
   sorted largest first, and its top entry is the prediction being explained.
4. **Contributions cover every feature**, are sorted by absolute size, and put a
   feature the model actually uses above one it ignores.
5. **The result always records its method**, so no explanation can be quoted
   without knowing how it was produced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluation import (
    EXPLAINER_BACKEND,
    SHAP_AVAILABLE,
    explain_prediction,
    permutation_feature_importance,
)
from src.models import get_naive_bayes, get_random_forest
from src.preprocessing import build_preprocessor
from tests.conftest import requires_raw_dataset


@pytest.fixture
def toy_frame() -> pd.DataFrame:
    """Three separable classes, where `signal` decides the label and `noise` does not."""
    rng = np.random.default_rng(seed=0)
    blocks = []
    for index, centre in enumerate((0.0, 6.0, 12.0)):
        blocks.append(
            pd.DataFrame(
                {
                    "signal": rng.normal(centre, 0.5, size=40),
                    "noise": rng.normal(0.0, 1.0, size=40),
                    "label": f"class_{index}",
                }
            )
        )
    return pd.concat(blocks, ignore_index=True)


@pytest.fixture
def fitted_forest(toy_frame):
    """A small forest fitted on the toy frame, with its training features."""
    X = toy_frame[["signal", "noise"]]
    y = toy_frame["label"]
    return get_random_forest(n_estimators=20).fit(X, y), X, y


# --------------------------------------------------------------------------
# permutation_feature_importance
# --------------------------------------------------------------------------


def test_permutation_importance_returns_one_row_per_feature(fitted_forest):
    model, X, y = fitted_forest

    table = permutation_feature_importance(model, X, y, n_repeats=3)

    assert isinstance(table, pd.DataFrame)
    assert set(table.index) == {"signal", "noise"}
    assert list(table.columns) == ["importance_mean", "importance_std"]


def test_permutation_importance_ranks_the_signal_above_the_noise(fitted_forest):
    model, X, y = fitted_forest

    table = permutation_feature_importance(model, X, y, n_repeats=5)

    assert table.index[0] == "signal"
    assert table.loc["signal", "importance_mean"] > table.loc["noise", "importance_mean"]


def test_permutation_importance_is_sorted_most_important_first(fitted_forest):
    model, X, y = fitted_forest

    means = permutation_feature_importance(model, X, y, n_repeats=3)["importance_mean"]

    assert list(means) == sorted(means, reverse=True)


def test_permutation_importance_is_reproducible_under_a_fixed_seed(fitted_forest):
    model, X, y = fitted_forest

    first = permutation_feature_importance(model, X, y, n_repeats=3)
    second = permutation_feature_importance(model, X, y, n_repeats=3)

    pd.testing.assert_frame_equal(first, second)


def test_permutation_importance_rejects_mismatched_lengths(fitted_forest):
    model, X, y = fitted_forest

    with pytest.raises(ValueError, match="must match"):
        permutation_feature_importance(model, X, y.iloc[:-1])


def test_permutation_importance_rejects_zero_repeats(fitted_forest):
    model, X, y = fitted_forest

    with pytest.raises(ValueError, match="at least 1"):
        permutation_feature_importance(model, X, y, n_repeats=0)


def test_permutation_importance_works_on_a_model_without_feature_importances(toy_frame):
    X = toy_frame[["signal", "noise"]]
    y = toy_frame["label"]
    model = get_naive_bayes().fit(X, y)

    table = permutation_feature_importance(model, X, y, n_repeats=3)

    assert not hasattr(model, "feature_importances_")   # the point of the test
    assert table.index[0] == "signal"


# --------------------------------------------------------------------------
# explain_prediction — the shared contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["auto", "permutation"])
def test_explain_prediction_returns_the_documented_keys(fitted_forest, method):
    model, X, _ = fitted_forest

    result = explain_prediction(model, X.iloc[[0]], background=X, method=method)

    assert set(result) >= {
        "prediction",
        "probability",
        "probabilities",
        "contributions",
        "top_feature",
        "method",
        "base_value",
    }


@pytest.mark.parametrize("method", ["auto", "permutation"])
def test_the_explained_prediction_matches_the_model(fitted_forest, method):
    model, X, _ = fitted_forest

    result = explain_prediction(model, X.iloc[[5]], background=X, method=method)

    assert result["prediction"] == model.predict(X.iloc[[5]])[0]


@pytest.mark.parametrize("method", ["auto", "permutation"])
def test_the_probability_breakdown_is_a_distribution(fitted_forest, method):
    model, X, _ = fitted_forest

    result = explain_prediction(model, X.iloc[[5]], background=X, method=method)
    probabilities = result["probabilities"]

    assert set(probabilities.index) == set(model.classes_)
    assert probabilities.sum() == pytest.approx(1.0)
    assert list(probabilities) == sorted(probabilities, reverse=True)
    assert probabilities.iloc[0] == pytest.approx(result["probability"])


@pytest.mark.parametrize("method", ["auto", "permutation"])
def test_contributions_cover_every_feature_sorted_by_size(fitted_forest, method):
    model, X, _ = fitted_forest

    contributions = explain_prediction(model, X.iloc[[5]], background=X, method=method)[
        "contributions"
    ]

    assert set(contributions.index) == set(X.columns)
    assert list(contributions.abs()) == sorted(contributions.abs(), reverse=True)


@pytest.mark.parametrize("method", ["auto", "permutation"])
def test_the_decisive_feature_beats_the_ignored_one(fitted_forest, method):
    model, X, _ = fitted_forest

    result = explain_prediction(model, X.iloc[[0]], background=X, method=method)

    assert result["top_feature"] == "signal"
    assert abs(result["contributions"]["signal"]) > abs(result["contributions"]["noise"])


@pytest.mark.parametrize("method", ["auto", "permutation"])
def test_the_result_records_which_backend_produced_it(fitted_forest, method):
    model, X, _ = fitted_forest

    result = explain_prediction(model, X.iloc[[0]], background=X, method=method)

    assert result["method"] in ("shap", "permutation")
    if method == "permutation":
        assert result["method"] == "permutation"


def test_the_backend_flag_matches_the_availability_flag():
    assert EXPLAINER_BACKEND == ("shap" if SHAP_AVAILABLE else "permutation")


# --------------------------------------------------------------------------
# explain_prediction — input handling
# --------------------------------------------------------------------------


def test_a_series_and_a_one_row_frame_explain_the_same_example(fitted_forest):
    model, X, _ = fitted_forest

    from_frame = explain_prediction(model, X.iloc[[7]], background=X, method="permutation")
    from_series = explain_prediction(model, X.iloc[7], background=X, method="permutation")

    assert from_frame["prediction"] == from_series["prediction"]


def test_a_plain_array_row_is_accepted(fitted_forest):
    model, X, _ = fitted_forest

    result = explain_prediction(
        model, X.iloc[7].to_numpy(), background=X, method="permutation"
    )

    assert result["prediction"] == model.predict(X.iloc[[7]])[0]


def test_more_than_one_row_is_rejected(fitted_forest):
    model, X, _ = fitted_forest

    with pytest.raises(ValueError, match="exactly one row"):
        explain_prediction(model, X.iloc[:3], background=X, method="permutation")


def test_an_unknown_method_is_rejected(fitted_forest):
    model, X, _ = fitted_forest

    with pytest.raises(ValueError, match="must be 'auto'"):
        explain_prediction(model, X.iloc[[0]], background=X, method="lime")


def test_the_fallback_without_background_says_what_is_missing(fitted_forest):
    model, X, _ = fitted_forest

    with pytest.raises(ValueError, match="needs `background`"):
        explain_prediction(model, X.iloc[[0]], method="permutation")


def test_a_model_without_predict_proba_is_rejected(toy_frame):
    from sklearn.svm import LinearSVC

    X = toy_frame[["signal", "noise"]]
    model = LinearSVC().fit(X, toy_frame["label"])

    with pytest.raises(AttributeError, match="predict_proba"):
        explain_prediction(model, X.iloc[[0]], background=X)


@pytest.mark.skipif(SHAP_AVAILABLE, reason="shap is installed in this environment")
def test_insisting_on_shap_without_shap_fails_clearly(fitted_forest):
    model, X, _ = fitted_forest

    with pytest.raises(ValueError, match="not installed"):
        explain_prediction(model, X.iloc[[0]], background=X, method="shap")


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="shap is not installed in this environment")
def test_shap_contributions_come_with_a_base_value(fitted_forest):
    model, X, _ = fitted_forest

    result = explain_prediction(model, X.iloc[[0]], background=X, method="shap")

    assert result["method"] == "shap"
    assert isinstance(result["base_value"], float)


def test_the_fallback_reports_no_base_value(fitted_forest):
    model, X, _ = fitted_forest

    result = explain_prediction(model, X.iloc[[0]], background=X, method="permutation")

    assert result["base_value"] is None


def test_both_backends_agree_on_the_decisive_feature(fitted_forest):
    model, X, _ = fitted_forest

    fallback = explain_prediction(model, X.iloc[[0]], background=X, method="permutation")
    automatic = explain_prediction(model, X.iloc[[0]], background=X)

    assert fallback["top_feature"] == automatic["top_feature"]


# --------------------------------------------------------------------------
# On the real dataset
# --------------------------------------------------------------------------


@requires_raw_dataset
def test_a_real_prediction_can_be_explained_through_a_pipeline(raw_data):
    features = list(FEATURE_COLUMNS)
    X = raw_data[features]
    y = raw_data[TARGET_COLUMN]
    pipeline = Pipeline(
        [("preproc", build_preprocessor(features)), ("model", get_naive_bayes())]
    ).fit(X, y)

    result = explain_prediction(
        pipeline, X.iloc[[0]], background=X.sample(50, random_state=0), method="permutation"
    )

    assert result["prediction"] in set(y)
    assert set(result["contributions"].index) == set(features)


@requires_raw_dataset
def test_permutation_importance_on_the_real_rows_ranks_the_seven_features(raw_data):
    features = list(FEATURE_COLUMNS)
    X = raw_data[features]
    y = raw_data[TARGET_COLUMN]
    model = get_random_forest(n_estimators=30).fit(X, y)

    table = permutation_feature_importance(model, X.head(300), y.head(300), n_repeats=3)

    assert list(table.index.sort_values()) == sorted(features)
    assert table["importance_mean"].max() > 0.0
