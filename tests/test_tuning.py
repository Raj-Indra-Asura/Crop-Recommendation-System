"""Tests for Week 8's hyperparameter search wrapper, :func:`tune_model`.

These are **smoke and contract tests**, not a proof that the search finds the
globally best settings. What they check:

1. **The wrapper runs and returns the documented dictionary** for both search
   strategies, on a small synthetic frame that keeps the suite fast.
2. **The protocol is honest.** The winner is chosen on held-out folds, the
   estimator passed in is left unfitted, the returned one is refitted on all the
   data, and the number of fits is exactly ``n_candidates * n_splits``.
3. **The search is reproducible** — the same call twice gives the same winner —
   and **randomised search costs what it is told to cost**, not what the size of
   the space would cost.
4. **Bad input fails immediately**, with a message that names the problem.
5. **On the real training rows a tuned model is at least as good as the default
   one**, measured on the same folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluation import (
    DEFAULT_N_ITER,
    DEFAULT_SEARCH_SCORING,
    SEARCH_STRATEGIES,
    cross_validated_accuracy,
    tune_model,
)
from src.models import get_decision_tree, get_random_forest
from src.preprocessing import build_preprocessor
from tests.conftest import requires_raw_dataset

SMALL_GRID = {"max_depth": [1, 2, 3]}


@pytest.fixture
def toy_frame() -> pd.DataFrame:
    """A tiny, clearly separable three-class frame: fast to search, easy to fit."""
    rng = np.random.default_rng(seed=0)
    blocks = []
    for index, centre in enumerate((0.0, 5.0, 10.0)):
        blocks.append(
            pd.DataFrame(
                {
                    "a": rng.normal(centre, 0.5, size=40),
                    "b": rng.normal(-centre, 0.5, size=40),
                    "label": f"class_{index}",
                }
            )
        )
    return pd.concat(blocks, ignore_index=True)


def features_and_labels(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a frame into its feature columns and its label column."""
    return frame.drop(columns=["label"]), frame["label"]


# --------------------------------------------------------------------------
# The returned dictionary
# --------------------------------------------------------------------------


def test_tune_model_returns_the_documented_keys(toy_frame):
    X, y = features_and_labels(toy_frame)

    result = tune_model(DecisionTreeClassifier(random_state=0), SMALL_GRID, X, y)

    assert set(result) >= {
        "best_estimator",
        "best_params",
        "best_score",
        "best_std",
        "cv_results",
        "n_candidates",
        "n_fits",
        "scoring",
        "search",
        "n_splits",
        "elapsed_seconds",
    }


def test_best_params_come_from_the_grid(toy_frame):
    X, y = features_and_labels(toy_frame)

    result = tune_model(DecisionTreeClassifier(random_state=0), SMALL_GRID, X, y)

    assert set(result["best_params"]) == {"max_depth"}
    assert result["best_params"]["max_depth"] in SMALL_GRID["max_depth"]


def test_best_score_is_a_probability_like_number(toy_frame):
    X, y = features_and_labels(toy_frame)

    result = tune_model(DecisionTreeClassifier(random_state=0), SMALL_GRID, X, y)

    assert 0.0 <= result["best_score"] <= 1.0
    assert result["best_std"] >= 0.0


def test_cv_results_holds_one_row_per_candidate_sorted_best_first(toy_frame):
    X, y = features_and_labels(toy_frame)

    result = tune_model(DecisionTreeClassifier(random_state=0), SMALL_GRID, X, y)
    table = result["cv_results"]

    assert isinstance(table, pd.DataFrame)
    assert len(table) == len(SMALL_GRID["max_depth"]) == result["n_candidates"]
    assert list(table["rank_test_score"]) == sorted(table["rank_test_score"])
    assert table.loc[0, "mean_test_score"] == pytest.approx(result["best_score"])


def test_n_fits_is_candidates_times_folds(toy_frame):
    X, y = features_and_labels(toy_frame)

    result = tune_model(DecisionTreeClassifier(random_state=0), SMALL_GRID, X, y, n_splits=3)

    assert result["n_splits"] == 3
    assert result["n_fits"] == result["n_candidates"] * 3


def test_the_protocol_is_recorded_in_the_result(toy_frame):
    X, y = features_and_labels(toy_frame)

    result = tune_model(
        DecisionTreeClassifier(random_state=0), SMALL_GRID, X, y, scoring="f1_macro"
    )

    assert result["scoring"] == "f1_macro"
    assert result["search"] == "grid"
    assert DEFAULT_SEARCH_SCORING == "accuracy"


# --------------------------------------------------------------------------
# What the search does to the estimators
# --------------------------------------------------------------------------


def test_the_estimator_passed_in_is_left_unfitted(toy_frame):
    X, y = features_and_labels(toy_frame)
    model = DecisionTreeClassifier(random_state=0)

    tune_model(model, SMALL_GRID, X, y)

    with pytest.raises(NotFittedError):
        model.predict(X)


def test_the_returned_estimator_is_fitted_and_carries_the_best_params(toy_frame):
    X, y = features_and_labels(toy_frame)

    result = tune_model(DecisionTreeClassifier(random_state=0), SMALL_GRID, X, y)
    best = result["best_estimator"]

    assert len(best.predict(X)) == len(y)          # fitted: no NotFittedError
    assert best.get_params()["max_depth"] == result["best_params"]["max_depth"]


def test_a_pipeline_is_searched_through_its_step_names(toy_frame):
    X, y = features_and_labels(toy_frame)
    pipeline = Pipeline(
        [("preproc", build_preprocessor(list(X.columns))), ("model", get_decision_tree())]
    )

    result = tune_model(pipeline, {"model__max_depth": [1, 3]}, X, y, n_splits=3)

    assert result["best_params"]["model__max_depth"] in (1, 3)
    assert result["best_estimator"].named_steps["model"].max_depth in (1, 3)


# --------------------------------------------------------------------------
# Randomised search
# --------------------------------------------------------------------------


def test_random_search_runs_and_reports_itself(toy_frame):
    X, y = features_and_labels(toy_frame)

    result = tune_model(
        DecisionTreeClassifier(random_state=0),
        {"max_depth": [1, 2, 3, 4, 5], "min_samples_leaf": [1, 2, 3, 4]},
        X,
        y,
        search="random",
        n_iter=6,
        n_splits=3,
    )

    assert result["search"] == "random"
    assert result["n_candidates"] == 6


def test_random_search_never_samples_more_than_the_grid_holds(toy_frame):
    X, y = features_and_labels(toy_frame)

    result = tune_model(
        DecisionTreeClassifier(random_state=0),
        SMALL_GRID,                       # only three combinations exist
        X,
        y,
        search="random",
        n_iter=50,
        n_splits=3,
    )

    assert result["n_candidates"] == 3


def test_random_search_is_cheaper_than_the_grid_it_samples_from(toy_frame):
    X, y = features_and_labels(toy_frame)
    space = {"max_depth": [1, 2, 3, 4, 5, 6], "min_samples_leaf": [1, 2, 3, 4, 5]}

    exhaustive = tune_model(DecisionTreeClassifier(random_state=0), space, X, y, n_splits=3)
    sampled = tune_model(
        DecisionTreeClassifier(random_state=0),
        space,
        X,
        y,
        search="random",
        n_iter=8,
        n_splits=3,
    )

    assert exhaustive["n_candidates"] == 30
    assert sampled["n_candidates"] == 8
    assert sampled["n_fits"] < exhaustive["n_fits"]


def test_the_default_n_iter_is_the_documented_one():
    assert DEFAULT_N_ITER == 20
    assert SEARCH_STRATEGIES == ("grid", "random")


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


def test_the_same_search_twice_gives_the_same_answer(toy_frame):
    X, y = features_and_labels(toy_frame)

    first = tune_model(DecisionTreeClassifier(random_state=0), SMALL_GRID, X, y)
    second = tune_model(DecisionTreeClassifier(random_state=0), SMALL_GRID, X, y)

    assert first["best_params"] == second["best_params"]
    assert first["best_score"] == pytest.approx(second["best_score"])


def test_random_search_is_reproducible_under_a_fixed_seed(toy_frame):
    X, y = features_and_labels(toy_frame)
    space = {"max_depth": [1, 2, 3, 4, 5], "min_samples_leaf": [1, 2, 3]}

    first = tune_model(
        DecisionTreeClassifier(random_state=0), space, X, y, search="random", n_iter=5, n_splits=3
    )
    second = tune_model(
        DecisionTreeClassifier(random_state=0), space, X, y, search="random", n_iter=5, n_splits=3
    )

    assert first["best_params"] == second["best_params"]


# --------------------------------------------------------------------------
# Rejecting bad input
# --------------------------------------------------------------------------


def test_mismatched_lengths_are_rejected(toy_frame):
    X, y = features_and_labels(toy_frame)

    with pytest.raises(ValueError, match="must match"):
        tune_model(DecisionTreeClassifier(), SMALL_GRID, X, y.iloc[:-1])


def test_an_unknown_search_strategy_is_rejected(toy_frame):
    X, y = features_and_labels(toy_frame)

    with pytest.raises(ValueError, match="must be one of"):
        tune_model(DecisionTreeClassifier(), SMALL_GRID, X, y, search="bayesian")


def test_an_empty_grid_is_rejected(toy_frame):
    X, y = features_and_labels(toy_frame)

    with pytest.raises(ValueError, match="must not be empty"):
        tune_model(DecisionTreeClassifier(), {}, X, y)


def test_a_non_dictionary_grid_entry_is_rejected(toy_frame):
    X, y = features_and_labels(toy_frame)

    with pytest.raises(ValueError, match="dictionaries"):
        tune_model(DecisionTreeClassifier(), [("max_depth", [1, 2])], X, y)


def test_a_useless_n_iter_is_rejected(toy_frame):
    X, y = features_and_labels(toy_frame)

    with pytest.raises(ValueError, match="at least 1"):
        tune_model(DecisionTreeClassifier(), SMALL_GRID, X, y, search="random", n_iter=0)


def test_a_single_fold_is_rejected(toy_frame):
    X, y = features_and_labels(toy_frame)

    with pytest.raises(ValueError, match="at least 2"):
        tune_model(DecisionTreeClassifier(), SMALL_GRID, X, y, n_splits=1)


# --------------------------------------------------------------------------
# On the real dataset
# --------------------------------------------------------------------------


@requires_raw_dataset
def test_tuning_a_forest_on_the_real_rows_does_not_lose_to_its_defaults(raw_data):
    X = raw_data[list(FEATURE_COLUMNS)]
    y = raw_data[TARGET_COLUMN]

    default_score = cross_validated_accuracy(get_random_forest(n_estimators=30), X, y, n_splits=3)
    tuned = tune_model(
        get_random_forest(n_estimators=30),
        {"max_features": ["sqrt", None]},
        X,
        y,
        n_splits=3,
    )

    # The default settings are inside the search space, so the winner can only
    # tie them or beat them - never lose to them.
    assert tuned["best_score"] >= default_score["mean"] - 1e-9
