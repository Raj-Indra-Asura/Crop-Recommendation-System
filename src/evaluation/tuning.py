"""Hyperparameter search: choosing settings with cross-validation, not by eye.

Every model in this project so far has run on its factory defaults. Week 6 and
Week 7 swept one setting at a time and *adopted none of them*, because reading a
winner off a curve is exactly the mistake a stated search protocol exists to
prevent: sweep long enough, look hard enough, and the number you pick is fitted
to the data you looked at.

:func:`tune_model` is that stated protocol, in one function. It wraps
scikit-learn's :class:`~sklearn.model_selection.GridSearchCV` and
:class:`~sklearn.model_selection.RandomizedSearchCV` so the rest of the project
never has to remember the arguments that make a search honest:

* the folds come from :func:`src.evaluation.metrics.build_cv`, so a search uses
  the *same* stratified splits and the *same* seed as every other measurement in
  the repository;
* the score of a candidate is always a **cross-validated mean over held-out
  folds**, never a training score;
* the test set is not involved at any point — ``X`` and ``y`` here are the
  training rows, and the winner is scored on ``data/processed/test.csv`` exactly
  once, afterwards, by the caller.

The two search strategies differ only in which candidates get tried:

* ``search="grid"`` tries **every** combination in ``param_grid``. Exhaustive,
  reproducible, and its cost is the product of the list lengths — a grid of
  3 x 4 x 2 values on 5 folds is 120 fits.
* ``search="random"`` samples ``n_iter`` combinations at random from the same
  space. The cost is fixed by ``n_iter`` instead of by the size of the space,
  which is what makes a large space affordable at all.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.data.split import DEFAULT_RANDOM_STATE
from src.evaluation.metrics import DEFAULT_CV_FOLDS, build_cv

#: The metric a search optimises unless told otherwise. Accuracy keeps this
#: week's searches comparable with the cross-validated accuracies quoted in
#: Weeks 4-7; ``"f1_macro"`` is the usual alternative and is accepted here.
DEFAULT_SEARCH_SCORING: str = "accuracy"

#: How many candidates :func:`tune_model` samples when ``search="random"``.
DEFAULT_N_ITER: int = 20

#: The search strategies this wrapper accepts.
SEARCH_STRATEGIES: tuple[str, ...] = ("grid", "random")


def _count_grid_candidates(param_grid: dict[str, Any] | list[dict[str, Any]]) -> int | None:
    """Return how many combinations a grid contains, or ``None`` if unknowable.

    A distribution object (anything without ``len``) has no finite size, which
    is precisely when ``search="random"`` is the only option.
    """
    grids = param_grid if isinstance(param_grid, list) else [param_grid]
    total = 0
    for grid in grids:
        combinations = 1
        for values in grid.values():
            try:
                combinations *= len(values)
            except TypeError:
                return None
        total += combinations
    return total


def _validate_param_grid(param_grid: dict[str, Any] | list[dict[str, Any]]) -> None:
    """Reject the shapes of ``param_grid`` that scikit-learn accepts too late."""
    grids = param_grid if isinstance(param_grid, list) else [param_grid]
    if not grids:
        raise ValueError("`param_grid` must not be empty; there would be nothing to search.")
    for grid in grids:
        if not isinstance(grid, dict):
            raise ValueError(
                f"`param_grid` entries must be dictionaries, got {type(grid).__name__}."
            )
        if not grid:
            raise ValueError("`param_grid` must not be empty; there would be nothing to search.")


def tune_model(
    model: BaseEstimator,
    param_grid: dict[str, Any] | list[dict[str, Any]],
    X: Any,
    y: Any,
    search: str = "grid",
    scoring: str = DEFAULT_SEARCH_SCORING,
    n_splits: int = DEFAULT_CV_FOLDS,
    n_iter: int = DEFAULT_N_ITER,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_jobs: int | None = None,
) -> dict[str, Any]:
    """Search ``param_grid`` for the settings that cross-validate best.

    The model is **not** modified: scikit-learn clones it before every fit, so
    the estimator passed in is still unfitted afterwards and the tuned copy is
    returned under ``"best_estimator"``.

    What the search actually does, for one candidate: split the training rows
    into ``n_splits`` stratified folds; fit the candidate on ``n_splits - 1`` of
    them and score it on the fold held out; repeat for every fold; average. The
    candidate with the best average wins, and is then refitted on **all** of
    ``X`` before being returned. That inner cross-validation is the whole point
    — it is why a search can compare dozens of candidates without any of them
    ever being scored on data it was fitted on, and why the test set can stay
    sealed while the choice is made.

    The winning score is still an **optimistic** estimate of future performance.
    Picking the maximum of many noisy numbers is itself a form of fitting, so
    ``"best_score"`` should be reported as "the best of N candidates on these
    folds", and the honest number for the tuned model is the one measured
    afterwards on data the search never touched.

    Args:
        model: An **unfitted** estimator, or a :class:`~sklearn.pipeline.Pipeline`.
            When it is a pipeline, ``param_grid`` keys use the usual
            ``"step__parameter"`` form, e.g. ``"model__n_estimators"``.
        param_grid: A dictionary mapping parameter names to the values to try,
            or a list of such dictionaries. With ``search="random"`` the values
            may also be distributions (anything exposing ``rvs``).
        X: Training features, one row per example.
        y: Training labels, used both to score candidates and to stratify folds.
        search: ``"grid"`` for an exhaustive
            :class:`~sklearn.model_selection.GridSearchCV`, or ``"random"`` for
            a :class:`~sklearn.model_selection.RandomizedSearchCV` of ``n_iter``
            samples. Defaults to ``"grid"``.
        scoring: Any scikit-learn scorer name — ``"accuracy"`` (the default),
            ``"f1_macro"``, ``"balanced_accuracy"`` and so on.
        n_splits: Folds in the inner cross-validation. Defaults to
            :data:`~src.evaluation.metrics.DEFAULT_CV_FOLDS` (5).
        n_iter: Candidates sampled when ``search="random"``. Ignored by the grid
            search. Defaults to :data:`DEFAULT_N_ITER` (20).
        random_state: Seed for both the fold shuffle and the candidate sampling,
            so a search is reproducible. Defaults to
            :data:`~src.data.split.DEFAULT_RANDOM_STATE`.
        n_jobs: Passed straight to scikit-learn. ``None`` (the default) runs the
            fits one at a time; ``-1`` uses every core.

    Returns:
        A dictionary with:

        * ``"best_estimator"`` — the winning candidate, refitted on all of ``X``;
        * ``"best_params"`` — the winning settings, as a plain dictionary;
        * ``"best_score"`` — its mean score over the held-out folds;
        * ``"best_std"`` — the standard deviation of that mean across folds, so
          the winner's margin can be compared against the noise;
        * ``"cv_results"`` — every candidate as a :class:`~pandas.DataFrame`,
          sorted best-first, with the parameters, the mean and standard
          deviation of the test-fold score, and the mean fit time;
        * ``"n_candidates"`` — how many settings were actually evaluated;
        * ``"n_fits"`` — ``n_candidates * n_splits``, the real cost of the search;
        * ``"scoring"``, ``"search"``, ``"n_splits"`` — the protocol, recorded
          so a result can never be quoted without it;
        * ``"elapsed_seconds"`` — wall-clock time of the search.

    Raises:
        ValueError: If ``X`` and ``y`` have different lengths, if ``search`` is
            not one of :data:`SEARCH_STRATEGIES`, if ``param_grid`` is empty, if
            ``n_iter`` is less than 1, or if ``n_splits`` is less than 2.
    """
    if len(X) != len(y):
        raise ValueError(f"`X` has {len(X)} rows but `y` has {len(y)} labels; they must match.")
    if search not in SEARCH_STRATEGIES:
        raise ValueError(f"`search` must be one of {SEARCH_STRATEGIES}, got {search!r}.")
    if n_iter < 1:
        raise ValueError(f"`n_iter` must be at least 1, got {n_iter}.")
    _validate_param_grid(param_grid)

    cv = build_cv(n_splits=n_splits, random_state=random_state)

    if search == "grid":
        searcher: GridSearchCV | RandomizedSearchCV = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
        )
    else:
        grid_size = _count_grid_candidates(param_grid)
        searcher = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter if grid_size is None else min(n_iter, grid_size),
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
            refit=True,
        )

    started = time.perf_counter()
    searcher.fit(X, y)
    elapsed = time.perf_counter() - started

    results = pd.DataFrame(searcher.cv_results_)
    columns = ["params", "mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time"]
    cv_results = results[columns].sort_values("rank_test_score").reset_index(drop=True)

    best_index = int(searcher.best_index_)
    n_candidates = int(len(results))

    return {
        "best_estimator": searcher.best_estimator_,
        "best_params": dict(searcher.best_params_),
        "best_score": float(searcher.best_score_),
        "best_std": float(np.asarray(results["std_test_score"])[best_index]),
        "cv_results": cv_results,
        "n_candidates": n_candidates,
        "n_fits": n_candidates * int(n_splits),
        "scoring": scoring,
        "search": search,
        "n_splits": int(n_splits),
        "elapsed_seconds": float(elapsed),
    }
