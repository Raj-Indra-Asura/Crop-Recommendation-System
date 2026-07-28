"""Evaluation: turning a fitted model into numbers you can defend.

This module is the project's single place for scoring a model. Week 4 gives it
two responsibilities:

* :func:`evaluate_model` — score an already-fitted model on some data, returning
  accuracy *and* the per-class breakdown that stops accuracy from being read on
  its own;
* :func:`cross_validated_accuracy` — score an *unfitted* model with k-fold
  cross-validation, so the number reported is a mean over several splits with a
  spread attached, rather than one lucky split.

It is written to be **extended, not replaced**: Week 5 compares real
classifiers through the same helpers, Week 6 tunes them, and Week 8 adds
precision, recall, F1 and the confusion matrix as first-class outputs. The
per-class report is already here in string form precisely so that "accuracy is
not the whole story" is visible from Week 4 rather than announced in Week 8.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.data.split import DEFAULT_RANDOM_STATE

#: Number of folds used whenever this project cross-validates. Five is the usual
#: compromise: each fold still holds 20% of the data (352 rows here, 16 per
#: crop), and the model is fitted five times rather than 1,760.
DEFAULT_CV_FOLDS: int = 5


def build_cv(
    n_splits: int = DEFAULT_CV_FOLDS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> StratifiedKFold:
    """Build the cross-validation splitter this project uses everywhere.

    The splitter is **stratified** for the same reason the train/test split is
    (see :func:`src.data.split.stratified_split`): with 22 classes, an
    unstratified fold can end up short of a crop entirely, and that fold's score
    would then measure the shuffle rather than the model. It shuffles before
    splitting, with a fixed seed, so folds are neither an artifact of the row
    order in the CSV nor different on every run.

    Args:
        n_splits: Number of folds. Defaults to :data:`DEFAULT_CV_FOLDS` (5).
        random_state: Seed for the shuffle. Defaults to the project-wide
            :data:`src.data.split.DEFAULT_RANDOM_STATE`.

    Returns:
        A configured :class:`~sklearn.model_selection.StratifiedKFold`.

    Raises:
        ValueError: If ``n_splits`` is less than 2.
    """
    if n_splits < 2:
        raise ValueError(f"`n_splits` must be at least 2, got {n_splits}.")
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def evaluate_model(model: BaseEstimator, X: Any, y: Any) -> dict[str, Any]:
    """Score a **fitted** model on a set of labelled examples.

    Returns two things deliberately together. ``accuracy`` is the share of rows
    predicted correctly — one number, easy to quote, and easy to misread.
    ``report`` is scikit-learn's per-class table of precision, recall and F1,
    which shows *where* the correct answers came from: a model can reach a
    respectable accuracy while never predicting some classes at all, and only
    the second output reveals it.

    The model must already be fitted; this function never fits, so it can be
    pointed at a test set without any risk of training on it.

    Args:
        model: A fitted classifier exposing ``predict``.
        X: Features to predict from — anything the model accepts, typically a
            dataframe of the seven crop features.
        y: The true labels for ``X``, of the same length.

    Returns:
        A dictionary with:

        * ``"accuracy"`` — :func:`~sklearn.metrics.accuracy_score` as a float;
        * ``"report"`` — :func:`~sklearn.metrics.classification_report` as a
          printable string;
        * ``"n_samples"`` — how many rows the score was computed on, because an
          accuracy without a sample size is not a claim anyone can check.

        Later weeks add keys to this dictionary; they do not rename these.

    Raises:
        ValueError: If ``X`` and ``y`` have different lengths.
        sklearn.exceptions.NotFittedError: If ``model`` has not been fitted.
    """
    if len(X) != len(y):
        raise ValueError(f"`X` has {len(X)} rows but `y` has {len(y)} labels; they must match.")

    predictions = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "report": classification_report(y, predictions, zero_division=0),
        "n_samples": int(len(y)),
    }


def cross_validated_accuracy(
    model: BaseEstimator,
    X: Any,
    y: Any,
    n_splits: int = DEFAULT_CV_FOLDS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Cross-validate an **unfitted** model and summarise its accuracy.

    k-fold cross-validation cuts the data into ``k`` equal parts, then fits the
    model ``k`` times: each time on ``k - 1`` parts and scored on the part left
    out. Every row is therefore predicted exactly once, by a model that did not
    see it, and the result is ``k`` scores instead of one.

    The spread of those scores matters as much as their mean. A single
    train/test split gives one number with no indication of how much of it was
    luck; five folds show whether "82%" means 81-83% or 68-95%, and a difference
    between two models that is smaller than that spread is not yet a difference.

    ``model`` is cloned by scikit-learn before each fit, so the object passed in
    is left unfitted and can be reused.

    Args:
        model: An unfitted estimator.
        X: Features, one row per example.
        y: Labels, used both to score and to stratify the folds.
        n_splits: Number of folds. Defaults to :data:`DEFAULT_CV_FOLDS` (5).
        random_state: Seed for the fold shuffle, so the same folds are drawn
            every run. Defaults to :data:`src.data.split.DEFAULT_RANDOM_STATE`.

    Returns:
        A dictionary with ``"scores"`` (the per-fold accuracies as a NumPy
        array), ``"mean"``, ``"std"`` and ``"n_splits"``.

    Raises:
        ValueError: If ``X`` and ``y`` have different lengths, or if
            ``n_splits`` is less than 2.
    """
    if len(X) != len(y):
        raise ValueError(f"`X` has {len(X)} rows but `y` has {len(y)} labels; they must match.")

    scores = cross_val_score(
        model,
        X,
        y,
        cv=build_cv(n_splits=n_splits, random_state=random_state),
        scoring="accuracy",
    )
    return {
        "scores": np.asarray(scores),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "n_splits": int(n_splits),
    }
