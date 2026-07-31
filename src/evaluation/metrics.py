"""Evaluation: turning a fitted model into numbers you can defend.

This module is the project's single place for scoring a model. Week 4 gives it
two responsibilities:

* :func:`evaluate_model` — score an already-fitted model on some data, returning
  accuracy *and* the per-class breakdown that stops accuracy from being read on
  its own;
* :func:`cross_validated_accuracy` — score an *unfitted* model with k-fold
  cross-validation, so the number reported is a mean over several splits with a
  spread attached, rather than one lucky split.

Week 8 adds :func:`confusion_frame` and extends :func:`evaluate_model` with
macro/weighted F1 and a labelled confusion matrix, alongside two new modules:
:mod:`src.evaluation.tuning` (hyperparameter search) and
:mod:`src.evaluation.explainability` (permutation importance and
single-prediction explanations).

It is written to be **extended, not replaced**: Week 5 compares real
classifiers through the same helpers, Week 6 tunes them, and Week 8 adds
precision, recall, F1 and the confusion matrix as first-class outputs. The
per-class report is already here in string form precisely so that "accuracy is
not the whole story" is visible from Week 4 rather than announced in Week 8.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
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
        * ``"report_dict"`` — the same report as nested dictionaries, for
          plotting or asserting on;
        * ``"macro_f1"`` / ``"weighted_f1"`` — F1 averaged over classes with
          equal weight, and weighted by class support (Week 8);
        * ``"macro_precision"`` / ``"macro_recall"`` — the two halves of macro F1;
        * ``"confusion_matrix"`` — a labelled :class:`~pandas.DataFrame` from
          :func:`confusion_frame` (Week 8);
        * ``"labels"`` — the class order used by that matrix;
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
    labels = sorted(set(np.asarray(y).tolist()) | set(np.asarray(predictions).tolist()))
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "report": classification_report(y, predictions, zero_division=0),
        "report_dict": classification_report(y, predictions, zero_division=0, output_dict=True),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, predictions, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_frame(y, predictions, labels=labels),
        "labels": labels,
        "n_samples": int(len(y)),
    }


def confusion_frame(y_true: Any, y_pred: Any, labels: list[Any] | None = None) -> pd.DataFrame:
    """Build a labelled confusion matrix: true classes down, predicted across.

    Cell ``(i, j)`` counts the examples whose true class is ``i`` and whose
    predicted class is ``j``. The diagonal is therefore the correct answers, and
    every off-diagonal cell is one specific mistake — not "the model got 2%
    wrong" but "three fields of muskmelon were sent watermelon's advice".

    With 22 classes the raw NumPy array from scikit-learn is unreadable, so this
    wrapper attaches the class names to both axes.

    Args:
        y_true: The true labels.
        y_pred: The predicted labels, of the same length.
        labels: Row/column order. Defaults to the sorted union of both inputs,
            so a class that is never predicted still gets a row of zeros.

    Returns:
        A :class:`~pandas.DataFrame` of counts, indexed by true class with
        predicted classes as columns.

    Raises:
        ValueError: If the two inputs have different lengths.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"`y_true` has {len(y_true)} labels but `y_pred` has {len(y_pred)}; they must match."
        )
    if labels is None:
        labels = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(
        matrix,
        index=pd.Index(labels, name="true"),
        columns=pd.Index(labels, name="predicted"),
    )


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
