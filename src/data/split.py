"""Splitting the dataset into a training set and a held-out test set (Week 3).

This module is the single place in the project that decides *how* the data is
divided. Every notebook and test imports :func:`stratified_split` instead of
calling :func:`sklearn.model_selection.train_test_split` directly, so that the
split rule — stratified, 20% held back, seeded with
:data:`DEFAULT_RANDOM_STATE` — is written down once and cannot silently differ
between two places in the course.

Two properties matter, and both are enforced here rather than left to the
caller:

**Stratification.** A plain random split can, by chance, put 26 of a crop's 100
rows in the test set and 74 in the training set. With 22 classes that kind of
drift is likely, and it makes the test score partly a measurement of which rows
happened to land where. Stratifying draws the split *within* each class, so
every crop keeps (as closely as integer arithmetic allows) the same share of
rows on both sides.

**A fixed seed.** ``train_test_split`` shuffles, and shuffling needs a source of
randomness. Left unseeded it draws a different split on every run, so two
students — or the same student on two days — would compute different accuracies
from identical code and be unable to tell a real improvement from noise. Fixing
``random_state`` makes the shuffle deterministic: the same rows land in the same
side of the split forever. The seed's *value* is arbitrary and carries no
meaning; what matters is that it is fixed and recorded. It must never be tuned
to obtain a nicer score — choosing the seed that maximises test accuracy is
simply overfitting the test set by hand.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.validate_schema import TARGET_COLUMN

#: The seed used everywhere in this project unless a caller overrides it. 42 is
#: a convention, nothing more — any constant would do, as long as it never
#: changes and is never chosen to flatter a result.
DEFAULT_RANDOM_STATE: int = 42

#: Share of the rows held back for testing. 20% of 2,200 rows is 440 rows, i.e.
#: 20 rows per crop — small enough to leave 80 rows per crop for training,
#: large enough that a per-class score is not decided by two or three rows.
DEFAULT_TEST_SIZE: float = 0.2


def stratified_split(
    frame: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    target: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a labelled dataframe into train and test parts, preserving class shares.

    The split is stratified on ``target``: each class contributes the same
    proportion of its rows to the test set, so the class balance measured in
    Week 2 survives into both halves.

    Both returned frames keep **all** original columns, features and target
    alike, and their indexes are reset to ``0..n-1``. Keeping the target
    attached means the two frames can be written to CSV and reloaded later
    without a second file recording which label belonged to which row; resetting
    the index means the saved CSVs do not carry a meaningless shuffled index.

    Args:
        frame: The labelled dataset to split. Not modified.
        test_size: Share of rows to hold back for testing, strictly between 0
            and 1. Defaults to :data:`DEFAULT_TEST_SIZE` (0.2).
        random_state: Seed for the shuffle. Defaults to
            :data:`DEFAULT_RANDOM_STATE` so that repeated runs produce a
            byte-identical split.
        target: Name of the column to stratify on. Defaults to
            :data:`src.data.validate_schema.TARGET_COLUMN` (``"label"``).

    Returns:
        A ``(train, test)`` tuple of new dataframes.

    Raises:
        KeyError: If ``target`` is absent from ``frame``.
        ValueError: If ``test_size`` is not strictly between 0 and 1, or if any
            class has fewer than two rows (stratification then cannot place a
            row on both sides of the split).
    """
    if target not in frame.columns:
        raise KeyError(
            f"Target column {target!r} not found in dataframe. "
            f"Available: {list(frame.columns)}"
        )
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"`test_size` must be strictly between 0 and 1, got {test_size}.")

    smallest_class_size = int(frame[target].value_counts().min())
    if smallest_class_size < 2:
        raise ValueError(
            "Stratified splitting needs at least 2 rows per class, but the rarest "
            f"class in {target!r} has {smallest_class_size}."
        )

    train, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=frame[target],
        shuffle=True,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def class_proportions(frame: pd.DataFrame, target: str = TARGET_COLUMN) -> pd.Series:
    """Report each class's share of the rows, sorted by class name.

    Used to check a split: if the same call on the training and test frames
    returns near-identical series, the split really was stratified.

    Args:
        frame: The dataframe to measure.
        target: Name of the class column. Defaults to ``"label"``.

    Returns:
        A series indexed by class value holding proportions that sum to 1,
        sorted alphabetically by class so two calls can be compared directly.

    Raises:
        KeyError: If ``target`` is absent from ``frame``.
    """
    if target not in frame.columns:
        raise KeyError(
            f"Target column {target!r} not found in dataframe. "
            f"Available: {list(frame.columns)}"
        )
    return frame[target].value_counts(normalize=True).sort_index()
