"""The dataset contract: what the raw Crop Recommendation data must look like.

Everything the rest of the project assumes about the data is written down here,
once, as executable checks. :func:`validate_dataset` is called by
:func:`src.data.data_loader.load_data` immediately after the CSV is read, so no
week of this course can accidentally work with data that has drifted from the
contract.

The recorded expectations are also written out in prose in
``docs/curriculum/week01/validation.md`` — in particular the *expected label
set*, which every later week that touches ``label`` must match against.
"""

from __future__ import annotations

import pandas as pd

#: The seven measurable growing conditions used as model inputs.
FEATURE_COLUMNS: tuple[str, ...] = (
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
)

#: The column we are trying to predict.
TARGET_COLUMN: str = "label"

#: Full column set, in the order the source CSV stores them.
EXPECTED_COLUMNS: tuple[str, ...] = (*FEATURE_COLUMNS, TARGET_COLUMN)

#: The published dataset contains exactly this many rows...
EXPECTED_ROW_COUNT: int = 2_200

#: ...spread evenly over exactly this many crop classes.
EXPECTED_LABEL_COUNT: int = 22

#: The exact 22 crop names, recorded in Week 1 and frozen from then on. Any
#: later week that encodes, filters or predicts labels must agree with this
#: set; a new, missing or misspelled crop must fail loudly rather than quietly
#: change what the model is trained to do.
EXPECTED_LABELS: frozenset[str] = frozenset(
    {
        "apple",
        "banana",
        "blackgram",
        "chickpea",
        "coconut",
        "coffee",
        "cotton",
        "grapes",
        "jute",
        "kidneybeans",
        "lentil",
        "maize",
        "mango",
        "mothbeans",
        "mungbean",
        "muskmelon",
        "orange",
        "papaya",
        "pigeonpeas",
        "pomegranate",
        "rice",
        "watermelon",
    }
)


class DatasetValidationError(ValueError):
    """Raised when the raw dataset does not match its expected contract.

    Failing loudly here is deliberate. A silently truncated, re-ordered or
    re-labelled CSV would produce a model that trains without error but is
    quietly wrong, which is far harder to debug than an exception at load time.
    """


def _check_columns(frame: pd.DataFrame) -> None:
    """Check the column names and their order."""
    actual_columns = tuple(frame.columns)
    if actual_columns != EXPECTED_COLUMNS:
        raise DatasetValidationError(
            f"Unexpected columns. Expected {list(EXPECTED_COLUMNS)}, got {list(actual_columns)}. "
            "Stray whitespace in a header (' ph') counts as a different name."
        )


def _check_row_count(frame: pd.DataFrame) -> None:
    """Check the row count against the known size of the published dataset."""
    if len(frame) != EXPECTED_ROW_COUNT:
        raise DatasetValidationError(
            f"Expected {EXPECTED_ROW_COUNT} rows, got {len(frame)}. "
            "The dataset may have been truncated, appended to or duplicated."
        )


def _check_features(frame: pd.DataFrame) -> None:
    """Check that every feature column is numeric and complete."""
    non_numeric = [
        column
        for column in FEATURE_COLUMNS
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise DatasetValidationError(
            f"Feature columns must be numeric, but these are not: {non_numeric}. "
            "A single stray non-numeric cell makes pandas read the whole column as text."
        )

    null_counts = frame[list(FEATURE_COLUMNS)].isna().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    if not columns_with_nulls.empty:
        raise DatasetValidationError(
            f"Feature columns must have no missing values, but found: "
            f"{columns_with_nulls.to_dict()}."
        )


def _check_labels(frame: pd.DataFrame) -> None:
    """Check the target column for nulls, and against the recorded label set."""
    null_label_count = int(frame[TARGET_COLUMN].isna().sum())
    if null_label_count:
        raise DatasetValidationError(
            f"The '{TARGET_COLUMN}' column must have no missing values, "
            f"but {null_label_count} row(s) are null."
        )

    actual_labels = set(frame[TARGET_COLUMN].unique())
    if len(actual_labels) != EXPECTED_LABEL_COUNT:
        raise DatasetValidationError(
            f"Expected {EXPECTED_LABEL_COUNT} distinct crops, got {len(actual_labels)}."
        )

    unexpected = sorted(actual_labels - EXPECTED_LABELS)
    missing = sorted(EXPECTED_LABELS - actual_labels)
    if unexpected or missing:
        raise DatasetValidationError(
            "Label set does not match the set recorded in Week 1 "
            "(docs/curriculum/week01/validation.md). "
            f"Unexpected: {unexpected}. Missing: {missing}."
        )


def validate_dataset(frame: pd.DataFrame) -> None:
    """Check that a dataframe matches the known contract of the raw dataset.

    The checks are exact rather than approximate: the published dataset is a
    fixed, versioned artifact, so any deviation means the file was modified,
    truncated or replaced.

    The contract is:

    * the columns are exactly ``N, P, K, temperature, humidity, ph, rainfall,
      label``, in that order;
    * the seven feature columns are numeric and contain no nulls;
    * ``label`` contains no nulls, and its unique values are exactly the 22
      crops recorded in :data:`EXPECTED_LABELS`;
    * there are exactly 2,200 rows.

    Args:
        frame: The dataframe to check, normally straight from the raw CSV.

    Raises:
        DatasetValidationError: If any part of the contract is violated. The
            message names the specific check that failed.
    """
    _check_columns(frame)
    _check_row_count(frame)
    _check_features(frame)
    _check_labels(frame)
