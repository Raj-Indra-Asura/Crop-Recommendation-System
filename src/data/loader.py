"""Load and validate the raw Crop Recommendation dataset.

This module is the single place in the project that knows *where* the raw CSV
lives and *what shape it must have*. Every notebook, test and downstream
pipeline imports from here instead of hard-coding a file path, so that moving
the dataset later means editing exactly one line.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# The repository root, resolved relative to this file (src/data/loader.py ->
# up three levels). Deriving it this way keeps the loader working no matter
# which directory the student runs Python or Jupyter from.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

RAW_DATA_PATH: Path = PROJECT_ROOT / "data" / "raw" / "Crop_recommendation.csv"

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


class DatasetValidationError(ValueError):
    """Raised when the raw dataset does not match its expected contract.

    Failing loudly here is deliberate. A silently truncated or re-ordered CSV
    would produce a model that trains without error but is quietly wrong, which
    is far harder to debug than an exception at load time.
    """


def validate_raw_data(frame: pd.DataFrame) -> None:
    """Check that a dataframe matches the known contract of the raw dataset.

    The checks are exact rather than approximate: the published dataset is a
    fixed, versioned artifact, so any deviation means the file was modified,
    truncated or replaced.

    Args:
        frame: The dataframe to check, normally straight from the raw CSV.

    Raises:
        DatasetValidationError: If the columns, row count, class count or
            null-ness of the data differ from the expected contract.
    """
    actual_columns = tuple(frame.columns)
    if actual_columns != EXPECTED_COLUMNS:
        raise DatasetValidationError(
            f"Unexpected columns. Expected {list(EXPECTED_COLUMNS)}, got {list(actual_columns)}."
        )

    if len(frame) != EXPECTED_ROW_COUNT:
        raise DatasetValidationError(
            f"Expected {EXPECTED_ROW_COUNT} rows, got {len(frame)}. "
            "The dataset may have been truncated or appended to."
        )

    null_counts = frame.isna().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    if not columns_with_nulls.empty:
        raise DatasetValidationError(
            f"Expected no missing values, but found: {columns_with_nulls.to_dict()}."
        )

    label_count = frame[TARGET_COLUMN].nunique()
    if label_count != EXPECTED_LABEL_COUNT:
        raise DatasetValidationError(
            f"Expected {EXPECTED_LABEL_COUNT} distinct crops, got {label_count}."
        )


def load_raw_data(path: Path | str | None = None, validate: bool = True) -> pd.DataFrame:
    """Read the raw Crop Recommendation CSV into a dataframe.

    Args:
        path: Optional override for the CSV location. Defaults to
            ``data/raw/Crop_recommendation.csv`` inside the repository.
        validate: When ``True`` (the default) the loaded data is checked
            against the expected contract before being returned.

    Returns:
        A dataframe with the seven feature columns plus the ``label`` target.

    Raises:
        FileNotFoundError: If the CSV is not present at the resolved path.
        DatasetValidationError: If ``validate`` is set and the contents do not
            match the expected contract.
    """
    csv_path = Path(path) if path is not None else RAW_DATA_PATH

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found at {csv_path}.\n"
            "This file is version-controlled and should already be present. "
            "Download 'Crop_recommendation.csv' from "
            "https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset "
            f"and place it at {RAW_DATA_PATH}."
        )

    frame = pd.read_csv(csv_path)

    if validate:
        validate_raw_data(frame)

    return frame
