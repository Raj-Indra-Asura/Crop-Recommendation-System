"""Tests for :mod:`src.data.data_loader` and :mod:`src.data.validate_schema`.

Two groups of tests live here:

* Contract tests that read the real committed CSV and assert its exact shape.
  These are skipped (loudly) if the dataset is missing.
* Behaviour tests that feed deliberately malformed in-memory frames to
  :func:`validate_dataset` and assert that it refuses them. These run
  everywhere, and are what protect every later week from silently loading data
  that no longer matches the Week 1 contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import (
    EXPECTED_COLUMNS,
    EXPECTED_LABEL_COUNT,
    EXPECTED_LABELS,
    EXPECTED_ROW_COUNT,
    FEATURE_COLUMNS,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    DatasetValidationError,
    load_data,
    validate_dataset,
)
from tests.conftest import requires_raw_dataset


@requires_raw_dataset
def test_raw_dataset_file_is_committed():
    assert RAW_DATA_PATH.is_file()


@requires_raw_dataset
def test_raw_dataset_has_expected_shape(raw_data):
    assert raw_data.shape == (EXPECTED_ROW_COUNT, len(EXPECTED_COLUMNS))


@requires_raw_dataset
def test_raw_dataset_has_expected_columns_in_order(raw_data):
    assert tuple(raw_data.columns) == EXPECTED_COLUMNS


@requires_raw_dataset
def test_raw_dataset_has_expected_number_of_crops(raw_data):
    assert raw_data[TARGET_COLUMN].nunique() == EXPECTED_LABEL_COUNT


@requires_raw_dataset
def test_raw_dataset_label_set_matches_recorded_set(raw_data):
    assert set(raw_data[TARGET_COLUMN].unique()) == set(EXPECTED_LABELS)


@requires_raw_dataset
def test_raw_dataset_has_no_missing_values(raw_data):
    assert not raw_data.isna().any().any()


@requires_raw_dataset
def test_raw_dataset_features_are_numeric(raw_data):
    features = raw_data.drop(columns=[TARGET_COLUMN])
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes)


@requires_raw_dataset
def test_load_data_validates_by_default():
    # Should not raise: the committed dataset must satisfy its own contract.
    assert len(load_data()) == EXPECTED_ROW_COUNT


def _contract_frame() -> pd.DataFrame:
    """Build a synthetic frame that satisfies the full contract.

    2,200 rows over the 22 recorded crops, 100 rows each — the same shape as
    the real file, but with values that are obviously synthetic so no test can
    mistake it for the dataset itself.
    """
    labels = sorted(EXPECTED_LABELS)
    rows_per_label = EXPECTED_ROW_COUNT // EXPECTED_LABEL_COUNT
    frame = pd.DataFrame(
        {
            column: np.arange(EXPECTED_ROW_COUNT, dtype=float)
            for column in FEATURE_COLUMNS
        }
    )
    frame[TARGET_COLUMN] = [label for label in labels for _ in range(rows_per_label)]
    return frame[list(EXPECTED_COLUMNS)]


def test_validate_accepts_a_frame_matching_the_contract():
    validate_dataset(_contract_frame())  # must not raise


def test_validate_rejects_wrong_columns():
    frame = _contract_frame().rename(columns={"N": "nitrogen"})
    with pytest.raises(DatasetValidationError, match="Unexpected columns"):
        validate_dataset(frame)


def test_validate_rejects_whitespace_in_column_names():
    frame = _contract_frame().rename(columns={"ph": " ph"})
    with pytest.raises(DatasetValidationError, match="Unexpected columns"):
        validate_dataset(frame)


def test_validate_rejects_reordered_columns():
    frame = _contract_frame()[["P", "N", "K", "temperature", "humidity", "ph", "rainfall", "label"]]
    with pytest.raises(DatasetValidationError, match="Unexpected columns"):
        validate_dataset(frame)


def test_validate_rejects_wrong_row_count():
    with pytest.raises(DatasetValidationError, match="rows"):
        validate_dataset(_contract_frame().head(100))


def test_validate_rejects_null_features():
    frame = _contract_frame()
    frame.loc[0, "ph"] = np.nan
    with pytest.raises(DatasetValidationError, match="no missing values"):
        validate_dataset(frame)


def test_validate_rejects_non_numeric_features():
    frame = _contract_frame()
    frame["rainfall"] = frame["rainfall"].astype(str)
    with pytest.raises(DatasetValidationError, match="must be numeric"):
        validate_dataset(frame)


def test_validate_rejects_null_labels():
    frame = _contract_frame()
    frame.loc[0, TARGET_COLUMN] = None
    with pytest.raises(DatasetValidationError, match="must have no missing values"):
        validate_dataset(frame)


def test_validate_rejects_unexpected_label_values():
    frame = _contract_frame()
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].replace({"rice": "Rice"})
    with pytest.raises(DatasetValidationError, match="Label set does not match"):
        validate_dataset(frame)


def test_validate_rejects_wrong_number_of_labels():
    frame = _contract_frame()
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].replace({"rice": "maize"})
    with pytest.raises(DatasetValidationError, match="distinct crops"):
        validate_dataset(frame)


def test_load_data_reports_missing_file_clearly(tmp_path):
    missing = tmp_path / "Crop_recommendation.csv"
    with pytest.raises(FileNotFoundError, match="Raw dataset not found"):
        load_data(path=missing)


def test_load_data_raises_on_malformed_csv(tmp_path):
    malformed = tmp_path / "Crop_recommendation.csv"
    _contract_frame().head(10).to_csv(malformed, index=False)
    with pytest.raises(DatasetValidationError):
        load_data(path=malformed)
