"""Tests for :mod:`src.data.loader`.

Two groups of tests live here:

* Contract tests that read the real committed CSV and assert its exact shape.
  These are skipped (loudly) if the dataset is missing.
* Behaviour tests for the loader itself, which build small in-memory frames and
  therefore run everywhere.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import (
    EXPECTED_COLUMNS,
    EXPECTED_LABEL_COUNT,
    EXPECTED_ROW_COUNT,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    DatasetValidationError,
    load_raw_data,
    validate_raw_data,
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
def test_raw_dataset_has_no_missing_values(raw_data):
    assert not raw_data.isna().any().any()


@requires_raw_dataset
def test_raw_dataset_features_are_numeric(raw_data):
    features = raw_data.drop(columns=[TARGET_COLUMN])
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes)


@requires_raw_dataset
def test_load_raw_data_validates_by_default():
    # Should not raise: the committed dataset must satisfy its own contract.
    assert len(load_raw_data()) == EXPECTED_ROW_COUNT


def _valid_frame() -> pd.DataFrame:
    """Build a minimal frame that satisfies every rule except row/label counts."""
    return pd.DataFrame(
        [[90, 42, 43, 25.0, 80.0, 6.5, 200.0, "rice"]],
        columns=list(EXPECTED_COLUMNS),
    )


def test_validate_rejects_wrong_columns():
    frame = _valid_frame().rename(columns={"N": "nitrogen"})
    with pytest.raises(DatasetValidationError, match="Unexpected columns"):
        validate_raw_data(frame)


def test_validate_rejects_wrong_row_count():
    with pytest.raises(DatasetValidationError, match="rows"):
        validate_raw_data(_valid_frame())


def test_load_raw_data_reports_missing_file_clearly(tmp_path):
    missing = tmp_path / "Crop_recommendation.csv"
    with pytest.raises(FileNotFoundError, match="Raw dataset not found"):
        load_raw_data(path=missing)
