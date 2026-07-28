"""Data access layer: locating, loading and validating the raw dataset."""

from src.data.loader import (
    EXPECTED_COLUMNS,
    EXPECTED_LABEL_COUNT,
    EXPECTED_ROW_COUNT,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    DatasetValidationError,
    load_raw_data,
    validate_raw_data,
)

__all__ = [
    "EXPECTED_COLUMNS",
    "EXPECTED_LABEL_COUNT",
    "EXPECTED_ROW_COUNT",
    "RAW_DATA_PATH",
    "TARGET_COLUMN",
    "DatasetValidationError",
    "load_raw_data",
    "validate_raw_data",
]
