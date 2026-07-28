"""Data access layer: locating, loading and validating the raw dataset."""

from src.data.data_loader import PROJECT_ROOT, RAW_DATA_PATH, load_data
from src.data.validate_schema import (
    EXPECTED_COLUMNS,
    EXPECTED_LABEL_COUNT,
    EXPECTED_LABELS,
    EXPECTED_ROW_COUNT,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    DatasetValidationError,
    validate_dataset,
)

__all__ = [
    "EXPECTED_COLUMNS",
    "EXPECTED_LABELS",
    "EXPECTED_LABEL_COUNT",
    "EXPECTED_ROW_COUNT",
    "FEATURE_COLUMNS",
    "PROJECT_ROOT",
    "RAW_DATA_PATH",
    "TARGET_COLUMN",
    "DatasetValidationError",
    "load_data",
    "validate_dataset",
]
