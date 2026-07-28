"""Data access layer: locating, loading, validating and splitting the dataset."""

from src.data.data_loader import PROJECT_ROOT, RAW_DATA_PATH, load_data
from src.data.split import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    class_proportions,
    stratified_split,
)
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
    "DEFAULT_RANDOM_STATE",
    "DEFAULT_TEST_SIZE",
    "EXPECTED_COLUMNS",
    "EXPECTED_LABELS",
    "EXPECTED_LABEL_COUNT",
    "EXPECTED_ROW_COUNT",
    "FEATURE_COLUMNS",
    "PROJECT_ROOT",
    "RAW_DATA_PATH",
    "TARGET_COLUMN",
    "DatasetValidationError",
    "class_proportions",
    "load_data",
    "stratified_split",
    "validate_dataset",
]
