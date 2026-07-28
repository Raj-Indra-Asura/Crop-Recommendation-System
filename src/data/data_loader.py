"""Load the raw Crop Recommendation dataset.

This module is the single place in the project that knows *where* the raw CSV
lives. Every notebook, test and downstream pipeline imports from here instead
of hard-coding a file path, so that moving the dataset later means editing
exactly one line.

Loading always runs the dataset contract defined in
:mod:`src.data.validate_schema` immediately after reading the file, so every
later week receives data that has already been checked.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.validate_schema import validate_dataset

# The repository root, resolved relative to this file (src/data/data_loader.py
# -> up three levels). Deriving it this way keeps the loader working no matter
# which directory the student runs Python or Jupyter from.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

RAW_DATA_PATH: Path = PROJECT_ROOT / "data" / "raw" / "Crop_recommendation.csv"


def load_data(path: Path | str | None = None, validate: bool = True) -> pd.DataFrame:
    """Read the raw Crop Recommendation CSV into a dataframe.

    Args:
        path: Optional override for the CSV location. Defaults to
            ``data/raw/Crop_recommendation.csv`` inside the repository.
        validate: When ``True`` (the default) the loaded data is passed to
            :func:`src.data.validate_schema.validate_dataset` before being
            returned.

    Returns:
        A dataframe with the seven feature columns plus the ``label`` target,
        2,200 rows deep.

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
        validate_dataset(frame)

    return frame
