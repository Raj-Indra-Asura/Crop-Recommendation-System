"""Feature preparation: the transformations applied between data and model.

Week 3 adds :mod:`src.preprocessing.preprocessor`, which builds the
:class:`~sklearn.compose.ColumnTransformer` used by
``notebooks/03_data_preparation.ipynb`` and by every model from Week 4 onward.
"""

from src.preprocessing.preprocessor import (
    NUMERIC_TRANSFORMER_NAME,
    build_preprocessing_pipeline,
    build_preprocessor,
)

__all__ = [
    "NUMERIC_TRANSFORMER_NAME",
    "build_preprocessing_pipeline",
    "build_preprocessor",
]
