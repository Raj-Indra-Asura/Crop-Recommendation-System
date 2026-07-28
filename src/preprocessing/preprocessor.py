"""Turning raw feature columns into model-ready numbers (Week 3).

The whole of this project's feature preparation is expressed as one
:class:`~sklearn.compose.ColumnTransformer`, built by :func:`build_preprocessor`.
Nothing else in the codebase is allowed to rescale a column by hand.

Why an object rather than a few lines of arithmetic
---------------------------------------------------
A ``ColumnTransformer`` is a *fittable* object with the standard scikit-learn
API:

* ``fit(X)`` — learn the parameters the transformation needs (for
  :class:`~sklearn.preprocessing.StandardScaler`, the mean and standard
  deviation of every column) and store them on the object.
* ``transform(X)`` — apply the stored parameters to any data, including data it
  has never seen.
* ``fit_transform(X)`` — both, as one call. It is a convenience, not a third
  operation, and it belongs to **training data only**.

That split is what makes the "fit on train, transform both" rule enforceable
instead of merely stated. The test set is transformed with the *training* mean
and standard deviation, so no information about the held-out rows can reach the
model — the concrete form of the data leakage Week 2 introduced.

It also means the transformation is a single object that can be handed to a
:class:`~sklearn.pipeline.Pipeline`, saved to disk with the model it was fitted
alongside, and reloaded by the API in Week 10, guaranteeing that a request is
preprocessed at serving time in exactly the way the training rows were.

Which models need this
----------------------
Scaling changes nothing about the information in the data, only its units. It
matters for algorithms that compare or sum feature values across columns —
k-nearest neighbours and SVMs (distances), logistic regression and neural
networks (gradient descent on a weighted sum), PCA (variance). It is irrelevant
to decision trees and their ensembles, which only ever ask "is this column above
this threshold?", a question whose answer is unchanged by rescaling. Weeks 4-6
compare both families, and the same preprocessor is used for both so that the
comparison is about the models rather than about their inputs.
"""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.validate_schema import FEATURE_COLUMNS

#: Name given to the scaling branch inside the returned ColumnTransformer.
#: Naming it explicitly keeps ``preprocessor.named_transformers_["numeric"]``
#: readable, and keeps the generated feature names stable for Week 7.
NUMERIC_TRANSFORMER_NAME: str = "numeric"


def build_preprocessor(
    numeric_features: Sequence[str] = FEATURE_COLUMNS,
) -> ColumnTransformer:
    """Build the unfitted feature preprocessor for this project.

    The returned transformer standardises every numeric feature: each column is
    replaced by ``(value - mean) / std``, where ``mean`` and ``std`` are learned
    from the data passed to ``fit``. On the fitted data the result therefore has
    mean 0 and standard deviation 1 per column, which puts all seven crop
    features — ``ph`` spanning about 6 units, ``K`` spanning 200 — into
    comparable units.

    Standardisation is a *linear* rescaling. It moves and stretches a column but
    does not reorder its values, remove its skew or delete its outliers; a
    right-tailed column stays right-tailed afterwards.

    All columns not named in ``numeric_features`` are dropped
    (``remainder="drop"``), which is deliberate: the target must never be routed
    through the feature preprocessor, and an unexpected extra column should
    disappear rather than reach the model unexamined.

    The object comes back **unfitted**. Fitting it is the caller's decision,
    precisely because *what* it is fitted on is the decision that data leakage
    turns on.

    Args:
        numeric_features: Names of the columns to standardise, in the order the
            output columns should appear. Defaults to the project's seven
            features, :data:`src.data.validate_schema.FEATURE_COLUMNS`.

    Returns:
        An unfitted :class:`~sklearn.compose.ColumnTransformer` whose
        ``transform`` returns a NumPy array with one column per entry of
        ``numeric_features``.

    Raises:
        ValueError: If ``numeric_features`` is empty or contains duplicates.
    """
    requested = list(numeric_features)
    if not requested:
        raise ValueError("`numeric_features` must name at least one column.")
    duplicates = sorted({name for name in requested if requested.count(name) > 1})
    if duplicates:
        raise ValueError(f"`numeric_features` contains duplicate column(s): {duplicates}.")

    return ColumnTransformer(
        transformers=[(NUMERIC_TRANSFORMER_NAME, StandardScaler(), requested)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_preprocessing_pipeline(
    numeric_features: Sequence[str] = FEATURE_COLUMNS,
) -> Pipeline:
    """Wrap :func:`build_preprocessor` in a one-step :class:`~sklearn.pipeline.Pipeline`.

    A ``Pipeline`` chains named steps and exposes them as a single estimator:
    calling ``fit`` fits every step in order, and calling ``transform`` (or, once
    a model is appended as the final step, ``predict``) pushes the data through
    all of them. With one step it changes nothing functionally — its value is
    that later weeks append to it rather than restructure anything.

    From Week 4 the pattern becomes ``Pipeline([("preprocess", ...), ("model",
    ...)])``, at which point ``pipeline.fit(X_train, y_train)`` fits the scaler
    and the model together, and cross-validation in Week 6 re-fits the scaler
    inside every fold — the only way to cross-validate a scaled model without
    leaking the validation fold's statistics into training.

    Args:
        numeric_features: Columns to standardise. Defaults to the project's
            seven features.

    Returns:
        An unfitted pipeline with a single step named ``"preprocess"``.

    Raises:
        ValueError: If ``numeric_features`` is empty or contains duplicates.
    """
    return Pipeline([("preprocess", build_preprocessor(numeric_features))])
