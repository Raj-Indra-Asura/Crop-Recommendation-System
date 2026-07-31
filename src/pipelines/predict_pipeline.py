"""Turn measurements into a crop recommendation, from the saved model (Week 9).

Run it from the repository root for a quick manual check::

    python -m src.pipelines.predict_pipeline

or import it::

    from src.pipelines.predict_pipeline import predict

    predict({"N": 90, "P": 42, "K": 43, "temperature": 25,
             "humidity": 80, "ph": 6.5, "rainfall": 200})

Train on demand, rather than crash
----------------------------------
``models/crop_model.joblib`` is **not** committed: a pickled estimator is a
binary that changes on every retrain, and it can only be reloaded by the exact
library versions that wrote it. Only ``data/raw/`` is version-controlled, and
the model is derived from it.

That makes a missing artifact the *normal* state of a fresh clone, not an error.
So :func:`load_pipeline` trains and saves one when the file is absent, which is
what lets Week 10's tests, the API and CI all start from a clean checkout. The
cost is a few seconds on the first call; every later call loads the file.

What the input dictionary has to contain
----------------------------------------
Exactly the seven features of :data:`src.data.validate_schema.FEATURE_COLUMNS`,
in any order, each a number. The dictionary is turned into a one-row dataframe
with the training column order before it reaches the pipeline, because a
``ColumnTransformer`` fitted on named columns will refuse — or worse, silently
mis-assign — an array whose columns arrived in a different order.
"""

from __future__ import annotations

import argparse
import json
import numbers
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import FEATURE_COLUMNS, MODEL_PATH
from src.pipelines.training_pipeline import train_pipeline

#: A valid request, used by the ``__main__`` demo and quoted in the Week 9 docs.
EXAMPLE_INPUT: dict[str, float] = {
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 25,
    "humidity": 80,
    "ph": 6.5,
    "rainfall": 200,
}


def load_pipeline(
    model_path: Path | str = MODEL_PATH,
    train_if_missing: bool = True,
) -> Pipeline:
    """Load the fitted pipeline, training and saving one first if it is absent.

    Args:
        model_path: Artifact to load. Defaults to :data:`src.config.MODEL_PATH`.
        train_if_missing: When ``True`` (the default) a missing file triggers a
            full training run that writes to ``model_path``. Set it to ``False``
            to require a pre-existing artifact.

    Returns:
        The fitted :class:`~sklearn.pipeline.Pipeline`.

    Raises:
        FileNotFoundError: If the file is absent and ``train_if_missing`` is
            ``False``.
    """
    path = Path(model_path)

    if not path.is_file():
        if not train_if_missing:
            raise FileNotFoundError(
                f"No trained model at {path}. Model artifacts are not committed; "
                "run `python -m src.pipelines.training_pipeline` to build one, or "
                "call load_pipeline(train_if_missing=True)."
            )
        train_pipeline(model_path=path)

    return joblib.load(path)


def _as_frame(input_dict: dict[str, Any]) -> pd.DataFrame:
    """Validate one request and turn it into a one-row dataframe.

    Args:
        input_dict: Mapping of feature name to value.

    Returns:
        A dataframe with one row and the columns of ``FEATURE_COLUMNS``, in the
        training order.

    Raises:
        TypeError: If ``input_dict`` is not a mapping, or a value is not numeric.
        ValueError: If a feature is missing or an unexpected key is present.
    """
    if not isinstance(input_dict, dict):
        raise TypeError(
            f"`input_dict` must be a dict of features, got {type(input_dict).__name__}."
        )

    expected = list(FEATURE_COLUMNS)
    missing = [name for name in expected if name not in input_dict]
    if missing:
        raise ValueError(f"Missing feature(s): {missing}. Required: {expected}.")
    unexpected = sorted(set(input_dict) - set(expected))
    if unexpected:
        raise ValueError(f"Unexpected feature(s): {unexpected}. Required: {expected}.")

    for name in expected:
        value = input_dict[name]
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"Feature {name!r} must be a number, got {value!r}.")

    return pd.DataFrame([{name: float(input_dict[name]) for name in expected}], columns=expected)


def predict(
    input_dict: dict[str, Any],
    model_path: Path | str = MODEL_PATH,
    pipeline: Pipeline | None = None,
) -> str:
    """Recommend one crop for one set of growing conditions.

    Args:
        input_dict: The seven features, e.g. :data:`EXAMPLE_INPUT`.
        model_path: Artifact to use when ``pipeline`` is not supplied.
        pipeline: An already-loaded pipeline, so a caller predicting many rows
            (or Week 10's API, which loads once at start-up) does not re-read
            the file on every call.

    Returns:
        The predicted crop label, one of the 22 in the dataset.

    Raises:
        TypeError: If the input is not a mapping of numbers.
        ValueError: If a feature is missing or unexpected.
    """
    frame = _as_frame(input_dict)
    model = load_pipeline(model_path) if pipeline is None else pipeline
    return str(model.predict(frame)[0])


def predict_proba(
    input_dict: dict[str, Any],
    model_path: Path | str = MODEL_PATH,
    pipeline: Pipeline | None = None,
    top_k: int | None = None,
) -> dict[str, float]:
    """Return the model's confidence in each crop, most likely first.

    A single label hides how close the decision was. Week 8's error analysis
    turned on exactly that: ``rice`` and ``jute`` are separated only by
    rainfall, so a runner-up probability is the honest part of the answer.

    Args:
        input_dict: The seven features.
        model_path: Artifact to use when ``pipeline`` is not supplied.
        pipeline: An already-loaded pipeline.
        top_k: Keep only the ``top_k`` most likely crops. ``None`` keeps all 22.

    Returns:
        An insertion-ordered mapping of crop label to probability, descending.

    Raises:
        TypeError: If the input is not a mapping of numbers.
        ValueError: If a feature is missing or unexpected, or ``top_k`` is not
            positive.
        AttributeError: If the loaded model has no ``predict_proba``.
    """
    if top_k is not None and top_k < 1:
        raise ValueError(f"`top_k` must be at least 1, got {top_k}.")

    frame = _as_frame(input_dict)
    model = load_pipeline(model_path) if pipeline is None else pipeline

    probabilities = model.predict_proba(frame)[0]
    ranked = sorted(
        zip(model.classes_, probabilities, strict=True),
        key=lambda pair: float(pair[1]),
        reverse=True,
    )
    if top_k is not None:
        ranked = ranked[:top_k]
    return {str(label): float(probability) for label, probability in ranked}


def main(argv: list[str] | None = None) -> int:
    """Predict from the command line, for a quick manual check.

    Args:
        argv: Command-line arguments, for testing. Defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success.
    """
    parser = argparse.ArgumentParser(description="Recommend a crop for one set of conditions.")
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "The seven features as a JSON object. Defaults to the example in "
            "the Week 9 docs."
        ),
    )
    parser.add_argument(
        "--model-path",
        default=str(MODEL_PATH),
        help=f"Artifact to load, training one if absent (default: {MODEL_PATH}).",
    )
    args = parser.parse_args(argv)

    features = EXAMPLE_INPUT if args.input is None else json.loads(args.input)

    pipeline = load_pipeline(args.model_path)
    label = predict(features, pipeline=pipeline)
    ranked = predict_proba(features, pipeline=pipeline, top_k=3)

    print(f"Input:      {features}")
    print(f"Prediction: {label}")
    print("Top 3:      " + ", ".join(f"{crop} {prob:.4f}" for crop, prob in ranked.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
