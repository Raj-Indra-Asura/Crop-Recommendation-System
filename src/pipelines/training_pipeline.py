"""Train the crop recommendation model end to end and save it (Week 9).

Run it from the repository root::

    python -m src.pipelines.training_pipeline

The script is the notebook work of Weeks 3-8 with the notebook removed. It
loads ``data/raw/Crop_recommendation.csv``, makes the same stratified 80/20
split as Week 3, builds one :class:`~sklearn.pipeline.Pipeline` holding the
Week 3 ``ColumnTransformer`` *and* the Week 8 final model, fits it on the
training rows, scores it on the held-out rows, and writes the fitted object to
``models/crop_model.joblib``.

Why one ``Pipeline`` instead of a scaler and a model side by side
-----------------------------------------------------------------
A pipeline is a single estimator. ``fit`` fits the scaler and then the model;
``predict`` applies the *fitted* scaler and then the model. That matters at
three separate moments:

* **Cross-validation** re-fits the scaler inside every fold, so a validation
  fold's mean and standard deviation never reach the training rows.
* **Serialization** saves one object. A scaler saved separately from its model
  is a pair that can be reunited in the wrong order, or with one half stale.
* **Serving** (Week 10) receives raw measurements in the units a farmer
  reports, and the pipeline applies exactly the transformation the training
  rows were given — the training/serving skew problem solved by construction
  rather than by discipline.

What ``joblib`` does and does not save
--------------------------------------
:func:`joblib.dump` pickles the fitted Python object: the learned parameters
(the scaler's means and scales, the model's per-class means, variances and
priors) plus references to the classes that hold them. It does **not** save the
code of those classes, nor the versions of scikit-learn, NumPy or Python that
defined them. Reloading the file therefore needs the same libraries installed —
which is why the artifact is rebuilt from source rather than committed, and why
``requirements.txt`` is pinned.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import (
    FEATURE_COLUMNS,
    FINAL_MODEL_NAME,
    FINAL_MODEL_PARAMS,
    MODEL_PATH,
    MODEL_STEP_NAME,
    PREPROCESSOR_STEP_NAME,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data import load_data, stratified_split
from src.evaluation import evaluate_model
from src.models.classical_models import CLASSICAL_MODEL_FACTORIES
from src.models.ensemble_models import ENSEMBLE_MODEL_FACTORIES
from src.preprocessing import build_preprocessor

#: Every model factory the project exposes, under the name used in the config.
#: Merging the two Week 5-7 registries here means the final model can be swapped
#: by editing one string in :mod:`src.config`, with no change to this script.
MODEL_FACTORIES: dict[str, Any] = {**CLASSICAL_MODEL_FACTORIES, **ENSEMBLE_MODEL_FACTORIES}


def build_model(
    name: str = FINAL_MODEL_NAME,
    params: dict[str, Any] | None = None,
):
    """Build the unfitted estimator named in the config.

    Args:
        name: Key into :data:`MODEL_FACTORIES`. Defaults to
            :data:`src.config.FINAL_MODEL_NAME` (``"naive_bayes"``).
        params: Keyword arguments for the factory. Defaults to
            :data:`src.config.FINAL_MODEL_PARAMS`.

    Returns:
        An unfitted scikit-learn classifier.

    Raises:
        KeyError: If ``name`` is not a known model.
    """
    if name not in MODEL_FACTORIES:
        raise KeyError(
            f"Unknown model {name!r}. Available: {sorted(MODEL_FACTORIES)}."
        )
    settings = dict(FINAL_MODEL_PARAMS if params is None else params)
    return MODEL_FACTORIES[name](**settings)


def build_training_pipeline(
    model_name: str = FINAL_MODEL_NAME,
    model_params: dict[str, Any] | None = None,
) -> Pipeline:
    """Build the unfitted, end-to-end pipeline: preprocessing then model.

    The two steps are named :data:`src.config.PREPROCESSOR_STEP_NAME` and
    :data:`src.config.MODEL_STEP_NAME`, so a caller — a hyperparameter search,
    a test, or Week 10's API — can reach either half by name rather than by
    position.

    Args:
        model_name: Which factory to use for the final step.
        model_params: Hyperparameters for that factory.

    Returns:
        An unfitted :class:`~sklearn.pipeline.Pipeline`.

    Raises:
        KeyError: If ``model_name`` is not a known model.
    """
    return Pipeline(
        [
            (PREPROCESSOR_STEP_NAME, build_preprocessor(FEATURE_COLUMNS)),
            (MODEL_STEP_NAME, build_model(model_name, model_params)),
        ]
    )


def save_pipeline(pipeline: Pipeline, path: Path | str = MODEL_PATH) -> Path:
    """Serialize a fitted pipeline to disk with :func:`joblib.dump`.

    The parent directory is created if it does not exist, because ``models/``
    holds no committed files and may be absent in a fresh clone.

    Args:
        pipeline: The fitted pipeline to save.
        path: Destination file. Defaults to :data:`src.config.MODEL_PATH`.

    Returns:
        The resolved path actually written to.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, destination)
    return destination


def train_pipeline(
    frame: pd.DataFrame | None = None,
    model_path: Path | str | None = MODEL_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    model_name: str = FINAL_MODEL_NAME,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit the full pipeline on the training split, score it, and save it.

    Every source of randomness in the run is seeded from ``random_state``: the
    split here, and the model's own ``random_state`` where it has one (naive
    Bayes does not — it is deterministic given its input). Two runs of this
    function on the same data therefore produce the same numbers.

    Args:
        frame: Labelled data to train on. Defaults to ``None``, which loads and
            validates ``data/raw/Crop_recommendation.csv``. Tests pass a small
            sample here to stay fast.
        model_path: Where to write the fitted pipeline. Pass ``None`` to skip
            saving and only return the fitted object.
        test_size: Share of rows held back for the held-out score.
        random_state: Seed for the split.
        model_name: Which model to put in the final step.
        model_params: That model's hyperparameters.

    Returns:
        A dictionary with:

        * ``"pipeline"`` — the fitted :class:`~sklearn.pipeline.Pipeline`;
        * ``"metrics"`` — the :func:`src.evaluation.evaluate_model` dictionary
          computed on the held-out rows;
        * ``"model_path"`` — the :class:`~pathlib.Path` written, or ``None``;
        * ``"n_train"`` / ``"n_test"`` — the two split sizes;
        * ``"model_name"`` / ``"model_params"`` — what was actually built, so a
          run's log records its own configuration.

    Raises:
        KeyError: If ``model_name`` is unknown, or the target column is absent.
        ValueError: If the split parameters or the data make a stratified split
            impossible.
    """
    data = load_data() if frame is None else frame

    train, test = stratified_split(data, test_size=test_size, random_state=random_state)
    X_train, y_train = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]
    X_test, y_test = test[list(FEATURE_COLUMNS)], test[TARGET_COLUMN]

    pipeline = build_training_pipeline(model_name, model_params)
    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(pipeline, X_test, y_test)

    saved_to = save_pipeline(pipeline, model_path) if model_path is not None else None

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "model_path": saved_to,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "model_name": model_name,
        "model_params": dict(FINAL_MODEL_PARAMS if model_params is None else model_params),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the training pipeline as a command-line script.

    Args:
        argv: Command-line arguments, for testing. Defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, so the shell (and CI) can branch on the exit code.
    """
    parser = argparse.ArgumentParser(description="Train and save the crop recommendation model.")
    parser.add_argument(
        "--model-path",
        default=str(MODEL_PATH),
        help=f"Where to write the fitted pipeline (default: {MODEL_PATH}).",
    )
    parser.add_argument(
        "--model-name",
        default=FINAL_MODEL_NAME,
        choices=sorted(MODEL_FACTORIES),
        help=f"Which model to put in the final step (default: {FINAL_MODEL_NAME}).",
    )
    args = parser.parse_args(argv)

    model_params = None if args.model_name == FINAL_MODEL_NAME else {}
    result = train_pipeline(
        model_path=args.model_path,
        model_name=args.model_name,
        model_params=model_params,
    )

    metrics = result["metrics"]
    print(f"Model:        {result['model_name']} {result['model_params']}")
    print(f"Train rows:   {result['n_train']}")
    print(f"Test rows:    {result['n_test']}")
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"Macro F1:     {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:  {metrics['weighted_f1']:.4f}")
    print(f"Saved to:     {result['model_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
