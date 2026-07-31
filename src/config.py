"""One place for the settings the whole project shares (Week 9).

Up to Week 8 every decision lived where it was made: the raw CSV path in
:mod:`src.data.data_loader`, the seed in :mod:`src.data.split`, the chosen
model's hyperparameters in a notebook cell. That is fine while the code is only
ever driven by a human reading a notebook, and it stops being fine the moment a
script, a test and (in Week 10) an API all have to agree on the same values.

This module is the agreement. It holds:

* **paths** — where the raw data is read from and where the trained artifact is
  written to;
* **the seed** — one ``RANDOM_STATE`` re-exported from :mod:`src.data.split`, so
  "fixed random_state end-to-end" means one constant rather than four copies of
  the number 42;
* **the final model's identity and hyperparameters** — the Week 8 decision
  (Gaussian naive Bayes, ``var_smoothing=1e-9``) written down as data.

Nothing here computes anything. Config is deliberately inert: reading it must
never load a file, fit a model or have any other side effect, so importing it
from a test or from a web server costs nothing and can never fail halfway.

Overriding a value is done by passing an argument, not by editing this file —
every function that uses a constant here also accepts it as a keyword argument
so that tests can point at a temporary directory instead of ``models/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.data_loader import PROJECT_ROOT, RAW_DATA_PATH
from src.data.split import DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE
from src.data.validate_schema import FEATURE_COLUMNS, TARGET_COLUMN

#: Directory the trained artifact is written to. It is version-controlled as an
#: empty directory; its contents are not (see ``.gitignore``).
MODELS_DIR: Path = PROJECT_ROOT / "models"

#: The one file the training pipeline writes and the predict pipeline reads.
MODEL_PATH: Path = MODELS_DIR / "crop_model.joblib"

#: The seed used by every shuffling, splitting or sampling step in the project.
#: Re-exported from :data:`src.data.split.DEFAULT_RANDOM_STATE` rather than
#: redefined, so the two can never drift apart.
RANDOM_STATE: int = DEFAULT_RANDOM_STATE

#: Share of rows held back for the test set, from Week 3.
TEST_SIZE: float = DEFAULT_TEST_SIZE

#: Key into :data:`src.models.classical_models.CLASSICAL_MODEL_FACTORIES` naming
#: the model Week 8 chose: Gaussian naive Bayes, level with a tuned random
#: forest on accuracy but far cheaper to fit and to serve.
FINAL_MODEL_NAME: str = "naive_bayes"

#: The chosen model's hyperparameters, as keyword arguments for its factory.
#: Week 8's search over twelve values of ``var_smoothing`` returned one
#: identical cross-validated score for all of them, so the tuned value *is* the
#: default — recorded here explicitly because "we kept the default" is a
#: decision, and an undocumented decision is indistinguishable from an accident.
FINAL_MODEL_PARAMS: dict[str, Any] = {"var_smoothing": 1e-9}

#: Step names inside the fitted pipeline. Used by tests and by Week 10 to reach
#: into the artifact without guessing at strings.
PREPROCESSOR_STEP_NAME: str = "preprocess"
MODEL_STEP_NAME: str = "model"

__all__ = [
    "FEATURE_COLUMNS",
    "FINAL_MODEL_NAME",
    "FINAL_MODEL_PARAMS",
    "MODELS_DIR",
    "MODEL_PATH",
    "MODEL_STEP_NAME",
    "PREPROCESSOR_STEP_NAME",
    "PROJECT_ROOT",
    "RANDOM_STATE",
    "RAW_DATA_PATH",
    "TARGET_COLUMN",
    "TEST_SIZE",
]
