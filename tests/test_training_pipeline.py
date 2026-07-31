"""Tests for Week 9's training pipeline, :mod:`src.pipelines.training_pipeline`.

The point of the week is that training is now *code that runs unattended*, so
these tests check the things a script has to guarantee and a notebook never did:

1. **Shape.** The built object is one unfitted ``Pipeline`` whose two steps are
   the Week 3 preprocessor and the Week 8 model, reachable by name.
2. **It produces an artifact.** Training on a small sample writes a file, and
   that file reloads into a working estimator.
3. **It reports honestly.** The returned metrics come from held-out rows and
   the split sizes add up to the data it was given.
4. **It is reproducible.** Two runs with the same seed give the same score;
   a different seed is allowed to differ.
5. **It fails loudly on bad configuration**, rather than half-training.

Every test writes to ``tmp_path``, never to the repository's ``models/``.
"""

from __future__ import annotations

import joblib
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from src.config import (
    FEATURE_COLUMNS,
    FINAL_MODEL_NAME,
    FINAL_MODEL_PARAMS,
    MODEL_STEP_NAME,
    PREPROCESSOR_STEP_NAME,
    TARGET_COLUMN,
)
from src.data import EXPECTED_LABELS
from src.pipelines.training_pipeline import (
    MODEL_FACTORIES,
    build_model,
    build_training_pipeline,
    main,
    save_pipeline,
    train_pipeline,
)
from tests.conftest import requires_raw_dataset

#: Rows kept per crop for the sampled fixture: enough for a stratified 80/20
#: split (8 train, 2 test per class) and small enough to keep the suite fast.
ROWS_PER_CLASS = 10


@pytest.fixture(scope="module")
def small_sample(raw_data):
    """A stratified slice of the real data: ``ROWS_PER_CLASS`` rows per crop."""
    return (
        raw_data.groupby(TARGET_COLUMN, group_keys=False)
        .head(ROWS_PER_CLASS)
        .reset_index(drop=True)
    )


# --- build_model / build_training_pipeline -----------------------------------


def test_build_model_returns_the_week8_final_model():
    model = build_model()
    assert isinstance(model, GaussianNB)
    assert model.var_smoothing == FINAL_MODEL_PARAMS["var_smoothing"]


def test_build_model_accepts_any_registered_name():
    assert set(MODEL_FACTORIES) >= {"naive_bayes", "random_forest", "decision_tree"}
    assert build_model("decision_tree", {}) is not None


def test_build_model_rejects_an_unknown_name():
    with pytest.raises(KeyError, match="Unknown model"):
        build_model("crystal_ball", {})


def test_pipeline_has_the_two_named_steps():
    pipeline = build_training_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == [PREPROCESSOR_STEP_NAME, MODEL_STEP_NAME]
    assert isinstance(pipeline.named_steps[PREPROCESSOR_STEP_NAME], ColumnTransformer)
    assert isinstance(pipeline.named_steps[MODEL_STEP_NAME], GaussianNB)


def test_pipeline_comes_back_unfitted(small_sample):
    pipeline = build_training_pipeline()
    with pytest.raises(NotFittedError):
        pipeline.predict(small_sample[list(FEATURE_COLUMNS)])


# --- train_pipeline ----------------------------------------------------------


@requires_raw_dataset
def test_training_writes_a_model_file(small_sample, tmp_path):
    destination = tmp_path / "crop_model.joblib"
    result = train_pipeline(frame=small_sample, model_path=destination)

    assert destination.is_file()
    assert destination.stat().st_size > 0
    assert result["model_path"] == destination


@requires_raw_dataset
def test_saved_file_reloads_and_predicts(small_sample, tmp_path):
    destination = tmp_path / "crop_model.joblib"
    train_pipeline(frame=small_sample, model_path=destination)

    reloaded = joblib.load(destination)
    assert isinstance(reloaded, Pipeline)

    predictions = reloaded.predict(small_sample[list(FEATURE_COLUMNS)].head(5))
    assert len(predictions) == 5
    assert set(predictions) <= set(EXPECTED_LABELS)


@requires_raw_dataset
def test_training_reports_split_sizes_and_metrics(small_sample, tmp_path):
    result = train_pipeline(frame=small_sample, model_path=tmp_path / "m.joblib")

    assert result["n_train"] + result["n_test"] == len(small_sample)
    assert result["model_name"] == FINAL_MODEL_NAME
    assert result["model_params"] == FINAL_MODEL_PARAMS

    metrics = result["metrics"]
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["n_samples"] == result["n_test"]
    assert {"macro_f1", "weighted_f1", "confusion_matrix", "report"} <= set(metrics)


@requires_raw_dataset
def test_training_can_skip_saving(small_sample, tmp_path):
    result = train_pipeline(frame=small_sample, model_path=None)

    assert result["model_path"] is None
    assert list(tmp_path.iterdir()) == []
    assert result["pipeline"].predict(small_sample[list(FEATURE_COLUMNS)].head(1)) is not None


@requires_raw_dataset
def test_training_is_reproducible(small_sample, tmp_path):
    first = train_pipeline(frame=small_sample, model_path=tmp_path / "a.joblib")
    second = train_pipeline(frame=small_sample, model_path=tmp_path / "b.joblib")

    assert first["metrics"]["accuracy"] == second["metrics"]["accuracy"]
    assert first["n_train"] == second["n_train"]

    features = small_sample[list(FEATURE_COLUMNS)]
    assert list(first["pipeline"].predict(features)) == list(second["pipeline"].predict(features))


@requires_raw_dataset
def test_a_different_seed_is_allowed_to_split_differently(small_sample, tmp_path):
    seeded = train_pipeline(frame=small_sample, model_path=None, random_state=42)
    other = train_pipeline(frame=small_sample, model_path=None, random_state=7)

    assert seeded["n_train"] == other["n_train"]  # only *which* rows may differ
    assert seeded["metrics"]["n_samples"] == other["metrics"]["n_samples"]


@requires_raw_dataset
def test_training_creates_a_missing_models_directory(small_sample, tmp_path):
    destination = tmp_path / "does" / "not" / "exist" / "crop_model.joblib"
    train_pipeline(frame=small_sample, model_path=destination)
    assert destination.is_file()


def test_training_rejects_an_unknown_model(small_sample, tmp_path):
    with pytest.raises(KeyError, match="Unknown model"):
        train_pipeline(
            frame=small_sample,
            model_path=tmp_path / "never_written.joblib",
            model_name="crystal_ball",
        )
    assert not (tmp_path / "never_written.joblib").exists()


def test_save_pipeline_returns_the_path_it_wrote(small_sample, tmp_path):
    pipeline = build_training_pipeline()
    pipeline.fit(small_sample[list(FEATURE_COLUMNS)], small_sample[TARGET_COLUMN])

    written = save_pipeline(pipeline, tmp_path / "explicit.joblib")

    assert written == tmp_path / "explicit.joblib"
    assert isinstance(joblib.load(written), Pipeline)


# --- the command-line entry point --------------------------------------------


@requires_raw_dataset
def test_main_trains_saves_and_exits_zero(tmp_path, capsys):
    destination = tmp_path / "cli.joblib"

    exit_code = main(["--model-path", str(destination)])

    assert exit_code == 0
    assert destination.is_file()

    printed = capsys.readouterr().out
    assert "Accuracy:" in printed
    assert str(destination) in printed
