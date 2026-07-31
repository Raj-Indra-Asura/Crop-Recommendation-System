"""Tests for Week 9's prediction pipeline, :mod:`src.pipelines.predict_pipeline`.

The central test is the **round trip**: train, save, reload, predict on a known
input, and confirm the answer is one of the 22 crops in the dataset. Around it:

1. **Train on demand.** A missing artifact is the normal state of a fresh clone,
   so :func:`load_pipeline` must build one rather than crash — and must *not*
   when the caller explicitly forbids it.
2. **Input validation.** A missing, extra or non-numeric feature is rejected
   with a message that names the problem, before it can reach a model that
   would otherwise silently mis-align columns.
3. **Column order does not matter to the caller**, because the request is
   rebuilt in the training order before it is passed on.
4. **Probabilities agree with the label**: the argmax of ``predict_proba`` is
   what ``predict`` returns.

Every test writes to ``tmp_path``, so none of them depend on — or overwrite —
``models/crop_model.joblib``.
"""

from __future__ import annotations

import pytest
from sklearn.pipeline import Pipeline

from src.config import FEATURE_COLUMNS
from src.data import EXPECTED_LABELS
from src.pipelines.predict_pipeline import (
    EXAMPLE_INPUT,
    load_pipeline,
    predict,
    predict_proba,
)
from src.pipelines.training_pipeline import train_pipeline
from tests.conftest import requires_raw_dataset


@pytest.fixture(scope="module")
def trained_model_path(tmp_path_factory):
    """Train the real pipeline once per module and return the saved artifact."""
    destination = tmp_path_factory.mktemp("models") / "crop_model.joblib"
    train_pipeline(model_path=destination)
    return destination


# --- the round trip ----------------------------------------------------------


@requires_raw_dataset
def test_round_trip_train_save_reload_predict(trained_model_path):
    assert trained_model_path.is_file()

    reloaded = load_pipeline(trained_model_path)
    assert isinstance(reloaded, Pipeline)

    label = predict(EXAMPLE_INPUT, model_path=trained_model_path)
    assert isinstance(label, str)
    assert label in EXPECTED_LABELS


@requires_raw_dataset
def test_prediction_is_stable_across_reloads(trained_model_path):
    first = predict(EXAMPLE_INPUT, model_path=trained_model_path)
    second = predict(EXAMPLE_INPUT, pipeline=load_pipeline(trained_model_path))
    assert first == second


@requires_raw_dataset
def test_feature_order_in_the_request_does_not_matter(trained_model_path):
    reversed_input = {name: EXAMPLE_INPUT[name] for name in reversed(list(EXAMPLE_INPUT))}
    assert list(reversed_input) != list(EXAMPLE_INPUT)

    pipeline = load_pipeline(trained_model_path)
    assert predict(reversed_input, pipeline=pipeline) == predict(EXAMPLE_INPUT, pipeline=pipeline)


# --- training on demand ------------------------------------------------------


@requires_raw_dataset
def test_load_pipeline_trains_when_the_artifact_is_missing(tmp_path):
    destination = tmp_path / "absent.joblib"
    assert not destination.exists()

    pipeline = load_pipeline(destination)

    assert isinstance(pipeline, Pipeline)
    assert destination.is_file()


def test_load_pipeline_can_refuse_to_train(tmp_path):
    with pytest.raises(FileNotFoundError, match="No trained model"):
        load_pipeline(tmp_path / "absent.joblib", train_if_missing=False)


# --- input validation --------------------------------------------------------


@requires_raw_dataset
def test_missing_feature_is_rejected(trained_model_path):
    incomplete = {name: value for name, value in EXAMPLE_INPUT.items() if name != "ph"}
    with pytest.raises(ValueError, match="Missing feature"):
        predict(incomplete, model_path=trained_model_path)


@requires_raw_dataset
def test_unexpected_feature_is_rejected(trained_model_path):
    extra = {**EXAMPLE_INPUT, "altitude": 300}
    with pytest.raises(ValueError, match="Unexpected feature"):
        predict(extra, model_path=trained_model_path)


@requires_raw_dataset
@pytest.mark.parametrize("bad_value", ["6.5", None, True, [6.5]])
def test_non_numeric_feature_is_rejected(trained_model_path, bad_value):
    with pytest.raises(TypeError, match="must be a number"):
        predict({**EXAMPLE_INPUT, "ph": bad_value}, model_path=trained_model_path)


@requires_raw_dataset
def test_non_mapping_input_is_rejected(trained_model_path):
    with pytest.raises(TypeError, match="must be a dict"):
        predict([90, 42, 43, 25, 80, 6.5, 200], model_path=trained_model_path)


# --- probabilities -----------------------------------------------------------


@requires_raw_dataset
def test_probabilities_cover_every_crop_and_sum_to_one(trained_model_path):
    ranked = predict_proba(EXAMPLE_INPUT, model_path=trained_model_path)

    assert set(ranked) == set(EXPECTED_LABELS)
    assert sum(ranked.values()) == pytest.approx(1.0)


@requires_raw_dataset
def test_probabilities_are_sorted_and_agree_with_the_label(trained_model_path):
    pipeline = load_pipeline(trained_model_path)
    ranked = predict_proba(EXAMPLE_INPUT, pipeline=pipeline)

    values = list(ranked.values())
    assert values == sorted(values, reverse=True)
    assert next(iter(ranked)) == predict(EXAMPLE_INPUT, pipeline=pipeline)


@requires_raw_dataset
def test_top_k_truncates(trained_model_path):
    ranked = predict_proba(EXAMPLE_INPUT, model_path=trained_model_path, top_k=3)
    assert len(ranked) == 3


@requires_raw_dataset
def test_top_k_must_be_positive(trained_model_path):
    with pytest.raises(ValueError, match="at least 1"):
        predict_proba(EXAMPLE_INPUT, model_path=trained_model_path, top_k=0)


# --- the example input -------------------------------------------------------


def test_example_input_names_exactly_the_seven_features():
    assert set(EXAMPLE_INPUT) == set(FEATURE_COLUMNS)
