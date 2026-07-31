"""Tests for Week 10's HTTP layer, :mod:`api.main` and :mod:`api.schemas`.

The point of these tests is the **boundary**, not the model. The model is
already covered by ``tests/test_predict_pipeline.py``; what is new this week is
everything that happens to a request before and after it:

1. a valid payload returns **200** with a crop label from the dataset;
2. an invalid payload — missing field, out-of-range value, wrong type, unknown
   key — returns **422**, and the model is never called;
3. an unexpected failure *inside* prediction returns **500**, not a traceback;
4. ``/health`` answers whether a model is loaded.

``TestClient`` drives the app in-process, so no port is opened and no server
process is started. The fitted pipeline is trained once into ``tmp_path`` and
injected with FastAPI's ``dependency_overrides``, so the suite never reads,
writes or trains ``models/crop_model.joblib``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app, get_pipeline
from api.schemas import EXAMPLE_REQUEST
from src.data import EXPECTED_LABELS
from src.pipelines.predict_pipeline import load_pipeline
from src.pipelines.training_pipeline import train_pipeline
from tests.conftest import requires_raw_dataset

pytestmark = requires_raw_dataset


@pytest.fixture(scope="module")
def trained_pipeline(tmp_path_factory):
    """Train the real pipeline once per module, away from ``models/``."""
    destination = tmp_path_factory.mktemp("models") / "crop_model.joblib"
    train_pipeline(model_path=destination)
    return load_pipeline(destination)


@pytest.fixture
def client(trained_pipeline):
    """A test client whose app serves the temporary, already-fitted pipeline.

    ``TestClient`` is deliberately *not* used as a context manager: that would
    run the lifespan handler, which loads (and on a clean clone trains)
    ``models/crop_model.joblib``. The override supplies the model instead.
    """
    app.dependency_overrides[get_pipeline] = lambda: trained_pipeline
    yield TestClient(app)
    app.dependency_overrides.clear()


# --- the happy path ----------------------------------------------------------


def test_valid_payload_returns_200_and_a_known_crop(client):
    response = client.post("/predict", json=EXAMPLE_REQUEST)

    assert response.status_code == 200
    body = response.json()
    assert body["crop"] in EXPECTED_LABELS


def test_response_carries_probabilities_that_agree_with_the_label(client):
    body = client.post("/predict", json=EXAMPLE_REQUEST).json()

    probabilities = body["probabilities"]
    assert probabilities, "the naive Bayes pipeline does expose predict_proba"
    assert body["crop"] == next(iter(probabilities)), "the label must be the ranking's head"
    assert body["confidence"] == pytest.approx(probabilities[body["crop"]])
    assert 0.0 <= body["confidence"] <= 1.0


def test_feature_order_in_the_json_body_does_not_change_the_answer(client):
    reversed_payload = dict(reversed(list(EXAMPLE_REQUEST.items())))

    first = client.post("/predict", json=EXAMPLE_REQUEST).json()
    second = client.post("/predict", json=reversed_payload).json()

    assert first == second


# --- 422: the client got it wrong --------------------------------------------


@pytest.mark.parametrize("missing", sorted(EXAMPLE_REQUEST))
def test_missing_field_returns_422(client, missing):
    payload = {key: value for key, value in EXAMPLE_REQUEST.items() if key != missing}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert missing in str(response.json()["detail"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ph", 20.0),  # above the 0-14 scale
        ("ph", -1.0),
        ("humidity", 150.0),  # a percentage cannot exceed 100
        ("N", -5.0),  # a nutrient concentration cannot be negative
        ("temperature", 500.0),
        ("rainfall", 100_000.0),
    ],
)
def test_out_of_range_value_returns_422(client, field, value):
    payload = {**EXAMPLE_REQUEST, field: value}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert field in str(response.json()["detail"])


def test_non_numeric_value_returns_422(client):
    response = client.post("/predict", json={**EXAMPLE_REQUEST, "ph": "slightly acidic"})

    assert response.status_code == 422


def test_unknown_field_returns_422(client):
    response = client.post("/predict", json={**EXAMPLE_REQUEST, "Nitrogen": 90})

    assert response.status_code == 422


def test_empty_body_returns_422(client):
    assert client.post("/predict", json={}).status_code == 422


def test_a_422_names_every_offending_field_at_once(client):
    payload = {**EXAMPLE_REQUEST, "ph": 99.0, "humidity": -1.0}
    del payload["N"]

    detail = client.post("/predict", json=payload).json()["detail"]

    offenders = {error["loc"][-1] for error in detail}
    assert {"N", "ph", "humidity"} <= offenders


def test_an_invalid_request_never_reaches_the_model(client, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("the model must not be consulted for an invalid request")

    monkeypatch.setattr("api.main.predict", explode)

    response = client.post("/predict", json={"N": 90})

    assert response.status_code == 422


# --- 500 and 503: the server got it wrong ------------------------------------


def test_unexpected_failure_returns_500_without_leaking_details(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("corrupted estimator internals")

    monkeypatch.setattr("api.main.predict", boom)

    response = client.post("/predict", json=EXAMPLE_REQUEST)

    assert response.status_code == 500
    assert response.json()["detail"] == "Prediction failed."
    assert "corrupted estimator internals" not in response.text


def test_predict_without_a_loaded_model_returns_503(monkeypatch):
    monkeypatch.setattr("api.main._pipeline", None)

    response = TestClient(app).post("/predict", json=EXAMPLE_REQUEST)

    assert response.status_code == 503


# --- /health -----------------------------------------------------------------


def test_health_reports_a_loaded_model(client, trained_pipeline, monkeypatch):
    monkeypatch.setattr("api.main._pipeline", trained_pipeline)

    body = client.get("/health").json()

    assert body == {"status": "ok", "model_loaded": True, "n_classes": len(EXPECTED_LABELS)}


def test_health_still_answers_when_no_model_is_loaded(monkeypatch):
    monkeypatch.setattr("api.main._pipeline", None)

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "model_loaded": False, "n_classes": 0}


# --- the contract itself -----------------------------------------------------


def test_openapi_schema_documents_both_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]

    assert "/predict" in paths
    assert "/health" in paths


def test_get_is_not_allowed_on_predict(client):
    assert client.get("/predict").status_code == 405
