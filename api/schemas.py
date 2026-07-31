"""The request and response contract of the crop API (Week 10).

A schema is the API's *promise*: these fields, these types, these ranges. It is
written once, here, and then does four jobs at the same time —

1. **validation** — FastAPI rejects a body that does not fit before any handler
   code runs, with HTTP 422 and a per-field explanation;
2. **parsing** — the JSON numbers arrive as a typed :class:`CropFeatures`
   object, not as an untrusted ``dict``;
3. **documentation** — the same model is rendered as the request body in the
   interactive docs at ``/docs``;
4. **serialization** — :class:`PredictionResponse` fixes the shape of the reply,
   so a client can rely on it.

Why validate here when Week 9 already validates?
------------------------------------------------
:func:`src.pipelines.predict_pipeline._as_frame` already rejects a missing,
unexpected or non-numeric feature. That check stays: it protects *every* caller
of the library, including Streamlit and the tests. What it cannot do is answer a
stranger politely. A ``TypeError`` raised deep inside a library is a crash — a
500 — and 500 means "the server is broken". Malformed input is not the server
being broken, it is the *client* being wrong, which is a 422. Validating at the
HTTP boundary is what converts one into the other.

Ranges, and what they are not
-----------------------------
The bounds below are generous physical limits (a pH cannot be 40; humidity is a
percentage), not the training ranges. The observed training minimum/maximum of
each column is quoted in the field description so that the gap is visible: a
request inside the bounds but far outside the training data is accepted and
answered with an over-confident label. Detecting *that* is out-of-distribution
detection, which this course does not cover — see
``docs/curriculum/week10/learning_notes.md`` §6.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

#: The example request used in the docs, the tests and the ``/docs`` page.
#: Identical to :data:`src.pipelines.predict_pipeline.EXAMPLE_INPUT`.
EXAMPLE_REQUEST: dict[str, float] = {
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 25,
    "humidity": 80,
    "ph": 6.5,
    "rainfall": 200,
}


class CropFeatures(BaseModel):
    """One set of growing conditions: the seven features the model was fitted on.

    ``extra="forbid"`` makes an unknown key an error rather than something
    silently dropped, because a client sending ``"Nitrogen"`` instead of ``"N"``
    has a bug it needs to be told about — and dropping the key would leave a
    *missing* required field anyway.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [EXAMPLE_REQUEST]},
    )

    N: float = Field(
        ...,
        ge=0,
        le=500,
        description="Nitrogen content of the soil, kg/ha. Training range: 0-140.",
    )
    P: float = Field(
        ...,
        ge=0,
        le=500,
        description="Phosphorus content of the soil, kg/ha. Training range: 5-145.",
    )
    K: float = Field(
        ...,
        ge=0,
        le=500,
        description="Potassium content of the soil, kg/ha. Training range: 5-205.",
    )
    temperature: float = Field(
        ...,
        ge=-20,
        le=60,
        description="Average temperature, degrees Celsius. Training range: 8.8-43.7.",
    )
    humidity: float = Field(
        ...,
        ge=0,
        le=100,
        description="Relative humidity, percent. Training range: 14.3-100.0.",
    )
    ph: float = Field(
        ...,
        ge=0,
        le=14,
        description="Soil pH on the 0-14 scale. Training range: 3.5-9.9.",
    )
    rainfall: float = Field(
        ...,
        ge=0,
        le=1000,
        description="Rainfall, millimetres. Training range: 20.2-298.6.",
    )

    def to_features(self) -> dict[str, float]:
        """Return the plain dictionary the prediction pipeline expects.

        Returns:
            Mapping of feature name to value, with exactly the seven keys of
            :data:`src.config.FEATURE_COLUMNS`.
        """
        return self.model_dump()


class PredictionResponse(BaseModel):
    """The answer to one ``POST /predict``.

    Attributes:
        crop: The recommended crop, one of the 22 labels in the dataset.
        confidence: The model's probability for ``crop``, between 0 and 1.
        probabilities: The most likely crops with their probabilities, highest
            first. Empty when the loaded model has no ``predict_proba`` — the
            label is the contract, the probabilities are a courtesy.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "crop": "jute",
                    "confidence": 0.7253,
                    "probabilities": {"jute": 0.7253, "rice": 0.2747, "coffee": 0.0},
                }
            ]
        }
    )

    crop: str = Field(..., description="The recommended crop label.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability assigned to `crop`, or 0.0 when unavailable.",
    )
    probabilities: dict[str, float] = Field(
        default_factory=dict,
        description="Top candidate crops and their probabilities, highest first.",
    )


class HealthResponse(BaseModel):
    """The answer to ``GET /health``.

    Attributes:
        status: ``"ok"`` when the process is up and a model is loaded.
        model_loaded: Whether the fitted pipeline is in memory and ready.
        n_classes: How many crop labels that pipeline can return.
    """

    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., description='"ok" when the service can serve predictions.')
    model_loaded: bool = Field(..., description="Whether a fitted pipeline is in memory.")
    n_classes: int = Field(..., ge=0, description="Number of crop labels the model knows.")
