r"""The crop recommendation HTTP API (Week 10).

Run it locally from the repository root::

    uvicorn api.main:app --host 127.0.0.1 --port 8000

then either open http://127.0.0.1:8000/docs and press *Try it out*, or::

    curl -X POST http://127.0.0.1:8000/predict \\
      -H "Content-Type: application/json" \\
      -d '{"N":90,"P":42,"K":43,"temperature":25,"humidity":80,"ph":6.5,"rainfall":200}'

What this module is, and is not
-------------------------------
It is a **transport layer**. It parses JSON, validates it against
:mod:`api.schemas`, calls :func:`src.pipelines.predict_pipeline.predict`, and
turns the answer back into JSON with a status code. It contains no modelling
logic whatsoever: no feature engineering, no thresholds, no ``sklearn`` import
beyond a type annotation. If a rule about crops ever needs changing, it changes
in ``src/``, and this file does not move.

Load the model once, not per request
------------------------------------
Reading and unpickling ``models/crop_model.joblib`` takes milliseconds and
training one takes seconds; doing either inside the request handler would pay
that cost on every call and let two concurrent requests race on the same file.
So the pipeline is loaded once at start-up (in the lifespan handler) and handed
to the endpoint through a dependency, which also gives the tests a seam: they
override :func:`get_pipeline` with a model trained into a temporary directory
and never touch ``models/`` at all.

Status codes this API can return
--------------------------------
* **200** — a prediction. The body is a :class:`~api.schemas.PredictionResponse`.
* **422** — *your* request was wrong: a missing field, a value out of range, a
  string where a number belongs, an unknown key. FastAPI produces this from the
  schema before any code below runs.
* **503** — the service is up but has no model in memory (start-up failed).
  Retriable, unlike a 422.
* **500** — *we* were wrong: an unexpected failure inside prediction. The client
  is told that it happened, and nothing about where — a traceback in a response
  body is a gift to an attacker. The traceback goes to the server log.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from sklearn.pipeline import Pipeline

from api.schemas import CropFeatures, HealthResponse, PredictionResponse
from src.config import MODEL_PATH
from src.pipelines.predict_pipeline import load_pipeline, predict, predict_proba

logger = logging.getLogger(__name__)

#: How many crops the response ranks. All 22 probabilities are computed; only
#: the leaders are interesting to a caller, and a short body is easier to read
#: in a terminal.
TOP_K = 3

#: The process-wide model, populated at start-up by :func:`load_model`.
_pipeline: Pipeline | None = None


def load_model(model_path: Path | str = MODEL_PATH) -> Pipeline:
    """Load the fitted pipeline into the module-level cache.

    Args:
        model_path: Artifact to load. A missing file is trained on demand by
            :func:`src.pipelines.predict_pipeline.load_pipeline`, which is what
            lets a clean clone start the server with no extra step.

    Returns:
        The fitted pipeline, also stored for :func:`get_pipeline`.
    """
    global _pipeline
    _pipeline = load_pipeline(model_path)
    return _pipeline


def get_pipeline() -> Pipeline:
    """Return the loaded pipeline, as a FastAPI dependency.

    Returns:
        The fitted pipeline loaded at start-up.

    Raises:
        HTTPException: 503 if no model is loaded, i.e. start-up failed. This is
            a *server* state, not a bad request, and it is worth retrying.
    """
    if _pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded; the service is not ready.",
        )
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model before the first request and release it after the last.

    A failure here is logged rather than raised so that ``/health`` stays
    reachable and can report ``model_loaded: false`` — a server that refuses to
    start tells an operator nothing except that it is missing.

    Args:
        app: The application being started. Unused, but part of the protocol.

    Yields:
        Control to the running application.
    """
    global _pipeline
    try:
        load_model()
        logger.info("Model loaded from %s", MODEL_PATH)
    except Exception:  # noqa: BLE001 - start-up must not take /health down with it
        logger.exception("Model could not be loaded from %s", MODEL_PATH)
        _pipeline = None
    yield
    _pipeline = None


app = FastAPI(
    title="Crop Recommendation API",
    description=(
        "Recommends a crop from seven soil and weather measurements, using the "
        "Gaussian naive Bayes pipeline trained in Week 9."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Report whether the service can serve predictions.

    Deliberately cheap and dependency-free: it must answer while the model is
    missing, which is exactly when someone is asking. Week 12's container and
    any load balancer poll this endpoint.

    Returns:
        The status, whether a model is in memory, and how many crops it knows.
    """
    loaded = _pipeline is not None
    n_classes = len(getattr(_pipeline, "classes_", ())) if loaded else 0
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        n_classes=n_classes,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict_crop(
    features: CropFeatures,
    pipeline: Annotated[Pipeline, Depends(get_pipeline)],
) -> PredictionResponse:
    """Recommend one crop for one set of growing conditions.

    Args:
        features: The seven validated measurements. Anything that does not fit
            :class:`~api.schemas.CropFeatures` never reaches this function —
            FastAPI has already answered 422.
        pipeline: The model loaded at start-up, injected by :func:`get_pipeline`.

    Returns:
        The recommended crop, its probability, and the top few alternatives.

    Raises:
        HTTPException: 500 if prediction fails unexpectedly. The cause is
            logged with its traceback and not returned to the caller.
    """
    payload = features.to_features()
    try:
        label = predict(payload, pipeline=pipeline)
        try:
            ranked = predict_proba(payload, pipeline=pipeline, top_k=TOP_K)
        except AttributeError:
            # A model without `predict_proba` is still a usable model here: the
            # label is the contract, the probabilities are a courtesy.
            ranked = {}
    except Exception as error:
        logger.exception("Prediction failed for payload %s", payload)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed.",
        ) from error

    return PredictionResponse(
        crop=label,
        confidence=ranked.get(label, 0.0),
        probabilities=ranked,
    )
