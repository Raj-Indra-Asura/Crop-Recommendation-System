"""HTTP interface to the Week 9 prediction pipeline (Week 10).

Two modules, deliberately small:

* :mod:`api.schemas` — the request and response *contract*, as Pydantic models.
* :mod:`api.main` — the FastAPI application: ``POST /predict`` and
  ``GET /health``.

Neither module knows how the model works. Both go through
:mod:`src.pipelines.predict_pipeline`, which is the only code that touches the
artifact — see ``docs/architecture.md``.
"""
