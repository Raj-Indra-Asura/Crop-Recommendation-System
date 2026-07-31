# Architecture

How a measurement becomes a recommendation, and which file is responsible for
each step. Written in Week 10, when the project first had more than one way in.

---

## The request flow

```
   ┌────────────────┐        ┌────────────────────┐
   │  HTTP client   │        │  Streamlit UI      │
   │  curl, Postman │        │  app/streamlit_app │
   │  browser, app  │        │  (demo only)       │
   └───────┬────────┘        └─────────┬──────────┘
           │ JSON over HTTP            │ in-process Python call
           v                           │
   ┌────────────────────────────┐      │
   │  FastAPI app  api/main.py  │      │
   │  POST /predict, GET /health│      │
   │  validates via api/schemas │      │
   └───────────┬────────────────┘      │
               │                       │
               └───────────┬───────────┘
                           v
        ┌──────────────────────────────────────┐
        │  src/pipelines/predict_pipeline.py   │
        │  predict() / predict_proba()         │
        │  load_pipeline() — trains on demand  │
        └───────────────────┬──────────────────┘
                            v
        ┌──────────────────────────────────────┐
        │  models/crop_model.joblib            │
        │  Pipeline: preprocess -> model       │
        │  (git-ignored, rebuilt from data)    │
        └───────────────────┬──────────────────┘
                            v
                   "jute", p = 0.7253
```

Both entry points depend on the **same** function. Neither depends on the
other: `api/` never imports `app/`, `app/` never imports `api/`, and `src/`
imports neither.

### Step by step, for one `POST /predict`

| # | Where | What happens | Failure mode |
| --- | --- | --- | --- |
| 1 | client | Sends a JSON body with the seven features | — |
| 2 | `api/schemas.py` | Pydantic parses and range-checks every field | **422** with a per-field message |
| 3 | `api/main.py` | Handler receives a typed `CropFeatures`, gets the model from `get_pipeline()` | **503** if no model is loaded |
| 4 | `src/pipelines/predict_pipeline.py` | `_as_frame()` rebuilds the row in the training column order | **500** (should be unreachable — step 2 already checked) |
| 5 | `models/crop_model.joblib` | `preprocess` scales, `model` predicts, `predict_proba` ranks | **500** |
| 6 | `api/main.py` | Wraps label + probabilities into `PredictionResponse` | — |
| 7 | client | Receives **200** and the JSON body | — |

The model is loaded **once**, in the FastAPI lifespan handler, not per request.
Streamlit does the same with `@st.cache_resource`.

---

## Why Streamlit does not call the API

`app/streamlit_app.py` calls `predict_pipeline.predict()` in its own process. It
does *not* make an HTTP request to `api/main.py`. This is a deliberate project
decision, not an oversight:

* the demo runs with nothing else running — one command, no port, no base URL,
  no CORS, no "connection refused" as a first-run experience;
* Week 12 containerises the **API**, and a UI that hard-codes an API address is
  a second deployment topology to configure and debug;
* there is no HTTP hop to pay for when both sides are the same Python process.

The trade-off is real and worth naming: the UI does not exercise the API, so a
broken request schema will not show up by clicking around. `tests/test_api.py`
is what covers that instead. In a system where the UI is a separate service —
or written in a language that cannot import `src/` — the arrow would point at
the API, and the reasoning would come out the other way.

Full argument: [`docs/curriculum/week10/learning_notes.md`](curriculum/week10/learning_notes.md) §5.

---

## Layers, and the rule between them

| Layer | Path | Knows about | Must never know about |
| --- | --- | --- | --- |
| Transport | `api/` | `src/pipelines`, HTTP | Streamlit, notebooks |
| Demo UI | `app/` | `src/pipelines`, Streamlit | `api/`, HTTP |
| Application | `src/pipelines/` | `src/`, the artifact | HTTP, sessions, widgets |
| Domain | `src/data`, `src/preprocessing`, `src/models`, `src/evaluation` | pandas, scikit-learn | anything above |
| Configuration | `src/config.py` | nothing — it is inert | everything |
| Artifact | `models/*.joblib` | — | — |

The rule is one-directional: **dependencies point downwards only.** A change to
the response JSON touches `api/` alone. A change to the model touches `src/`
alone, and both entry points get it for free.

---

## Runtime shapes

| Command | Process | Serves |
| --- | --- | --- |
| `python -m src.pipelines.training_pipeline` | one-shot | writes `models/crop_model.joblib` |
| `python -m src.pipelines.predict_pipeline` | one-shot | one prediction on stdout |
| `uvicorn api.main:app` | long-running server | `POST /predict`, `GET /health`, `/docs` |
| `streamlit run app/streamlit_app.py` | long-running server | the demo form on port 8501 |
| `pytest` | one-shot | the whole suite, including the API via `TestClient` |

`TestClient` drives the FastAPI app in-process: the tests open no port and start
no server, which is why they are fast and cannot collide with a running dev
server on 8000.

---

## What is not here yet

* **No container.** Both servers run from a checkout with the pinned
  environment installed. Week 12.
* **No authentication, rate limiting or TLS.** The API is open, and is meant for
  `127.0.0.1`.
* **No persistence.** Requests and predictions are not stored, so there is
  nothing to monitor for drift with.
* **No model versioning.** One filename, overwritten by the next training run;
  the response says nothing about which model answered.
* **No batch endpoint.** One row per request.
