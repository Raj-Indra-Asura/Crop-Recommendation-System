# Architecture

How a measurement becomes a recommendation, and which file is responsible for
each step. Written in Week 10, when the project first had more than one way in;
completed in Week 12 with the training path and the production gap.

---

## The whole system, in one picture

Two paths run through this repository. The **training path** runs offline,
occasionally, and produces an artifact. The **serving path** runs per request
and consumes it. They meet at exactly one file.

```
  TRAINING (offline, occasional)                SERVING (online, per request)
  ──────────────────────────────                ─────────────────────────────

  data/raw/Crop_recommendation.csv                 HTTP client        Streamlit
  2,200 rows · committed · read-only               curl/Postman       app/
          │                                        browser/app        streamlit_app
          │ src/data/data_loader.py  (W1)               │                  │
          │ validates columns, dtypes,                  │ JSON/HTTP        │
          │ row count, label set                        v                  │
          v                                    ┌──────────────────┐        │
  src/data/split.py         (W3)               │ api/main.py      │        │
  stratified 80/20 → 1,760 / 440               │ api/schemas.py   │        │
  data/processed/{train,test}.csv              │ POST /predict    │        │
          │                                    │ GET  /health     │        │
          v                                    └────────┬─────────┘        │
  src/preprocessing/preprocessor.py (W3)                │                  │
  ColumnTransformer, fitted on TRAIN only               └─────────┬────────┘
          │                                                       │
          v                                                       v
  src/models/classical_models.py  (W5)          src/pipelines/predict_pipeline.py (W9)
  GaussianNB(var_smoothing=1e-9)                predict() / predict_proba()
  chosen in W8 over 12 alternatives             load_pipeline() — trains on demand
          │                                                       │
          v                                                       │
  src/pipelines/training_pipeline.py (W9)                         │
  Pipeline([preprocess, model]).fit()                             │
  accuracy 0.9955 · macro F1 0.9954                               │
          │                                                       │
          └──────────►  models/crop_model.joblib  ◄───────────────┘
                        git-ignored · rebuilt from data + code + pins
                                        │
                                        v
                             "jute", p = 0.7253
```

`src/config.py` (W9) is off the diagram on purpose: it is inert. It holds the
paths, `RANDOM_STATE = 42` and the chosen hyperparameters, and every box above
reads from it rather than hard-coding its own.

The whole training path is also run **inside** `deployment/Dockerfile` (W11), at
build time, so a started container reads a file rather than fitting a model.

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

### Step by step, for one `python -m src.pipelines.training_pipeline`

| # | Where | What happens | Failure mode |
| --- | --- | --- | --- |
| 1 | `data/raw/Crop_recommendation.csv` | 2,200 committed rows, never written to | — |
| 2 | `src/data/data_loader.py` | `load_data()` reads it and calls `validate_dataset()`: columns, dtypes, row count, label set, no nulls | Raises, naming the file and what is wrong |
| 3 | `src/data/split.py` | `stratified_split()` — 80/20 with `RANDOM_STATE`, class proportions preserved: 1,760 train, 440 test (20 per crop) | Raises if a class is too small to stratify |
| 4 | `src/preprocessing/preprocessor.py` | Builds the `ColumnTransformer`; it is **not** fitted here — the `Pipeline` fits it on the training fold only | Leakage, if ever fitted before step 3 |
| 5 | `src/models/classical_models.py` | `get_naive_bayes()` — the model and hyperparameters named by `FINAL_MODEL_NAME` / `FINAL_MODEL_PARAMS` in `src/config.py` | `KeyError` on an unknown model name |
| 6 | `src/pipelines/training_pipeline.py` | `train_pipeline()` fits `Pipeline([preprocess, model])` on the training rows, scores it on the held-out rows, prints accuracy and F1 | — |
| 7 | `models/crop_model.joblib` | `joblib.dump` of the whole fitted `Pipeline` — preprocessing included | Directory created if absent |

Steps 1-7 also run inside the image build (Week 11), which is why a container
starts by reading a file. Nothing in the serving path ever fits anything: if the
artifact is missing, `load_pipeline()` re-runs this whole path once, and then
serves.

---

## Why Streamlit does not call the API

`app/streamlit_app.py` calls `predict_pipeline.predict()` in its own process. It
does *not* make an HTTP request to `api/main.py`. This is a deliberate project
decision, not an oversight:

* the demo runs with nothing else running — one command, no port, no base URL,
  no CORS, no "connection refused" as a first-run experience;
* Week 11 containerises the **API** alone, and a UI that hard-codes an API
  address is a second deployment topology to configure and debug;
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
| `docker run -p 8000:8000 crop-api` | long-running container | the same API, from an image built by `deployment/Dockerfile` (Week 11) |

`TestClient` drives the FastAPI app in-process: the tests open no port and start
no server, which is why they are fast and cannot collide with a running dev
server on 8000.

---

## What is not here yet

* **No public deployment.** Week 11 added a container
  ([`docs/deployment_guide.md`](deployment_guide.md)), so the API runs anywhere
  Docker runs — but on your machine only: no registry, no host, no domain, no
  TLS. The Streamlit demo is not containerised and still needs a checkout with
  the pinned environment installed.
* **No authentication, rate limiting or TLS.** The API is open, and is meant for
  `127.0.0.1`.
* **No persistence.** Requests and predictions are not stored, so there is
  nothing to monitor for drift with.
* **No model versioning.** One filename, overwritten by the next training run;
  the response says nothing about which model answered.
* **No batch endpoint.** One row per request.

Week 12 reviewed that list rather than shortening it. Each item is a deliberate
scope boundary, and
[`docs/curriculum/week12/learning_notes.md`](curriculum/week12/learning_notes.md)
§5 says what building each one would involve, in the order it would be worth
doing:

| Missing | What it would take | Why it is not here |
| --- | --- | --- |
| Request logging | One structured JSON line per prediction, to a file or a log service | Everything below needs it; it is the first thing to add, and also a privacy decision |
| Model version in the response | `MODEL_VERSION` in `src/config.py`, embedded in the artifact, returned by `/predict` and `/health` | Ten lines, and the difference between "the model was wrong" and "*which* model was wrong" |
| Authentication, rate limiting | An API key check and a per-caller budget in `api/main.py` | Meaningless while the service is bound to localhost |
| A public address | Registry push, a host, DNS, a certificate, and the ongoing duty of operating it | A course of its own; Week 10's notes promised it to Week 12 and Week 12 withdraws the promise rather than faking it |
| Drift monitoring | Weekly comparison of request feature statistics against Week 2's training statistics | Needs stored requests, which needs logging |
| A model registry | Artifacts stored against data hash, commit, metrics; one marked `production`; rollback as a pointer change | Needs more than one model worth naming |

The rule this table follows is the same one the repository has followed since
Week 1: name the boundary precisely, and do not describe a capability that does
not exist.
