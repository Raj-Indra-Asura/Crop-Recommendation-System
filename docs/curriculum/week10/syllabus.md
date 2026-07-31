# Week 10 — Serving the Model Over HTTP

> 🗺 [Roadmap](../README.md) › [Part IV — Production (Weeks 9-11)](../README.md#part-iv--production-weeks-9-11) › [Chapter 10 — Serving the Model Over HTTP](README.md) › **§10.1 Syllabus**

## Title

**From `predict()` to `POST /predict`: an API, a demo UI, and the difference
between a 422 and a 500**

## Learning objectives

By the end of this week a student should be able to:

1. Say what an **API** is, and what a **web API** adds: a contract another
   program can call across a network, in a language it does not have to share.
2. Describe a **request/response** cycle and name its parts — method, path,
   headers, body, status code — for one real `POST /predict`.
3. Choose the right **HTTP method** (`GET` for reading, `POST` for sending a
   body) and read a **status code** by its class (2xx, 4xx, 5xx).
4. Explain what **FastAPI** does with a type hint: validate the request, parse
   it into a typed object, and publish interactive documentation at `/docs`.
5. Write a **Pydantic** model with field types and ranges, and predict exactly
   which requests it will reject.
6. State the difference between **422** and **500** in one sentence each — *you
   sent something invalid* vs *we failed* — and say who has to act in each case.
7. Explain why the model is loaded **once at start-up**, not per request, and
   what a dependency (`Depends`) buys the tests.
8. Say what **Streamlit** is good for (a demo a human can click in an afternoon)
   and what it is not (the production frontend for real traffic).
9. Defend the project's decision that **Streamlit calls `predict()` directly**
   rather than calling the API over HTTP, and name the cost of that choice.
10. Point at the **separation of concerns**: `api/` and `app/` both depend on
    `src/pipelines/predict_pipeline.py` and never on each other.
11. Test an API with **`TestClient`** without starting a server or opening a
    port.

## Prerequisites

Weeks 1-9, in full. This week assumes and does **not** re-explain:

* the seven feature columns and the dataset contract
  ([Week 1 notes](../week01/learning_notes.md));
* the `Pipeline`, the saved artifact and train-on-demand
  ([Week 9 notes](../week09/learning_notes.md));
* `predict()` / `predict_proba()` and the input validation they already do
  ([Week 9 notes](../week09/learning_notes.md) §7.2);
* the `rice` / `jute` confusion pair, which is what the returned probabilities
  keep showing ([Week 8 notes](../week08/learning_notes.md) §7).

Five new pins, all required: `fastapi==0.115.6`, `pydantic==2.10.4`,
`uvicorn==0.34.0`, `streamlit==1.41.1`, `httpx==0.28.1`. Nothing optional this
week.

## Concepts covered

| Concept | First introduced |
| --- | --- |
| API, and what a *web* API adds | Week 10 |
| Client / server | Week 10 |
| Request/response cycle | Week 10 |
| HTTP methods (`GET`, `POST`) | Week 10 |
| Status codes: 2xx / 4xx / 5xx | Week 10 |
| REST resources and statelessness | Week 10 |
| JSON as the wire format | Week 10 |
| API contract / schema | Week 10 |
| Pydantic model, field constraints | Week 10 |
| Automatic validation from type hints | Week 10 |
| 422 vs 500 — whose fault it is | Week 10 |
| Not leaking internals in an error | Week 10 |
| OpenAPI and the `/docs` page | Week 10 |
| ASGI, and framework vs server (FastAPI vs uvicorn) | Week 10 |
| Load the model once, at start-up | Week 10 |
| Dependency injection (`Depends`) as a test seam | Week 10 |
| Health endpoint | Week 10 |
| Separation of concerns / layered dependencies | Week 3 (pipeline), Week 10 (services) |
| Demo UI vs production frontend | Week 10 |
| In-process call vs network call | Week 10 |
| Testing an API with `TestClient` | Week 10 |
| Out-of-distribution input | Week 10 |

Each also appears in [`docs/ml_concepts.md`](../../ml_concepts.md).

## Connection to the previous week

Week 9 ended with a program instead of a notebook: `predict({...})` returns
`jute`. Its stated limitation was blunt — *"there is no HTTP endpoint, no JSON
request/response contract, no port, no server. Everything this week requires a
Python interpreter on the same machine, with the repository importable."*

Week 10 removes exactly that limitation, and changes nothing else. The same
artifact, the same `predict()`, the same 0.9955 accuracy; what is new is a way
in for a program that is not Python, and a way in for a human who does not have
a terminal.

## Connection to the ML lifecycle

```
Frame the problem -> Get the data -> Explore -> Prepare -> Model
    -> Evaluate -> Improve -> Productionize -> [Deploy] -> Monitor
```

Week 10 is the first half of **deploy**: the model is now reachable over a
network *on this machine*. Week 12 supplies the second half — a container and a
public address. Between them, nothing about the model changes; deployment is a
software-engineering problem that happens to have a model inside it.

## Expected student outcome

### The student CAN, after this week

* **Run the API locally.** `uvicorn api.main:app --host 127.0.0.1 --port 8000`
  starts a server that loads the model once (training one first if the clone is
  clean) and logs `Application startup complete`.
* **Get a prediction over HTTP**, with `curl`, Postman, the browser, or any
  other language:
  ```
  {"crop":"jute","confidence":0.7253,"probabilities":{"jute":0.7253,"rice":0.2747,"coffee":5.25e-08}}
  ```
* **Read the interactive docs** at `http://127.0.0.1:8000/docs`, see the seven
  fields with their ranges, and send a request without writing any client code.
* **Run the demo UI.** `streamlit run app/streamlit_app.py` gives a form with
  seven numeric inputs and a button, and shows the recommended crop with its
  confidence.
* **Explain 422 vs 500.** A 422 is the client's problem — a missing field, a pH
  of 99 — and the fix is to send a different request. A 500 is the server's
  problem, the request was fine, and no amount of retrying with the same body
  helps.
* **Say why the UI does not call the API**, and what that costs (the UI never
  exercises the HTTP contract; the tests have to).
* Run `pytest tests/test_api.py` (27 tests) with no server running.

### The student CANNOT yet

* **Containerise or deploy any of this publicly.** Both commands need a
  checkout, Python 3.11/3.12 and the pinned environment. There is no image, no
  `Dockerfile`, no host, no domain, no HTTPS. That is Week 12.
* **Serve real traffic safely.** No authentication, no API keys, no rate
  limiting, no CORS policy, no request size limits, no TLS. The API binds to
  `127.0.0.1` for a reason.
* **Handle load.** One uvicorn worker, one process, no queue, no autoscaling; a
  Streamlit server keeps per-session state in memory and cannot be scaled by
  adding replicas behind a naive load balancer.
* **Batch.** One row per request; there is no `POST /predict/batch`.
* **Know which model answered.** The response carries no model version, and
  nothing is logged for later analysis — so **monitoring and drift detection**
  remain impossible, as they were in Week 9.
* **Detect out-of-distribution input.** A request inside the schema's bounds but
  far outside the training data is answered confidently and wrongly.

## Deliverables for the week

* `api/schemas.py` — `CropFeatures` (seven fields, typed and range-checked,
  `extra="forbid"`), `PredictionResponse` and `HealthResponse`.
* `api/main.py` — the FastAPI app: `POST /predict`, `GET /health`, a lifespan
  handler that loads the model once, and `get_pipeline()` as the test seam.
* `app/streamlit_app.py` — the seven-field form, calling
  `predict_pipeline.predict()` in-process and displaying the crop, its
  confidence and the top three candidates.
* `tests/test_api.py` (27 tests) — 200 on a valid payload, 422 on missing,
  out-of-range, mistyped and unknown fields, 500 on an unexpected failure, 503
  with no model, and `/health`. 404 passed and 1 skipped in the whole suite.
* `docs/architecture.md` — the request flow and the layering rule.
* This week's four curriculum documents.
* Updated `requirements.txt` (five new pins), `docs/ml_concepts.md` and the
  README progress table.

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| ◀ [Chapter 10 — Serving the Model Over HTTP](README.md) | [Chapter 10 — Serving the Model Over HTTP](README.md) · 🗺 [Roadmap](../README.md) | [§10.2 Learning notes](learning_notes.md) ▶ |

<!-- nav:end -->
