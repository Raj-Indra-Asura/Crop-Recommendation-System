# Week 10 — Learning Notes

Nine weeks produced a function. This week produces a *service*: something that
is running, listening, and answering questions asked by programs that were not
written by us.

Nothing about the model changes. Everything about who can reach it does.

---

## 1. What an API is

### 1.1 The general idea

An **API** — Application Programming Interface — is the set of things one piece
of software promises another piece of software it can do, plus the exact way to
ask. You have been using APIs since Week 1 without the word:

```python
frame = load_data()                       # the API of src.data
model.fit(X, y)                           # the API of scikit-learn
predict({"N": 90, ..., "rainfall": 200})  # the API of Week 9
```

`predict()` promises: *give me a dictionary with these seven numeric keys, and
I give you back one of 22 strings.* That promise is the interface. How it keeps
it — a `ColumnTransformer`, a `GaussianNB`, a `joblib` file — is the
implementation, and callers do not need to know any of it. That split is the
whole point of an interface: the implementation can change (Week 8 swapped the
random forest for naive Bayes) without a single caller changing.

### 1.2 What a *web* API adds

The Week 9 API has one restriction that has nothing to do with crops: the
caller must be **Python code, in the same process, on the same machine, with
this repository importable**. That rules out:

* the agronomy team's mobile app, written in Kotlin;
* a Node.js dashboard;
* a colleague on another laptop;
* a scheduled job in another company's system;
* anyone who does not have — and should not have — a copy of the model file.

A **web API** removes that restriction by putting a network protocol in the
middle. The caller sends text over TCP; the server replies with text. Neither
side has to share a language, a library, an operating system, or a machine. The
model file never leaves the server.

The cost is equally real, and worth stating up front:

| In-process call | Network call |
| --- | --- |
| Microseconds | Milliseconds at best |
| Types are Python objects | Everything becomes text, and must be re-parsed |
| A bad argument is a `TypeError` | A bad argument is a status code |
| Cannot fail "in transit" | Can time out, drop, be retried, arrive twice |
| Caller is trusted (it is your code) | Caller is a stranger until validated |

Week 10 exists to pay those costs deliberately rather than discover them later.

### 1.3 Client and server

* The **server** is the program that is already running and waiting. Here:
  `uvicorn` hosting `api.main:app`, listening on port 8000.
* The **client** is whoever speaks first. Here: `curl`, Postman, a browser, a
  test, another service.

The relationship is asymmetric and *pull*-shaped: the server never contacts the
client first, and does nothing at all until asked. "The server is up" and "the
server is answering correctly" are different claims — which is why §8 adds a
health endpoint.

---

## 2. HTTP: request, response, methods, status codes

### 2.1 The cycle

Every interaction is exactly one **request** and one **response**. Here is the
real one this week produces:

**Request**

```
POST /predict HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{"N":90,"P":42,"K":43,"temperature":25,"humidity":80,"ph":6.5,"rainfall":200}
```

| Part | Value here | What it is for |
| --- | --- | --- |
| Method | `POST` | The verb: what kind of action this is |
| Path | `/predict` | Which thing on the server is being addressed |
| Headers | `Content-Type: application/json` | Metadata; here, "the body is JSON" |
| Body | the seven features | The data being sent |

**Response**

```
HTTP/1.1 200 OK
content-type: application/json

{"crop":"jute","confidence":0.725342776978384,
 "probabilities":{"jute":0.725342776978384,"rice":0.2746571705151149,
                  "coffee":5.250650112483512e-08}}
```

| Part | Value here | What it is for |
| --- | --- | --- |
| Status code | `200` | Did it work, and if not, whose fault |
| Headers | `content-type` | Metadata about the reply |
| Body | the prediction | The data being returned |

That is the entire protocol. Everything else in this document is a consequence.

### 2.2 Methods

The method is the verb, and choosing it correctly is not decoration — caches,
proxies, browsers and retry logic all behave differently per method.

| Method | Means | Body? | Safe to repeat? | In this project |
| --- | --- | --- | --- | --- |
| `GET` | Read something | No | Yes | `/health`, `/docs` |
| `POST` | Send data, cause something | Yes | Not assumed | `/predict` |
| `PUT` | Replace something wholesale | Yes | Yes | — |
| `PATCH` | Modify part of something | Yes | Not assumed | — |
| `DELETE` | Remove something | No | Yes | — |

**Why is `/predict` a `POST` when it changes nothing on the server?** Two
reasons, and the first is the practical one: `GET` has no body, so the seven
features would have to go in the URL as a query string — visible in every proxy
log, length-limited, and awkward for anything structured. The second is
convention: `POST` says "here is data, do something with it", which is exactly
what is happening. A purist would note that prediction is *safe* (it changes no
state) and might argue for `GET`; every ML API in practice uses `POST`.

Ask for a method an endpoint does not implement and the framework answers
**405 Method Not Allowed** — `GET /predict` does exactly that, and
`tests/test_api.py` asserts it.

### 2.3 Status codes

Read the first digit first. It tells you *who has to do something*.

| Class | Meaning | Who acts | Examples seen this week |
| --- | --- | --- | --- |
| **2xx** | It worked | Nobody | `200 OK` |
| **3xx** | Look elsewhere | Client follows a redirect | `307` (a missing trailing slash) |
| **4xx** | **The client was wrong** | The caller: send a different request | `404`, `405`, `422` |
| **5xx** | **The server was wrong** | The server's owner: read the logs | `500`, `503` |

The ones this API can return:

| Code | Name | When |
| --- | --- | --- |
| 200 | OK | A prediction was made |
| 404 | Not Found | A path that does not exist, e.g. `/predictt` |
| 405 | Method Not Allowed | Right path, wrong verb |
| 422 | Unprocessable Entity | Well-formed JSON, but it does not satisfy the schema |
| 500 | Internal Server Error | We crashed |
| 503 | Service Unavailable | We are up but not ready (no model loaded) |

§6 is entirely about the 422/500 boundary, because getting it wrong is the most
common failure of a first API.

### 2.4 JSON

**JSON** is the wire format: text, with objects, arrays, strings, numbers,
booleans and `null`. It is used because every language can read and write it,
and a human can read it in a terminal.

What it is *not* is Python. It has no tuples, no `NaN`, no `datetime`, and its
numbers are just numbers — which is why `1e-08` above is a float, why NumPy
types must be converted (`float(probability)` in Week 9's `predict_proba`), and
why a `numpy.float32` sneaking into a response is a classic serialization
error. Pydantic converts on the way out; Week 9 already converted on the way in.

---

## 3. REST, briefly and honestly

**REST** is a style for web APIs, not a standard. The parts that matter here:

1. **Resources have paths.** A path names a *thing* (`/predict`, `/health`),
   and the method says what to do with it. Not `/doPredictionNow`.
2. **Statelessness.** Every request carries everything needed to answer it. The
   server keeps no memory of the previous call, so any request can go to any
   replica, in any order — which is what makes horizontal scaling possible at
   all. Our API is stateless: it holds a model in memory (that is not *client*
   state), and nothing about a caller.
3. **Uniform, predictable use of methods and status codes**, as above.
4. **A representation, not the object.** The client receives JSON describing the
   prediction, never a Python object or a pickle.

Purists will note that a strictly RESTful design models nouns and would perhaps
have `POST /recommendations`. That argument is worth exactly one sentence in a
course this size; every serving framework in industry uses `/predict`, and
consistency with the ecosystem wins.

---

## 4. Why FastAPI

The alternatives were Flask (older, simpler, no typing), Django REST Framework
(heavy, aimed at database-backed applications), or a hand-rolled HTTP server
(instructive once, painful forever). FastAPI is the ML-serving default now, for
three concrete reasons.

### 4.1 Type hints become validation

This is the whole trick. Write a type:

```python
@app.post("/predict")
def predict_crop(features: CropFeatures) -> PredictionResponse:
```

and FastAPI reads the annotation at import time and generates the code that
parses the request body, checks every field, converts types and produces a 422
with a per-field explanation when something does not fit. Compare Flask, where
the same guarantees are forty lines of hand-written `if "N" not in body:` — code
that is boring to write, easy to get subtly wrong, and never quite complete.

The type hint is not documentation of intent. It is the enforcement mechanism.

### 4.2 Pydantic does the validating

**Pydantic** is the library FastAPI uses for that. A Pydantic model is a class
whose attributes carry types and constraints:

```python
class CropFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ph: float = Field(..., ge=0, le=14, description="Soil pH on the 0-14 scale.")
    ...
```

* `...` (`Ellipsis`) means **required** — no default, so omitting it is an error.
* `ge` / `le` are **inclusive bounds**. `ph=99` is refused before any of our
  code runs.
* `float` means *coercible to float*: `6.5` and `6` are accepted, `"6.5"`
  is accepted in Pydantic's lenient JSON mode, `"slightly acidic"` is not.
* `extra="forbid"` rejects unknown keys. Without it, a client sending
  `"Nitrogen": 90` would have its typo silently dropped — and then fail on the
  *missing* `N` with a confusing message, or worse, get a prediction from a
  default. Failing loudly on the actual mistake is kinder.

Validation is **declarative**: the schema says what is acceptable, not how to
check it. That is why the file reads like a specification, and why the
specification cannot drift from the behaviour — it *is* the behaviour.

### 4.3 The documentation writes itself

From the same models, FastAPI generates an **OpenAPI** description (JSON, at
`/openapi.json`) and two human pages: **`/docs`** (Swagger UI — interactive, with
a *Try it out* button that sends real requests) and `/redoc`.

This matters more than it sounds. Hand-written API documentation is wrong within
a month, because the code changes and the document does not. Here there is only
one source, so `/docs` cannot be out of date; the field descriptions in
`api/schemas.py` are what a stranger reads in their browser.

### 4.4 FastAPI is not the server

A distinction beginners trip on:

* **FastAPI** is a *framework*: a library that turns your functions into an
  **ASGI** application object (`app`). It does not open a socket.
* **uvicorn** is the *server*: it listens on a TCP port, speaks HTTP, and calls
  the ASGI application for each request.

Hence `uvicorn api.main:app` — "import `app` from `api.main`, and serve it". The
same app object can be served by a different ASGI server without changing a
line, and in Week 12 the container's start command is exactly this line.

ASGI is the asynchronous successor to WSGI (Flask's interface). Our endpoints
are ordinary `def` functions, not `async def`, which is deliberate:
scikit-learn's `predict` is CPU-bound and blocking, and Starlette runs plain
`def` endpoints in a thread pool so a slow prediction cannot block the event
loop. Declaring them `async def` and then calling blocking code inside would be
the actual mistake.

---

## 5. Why Streamlit — and the decision it does *not* get to make

### 5.1 What Streamlit is for

**Streamlit** turns a Python script into a web page. There is no HTML, no CSS,
no JavaScript, no routing, no build step: `st.number_input(...)` is a form
field, `st.bar_chart(...)` is a chart, and the script runs top to bottom on
every interaction.

For this project that is exactly right. The audience is a human who wants to
type seven numbers and see a crop, the developer is a data scientist, and the
budget is an afternoon. Building the same thing in React would take a week and
teach nothing about ML.

### 5.2 What Streamlit is *not*

**Streamlit is a demo tool, not the production frontend for real traffic.** This
is not a slight; it is the tool's own design centre. Concretely:

* **The whole script re-runs on every widget change.** That is a delightful
  programming model and a terrible request-handling model — which is why
  `get_pipeline()` is wrapped in `@st.cache_resource`; without it, every click
  would reload (or, on a clean clone, *retrain*) the model.
* **Session state lives in the server's memory**, so two replicas behind a
  round-robin load balancer will disagree about who you are unless it is
  configured for sticky sessions.
* **No authentication, no authorisation, no roles**, out of the box.
* **The client is coupled to the server** over a websocket; it is not a static
  page a CDN can serve.
* **Layout control is limited** on purpose, so a designer cannot deliver a
  brand-accurate UI in it.

The healthy pattern in industry: Streamlit for internal tools, demos and
stakeholder review; a real frontend (React, Vue, a mobile app) plus the API for
customers. Both live in this repository, and only the API is meant to be
exposed.

### 5.3 The decision: Streamlit calls `predict()` directly

There are two ways to wire the UI, and this project has chosen one:

```
(A) chosen:     Streamlit  ->  predict_pipeline.predict()  ->  model
(B) not chosen: Streamlit  ->  HTTP  ->  FastAPI  ->  predict_pipeline  ->  model
```

`app/streamlit_app.py` **imports `predict()` and calls it in its own process.**
It never sends an HTTP request to `api/main.py`, and the API does not have to be
running for the UI to work.

**Why:**

1. **Week 11's and Week 12's Docker image stays simple.** Containerising is
   hard enough the first time; option (B) would mean either two images and a
   network between them, or one image running two servers under a process
   manager. Option (A) leaves the container with exactly one job: serve the API.
2. **The Streamlit app runs on its own.** One command, no port, no base URL, no
   CORS configuration, no "connection refused" as a beginner's first
   experience, no second terminal window.
3. **There is nothing to gain from the hop.** Both processes would load the same
   artifact from the same disk. Serialising a dictionary to JSON, sending it
   through the loopback interface and parsing it back buys latency and a new
   class of failure, in exchange for nothing this project needs.
4. **The layering rule is preserved either way.** Both entry points still depend
   on `src/pipelines/predict_pipeline.py` and on nothing of each other's, which
   is the property that actually matters (§7).

**What it costs**, stated plainly rather than hidden:

* The UI does not exercise the HTTP contract, so a broken request schema will
  not be caught by clicking around. `tests/test_api.py` is what covers that, and
  it has to be good because it is the only thing watching.
* Each process holds its own copy of the model in memory. At 6.3 KB, irrelevant
  here; at 6 GB, it would be the deciding argument for (B).
* If the UI ever needs to run somewhere that cannot import `src/` — a different
  machine, a different language, a locked-down environment — it must switch to
  (B). The switch is small (replace two calls with `httpx.post`) precisely
  because the UI's logic is one function call deep.

That last point is the general lesson: **choose the simpler topology until
something concrete forces the complex one**, and write down what that something
would be.

---

## 6. Input validation and error handling

### 6.1 Two validations, two audiences

Week 9's `_as_frame()` already rejects a missing, unexpected or non-numeric
feature. So why validate again in `api/schemas.py`?

Because they protect different people. The Week 9 check protects **every caller
of the library** — Streamlit, tests, a future batch job — and its failure mode
is a Python exception, which is right for a programmer. The schema protects
**the HTTP boundary**, where the caller is a stranger, and its failure mode has
to be a status code and a message, because a stranger cannot read your
traceback.

Keep both. The inner check is the invariant; the outer check is the manners.

### 6.2 422 — you sent something invalid

A **422 Unprocessable Entity** means: *the request was well-formed HTTP and
valid JSON, but its contents do not satisfy the contract.* It is a 4xx, so it is
the **client's** problem and the client can fix it.

FastAPI produces it automatically, before any of our code runs, and it names
every offending field at once rather than the first:

```json
{"detail":[{"type":"missing","loc":["body","rainfall"],"msg":"Field required",
            "input":{"N":90,"P":42,"K":43,"temperature":25,"humidity":80,"ph":6.5}}]}
```

```json
{"detail":[{"type":"less_than_equal","loc":["body","ph"],
            "msg":"Input should be less than or equal to 14","input":99,"ctx":{"le":14.0}}]}
```

Note `loc`: it points at the exact field. A client can render this next to the
right form input without parsing English.

What earns a 422 here: a missing field, an unknown field, a string where a
number belongs, `null`, and any value outside the declared range — pH 20,
humidity 150, negative nitrogen.

(Strictly: malformed *JSON* — a truncated body, a stray comma — is a **400 Bad
Request**, because the server could not even parse it. 422 means "parsed fine,
means nothing".)

### 6.3 500 — we failed

A **500 Internal Server Error** means: *your request was fine; we broke.* It is
a 5xx, so it is the **server owner's** problem, and the client retrying the
identical request will not help.

In `api/main.py`:

```python
except Exception as error:
    logger.exception("Prediction failed for payload %s", payload)
    raise HTTPException(status_code=500, detail="Prediction failed.") from error
```

Two rules are visible there:

* **Log the detail, return none of it.** A traceback in a response body tells a
  stranger your file paths, library versions and internal structure — a genuine
  security finding, not a style preference. The client is told *that* it failed;
  the operator gets the stack trace in the server log.
* **`from error`** keeps the original exception chained for that log.

If you find yourself needing a 500 for bad *input*, the schema is wrong. A
correct API answers 422 for everything a client could have sent incorrectly, and
500 only for things that are genuinely the server's fault. The test
`test_unexpected_failure_returns_500_without_leaking_details` forces a real
failure to prove the branch exists and stays quiet.

### 6.4 503 — up, but not ready

Loading a model can fail: the artifact is missing and training also failed, the
file is corrupt, the disk is unreadable. Crashing at start-up would be one
choice; this API instead logs the failure, keeps running, and

* answers `GET /health` with `{"status":"degraded","model_loaded":false,...}`,
* answers `POST /predict` with **503 Service Unavailable**.

The distinction is operational: 503 tells a client (and a load balancer)
"correct request, retry later", where 500 says "something is wrong that
retrying will not fix".

### 6.5 The validation the schema cannot do

Ranges are not distributions. This request is accepted:

```json
{"N": 0, "P": 0, "K": 0, "temperature": 55, "humidity": 3, "ph": 13.9, "rainfall": 0}
```

Every field is inside its bounds; the combination describes a place that appears
nowhere in the 2,200 training rows. The model will answer, and it will answer
*confidently*, because a classifier trained on 22 crops has no vocabulary for
"none of these" — it returns the least-bad of the alternatives it knows.

This is **out-of-distribution input**, and detecting it is a real subfield
(density estimation on the inputs, distance to training data, calibrated
confidence thresholds, a "reject" option). This course does not cover it. What
the course *does* insist on is that you know the hole is there: the API's job is
to reject what is impossible, and a domain expert's job is to notice what is
merely absurd.

---

## 7. Separation of concerns

The layering, and the one rule:

```
api/  ─┐                        dependencies point DOWNWARDS only
app/  ─┼─>  src/pipelines/  ─>  src/{data,preprocessing,models,evaluation}
       │           │
       │           └─────────>  models/crop_model.joblib
       └─ never imports the other
```

* **`api/` does not know Streamlit exists.** Grep it for `streamlit`: nothing.
* **`app/` does not know the API exists.** Grep it for `httpx`, `requests`,
  `8000`: nothing.
* **`src/` knows about neither.** No `fastapi` import, no `st.` call, no HTTP
  status code anywhere under `src/`.

Why this is worth being strict about:

1. **One place to change a rule.** If the model gains a feature, `src/` and the
   schema change; the UI's rendering code does not care how a crop was chosen.
2. **Each layer is testable alone.** `tests/test_api.py` tests HTTP with a
   trained-in-`tmp_path` model injected; `tests/test_predict_pipeline.py` tests
   prediction with no HTTP anywhere.
3. **Entry points are cheap to add.** A CLI, a batch job, a message-queue
   consumer — each is a thin adapter over the same `predict()`. If the model
   logic lived inside the HTTP handler, every new entry point would have to
   copy it, and the copies would drift.
4. **The dependency arrow survives redeployment.** Splitting the UI into a
   separate service later (option (B) in §5.3) changes one file, because there
   is nothing to disentangle.

The failure this prevents has a name: a **fat controller** — business logic
inside a request handler, where it cannot be reached by anything that is not an
HTTP request.

`docs/architecture.md` draws the same picture with the per-request steps.

---

## 8. Serving concerns the model does not have

### 8.1 Load once, at start-up

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()      # before the first request
    yield
    ...               # after the last one
```

A **lifespan** handler runs code once at start-up and once at shutdown. Loading
the model there rather than inside the handler means:

* the cost (milliseconds to load, ~2 s to train on a clean clone) is paid once,
  not per request;
* two concurrent requests cannot race to read or write the same file;
* the process is either ready or visibly not ready, rather than discovering the
  problem on the first customer's request.

Streamlit does the same thing with `@st.cache_resource`, for the same reason,
because its script re-runs constantly.

### 8.2 `Depends` as a seam

```python
def get_pipeline() -> Pipeline: ...

def predict_crop(features: CropFeatures,
                 pipeline: Annotated[Pipeline, Depends(get_pipeline)]):
```

**Dependency injection**: the endpoint declares *what it needs*, and FastAPI
supplies it. The payoff is in the tests —

```python
app.dependency_overrides[get_pipeline] = lambda: trained_pipeline
```

— which lets the suite serve a model trained into `tmp_path` and never read,
write or train `models/crop_model.joblib`. Without the seam, the tests would
depend on the developer's filesystem, which is how test suites become slow and
flaky.

### 8.3 A health endpoint

`GET /health` answers whether the service can actually serve, not merely whether
a process is alive:

```json
{"status":"ok","model_loaded":true,"n_classes":22}
```

It is cheap, it takes no dependencies (so it still answers when the model is
missing — which is exactly when it is asked), and it is what Week 12's
container health check and any load balancer will poll. "The port is open" is
not the same claim as "the model is loaded".

---

## 9. Testing an API

`fastapi.testclient.TestClient` (built on `httpx`) drives the application
**in-process**: no port is opened, no server started, nothing to clean up, and
a full request/response cycle runs in milliseconds.

```python
response = client.post("/predict", json=EXAMPLE_REQUEST)
assert response.status_code == 200
assert response.json()["crop"] in EXPECTED_LABELS
```

Two details in `tests/test_api.py` worth copying:

* **`TestClient(app)` is deliberately not used as a context manager.** Entering
  the context would run the lifespan handler, which loads — and on a clean clone
  *trains* — `models/crop_model.joblib`. The dependency override supplies the
  model instead.
* **Assert on the label's membership, not its value.** `crop in EXPECTED_LABELS`
  survives a retrain; `crop == "jute"` turns every model improvement into a test
  failure. Test the contract, not the prediction.

What is tested this week: the happy path, the shape of the response, feature
order in the JSON body, six ways to earn a 422, the 500 branch (with the
assertion that the internal message does *not* appear in the body), the 503
branch, `/health` in both states, 405 on `GET /predict`, and that
`/openapi.json` documents both endpoints.

---

## 10. What this week does not give you

* **A deployment.** `127.0.0.1` is not the internet. No container, no host, no
  domain, no TLS certificate. Week 12.
* **Security.** No authentication, no API keys, no rate limiting, no CORS
  policy, no body size limit. An open `/predict` on a public address is free
  compute for whoever finds it.
* **Concurrency and scale.** One uvicorn worker; a queue of slow requests is a
  queue.
* **Observability.** Requests are logged by uvicorn and then forgotten. No
  latency metrics, no prediction log, and therefore no way to detect drift —
  the Week 9 gap, still open.
* **Model versioning in the response.** The client cannot tell which artifact
  answered.
* **Batch prediction.** One row per request; 10,000 rows means 10,000 requests.
* **Out-of-distribution detection**, per §6.5.

---

## Summary

| Question | Answer |
| --- | --- |
| What is an API? | A promise about how to ask software for something, and what comes back |
| What does a *web* API add? | Any language, any machine, at the price of latency, parsing and untrusted callers |
| What is in a request? | Method, path, headers, body |
| What is in a response? | Status code, headers, body |
| Why `POST /predict`? | It carries a body; `GET` would put seven features in the URL |
| Why FastAPI? | Type hints become validation, parsing and `/docs`, from one source |
| What does Pydantic do? | Declares the contract and enforces it before your code runs |
| FastAPI vs uvicorn? | Framework that builds the ASGI app vs server that listens on a port |
| Why Streamlit? | A clickable demo in an afternoon, in pure Python |
| Why *not* Streamlit for production? | Re-runs the script per interaction, in-memory session state, no auth, no design control |
| Does the UI call the API? | No — it calls `predict()` in-process, so Week 12's image only contains the API and the UI runs alone |
| 422 vs 500? | *You* sent something invalid vs *we* failed; 4xx the caller acts, 5xx the operator acts |
| Why not return the traceback? | It leaks paths, versions and structure to a stranger |
| Why load the model at start-up? | Pay the cost once, avoid races, be ready or visibly not ready |
| Why `Depends`? | It is the seam that lets tests inject a temporary model |
| Why `/health`? | "Port open" is not "model loaded" |
| Why `TestClient`? | Full HTTP semantics, in-process, no port, milliseconds |
