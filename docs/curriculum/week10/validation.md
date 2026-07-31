# Week 10 — Validation

> 🗺 [Roadmap](../README.md) › [Part IV — Production (Weeks 9-11)](../README.md#part-iv--production-weeks-9-11) › [Chapter 10 — Serving the Model Over HTTP](README.md) › **§10.4 Validation**

Run these in order from the repository root. Each step lists the exact command
and the output captured from a real run on 2026-07-31 (Python 3.12.3, the pinned
`requirements.txt`).

Probabilities come from a seeded pipeline and should match yours exactly. PIDs,
port numbers, timings and timestamps will not.

**About the two servers.** `uvicorn` and `streamlit` are long-running: started in
the foreground they never return, and the terminal is stuck until you press
`Ctrl-C`. Every step below therefore starts them **in the background** (`&`),
sleeps long enough for start-up, sends the requests, and then **kills** the
process. Never leave one running while starting another on the same port.

---

## Step 0 — Environment

```bash
python --version
pip install -r requirements.txt
python -c "import fastapi, pydantic, uvicorn, streamlit, httpx; \
print(fastapi.__version__, pydantic.VERSION, uvicorn.__version__, streamlit.__version__, httpx.__version__)"
```

```
Python 3.12.3
0.115.6 2.10.4 0.34.0 1.41.1 0.28.1
```

The five new pins this week:

```
fastapi==0.115.6
pydantic==2.10.4
uvicorn==0.34.0
streamlit==1.41.1
httpx==0.28.1
```

`pydantic` and `httpx` arrive as dependencies of `fastapi` and its test client;
they are named explicitly because `api/schemas.py` imports Pydantic directly and
`tests/test_api.py` runs on `httpx`.

---

## Step 1 — Lint and the whole suite

```bash
ruff check .
```

```
All checks passed!
```

```bash
pytest -q
```

```
.........................s.............................................. [ 88%]
.............................................                            [100%]
404 passed, 1 skipped in 28.66s
```

404 = Week 9's 377, plus this week's 27. The skip is the pre-existing optional
test from Week 8.

---

## Step 2 — Start the API and get a prediction

Start from a clean `models/` to prove the clean-clone path works: the lifespan
handler trains an artifact on demand before the server reports ready.

```bash
rm -f models/crop_model.joblib
uvicorn api.main:app --host 127.0.0.1 --port 8000 &
sleep 12
```

```
INFO:     Started server process [6588]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

`sleep 2` is enough when `models/crop_model.joblib` already exists; the longer
sleep here covers the ~2 s training run on a clean clone plus interpreter
start-up. If the next command prints `Connection refused`, the server had not
finished starting — sleep longer, do not conclude it is broken.

### 2a — Health

```bash
curl -s http://127.0.0.1:8000/health
```

```
{"status":"ok","model_loaded":true,"n_classes":22}
```

### 2b — A prediction

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"N":90,"P":42,"K":43,"temperature":25,"humidity":80,"ph":6.5,"rainfall":200}'
```

```
{"crop":"jute","confidence":0.725342776978384,"probabilities":{"jute":0.725342776978384,"rice":0.2746571705151149,"coffee":5.250650112483512e-08}}
```

The same answer as Week 9's `predict(EXAMPLE_INPUT)` — `jute` at 0.7253 with
`rice` second at 0.2747 — now reachable by any program that can speak HTTP. The
Week 8 `rice`/`jute` confusion pair is visible in the response body.

### 2c — The interactive docs

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/docs
```

```
200
```

Open `http://127.0.0.1:8000/docs` in a browser while the server runs: the seven
fields appear with their ranges and descriptions, and *Try it out* sends a real
request. Nothing about that page was written by hand — it is generated from
`api/schemas.py`.

```bash
curl -s http://127.0.0.1:8000/openapi.json | python -c "import json,sys; print(list(json.load(sys.stdin)['paths']))"
```

```
['/health', '/predict']
```

---

## Step 3 — Make the API say 422

Still the same running server.

### 3a — A missing field

```bash
curl -s -i -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"N":90,"P":42,"K":43,"temperature":25,"humidity":80,"ph":6.5}'
```

```
HTTP/1.1 422 Unprocessable Entity
date: Fri, 31 Jul 2026 14:51:10 GMT
server: uvicorn
content-length: 151
content-type: application/json

{"detail":[{"type":"missing","loc":["body","rainfall"],"msg":"Field required","input":{"N":90,"P":42,"K":43,"temperature":25,"humidity":80,"ph":6.5}}]}
```

`loc` names the field. No handler code ran.

### 3b — An out-of-range value

```bash
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"N":90,"P":42,"K":43,"temperature":25,"humidity":80,"ph":99,"rainfall":200}'
```

```
{"detail":[{"type":"less_than_equal","loc":["body","ph"],"msg":"Input should be less than or equal to 14","input":99,"ctx":{"le":14.0}}]}
```

### 3c — The wrong type

```bash
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"N":90,"P":42,"K":43,"temperature":25,"humidity":80,"ph":"acidic","rainfall":200}'
```

```
{"detail":[{"type":"float_parsing","loc":["body","ph"],"msg":"Input should be a valid number, unable to parse string as a number","input":"acidic"}]}
```

### 3d — An unknown field

```bash
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"N":90,"P":42,"K":43,"temperature":25,"humidity":80,"ph":6.5,"rainfall":200,"Nitrogen":1}'
```

```
{"detail":[{"type":"extra_forbidden","loc":["body","Nitrogen"],"msg":"Extra inputs are not permitted","input":1}]}
```

`extra="forbid"` is what makes a typo an error instead of a silent drop.

### 3e — The wrong path and the wrong method

```bash
curl -s -w "\n%{http_code}\n" http://127.0.0.1:8000/predictt
curl -s -w "\n%{http_code}\n" http://127.0.0.1:8000/predict
```

```
{"detail":"Not Found"}
404
{"detail":"Method Not Allowed"}
405
```

The second is a `GET` on a `POST`-only endpoint.

---

## Step 4 — What the schema cannot catch

```bash
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"N":0,"P":0,"K":0,"temperature":55,"humidity":3,"ph":13.9,"rainfall":0}'
```

```
{"crop":"pigeonpeas","confidence":0.9999996088122228,"probabilities":{"pigeonpeas":0.9999996088122228,"mothbeans":3.9118777508700714e-07,"maize":2.410298549500865e-106}}
```

Every field is inside its declared range, so this is a **200**. The combination
— no nutrients, 55 °C, 3% humidity, pH 13.9, no rain — appears nowhere in the
2,200 training rows, and the model answers with **99.99997% confidence**
anyway. That number is not knowledge; it is the shape of a classifier that has
no way to say "none of these". Out-of-distribution detection is not in this
course, but the hole is real — see
[`learning_notes.md`](learning_notes.md) §6.5.

---

## Step 5 — Stop the server

```bash
kill %1        # or: kill <PID printed at start-up>
```

The uvicorn log for the whole session, one line per request:

```
INFO:     127.0.0.1:44076 - "GET /predictt HTTP/1.1" 404 Not Found
INFO:     127.0.0.1:44090 - "GET /predict HTTP/1.1" 405 Method Not Allowed
INFO:     127.0.0.1:44102 - "POST /predict HTTP/1.1" 422 Unprocessable Entity
INFO:     127.0.0.1:44112 - "POST /predict HTTP/1.1" 422 Unprocessable Entity
```

Confirm the artifact the clean-clone start-up built:

```bash
ls -la models/
```

```
total 16
drwxrwxr-x  2 runner runner 4096 Jul 31 14:57 .
drwxr-xr-x 13 runner runner 4096 Jul 31 14:49 ..
-rw-rw-r--  1 runner runner    0 Jul 31 14:44 .gitkeep
-rw-rw-r--  1 runner runner 6329 Jul 31 14:57 crop_model.joblib
```

6,329 bytes — byte-for-byte the Week 9 artifact, because nothing about training
changed this week. It is still git-ignored.

---

## Step 6 — The API tests, with no server running

```bash
pytest tests/test_api.py -q
```

```
...........................                                              [100%]
27 passed in 1.67s
```

`TestClient` drives the app in-process, so these 27 tests open
no port and cannot collide with a dev server on 8000. They cover 200 on a valid
payload, 422 on missing / out-of-range / mistyped / unknown / empty bodies, 500
on an unexpected failure (asserting the internal message does **not** leak into
the body), 503 with no model loaded, `/health` in both states, 405 on
`GET /predict`, and the OpenAPI document.

---

## Step 7 — The Streamlit app

First the check the agent can make: the module imports cleanly, with no syntax
error and no missing dependency. It draws nothing, because the page is only
rendered under a Streamlit runtime.

```bash
python -c "import app.streamlit_app; print('import ok')"
```

```
import ok
```

Then the server, backgrounded like the API:

```bash
streamlit run app/streamlit_app.py --server.headless true --server.port 8501 &
sleep 15
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8501
```

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://10.1.0.40:8501

200
```

```bash
kill %1
```

### The human-verified part

A 200 from the root URL proves the server serves a page. It does **not** prove
the form works — that is yours to check. Run it without `--server.headless`:

```bash
streamlit run app/streamlit_app.py
```

and confirm, in the browser:

1. Seven numeric inputs appear in two columns, pre-filled with the example
   (N 90, P 42, K 43, temperature 25, humidity 80, pH 6.5, rainfall 200).
2. Pressing **Recommend a crop** shows `Recommended crop: jute` with a
   confidence of **72.5%** and a bar chart with `rice` close behind — the same
   numbers as Step 2b, because it is the same function on the same artifact.
3. The API is **not** running, and the app works anyway. That is the §5.3
   decision, visible: the UI calls `predict()` in-process and never sends an
   HTTP request.
4. Typing a pH of 20 is impossible — the widget clamps it at 14, matching the
   bound in `api/schemas.py`.

---

## Definition of Done — Week 10

| Requirement | Status |
| --- | --- |
| `docs/curriculum/week10/{syllabus,learning_notes,exercises,validation}.md` exist | ✅ |
| New code has docstrings and passes lint (`ruff check .`) | ✅ All checks passed |
| New behaviour has tests and `pytest` passes | ✅ 404 passed, 1 skipped (377 + 27) |
| `requirements.txt` updated, every dependency pinned | ✅ `fastapi`, `pydantic`, `uvicorn`, `streamlit`, `httpx` |
| README progress table updated | ✅ |
| `docs/ml_concepts.md` has an entry per new concept | ✅ |
| `validation.md` commands actually run, real output pasted | ✅ this file |
| `docs/architecture.md` records the request flow | ✅ UI/client -> API -> pipeline -> model |
| Valid payload returns 200 with a crop label | ✅ Step 2b, and `tests/test_api.py` |
| Invalid payload returns 422 | ✅ Step 3, five ways |
| API and UI depend only on `src/pipelines/predict_pipeline.py` | ✅ no import of `app` in `api/`, none of `api` in `app/` |
| Streamlit runs without the API | ✅ Step 7, human-verified |

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| ◀ [§10.3 Exercises](exercises.md) | [Chapter 10 — Serving the Model Over HTTP](README.md) · 🗺 [Roadmap](../README.md) | [Chapter 11 — Containerization and Continuous Integration](../week11/README.md) ▶ |

<!-- nav:end -->
