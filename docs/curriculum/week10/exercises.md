# Week 10 — Exercises

> 🗺 [Roadmap](../README.md) › [Part IV — Production (Weeks 9-11)](../README.md#part-iv--production-weeks-9-11) › [Chapter 10 — Serving the Model Over HTTP](README.md) › **§10.3 Exercises**

Work through these after reading [`learning_notes.md`](learning_notes.md) and
running the commands in [`validation.md`](validation.md). Nothing here needs a
new dependency beyond this week's five pins, and nothing needs a notebook.

Two servers are involved. Remember the rule from `validation.md`: start them in
the **background**, do the work, then kill them. If a command hangs with no
prompt, you started one in the foreground — `Ctrl-C`.

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 &
sleep 5
# ... your requests ...
kill %1
```

Most Python exercises start from the same imports:

```python
from fastapi.testclient import TestClient

from api.main import app, get_pipeline
from api.schemas import EXAMPLE_REQUEST, CropFeatures, PredictionResponse
from src.pipelines.predict_pipeline import EXAMPLE_INPUT, predict, predict_proba
```

Never commit a modified `api/` or `app/` from an exercise — use `git diff` and
`git checkout --` when you are done.

---

## Exercise 1 — Read one request and one response

**Goal:** see every part of the HTTP cycle, not just the JSON.

Start the server, then:

```bash
curl -v -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"N":90,"P":42,"K":43,"temperature":25,"humidity":80,"ph":6.5,"rainfall":200}'
```

1. In the `-v` output, lines starting `>` are what you sent and `<` are what came
   back. Name the method, path, one request header, the status code and one
   response header.
2. Remove `-H "Content-Type: application/json"` and run it again. What status
   code do you get, and why is *that* code the right one for the mistake you
   made?
3. Send the same body to `/health` with `-X POST`. Which status code, and which
   part of the request did the server object to?
4. Which of the three failures above are 4xx and which 5xx? Say in one sentence
   per case who has to change something.

---

## Exercise 2 — Make the schema reject you

**Goal:** predict Pydantic's behaviour before running it.

For each payload below, **write down first** whether it returns 200 or 422 and
which field is named, then check with `curl`.

| # | Change to `EXAMPLE_REQUEST` | Your prediction | Actual |
| --- | --- | --- | --- |
| a | `"ph": 7` (an integer, not a float) | | |
| b | `"ph": "6.5"` (a numeric string) | | |
| c | `"ph": null` | | |
| d | `"humidity": 100.0001` | | |
| e | `"N": 0` | | |
| f | `"N": -0.0` | | |
| g | drop `"K"` and set `"ph": 99` | | |
| h | `{}` | | |

1. Which of (a)-(c) are accepted, and what does that tell you about the
   difference between *type checking* and *type coercion*?
2. In (g), how many entries does `detail` contain? Why is answering about every
   bad field at once better for a client than stopping at the first?
3. `"N": 1e400` parses as JSON. What happens, and which layer objects?

---

## Exercise 3 — Change a bound and watch the contract move

**Goal:** see that the schema is a single source of truth.

In `api/schemas.py`, change `ph`'s upper bound from `le=14` to `le=9`.

1. Restart the server. Send the example request (pH 6.5) — still 200? Now send
   pH 12.
2. Reload `http://127.0.0.1:8000/docs`. What changed on the page, and how many
   files did you have to edit to change it?
3. Run `pytest tests/test_api.py`. Which test fails, and is that failure correct
   behaviour from the test suite or a badly written test?
4. Revert the change. Then argue, in three sentences, whether the *real* bound
   should be the 0-14 pH scale or the training range (3.5-9.9). What does the
   API owe a caller whose soil is at pH 11?

---

## Exercise 4 — 422 vs 500, deliberately

**Goal:** produce both, and never confuse them again.

1. Produce a 422 with a request that is *nearly* right (one field misspelled).
   Write the one-sentence explanation you would give the client's developer.
2. Now produce a 500. Temporarily edit `api/main.py` so that `predict_crop`
   raises `RuntimeError("boom")` as its first statement, restart, and send a
   *valid* request. What does the client see? What does the terminal running
   uvicorn show that the client does not?
3. Why is that asymmetry a security requirement rather than a matter of taste?
   Name two specific things a traceback would tell a stranger about your server.
4. Revert. Now break it differently: point the server at a corrupt artifact.

   ```bash
   printf 'not a model' > /tmp/bad.joblib
   ```

   Edit `api/main.py`'s lifespan to call `load_model("/tmp/bad.joblib")`,
   restart, and check `GET /health` and `POST /predict`. Which codes come back,
   and why is a 503 more useful to a load balancer than a 500 here?
5. Revert all edits (`git checkout -- api/main.py`) and confirm with
   `pytest tests/test_api.py`.

---

## Exercise 5 — The API is not the model

**Goal:** prove the layering claim rather than trust it.

1. Run these and explain each result:

   ```bash
   grep -rn "streamlit" api/ ; echo "exit: $?"
   grep -rn "httpx\|requests\|8000" app/ ; echo "exit: $?"
   grep -rn "fastapi\|streamlit" src/ ; echo "exit: $?"
   ```

2. `api/main.py` imports `sklearn.pipeline.Pipeline`. Is that a violation of
   "the API contains no modelling logic"? Argue both sides in two sentences,
   then say what you would do if the rule had to be absolute.
3. Suppose the model gains an eighth feature, `soil_type`, which is
   *categorical*. List every file that must change, in dependency order, and
   name the one place where the change would be invisible to a caller of
   `predict()`.
4. Add a third entry point without touching `api/` or `app/`: a script
   `/tmp/predict_cli.py` that reads a JSON object on stdin and prints the crop.
   How many lines is it, and what does that number say about the layering?

---

## Exercise 6 — Load once, not per request

**Goal:** measure the thing the lifespan handler avoids.

1. Time a warm request:

   ```bash
   curl -s -o /dev/null -w "%{time_total}\n" -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"N":90,"P":42,"K":43,"temperature":25,"humidity":80,"ph":6.5,"rainfall":200}'
   ```

2. In Python, time `load_pipeline()` on its own (`time.perf_counter()` around
   it, model file present). Roughly what fraction of a request would loading be
   if it happened per call?
3. Now delete `models/crop_model.joblib` and time `load_pipeline()` again. Which
   is the number that actually justifies the lifespan handler?
4. `@st.cache_resource` in `app/streamlit_app.py` solves the same problem for a
   different reason. Delete the decorator, run the app, move a slider a few
   times, and describe what you observe. Put it back.

---

## Exercise 7 — Test the boundary, not the prediction

**Goal:** learn what an API test should assert.

1. In `tests/test_api.py`, change
   `assert body["crop"] in EXPECTED_LABELS` to `assert body["crop"] == "jute"`.
   The test still passes. Explain in one sentence why it is nonetheless a worse
   test, and name a legitimate change that would break it.
2. Write a new test asserting that the response contains **exactly** the keys
   `crop`, `confidence` and `probabilities`. Why is a test on the response
   *shape* more valuable than one on its values?
3. `client` deliberately does not use `TestClient` as a context manager. Change
   it to `with TestClient(app) as client:` and run the suite with
   `models/crop_model.joblib` deleted. What happens, how long does it take, and
   what did the override protect you from?
4. Revert both edits.

---

## Exercise 8 — Be a different client

**Goal:** feel the point of a *web* API.

1. Call the API from Python without `TestClient`:

   ```python
   import httpx
   print(httpx.post("http://127.0.0.1:8000/predict", json=EXAMPLE_REQUEST).json())
   ```

2. Call it from a shell with no Python involvement (`curl`), and from the
   browser via `/docs`. Three clients, one contract, zero shared code — what
   exactly is shared between them?
3. Sketch (do not write) how a Kotlin mobile app would call this. Which files in
   this repository would it need? Which would it need under Week 9's
   arrangement?
4. Stop the server and re-run the `httpx` call. What does the failure look like
   from the client's side, and which of the failure modes in `learning_notes.md`
   §1.2 does it illustrate?

---

## Exercise 9 — The decision, re-argued

**Goal:** hold the §5.3 decision up to a case where it flips.

The project chose (A) Streamlit calls `predict()` in-process, over (B) Streamlit
calls the API over HTTP.

1. Rewrite the Streamlit prediction call as option (B) in a scratch copy
   (`cp app/streamlit_app.py /tmp/`, then use `httpx.post`). Start both servers
   and confirm it works. Then stop the API and describe the user's experience.
2. List everything the UI now needs to be told that it did not need before (base
   URL, timeout, retry, error mapping…). How many of those are configuration
   that would differ between your laptop and a deployment?
3. Give a concrete scenario where (B) is clearly right — think about model size,
   about who is allowed to hold the artifact, and about who writes the UI.
4. Which single sentence in `docs/architecture.md` would you have to change to
   switch the project to (B)?

---

## Exercise 10 — Out-of-distribution, on purpose

**Goal:** locate the limit of validation.

1. Send the absurd-but-in-range payload from `validation.md` Step 4. Record the
   crop and the confidence.
2. Construct two more payloads that are physically impossible in combination but
   pass the schema. Does the confidence ever drop below 0.9?
3. Why does a 22-class classifier report high confidence on inputs it has never
   seen anything like? Answer in terms of what `predict_proba` normalises over.
4. Propose — in prose, no code — one cheap guard the API could add, and state
   what it would cost in false alarms. Would you return a 422, a 200 with a
   warning field, or something else? Defend the status code you chose.

---

## Exercise 11 — Extend the contract

**Goal:** add an endpoint without disturbing the layers.

Add `GET /crops`, returning the 22 labels the model can predict.

1. Which layer does the list come from? (Hint: `pipeline.classes_`, or
   `src.data.EXPECTED_LABELS` — argue for one.)
2. Write the Pydantic response model first, then the endpoint. Why that order?
3. Add two tests: the status code and the length of the list. Should the second
   assert `== 22` or `== len(EXPECTED_LABELS)`?
4. Check `/docs` — what did you have to write to document it?
5. Decide whether to keep it. Is a new endpoint justified by "a client might
   want it", or should an API only grow when a caller actually asks?

---

## Exercise 12 — Read the week back

**Goal:** confirm the week's claims in your own words.

Answer without looking anything up:

1. A client sends `{"N": 90, ..., "ph": 6.5, "rainfall": "lots"}`. Which status
   code, produced by which layer, and did any of our code run?
2. The server has been up for a day and starts returning 500 on every request.
   Name three things you would check, in order.
3. Why is `/health` written so that it takes no dependencies?
4. A colleague proposes deleting `api/schemas.py` because "`predict()` already
   validates". Give the two-sentence rebuttal.
5. Why does the Streamlit app not have a `Dockerfile` in its future, while the
   API does?
6. What, precisely, is still missing before this could serve a real user? Name
   five things, and say which week fixes each.

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| ◀ [§10.2 Learning notes](learning_notes.md) | [Chapter 10 — Serving the Model Over HTTP](README.md) · 🗺 [Roadmap](../README.md) | [§10.4 Validation](validation.md) ▶ |

<!-- nav:end -->
