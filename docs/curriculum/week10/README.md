# Chapter 10 — Serving the Model Over HTTP

> 🗺 [Roadmap](../README.md) › [Part IV — Production (Weeks 9-11)](../README.md#part-iv--production-weeks-9-11) › **Chapter 10 — Serving the Model Over HTTP**

**From `predict()` to `POST /predict`: an API, a demo UI, and the difference
between them**

Part IV — Production (Weeks 9-11) · Chapter 10 of 12 · the curriculum calls this
**Week 10**, and the directory is `docs/curriculum/week10/`.

---

## Before you open this chapter

* **You should already have finished:** Chapter 9: `predict({...})` works from a
  plain dictionary against a saved artifact.
* **Previous chapter:** [Chapter 9 — Productionizing the
  Model](../week09/README.md)
* **Environment:** the virtual environment from Chapter 1, with
  `requirements.txt` installed.

## Read this chapter in this order

Top to bottom. Every row assumes the rows above it are done.

| Step | What | Where | Why here |
| --- | --- | --- | --- |
| 1 | §10.1 Syllabus | [`syllabus.md`](syllabus.md) | objectives, prerequisites, concepts, deliverables — read first, it is the map of the chapter |
| 2 | §10.2 Learning notes | [`learning_notes.md`](learning_notes.md) | the teaching text: what, why, where, how, and the common mistakes — read top to bottom |
| 3 | Code | [`api/schemas.py`](../../../api/schemas.py) | `CropFeatures`, `PredictionResponse`, `HealthResponse` |
| 4 | Code | [`api/main.py`](../../../api/main.py) | `POST /predict`, `GET /health`, the lifespan loader and the test seam |
| 5 | Code | [`app/streamlit_app.py`](../../../app/streamlit_app.py) | the seven-field demo form |
| 6 | Code | [`tests/test_api.py`](../../../tests/test_api.py) | 200, 422, 500, 503 and `/health` |
| 7 | Code | [`docs/architecture.md`](../../architecture.md) | the request flow and the layering rule |
| 8 | §10.3 Exercises | [`exercises.md`](exercises.md) | beginner, then intermediate, then challenge — do them before moving on |
| 9 | §10.4 Validation | [`validation.md`](validation.md) | run the commands, compare against the real recorded output |

## What this chapter adds to the repository

* [`api/schemas.py`](../../../api/schemas.py) — `CropFeatures`,
  `PredictionResponse`, `HealthResponse`
* [`api/main.py`](../../../api/main.py) — `POST /predict`, `GET /health`, the
  lifespan loader and the test seam
* [`app/streamlit_app.py`](../../../app/streamlit_app.py) — the seven-field demo
  form
* [`tests/test_api.py`](../../../tests/test_api.py) — 200, 422, 500, 503 and
  `/health`
* [`docs/architecture.md`](../../architecture.md) — the request flow and the
  layering rule

The full list, including documentation updates, is in the
[syllabus](syllabus.md#deliverables-for-the-week).

## Do not move on until

* [ ] You can start the API, call `/predict` with curl, and explain each status
  code it can return.
* [ ] You can say why validation belongs at the edge and not inside the model
  code.
* [ ] The exercises in [`exercises.md`](exercises.md) are done, not skimmed.
* [ ] Every command in [`validation.md`](validation.md) produced the output
  recorded there.

Then go to [Chapter 11 — Containerization and Continuous
Integration](../week11/README.md).

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| ◀ [§9.4 Validation](../week09/validation.md) | 🗺 [Roadmap](../README.md) | [§10.1 Syllabus](syllabus.md) ▶ |

<!-- nav:end -->
