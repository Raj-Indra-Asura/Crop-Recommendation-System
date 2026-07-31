# Chapter 9 — Productionizing the Model

> 🗺 [Roadmap](../README.md) › [Part IV — Production (Weeks 9-11)](../README.md#part-iv--production-weeks-9-11) › **Chapter 9 — Productionizing the Model**

**From notebook to pipeline: one fitted object, one saved artifact, one
prediction function**

Part IV — Production (Weeks 9-11) · Chapter 9 of 12 · the curriculum calls this
**Week 9**, and the directory is `docs/curriculum/week09/`.

---

## Before you open this chapter

* **You should already have finished:** Chapter 8: a final model and its
  hyperparameters have been chosen and defended.
* **Previous chapter:** [Chapter 8 — Model Evaluation &
  Explainability](../week08/README.md)
* **Environment:** the virtual environment from Chapter 1, with
  `requirements.txt` installed.

## Read this chapter in this order

Top to bottom. Every row assumes the rows above it are done.

| Step | What | Where | Why here |
| --- | --- | --- | --- |
| 1 | §9.1 Syllabus | [`syllabus.md`](syllabus.md) | objectives, prerequisites, concepts, deliverables — read first, it is the map of the chapter |
| 2 | §9.2 Learning notes | [`learning_notes.md`](learning_notes.md) | the teaching text: what, why, where, how, and the common mistakes — read top to bottom |
| 3 | Code | [`src/config.py`](../../../src/config.py) | paths, seed, chosen model name and hyperparameters — all inert |
| 4 | Code | [`src/pipelines/training_pipeline.py`](../../../src/pipelines/training_pipeline.py) | `build_model()`, `train_pipeline()`, `save_pipeline()` and `python -m src.pipelines.training_pipeline` |
| 5 | Code | [`src/pipelines/predict_pipeline.py`](../../../src/pipelines/predict_pipeline.py) | `load_pipeline()` (train-on-demand), `predict()` and `predict_proba()` |
| 6 | Code | [`tests/test_training_pipeline.py`](../../../tests/test_training_pipeline.py) | the pipeline trains, scores and saves |
| 7 | Code | [`tests/test_predict_pipeline.py`](../../../tests/test_predict_pipeline.py) | the artifact reloads and predicts from a plain dict |
| 8 | §9.3 Exercises | [`exercises.md`](exercises.md) | beginner, then intermediate, then challenge — do them before moving on |
| 9 | §9.4 Validation | [`validation.md`](validation.md) | run the commands, compare against the real recorded output |

## What this chapter adds to the repository

* [`src/config.py`](../../../src/config.py) — paths, seed, chosen model name and
  hyperparameters — all inert
* [`src/pipelines/training_pipeline.py`](../../../src/pipelines/training_pipeline.py)
  — `build_model()`, `train_pipeline()`, `save_pipeline()` and `python -m
  src.pipelines.training_pipeline`
* [`src/pipelines/predict_pipeline.py`](../../../src/pipelines/predict_pipeline.py)
  — `load_pipeline()` (train-on-demand), `predict()` and `predict_proba()`
* [`tests/test_training_pipeline.py`](../../../tests/test_training_pipeline.py)
  — the pipeline trains, scores and saves
* [`tests/test_predict_pipeline.py`](../../../tests/test_predict_pipeline.py) —
  the artifact reloads and predicts from a plain dict

The full list, including documentation updates, is in the
[syllabus](syllabus.md#deliverables-for-the-week).

## Do not move on until

* [ ] `python -m src.pipelines.training_pipeline` writes
  `models/crop_model.joblib` and reports 99.55%.
* [ ] You can explain why preprocessing lives inside the saved pipeline rather
  than beside it.
* [ ] The exercises in [`exercises.md`](exercises.md) are done, not skimmed.
* [ ] Every command in [`validation.md`](validation.md) produced the output
  recorded there.

Then go to [Chapter 10 — Serving the Model Over HTTP](../week10/README.md).

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| ◀ [§8.4 Validation](../week08/validation.md) | 🗺 [Roadmap](../README.md) | [§9.1 Syllabus](syllabus.md) ▶ |

<!-- nav:end -->
