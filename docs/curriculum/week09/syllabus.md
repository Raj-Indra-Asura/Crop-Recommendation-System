# Week 9 — Productionizing the Model

> 🗺 [Roadmap](../README.md) › [Part IV — Production (Weeks 9-11)](../README.md#part-iv--production-weeks-9-11) › [Chapter 9 — Productionizing the Model](README.md) › **§9.1 Syllabus**

## Title

**From notebook to pipeline: one fitted object, one saved artifact, one
`predict()`**

## Learning objectives

By the end of this week a student should be able to:

1. Name the three things a notebook cannot guarantee — **hidden state**,
   **execution order** and **no reuse** — and give a concrete example of each
   going wrong.
2. Explain what a **`Pipeline`** is: a single estimator that chains named steps,
   so `fit` fits them in order and `predict` applies the *fitted* ones in order.
3. Say why the Week 3 **`ColumnTransformer`** belongs *inside* that pipeline
   rather than beside it, in terms of cross-validation, serialization and
   serving.
4. Serialize a fitted pipeline with **`joblib.dump`**, reload it with
   `joblib.load`, and state precisely **what the file contains** (learned
   parameters plus class references) and **what it does not** (the code of those
   classes, the library versions, the training data).
5. Explain why **`joblib` alone does not guarantee reproducibility**, and name
   the three other things that must be fixed: the data, the seed, and the pinned
   environment.
6. Describe **config over hardcoding** — one module holding paths, the seed and
   the chosen hyperparameters — and say what breaks when the same constant is
   written down in four places.
7. Run a training script from the command line
   (`python -m src.pipelines.training_pipeline`) and read its report.
8. Call `predict({...})` from any Python program and get back one of the 22 crop
   labels, and explain why the entry point **trains on demand** when the artifact
   is missing rather than crashing.
9. Say why `models/*.joblib` is **git-ignored** while `data/raw/` is committed.

## Prerequisites

Weeks 1-8, in full. This week assumes and does **not** re-explain:

* the dataset contract and the seven feature columns
  ([Week 1 notes](../week01/learning_notes.md));
* the `ColumnTransformer`, the stratified 80/20 split and `random_state=42`
  ([Week 3 notes](../week03/learning_notes.md));
* accuracy, stratified folds and the 4.55% baseline
  ([Week 4 notes](../week04/learning_notes.md));
* Gaussian naive Bayes and `var_smoothing`
  ([Week 5 notes](../week05/learning_notes.md));
* the final-model decision and the `rice -> jute` confusion pair
  ([Week 8 notes](../week08/learning_notes.md)).

One new pin: `joblib==1.5.3`. It was already installed as a scikit-learn
dependency; this week calls it directly, so it is named explicitly in
`requirements.txt`. No new *optional* dependency — everything below runs on the
pinned environment alone.

## Concepts covered

| Concept | First introduced |
| --- | --- |
| Notebook hidden state and execution order | Week 9 |
| Research code vs production code | Week 9 |
| `Pipeline` as one estimator | Week 3 (built), Week 9 (used end to end) |
| Preprocessing bundled with the model | Week 3 (argued), Week 9 (shipped) |
| Model serialization / pickling | Week 9 |
| `joblib.dump` / `joblib.load` | Week 9 |
| What an artifact does *not* carry | Week 9 |
| Reproducibility: data + seed + code + environment | Week 3 (seed), Week 9 (all four) |
| Pinned requirements, "works on my machine" | Week 1 (pinned), Week 9 (explained) |
| Config module over scattered constants | Week 9 |
| Entry point / `python -m` script | Week 9 |
| Derived artifacts are not version-controlled | Week 9 |
| Train-on-demand for a clean clone | Week 9 |
| Training/serving skew | Week 9 |

Each also appears in [`docs/ml_concepts.md`](../../ml_concepts.md).

## Connection to the previous week

Week 8 ended by choosing a model — Gaussian naive Bayes, 99.55% on the 440
held-out rows — and then admitted what it had *not* done: nothing was
serialised, `models/` was empty, and the only way to obtain a prediction was to
re-run a notebook from the top.

Week 9 pays that debt exactly. It changes **no** modelling decision: the same
split, the same seed, the same preprocessor, the same estimator with the same
`var_smoothing=1e-9`, and the same 0.9955 accuracy. What changes is *who can run
it* — a shell, a test, a scheduler, and next week a web server.

## Connection to the ML lifecycle

```
Frame the problem -> Get the data -> Explore -> Prepare -> Model
    -> Evaluate -> Improve -> [Productionize] -> Deploy -> Monitor
```

Week 9 is the **productionize** stage, and it is the hinge of the course. Before
it, the deliverable of every week was a number in a notebook. After it, the
deliverable is a program. Week 10 puts HTTP in front of `predict()`, Week 11 a
UI, Week 12 a container — and all three import the same two functions written
this week.

## Expected student outcome

### The student CAN, after this week

* **Train from a shell.** `python -m src.pipelines.training_pipeline` loads
  `data/raw/`, splits it, fits the pipeline, prints accuracy 0.9955 / macro F1
  0.9954 on 440 held-out rows, and writes `models/crop_model.joblib` (~6.3 KB,
  about 2.3 s end to end).
* **Predict from any Python program.**
  `predict({"N": 90, "P": 42, ..., "rainfall": 200})` returns a crop label —
  `jute` for that input, with `rice` second at 0.2747, the Week 8 confusion pair
  showing up again in a place where it can now be *seen*.
* **Explain why `joblib` alone is not reproducibility.** The file holds learned
  numbers and references to `sklearn.pipeline.Pipeline`, `StandardScaler` and
  `GaussianNB` — not their code, not their versions, not the rows they were
  fitted on. Reload it under a different scikit-learn and it may warn,
  mis-behave, or refuse. Reproducibility needs all four of: the committed data,
  the fixed seed, the versioned code, and the pinned environment.
* **Point at one config module** for the artifact path, the seed and the chosen
  hyperparameters, and change any of them in exactly one place.
* Run `pytest tests/test_training_pipeline.py tests/test_predict_pipeline.py`
  (32 tests) from a clean clone with no model file present.

### The student CANNOT yet

* **Reach this model over a network.** There is no HTTP endpoint, no JSON
  request/response contract, no port, no server. Everything this week requires a
  Python interpreter on the same machine, with the repository importable. That
  is Week 10.
* Build a UI (Week 11), or containerise and deploy anything (Week 12).
* **Version or track models.** One filename, overwritten on every run. No model
  registry, no experiment tracking, no metadata written next to the artifact
  recording which data and which library versions produced it.
* **Retrain automatically or on a schedule.** The pipeline runs when a human (or
  a test) runs it.
* **Monitor** a served model, or detect drift.
* Serve the model to a *different* language or runtime: a pickle is Python, and
  a Python of roughly the same version at that. Exporting to ONNX or PMML is out
  of scope for this course.

## Deliverables for the week

* `src/config.py` — paths (`MODEL_PATH`, `MODELS_DIR`), `RANDOM_STATE`,
  `TEST_SIZE`, `FINAL_MODEL_NAME`, `FINAL_MODEL_PARAMS` and the pipeline step
  names, all inert.
* `src/pipelines/training_pipeline.py` — `build_model()`,
  `build_training_pipeline()`, `train_pipeline()`, `save_pipeline()` and a
  `main()` runnable as `python -m src.pipelines.training_pipeline`.
* `src/pipelines/predict_pipeline.py` — `load_pipeline()` (train-on-demand),
  `predict()`, `predict_proba()` and a `main()` for a quick manual check.
* `tests/test_training_pipeline.py` (15 tests) and
  `tests/test_predict_pipeline.py` (17 tests) — 377 passed and 1 skipped in the
  whole suite.
* `.gitignore` — `models/*.joblib` stays ignored; the artifact is rebuilt, never
  committed.
* This week's four curriculum documents.
* Updated `requirements.txt` (`joblib==1.5.3`), `docs/ml_concepts.md` and the
  README progress table.

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| ◀ [Chapter 9 — Productionizing the Model](README.md) | [Chapter 9 — Productionizing the Model](README.md) · 🗺 [Roadmap](../README.md) | [§9.2 Learning notes](learning_notes.md) ▶ |

<!-- nav:end -->
