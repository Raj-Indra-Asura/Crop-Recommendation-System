# Chapter 4 — Baseline Models

> 🗺 [Roadmap](../README.md) › [Part II — Modelling (Weeks 4-7)](../README.md#part-ii--modelling-weeks-4-7) › **Chapter 4 — Baseline Models**

**Establishing what "good" means, before building anything real**

Part II — Modelling (Weeks 4-7) · Chapter 4 of 12 · the curriculum calls this
**Week 4**, and the directory is `docs/curriculum/week04/`.

---

## Before you open this chapter

* **You should already have finished:** Chapter 3: you have a stratified split
  and a train-fitted preprocessor.
* **Previous chapter:** [Chapter 3 — Data Preparation](../week03/README.md)
* **Environment:** the virtual environment from Chapter 1, with
  `requirements.txt` installed.

## Read this chapter in this order

Top to bottom. Every row assumes the rows above it are done.

| Step | What | Where | Why here |
| --- | --- | --- | --- |
| 1 | §4.1 Syllabus | [`syllabus.md`](syllabus.md) | objectives, prerequisites, concepts, deliverables — read first, it is the map of the chapter |
| 2 | §4.2 Learning notes | [`learning_notes.md`](learning_notes.md) | the teaching text: what, why, where, how, and the common mistakes — read top to bottom |
| 3 | Code | [`src/models/baseline.py`](../../../src/models/baseline.py) | `get_baseline_model(strategy)` |
| 4 | Code | [`src/evaluation/metrics.py`](../../../src/evaluation/metrics.py) | `evaluate_model()`, `cross_validated_accuracy()` and `build_cv()`, extended in later chapters, never replaced |
| 5 | Code | [`tests/test_baseline.py`](../../../tests/test_baseline.py) | the factory, the folds and the 1/22 result on the real data |
| 6 | Notebook | [`notebooks/04_baseline_models.ipynb`](../../../notebooks/04_baseline_models.ipynb) | run it cell by cell: the number every future model must beat |
| 7 | §4.3 Exercises | [`exercises.md`](exercises.md) | beginner, then intermediate, then challenge — do them before moving on |
| 8 | §4.4 Validation | [`validation.md`](validation.md) | run the commands, compare against the real recorded output |

## What this chapter adds to the repository

* [`src/models/baseline.py`](../../../src/models/baseline.py) —
  `get_baseline_model(strategy)`
* [`src/evaluation/metrics.py`](../../../src/evaluation/metrics.py) —
  `evaluate_model()`, `cross_validated_accuracy()` and `build_cv()`, extended in
  later chapters, never replaced
* [`tests/test_baseline.py`](../../../tests/test_baseline.py) — the factory, the
  folds and the 1/22 result on the real data
* [`notebooks/04_baseline_models.ipynb`](../../../notebooks/04_baseline_models.ipynb)
  — the number every future model must beat

The full list, including documentation updates, is in the
[syllabus](syllabus.md#deliverables-for-the-week).

## Do not move on until

* [ ] You can say what the baseline accuracy is (4.55%) and why it is 1/22.
* [ ] You can explain cross-validation in terms of folds, and why a single split
  is a noisier estimate.
* [ ] The exercises in [`exercises.md`](exercises.md) are done, not skimmed.
* [ ] Every command in [`validation.md`](validation.md) produced the output
  recorded there.

Then go to [Chapter 5 — Classification Models](../week05/README.md).

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| ◀ [§3.4 Validation](../week03/validation.md) | 🗺 [Roadmap](../README.md) | [§4.1 Syllabus](syllabus.md) ▶ |

<!-- nav:end -->
