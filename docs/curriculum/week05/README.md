# Chapter 5 — Classification Models

> 🗺 [Roadmap](../README.md) › [Part II — Modelling (Weeks 4-7)](../README.md#part-ii--modelling-weeks-4-7) › **Chapter 5 — Classification Models**

**The first real algorithms: three ways to draw a boundary, compared fairly**

Part II — Modelling (Weeks 4-7) · Chapter 5 of 12 · the curriculum calls this
**Week 5**, and the directory is `docs/curriculum/week05/`.

---

## Before you open this chapter

* **You should already have finished:** Chapter 4: you have a baseline number
  and a fold strategy to compare against.
* **Previous chapter:** [Chapter 4 — Baseline Models](../week04/README.md)
* **Environment:** the virtual environment from Chapter 1, with
  `requirements.txt` installed.

## Read this chapter in this order

Top to bottom. Every row assumes the rows above it are done.

| Step | What | Where | Why here |
| --- | --- | --- | --- |
| 1 | §5.1 Syllabus | [`syllabus.md`](syllabus.md) | objectives, prerequisites, concepts, deliverables — read first, it is the map of the chapter |
| 2 | §5.2 Learning notes | [`learning_notes.md`](learning_notes.md) | the teaching text: what, why, where, how, and the common mistakes — read top to bottom |
| 3 | Code | [`src/models/classical_models.py`](../../../src/models/classical_models.py) | `get_logistic_regression()`, `get_knn()`, `get_naive_bayes()` and the `CLASSICAL_MODEL_FACTORIES` registry |
| 4 | Code | [`tests/test_classical_models.py`](../../../tests/test_classical_models.py) | factories, the shared training loop and algorithm-specific behaviour |
| 5 | Notebook | [`notebooks/05_classification_models.ipynb`](../../../notebooks/05_classification_models.ipynb) | run it cell by cell: Part 1 — the four-row results table Chapters 6-8 extend |
| 6 | §5.3 Exercises | [`exercises.md`](exercises.md) | beginner, then intermediate, then challenge — do them before moving on |
| 7 | §5.4 Validation | [`validation.md`](validation.md) | run the commands, compare against the real recorded output |

## What this chapter adds to the repository

* [`src/models/classical_models.py`](../../../src/models/classical_models.py) —
  `get_logistic_regression()`, `get_knn()`, `get_naive_bayes()` and the
  `CLASSICAL_MODEL_FACTORIES` registry
* [`tests/test_classical_models.py`](../../../tests/test_classical_models.py) —
  factories, the shared training loop and algorithm-specific behaviour
* [`notebooks/05_classification_models.ipynb`](../../../notebooks/05_classification_models.ipynb)
  — Part 1 — the four-row results table Chapters 6-8 extend

The full list, including documentation updates, is in the
[syllabus](syllabus.md#deliverables-for-the-week).

## Do not move on until

* [ ] You can describe how each of the three algorithms decides, in one sentence
  each.
* [ ] You can say which ones need scaled features and why.
* [ ] The exercises in [`exercises.md`](exercises.md) are done, not skimmed.
* [ ] Every command in [`validation.md`](validation.md) produced the output
  recorded there.

Then go to [Chapter 6 — Margin-based and Tree-based
Models](../week06/README.md).

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| ◀ [§4.4 Validation](../week04/validation.md) | 🗺 [Roadmap](../README.md) | [§5.1 Syllabus](syllabus.md) ▶ |

<!-- nav:end -->
