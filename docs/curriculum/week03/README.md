# Chapter 3 — Data Preparation

> 🗺 [Roadmap](../README.md) › [Part I — Foundations (Weeks 1-3)](../README.md#part-i--foundations-weeks-1-3) › **Chapter 3 — Data Preparation**

**Turning raw data into model-ready data, correctly**

Part I — Foundations (Weeks 1-3) · Chapter 3 of 12 · the curriculum calls this
**Week 3**, and the directory is `docs/curriculum/week03/`.

---

## Before you open this chapter

* **You should already have finished:** Chapter 2: you know the distributions,
  the class balance and the scale differences between features.
* **Previous chapter:** [Chapter 2 — Exploratory Data
  Analysis](../week02/README.md)
* **Environment:** the virtual environment from Chapter 1, with
  `requirements.txt` installed.

## Read this chapter in this order

Top to bottom. Every row assumes the rows above it are done.

| Step | What | Where | Why here |
| --- | --- | --- | --- |
| 1 | §3.1 Syllabus | [`syllabus.md`](syllabus.md) | objectives, prerequisites, concepts, deliverables — read first, it is the map of the chapter |
| 2 | §3.2 Learning notes | [`learning_notes.md`](learning_notes.md) | the teaching text: what, why, where, how, and the common mistakes — read top to bottom |
| 3 | Code | [`src/data/split.py`](../../../src/data/split.py) | `stratified_split()` and `class_proportions()` |
| 4 | Code | [`src/preprocessing/preprocessor.py`](../../../src/preprocessing/preprocessor.py) | `build_preprocessor()` and `build_preprocessing_pipeline()` |
| 5 | Code | [`tests/test_preprocessing.py`](../../../tests/test_preprocessing.py) | the split is stratified, reproducible, and the test rows are provably not part of the fit |
| 6 | Notebook | [`notebooks/03_data_preparation.ipynb`](../../../notebooks/03_data_preparation.ipynb) | run it cell by cell: the full preparation, writing five files to `data/processed/` |
| 7 | §3.3 Exercises | [`exercises.md`](exercises.md) | beginner, then intermediate, then challenge — do them before moving on |
| 8 | §3.4 Validation | [`validation.md`](validation.md) | run the commands, compare against the real recorded output |

## What this chapter adds to the repository

* [`src/data/split.py`](../../../src/data/split.py) — `stratified_split()` and
  `class_proportions()`
* [`src/preprocessing/preprocessor.py`](../../../src/preprocessing/preprocessor.py)
  — `build_preprocessor()` and `build_preprocessing_pipeline()`
* [`tests/test_preprocessing.py`](../../../tests/test_preprocessing.py) — the
  split is stratified, reproducible, and the test rows are provably not part of
  the fit
* [`notebooks/03_data_preparation.ipynb`](../../../notebooks/03_data_preparation.ipynb)
  — the full preparation, writing five files to `data/processed/`

The full list, including documentation updates, is in the
[syllabus](syllabus.md#deliverables-for-the-week).

## Do not move on until

* [ ] You can explain why the scaler is fitted on the training set only, and
  what breaks if it is not.
* [ ] You can reproduce the same split twice and show the class proportions are
  preserved.
* [ ] The exercises in [`exercises.md`](exercises.md) are done, not skimmed.
* [ ] Every command in [`validation.md`](validation.md) produced the output
  recorded there.

Then go to [Chapter 4 — Baseline Models](../week04/README.md).

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| ◀ [§2.4 Validation](../week02/validation.md) | 🗺 [Roadmap](../README.md) | [§3.1 Syllabus](syllabus.md) ▶ |

<!-- nav:end -->
