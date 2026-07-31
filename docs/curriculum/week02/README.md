# Chapter 2 — Exploratory Data Analysis

> 🗺 [Roadmap](../README.md) › [Part I — Foundations (Weeks 1-3)](../README.md#part-i--foundations-weeks-1-3) › **Chapter 2 — Exploratory Data Analysis**

**Looking before leaping: understanding the data statistically and visually**

Part I — Foundations (Weeks 1-3) · Chapter 2 of 12 · the curriculum calls this
**Week 2**, and the directory is `docs/curriculum/week02/`.

---

## Before you open this chapter

* **You should already have finished:** Chapter 1: you can load the dataframe
  and you know what a feature and a label are.
* **Previous chapter:** [Chapter 1 — Framing the Problem and Meeting the
  Data](../week01/README.md)
* **Environment:** the virtual environment from Chapter 1, with
  `requirements.txt` installed.

## Read this chapter in this order

Top to bottom. Every row assumes the rows above it are done.

| Step | What | Where | Why here |
| --- | --- | --- | --- |
| 1 | §2.1 Syllabus | [`syllabus.md`](syllabus.md) | objectives, prerequisites, concepts, deliverables — read first, it is the map of the chapter |
| 2 | §2.2 Learning notes | [`learning_notes.md`](learning_notes.md) | the teaching text: what, why, where, how, and the common mistakes — read top to bottom |
| 3 | Code | [`src/utils/eda.py`](../../../src/utils/eda.py) | nine documented exploration helpers |
| 4 | Code | [`tests/test_eda.py`](../../../tests/test_eda.py) | every helper run on a small synthetic frame |
| 5 | Notebook | [`notebooks/02_EDA.ipynb`](../../../notebooks/02_EDA.ipynb) | run it cell by cell: the full exploration, ending in four written findings |
| 6 | §2.3 Exercises | [`exercises.md`](exercises.md) | beginner, then intermediate, then challenge — do them before moving on |
| 7 | §2.4 Validation | [`validation.md`](validation.md) | run the commands, compare against the real recorded output |

## What this chapter adds to the repository

* [`src/utils/eda.py`](../../../src/utils/eda.py) — nine documented exploration
  helpers
* [`tests/test_eda.py`](../../../tests/test_eda.py) — every helper run on a
  small synthetic frame
* [`notebooks/02_EDA.ipynb`](../../../notebooks/02_EDA.ipynb) — the full
  exploration, ending in four written findings

The full list, including documentation updates, is in the
[syllabus](syllabus.md#deliverables-for-the-week).

## Do not move on until

* [ ] You can describe the class balance, the strongest correlations and the
  outliers, and say what each means for modelling.
* [ ] You can explain data leakage and name where it could enter this project.
* [ ] The exercises in [`exercises.md`](exercises.md) are done, not skimmed.
* [ ] Every command in [`validation.md`](validation.md) produced the output
  recorded there.

Then go to [Chapter 3 — Data Preparation](../week03/README.md).

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| ◀ [§1.4 Validation](../week01/validation.md) | 🗺 [Roadmap](../README.md) | [§2.1 Syllabus](syllabus.md) ▶ |

<!-- nav:end -->
