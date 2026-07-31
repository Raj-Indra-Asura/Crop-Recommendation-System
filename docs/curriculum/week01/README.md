# Chapter 1 — Framing the Problem and Meeting the Data

> 🗺 [Roadmap](../README.md) › [Part I — Foundations (Weeks 1-3)](../README.md#part-i--foundations-weeks-1-3) › **Chapter 1 — Framing the Problem and Meeting the Data**

**From a vague wish to a machine learning problem statement**

Part I — Foundations (Weeks 1-3) · Chapter 1 of 12 · the curriculum calls this
**Week 1**, and the directory is `docs/curriculum/week01/`.

---

## Before you open this chapter

* **You should already have finished:** Basic Python and a terminal. No machine
  learning, statistics or pandas knowledge is assumed — this is where everything
  ML-specific starts.
* **New here?** Read the [roadmap](../README.md) first — it explains how the
  whole book is meant to be read.
* **Environment:** nothing yet — this chapter builds it, from `venv` up.

## Read this chapter in this order

Top to bottom. Every row assumes the rows above it are done.

| Step | What | Where | Why here |
| --- | --- | --- | --- |
| 1 | §1.1 Syllabus | [`syllabus.md`](syllabus.md) | objectives, prerequisites, concepts, deliverables — read first, it is the map of the chapter |
| 2 | §1.2 Learning notes | [`learning_notes.md`](learning_notes.md) | the teaching text: what, why, where, how, and the common mistakes — read top to bottom |
| 3 | Code | [`src/data/data_loader.py`](../../../src/data/data_loader.py) | `load_data()`, the single entry point to the data |
| 4 | Code | [`src/data/validate_schema.py`](../../../src/data/validate_schema.py) | `validate_dataset()`, the dataset contract |
| 5 | Code | [`tests/test_data_loader.py`](../../../tests/test_data_loader.py) | the contract enforced automatically |
| 6 | Code | [`requirements.txt`](../../../requirements.txt) | nine pinned packages |
| 7 | Notebook | [`notebooks/01_problem_definition.ipynb`](../../../notebooks/01_problem_definition.ipynb) | run it cell by cell: the written problem framing plus a first look at the dataframe |
| 8 | §1.3 Exercises | [`exercises.md`](exercises.md) | beginner, then intermediate, then challenge — do them before moving on |
| 9 | §1.4 Validation | [`validation.md`](validation.md) | run the commands, compare against the real recorded output |

## What this chapter adds to the repository

* [`src/data/data_loader.py`](../../../src/data/data_loader.py) — `load_data()`,
  the single entry point to the data
* [`src/data/validate_schema.py`](../../../src/data/validate_schema.py) —
  `validate_dataset()`, the dataset contract
* [`tests/test_data_loader.py`](../../../tests/test_data_loader.py) — the
  contract enforced automatically
* [`requirements.txt`](../../../requirements.txt) — nine pinned packages
* [`notebooks/01_problem_definition.ipynb`](../../../notebooks/01_problem_definition.ipynb)
  — the written problem framing plus a first look at the dataframe

The full list, including documentation updates, is in the
[syllabus](syllabus.md#deliverables-for-the-week).

## Do not move on until

* [ ] You can state the problem as supervised multiclass classification, naming
  the seven inputs, the one output and the success measure.
* [ ] `pytest tests/test_data_loader.py` passes in your own virtual environment.
* [ ] The exercises in [`exercises.md`](exercises.md) are done, not skimmed.
* [ ] Every command in [`validation.md`](validation.md) produced the output
  recorded there.

Then go to [Chapter 2 — Exploratory Data Analysis](../week02/README.md).

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| ◀ [Roadmap — how to read this book](../README.md) | 🗺 [Roadmap](../README.md) | [§1.1 Syllabus](syllabus.md) ▶ |

<!-- nav:end -->
