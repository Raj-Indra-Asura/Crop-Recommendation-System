# Chapter 6 — Margin-based and Tree-based Models

> 🗺 [Roadmap](../README.md) › [Part II — Modelling (Weeks 4-7)](../README.md#part-ii--modelling-weeks-4-7) › **Chapter 6 — Margin-based and Tree-based Models**

**Two more ways to draw a boundary, and the first honest look at overfitting**

Part II — Modelling (Weeks 4-7) · Chapter 6 of 12 · the curriculum calls this
**Week 6**, and the directory is `docs/curriculum/week06/`.

---

## Before you open this chapter

* **You should already have finished:** Chapter 5: three models are already in
  the comparison table, evaluated on identical folds.
* **Previous chapter:** [Chapter 5 — Classification Models](../week05/README.md)
* **Environment:** the virtual environment from Chapter 1, with
  `requirements.txt` installed.

## Read this chapter in this order

Top to bottom. Every row assumes the rows above it are done.

| Step | What | Where | Why here |
| --- | --- | --- | --- |
| 1 | §6.1 Syllabus | [`syllabus.md`](syllabus.md) | objectives, prerequisites, concepts, deliverables — read first, it is the map of the chapter |
| 2 | §6.2 Learning notes | [`learning_notes.md`](learning_notes.md) | the teaching text: what, why, where, how, and the common mistakes — read top to bottom |
| 3 | Code | [`src/models/classical_models.py`](../../../src/models/classical_models.py) | `get_svm()` and `get_decision_tree()` added beside the Chapter 5 factories |
| 4 | Code | [`src/utils/visualization.py`](../../../src/utils/visualization.py) | `plot_decision_boundary(model, X_2d, y)` |
| 5 | Code | [`tests/test_classical_models.py`](../../../tests/test_classical_models.py) | extended: support vectors, kernels, tree depth and the widening train/validation gap |
| 6 | Notebook | [`notebooks/05_classification_models.ipynb`](../../../notebooks/05_classification_models.ipynb) | run it cell by cell: Part 2 (§8-§15) — a six-row results table |
| 7 | §6.3 Exercises | [`exercises.md`](exercises.md) | beginner, then intermediate, then challenge — do them before moving on |
| 8 | §6.4 Validation | [`validation.md`](validation.md) | run the commands, compare against the real recorded output |

## What this chapter adds to the repository

* [`src/models/classical_models.py`](../../../src/models/classical_models.py) —
  `get_svm()` and `get_decision_tree()` added beside the Chapter 5 factories
* [`src/utils/visualization.py`](../../../src/utils/visualization.py) —
  `plot_decision_boundary(model, X_2d, y)`
* [`tests/test_classical_models.py`](../../../tests/test_classical_models.py) —
  extended: support vectors, kernels, tree depth and the widening
  train/validation gap
* [`notebooks/05_classification_models.ipynb`](../../../notebooks/05_classification_models.ipynb)
  — Part 2 (§8-§15) — a six-row results table

The full list, including documentation updates, is in the
[syllabus](syllabus.md#deliverables-for-the-week).

## Do not move on until

* [ ] You can show overfitting with a tree-depth sweep and read the
  train/validation gap.
* [ ] You can explain what a support vector is and what the RBF kernel buys you.
* [ ] The exercises in [`exercises.md`](exercises.md) are done, not skimmed.
* [ ] Every command in [`validation.md`](validation.md) produced the output
  recorded there.

Then go to [Chapter 7 — Ensembles](../week07/README.md).

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| ◀ [§5.4 Validation](../week05/validation.md) | 🗺 [Roadmap](../README.md) | [§6.1 Syllabus](syllabus.md) ▶ |

<!-- nav:end -->
