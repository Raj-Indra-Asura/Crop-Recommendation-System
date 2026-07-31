# Chapter 7 — Ensembles

> 🗺 [Roadmap](../README.md) › [Part II — Modelling (Weeks 4-7)](../README.md#part-ii--modelling-weeks-4-7) › **Chapter 7 — Ensembles**

**Many weak models beat one strong one — bagging, boosting, and feature
importance**

Part II — Modelling (Weeks 4-7) · Chapter 7 of 12 · the curriculum calls this
**Week 7**, and the directory is `docs/curriculum/week07/`.

---

## Before you open this chapter

* **You should already have finished:** Chapter 6: you have five single models
  compared, and you have seen a tree overfit.
* **Previous chapter:** [Chapter 6 — Margin-based and Tree-based
  Models](../week06/README.md)
* **Environment:** the virtual environment from Chapter 1, with
  `requirements.txt` installed.

## Read this chapter in this order

Top to bottom. Every row assumes the rows above it are done.

| Step | What | Where | Why here |
| --- | --- | --- | --- |
| 1 | §7.1 Syllabus | [`syllabus.md`](syllabus.md) | objectives, prerequisites, concepts, deliverables — read first, it is the map of the chapter |
| 2 | §7.2 Learning notes | [`learning_notes.md`](learning_notes.md) | the teaching text: what, why, where, how, and the common mistakes — read top to bottom |
| 3 | Code | [`src/models/ensemble_models.py`](../../../src/models/ensemble_models.py) | `get_random_forest()`, `get_gradient_boosting()`, the XGBoost fallback and the label adapter |
| 4 | Code | [`tests/test_ensemble_models.py`](../../../tests/test_ensemble_models.py) | variance reduction, decorrelation, error correction, and importance limits — with and without XGBoost installed |
| 5 | Notebook | [`notebooks/06_model_selection.ipynb`](../../../notebooks/06_model_selection.ipynb) | run it cell by cell: Part 1 (§0-§7) — an eight-row results table with feature importances |
| 6 | §7.3 Exercises | [`exercises.md`](exercises.md) | beginner, then intermediate, then challenge — do them before moving on |
| 7 | §7.4 Validation | [`validation.md`](validation.md) | run the commands, compare against the real recorded output |

## What this chapter adds to the repository

* [`src/models/ensemble_models.py`](../../../src/models/ensemble_models.py) —
  `get_random_forest()`, `get_gradient_boosting()`, the XGBoost fallback and the
  label adapter
* [`tests/test_ensemble_models.py`](../../../tests/test_ensemble_models.py) —
  variance reduction, decorrelation, error correction, and importance limits —
  with and without XGBoost installed
* [`notebooks/06_model_selection.ipynb`](../../../notebooks/06_model_selection.ipynb)
  — Part 1 (§0-§7) — an eight-row results table with feature importances

The full list, including documentation updates, is in the
[syllabus](syllabus.md#deliverables-for-the-week).

## Do not move on until

* [ ] You can explain bagging and boosting mechanically, not just by name.
* [ ] You can state two ways `feature_importances_` misleads you.
* [ ] The exercises in [`exercises.md`](exercises.md) are done, not skimmed.
* [ ] Every command in [`validation.md`](validation.md) produced the output
  recorded there.

Then go to [Chapter 8 — Model Evaluation & Explainability](../week08/README.md).

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| ◀ [§6.4 Validation](../week06/validation.md) | 🗺 [Roadmap](../README.md) | [§7.1 Syllabus](syllabus.md) ▶ |

<!-- nav:end -->
