# Chapter 8 — Model Evaluation & Explainability

> 🗺 [Roadmap](../README.md) › [Part III — Evaluation and Explanation (Week 8)](../README.md#part-iii--evaluation-and-explanation-week-8) › **Chapter 8 — Model Evaluation & Explainability**

**Past accuracy: confusion matrices, honest hyperparameter search, and
explaining a prediction**

Part III — Evaluation and Explanation (Week 8) · Chapter 8 of 12 · the
curriculum calls this **Week 8**, and the directory is
`docs/curriculum/week08/`.

---

## Before you open this chapter

* **You should already have finished:** Chapter 7: eight models are compared on
  cross-validation, and the test set is still sealed.
* **Previous chapter:** [Chapter 7 — Ensembles](../week07/README.md)
* **Environment:** the virtual environment from Chapter 1, with
  `requirements.txt` installed.

## Read this chapter in this order

Top to bottom. Every row assumes the rows above it are done.

| Step | What | Where | Why here |
| --- | --- | --- | --- |
| 1 | §8.1 Syllabus | [`syllabus.md`](syllabus.md) | objectives, prerequisites, concepts, deliverables — read first, it is the map of the chapter |
| 2 | §8.2 Learning notes | [`learning_notes.md`](learning_notes.md) | the teaching text: what, why, where, how, and the common mistakes — read top to bottom |
| 3 | Code | [`src/evaluation/tuning.py`](../../../src/evaluation/tuning.py) | `tune_model()` over grid and randomised search |
| 4 | Code | [`src/evaluation/explainability.py`](../../../src/evaluation/explainability.py) | `permutation_feature_importance()` and `explain_prediction()` |
| 5 | Code | [`src/evaluation/metrics.py`](../../../src/evaluation/metrics.py) | `confusion_frame()`, macro/weighted F1 and the confusion matrix |
| 6 | Code | [`tests/test_tuning.py`](../../../tests/test_tuning.py) | the searches respect the project's folds and seed |
| 7 | Code | [`tests/test_explainability.py`](../../../tests/test_explainability.py) | permutation importance and the documented SHAP fallback |
| 8 | Notebook | [`notebooks/06_model_selection.ipynb`](../../../notebooks/06_model_selection.ipynb) | run it cell by cell: Part 2 (§8-§15) — the test set opened once and the final-model decision |
| 9 | Notebook | [`notebooks/07_model_explainability.ipynb`](../../../notebooks/07_model_explainability.ipynb) | run it cell by cell: the correlation trap, SHAP plots, and one prediction explained in plain language |
| 10 | §8.3 Exercises | [`exercises.md`](exercises.md) | beginner, then intermediate, then challenge — do them before moving on |
| 11 | §8.4 Validation | [`validation.md`](validation.md) | run the commands, compare against the real recorded output |

## What this chapter adds to the repository

* [`src/evaluation/tuning.py`](../../../src/evaluation/tuning.py) —
  `tune_model()` over grid and randomised search
* [`src/evaluation/explainability.py`](../../../src/evaluation/explainability.py)
  — `permutation_feature_importance()` and `explain_prediction()`
* [`src/evaluation/metrics.py`](../../../src/evaluation/metrics.py) —
  `confusion_frame()`, macro/weighted F1 and the confusion matrix
* [`tests/test_tuning.py`](../../../tests/test_tuning.py) — the searches respect
  the project's folds and seed
* [`tests/test_explainability.py`](../../../tests/test_explainability.py) —
  permutation importance and the documented SHAP fallback
* [`notebooks/06_model_selection.ipynb`](../../../notebooks/06_model_selection.ipynb)
  — Part 2 (§8-§15) — the test set opened once and the final-model decision
* [`notebooks/07_model_explainability.ipynb`](../../../notebooks/07_model_explainability.ipynb)
  — the correlation trap, SHAP plots, and one prediction explained in plain
  language

The full list, including documentation updates, is in the
[syllabus](syllabus.md#deliverables-for-the-week).

## Do not move on until

* [ ] You can read a confusion matrix and describe the two rows the model gets
  wrong.
* [ ] You can explain why the test set is opened once, at the end, and never
  tuned against.
* [ ] The exercises in [`exercises.md`](exercises.md) are done, not skimmed.
* [ ] Every command in [`validation.md`](validation.md) produced the output
  recorded there.

Then go to [Chapter 9 — Productionizing the Model](../week09/README.md).

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| ◀ [§7.4 Validation](../week07/validation.md) | 🗺 [Roadmap](../README.md) | [§8.1 Syllabus](syllabus.md) ▶ |

<!-- nav:end -->
