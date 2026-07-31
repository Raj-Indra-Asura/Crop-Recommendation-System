# Week 8 — Model Evaluation & Explainability

## Title

**Past accuracy: confusion matrices, honest hyperparameter search, and
explaining one prediction**

## Learning objectives

By the end of this week a student should be able to:

1. **Read a multiclass confusion matrix**: say what a cell `(i, j)` counts, why
   the diagonal is the correct answers, and read the two off-diagonal cells this
   project's final models produce (`rice -> jute`, `blackgram -> maize`).
2. Define **precision**, **recall** and **F1** per class in one sentence each,
   and say which of the two a false "jute" recommendation damages.
3. **Explain macro vs weighted averaging** — equal weight per class vs weight by
   support — say when the two diverge, and explain why they are almost identical
   here (**0.9954** vs **0.9954** for the final model on a test set with exactly
   20 rows per crop).
4. Describe what a **hyperparameter search** is, and why **cross-validation must
   live inside it**: candidates are ranked on held-out folds, so the test set is
   never consulted while choosing.
5. Say when to use **`GridSearchCV`** and when **`RandomizedSearchCV`**: cost is
   the product of the grid vs a number you choose (here **24 candidates / 120
   fits** against **20 draws from a 300-candidate space**).
6. Explain why a search's winning score is **optimistic**, and why a +0.17-point
   "improvement" sitting inside a ±0.60-point fold spread is not an improvement.
7. **Choose a final model on more than accuracy** — interpretability, training
   and serving cost, tuning risk and error pattern — and defend the choice made
   here: **Gaussian naive Bayes**, tied at 99.55% with the tuned forest.
8. Explain **permutation importance** and why it is more trustworthy than
   `feature_importances_`: computed on held-out data, in units of score lost,
   and available for models with no built-in importances at all.
9. State permutation importance's **correlation trap** and quote the measurement:
   shuffling `P` costs 17.9 points, `K` 43.3, the pair together **56.5**.
10. Explain what **SHAP** adds — per-row, signed, additive attributions — and
    read both a summary bar plot and a beeswarm.
11. **Explain one prediction end to end in plain language**, naming the deciding
    measurement, the runner-up class and its probability.
12. Say which explainer produced a given number, and what the **fallback** is
    when `shap` will not install.

## Prerequisites

Weeks 1-7, in full. This week assumes and does **not** re-explain:

* the dataset contract and the 22-crop label set
  ([Week 1 notes](../week01/learning_notes.md));
* class separation and the **0.74 correlation between `P` and `K`**
  ([Week 2 notes](../week02/learning_notes.md)) — §2 of the explainability
  notebook is unreadable without it;
* the stratified 80/20 split, `random_state=42`, and why
  `data/processed/test.csv` was sealed
  ([Week 3 notes](../week03/learning_notes.md));
* accuracy, the 4.55% baseline, `StratifiedKFold` and `cross_val_score`
  ([Week 4 notes](../week04/learning_notes.md));
* Gaussian naive Bayes and its `var_smoothing`
  ([Week 5 notes](../week05/learning_notes.md));
* overfitting, bias and variance ([Week 6 notes](../week06/learning_notes.md));
* random forests, `feature_importances_` and its three limitations
  ([Week 7 notes](../week07/learning_notes.md)).

One new, **optional** dependency: `shap==0.46.0`. `explain_prediction()` uses it
when it is importable and falls back to per-sample permutation plus the raw
`predict_proba` breakdown when it is not, so no environment is blocked on the
install — the same arrangement Week 7 made for `xgboost`. Everything else was
pinned in Week 1.

## Concepts covered

| Concept | First introduced |
| --- | --- |
| Confusion matrix (multiclass) | Week 8 |
| Precision, recall, F1, support | Week 4 (printed), Week 8 (taught) |
| Macro vs weighted averaging | Week 8 |
| Error analysis, confusion pairs | Week 8 |
| Hyperparameter vs parameter | Week 5 (named), Week 8 (searched) |
| `GridSearchCV`, `RandomizedSearchCV` | Week 8 |
| Inner cross-validation in a search | Week 8 |
| Optimism of a selected maximum | Week 8 |
| Held-out test evaluation, opened once | Week 3 (sealed), Week 8 (opened) |
| Final model selection on multiple criteria | Week 8 |
| Permutation importance | Week 7 (named), Week 8 (used) |
| Correlated-feature trap in importance | Week 2 (correlation), Week 8 (measured) |
| SHAP, Shapley values, additivity | Week 7 (named), Week 8 (used) |
| `TreeExplainer` vs `KernelExplainer` | Week 8 |
| Base value, local explanation | Week 8 |
| Optional-dependency fallback, stated in advance | Week 7 (xgboost), Week 8 (shap) |

Each also appears in [`docs/ml_concepts.md`](../../ml_concepts.md) and
[`docs/glossary.md`](../../glossary.md).

## Connection to the previous week

Week 7 ended with two open admissions. It had adopted **no** tuned
hyperparameter, because reading a winner off a sweep is not a protocol; and it
had plotted `feature_importances_` while listing three reasons not to trust it,
naming permutation importance and SHAP as the Week 8 replacements.

Week 8 pays both debts, and then does the thing every earlier week refused to
do: it opens `data/processed/test.csv`. That file has been untouched since Week 3
precisely so that this measurement would mean something, and it is read **after**
the model decisions rather than as an input to them.

The week's centre of gravity, though, is not the search and not the test score.
Both finalists land at 99.55%, as Week 4 warned they would — this dataset is
separable enough that accuracy stopped discriminating several weeks ago. The
content is therefore in the **two rows that are wrong**: which crops get confused
with which, why those particular pairs, and how the model's own probabilities
signalled the doubt.

## Connection to the ML lifecycle

```
Frame the problem -> Get the data -> Explore -> Prepare -> Model
    -> [Evaluate] -> [Improve] -> Productionize -> Deploy -> Monitor
```

Week 8 completes the **evaluate** and **improve** stages, and it is the last week
of the analysis half of the course. After it the artefact stops being a notebook
result and starts being software: Week 9 packages the chosen model, Week 10
serves it, Week 11 puts a UI on it, Week 12 containerises and deploys it.

The explainability work is also the first piece of the **monitor** stage.
"Route any prediction whose runner-up exceeds 15% to a human" is an operational
rule, and it exists only because the model was made to say *why*.

## Expected student outcome

### The student CAN, after this week

* **Read a confusion matrix.** Cell `(i, j)` is "true class `i`, predicted class
  `j`"; the diagonal is correct; off-diagonal cells name specific confusions. On
  the 440 held-out rows the tuned forest produces exactly two: one
  `rice -> jute` and one `blackgram -> maize`. Gaussian naive Bayes produces two
  `rice -> jute`.
* **Explain macro vs weighted F1.** Macro averages the 22 per-class F1 scores
  with equal weight; weighted averages them by support. They diverge when the
  classes are imbalanced, and reporting only the weighted number would then hide
  failure on the small classes. Here the test set has 20 rows per crop by
  construction, so the two agree to four decimal places (0.9954 / 0.9954) — and
  that agreement is a statement about the split, not a compliment to the model.
* **Explain why a specific prediction was made.** For test row 378: the model
  said **jute at 0.84**, with **rice second at 0.16**; the decisive feature was
  **`rainfall = 186.75 mm`**, against training averages of 237 mm for rice and
  176 mm for jute, while all six other measurements are nearly identical between
  the two crops. The true crop was rice — an ambiguous reading answered with the
  more likely crop, not a malfunction.
* **Name and justify the final chosen model.** **Gaussian naive Bayes**: tied at
  99.55% test accuracy and 0.9954 macro F1 with the tuned random forest, ~40x
  cheaper to fit, 308 stored numbers against a hundred trees, directly
  interpretable, and with **no hyperparameter that changes its behaviour on this
  data** — twelve values of `var_smoothing` across five orders of magnitude give
  a single identical CV score. The tuned forest is the recorded runner-up for the
  day the data stops being Gaussian and separable.
* **Run a hyperparameter search honestly** with
  `tune_model(model, param_grid, X, y)`, quote the protocol (5 stratified folds,
  seed 42, training rows only) and say why the winning score is optimistic.
* **Rank features on held-out data** with `permutation_feature_importance()`, say
  why it beats `feature_importances_`, and avoid the correlation trap.
* Run `pytest tests/test_tuning.py tests/test_explainability.py` and execute both
  notebooks end to end, with or without `shap` installed.

### The student CANNOT yet

* **Package this into a reusable production pipeline.** Nothing is serialised:
  `models/` is still empty, there is no `predict()` entry point outside a
  notebook, no versioned artifact, and no way for another program to import the
  chosen model. Week 9 onwards.
* Serve a prediction over HTTP, or build a UI for one (Weeks 10-11).
* Monitor a deployed model, or detect drift (Week 12 and beyond).
* Use a custom scorer, a cost-sensitive metric or a probability calibration
  curve; every search this week optimises plain accuracy.
* Run **nested** cross-validation, where the whole search is itself
  cross-validated to estimate the *tuning procedure* without bias. This week uses
  one inner loop and a single held-out set — enough to be honest, not enough to
  publish.
* Claim anything **causal**. Every number here describes the fitted model. "The
  model relies on rainfall" is not "rainfall causes rice".

## Deliverables for the week

* `src/evaluation/tuning.py` — `tune_model()` over `GridSearchCV` /
  `RandomizedSearchCV`, with the project's folds and seed built in.
* `src/evaluation/explainability.py` — `permutation_feature_importance()` and
  `explain_prediction()`, plus the `SHAP_AVAILABLE` / `EXPLAINER_BACKEND` flags
  and the documented fallback.
* `src/evaluation/metrics.py` — `confusion_frame()`, and macro/weighted F1 and
  the confusion matrix added to `evaluate_model()`'s result.
* `tests/test_tuning.py` (22 tests) and `tests/test_explainability.py` (32, one
  of which skips depending on whether `shap` is installed) — 346 in the whole
  suite.
* `notebooks/06_model_selection.ipynb` — Part 2 (§8-§15), committed with
  executed output: the searches, the test set opened once, confusion matrices,
  the error analysis and the written final-model decision.
* `notebooks/07_model_explainability.ipynb` — permutation importance, the
  correlation trap, SHAP summary and beeswarm plots, and one prediction explained
  end to end in plain language.
* This week's four curriculum documents.
* Updated `requirements.txt` (optional `shap==0.46.0`),
  `src/evaluation/__init__.py`, `docs/ml_concepts.md`, `docs/glossary.md` and the
  README progress table.
