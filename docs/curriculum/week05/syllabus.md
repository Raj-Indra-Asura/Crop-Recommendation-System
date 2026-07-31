# Week 5 — Classification Models

> 🗺 [Roadmap](../README.md) › [Part II — Modelling (Weeks 4-7)](../README.md#part-ii--modelling-weeks-4-7) › [Chapter 5 — Classification Models](README.md) › **§5.1 Syllabus**

## Title

**The first real algorithms: three ways to draw a boundary, compared fairly**

## Learning objectives

By the end of this week a student should be able to:

1. Name and write the scikit-learn training loop — `model.fit(X_train,
   y_train)` then `model.predict(X_test)` — and explain why every model from
   here to Week 12 reuses it unchanged.
2. Explain what logistic regression computes: a weighted sum per class, turned
   into probabilities by softmax, and therefore a **linear** (flat) decision
   boundary.
3. Distinguish the multinomial/softmax formulation from one-vs-rest, and say
   which scikit-learn uses here.
4. Explain what `C` controls in logistic regression, and why it is left at its
   default this week.
5. Describe how KNN predicts — distance to stored training rows, a vote among
   the `k` nearest — and why `fit` is nearly free while `predict` is not.
6. Predict the effect of `k`: `k = 1` overfits and memorises, large `k`
   underfits towards the baseline, and read the notebook's sweep as that curve.
7. Explain why KNN is the model most sensitive to feature scaling, and state the
   curse of dimensionality conceptually: distances concentrate as columns
   multiply, so "nearest" stops meaning "similar".
8. State the naive Bayes independence assumption, show that it is false on this
   dataset (`P`/`K` correlate at 0.74), and explain why the model still ranks
   classes correctly while its probabilities are overconfident.
9. Compare several models **fairly** — same rows, same folds, same seed, same
   metric — and explain why a difference smaller than the fold-to-fold spread is
   not yet a difference.
10. State this week's result from the results table: all three beat the 4.55%
    baseline, and Gaussian naive Bayes leads at **99.49%**.
11. Argue when they would prefer KNN over logistic regression, and when the
    reverse.

## Prerequisites

Weeks 1-4, in full. This week assumes and does **not** re-explain:

* the dataset contract and the 22-crop label set
  ([Week 1 notes §8](../week01/learning_notes.md));
* class balance and class separation — the reason every score here is so high
  ([Week 2 notes §2, §5](../week02/learning_notes.md));
* data leakage and train-only fitting (Week 2 §6, Week 3 §5);
* the stratified 80/20 split, `random_state=42`, and `data/processed/train.csv`
  ([Week 3 notes](../week03/learning_notes.md));
* `ColumnTransformer`, `Pipeline` and the `fit`/`transform` API (Week 3 §5-§6);
* accuracy, the 4.55% baseline, `StratifiedKFold` and `cross_val_score`
  ([Week 4 notes](../week04/learning_notes.md)).

No new dependencies: `scikit-learn==1.6.1` was pinned in Week 1.

## Concepts covered

| Concept | First introduced |
| --- | --- |
| The `fit` / `predict` training loop, named | Week 5 |
| Logistic regression (multiclass) | Week 5 |
| Softmax | Week 5 |
| One-vs-rest (OvR) | Week 5 |
| Linear decision boundary | Week 5 |
| Regularisation strength `C` | Week 5 (named), a later week (tuned) |
| k-nearest neighbours | Week 5 |
| Lazy (instance-based) learning | Week 5 |
| Effect of `k` on the decision boundary | Week 5 |
| Curse of dimensionality | Week 5 (conceptual) |
| Naive Bayes / Gaussian naive Bayes | Week 5 |
| Conditional independence assumption | Week 5 |
| Generative vs. discriminative models | Week 5 (brief) |
| Fair model comparison (shared folds) | Week 5 |
| Results table as a running record | Week 5 |
| Underfitting and overfitting, seen in `k` | Week 5 (observed), a later week (treated) |

Each also appears in [`docs/ml_concepts.md`](../../ml_concepts.md) and
[`docs/glossary.md`](../../glossary.md).

## Connection to the previous weeks

Week 4 refused to train a real model until there was a number to judge one
against. That number now exists — 4.55% — so this week finally spends it.

Three earlier results are cashed in here:

* **The Week 3 preprocessor.** Logistic regression and KNN both need features on
  a common scale, so every model this week is wrapped as
  `Pipeline([("preprocess", ...), ("model", ...)])` and the scaler is re-fitted
  inside each cross-validation fold.
* **The Week 4 protocol.** `cross_validated_accuracy` supplies the identical
  five stratified folds to all four candidates, which is what makes the
  comparison a comparison rather than four unrelated experiments.
* **Week 2's correlation matrix.** The 0.74 between `P` and `K` is the concrete
  evidence that naive Bayes' independence assumption is false here — and the
  model wins anyway, which is the week's most useful surprise.

## Connection to the ML lifecycle

```
Frame the problem -> Get the data -> Explore -> Prepare -> [Model]
    -> Evaluate -> Improve -> Productionize -> Deploy -> Monitor
```

Week 5 is the **model** stage, entered only now that evaluation is in place. The
order is the point: the metric, the protocol and the floor were all fixed before
any candidate existed, so no result this week can be used to reshape the
standard it is judged by.

## Expected student outcome

### The student CAN, after this week

* **Train and compare three classical models.** Logistic regression, KNN and
  Gaussian naive Bayes, each through
  `Pipeline([("preprocess", ...), ("model", ...)])`, cross-validated on the same
  five folds.
* **State which currently performs best and by how much it beats the baseline.**
  Gaussian naive Bayes, **99.49%** (±0.42) against the 4.55% floor; logistic
  regression 96.82% (±0.66); KNN with `k = 5` 96.53% (±1.21). All three clear
  the baseline by more than 90 percentage points.
* **Explain when they would prefer KNN over logistic regression** — curved or
  fragmented class regions, few features, plentiful rows, training that must be
  instantaneous — and when the reverse: explainability, many features, a small
  fast model, calibrated probabilities.
* Write the `fit`/`predict` loop from memory and say what each call does.
* Explain a linear decision boundary, the softmax, the effect of `k`, the
  independence assumption, and why standardisation matters to two of the three
  models and not the third.
* Read the results table correctly: quote the mean *with* its spread, and refuse
  to call the 0.3-point gap between logistic regression and KNN a win.
* Run `pytest tests/test_classical_models.py` and execute
  `notebooks/05_classification_models.ipynb` end to end.

### The student CANNOT yet

* **Tune a hyperparameter systematically.** `C = 1.0`, `k = 5` and
  `var_smoothing = 1e-9` are scikit-learn's defaults, not choices supported by
  evidence. The `k` sweep in the notebook is a demonstration of a curve, not a
  search, and nothing from it is adopted. Grid search, random search and
  validation curves are a later week.
* **Use ensembles.** No random forest, no gradient boosting, no voting or
  stacking classifier — a later week.
* Decide reliably between two models that are a fraction of a point apart;
  repeated cross-validation and significance testing are a later week.
* Say which features drive a prediction — **Week 7**.
* Interpret precision, recall, F1 or a confusion matrix — **Week 8**.
* Quote a test-set score. `data/processed/test.csv` stays unopened until Week 8.

## Deliverables for the week

* `src/models/classical_models.py` — `get_logistic_regression()`, `get_knn()`
  and `get_naive_bayes()`, each returning an unfitted estimator, plus the
  `CLASSICAL_MODEL_FACTORIES` registry the notebook iterates over.
* `tests/test_classical_models.py` — 57 tests covering the factories, the shared
  training loop, algorithm-specific behaviour (memorisation at `k = 1`, scaling
  sensitivity, dimensionality dilution) and the comparison against the baseline
  on the real data.
* `notebooks/05_classification_models.ipynb` — committed with executed output,
  ending in the four-row results table that Weeks 6-8 extend.
* This week's four curriculum documents.
* Updated `docs/ml_concepts.md`, `docs/glossary.md` and the README progress
  table.

No change to `requirements.txt`.

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| ◀ [Chapter 5 — Classification Models](README.md) | [Chapter 5 — Classification Models](README.md) · 🗺 [Roadmap](../README.md) | [§5.2 Learning notes](learning_notes.md) ▶ |

<!-- nav:end -->
