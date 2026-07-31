# Week 7 — Ensembles

## Title

**Many weak models beat one strong one — bagging, boosting, and the first look
at feature importance**

## Learning objectives

By the end of this week a student should be able to:

1. **Explain bagging and boosting in their own words**, and say what each one
   mainly reduces: bagging fits models **in parallel** on random resamples and
   averages them, cutting **variance**; boosting fits models **in sequence**,
   each on the errors of the ones before it, cutting **bias**.
2. State the condition that makes any ensemble work — **averaging cancels only
   the errors the members do not share** — and explain why 100 identically
   fitted trees are worth exactly one tree.
3. Describe the two sources of randomness in a random forest: **bootstrap
   sampling** of the rows (~63% distinct per tree, the rest out-of-bag) and
   **feature randomness** at every node (`max_features="sqrt"`, so 2 of this
   project's 7).
4. Explain **why the forest's trees are deliberately left unpruned**, which is
   the opposite of the advice Week 6 gave for a single tree.
5. Explain why `n_estimators` in a forest is a **budget rather than a
   bias-variance dial**, while `n_estimators` in a booster is a real one.
6. Describe gradient boosting as **sequential error-correction** — fit a weak
   tree, look at what is still wrong, fit the next tree on that, add a shrunken
   version of it — without the mathematics, and say what `learning_rate` trades
   against.
7. **Read a feature-importance plot**: rank the seven features, quote that
   `rainfall` (0.23) and `humidity` (0.22) lead and `ph` (0.05) trails, and say
   what "mean decrease in impurity" counts.
8. State the **three limitations** of `feature_importances_` — measured on the
   training data, biased towards high-cardinality features, and credit split
   arbitrarily between correlated features — and name what Week 8 replaces it
   with.
9. **State whether the ensembles beat the Week 5 and Week 6 models on this
   dataset**: they beat all of them except Gaussian naive Bayes, which they draw
   with, so the leader is unchanged at **99.49%**.
10. Explain why "ensembles usually win" and "naive Bayes wins here" are both
    true, in terms of this dataset's separability and the headroom left above
    99.5%.
11. Say what the XGBoost-or-scikit-learn fallback is for, and check which
    backend an environment is using.

## Prerequisites

Weeks 1-6, in full. This week assumes and does **not** re-explain:

* the dataset contract and the 22-crop label set
  ([Week 1 notes](../week01/learning_notes.md));
* class separation, and the 0.74 correlation between `P` and `K`
  ([Week 2 notes](../week02/learning_notes.md));
* the stratified 80/20 split, `random_state=42` and `data/processed/train.csv`
  ([Week 3 notes](../week03/learning_notes.md));
* `Pipeline`, `ColumnTransformer` and re-fitting the scaler inside every fold
  (Week 3 §5-§6);
* accuracy, the 4.55% baseline and `StratifiedKFold`
  ([Week 4 notes](../week04/learning_notes.md));
* the `fit`/`predict` loop and fair comparison on shared folds
  ([Week 5 notes](../week05/learning_notes.md));
* **decision trees, splits, purity, `max_depth`, bias, variance and the
  overfitting curve** ([Week 6 notes](../week06/learning_notes.md)) — this week
  is unreadable without them, because a random forest is a crowd of exactly the
  trees Week 6 fitted, and boosting is a chain of deliberately stunted ones.

One new, **optional** dependency: `xgboost==2.1.3`.
`get_gradient_boosting()` uses it when it is importable and falls back to
scikit-learn's `GradientBoostingClassifier` when it is not, so no environment is
blocked on the install. Everything else was pinned in Week 1.

## Concepts covered

| Concept | First introduced |
| --- | --- |
| Ensemble learning | Week 7 |
| Bagging (bootstrap aggregating) | Week 7 |
| Bootstrap sample, out-of-bag rows | Week 7 |
| Boosting | Week 7 |
| Weak learner, stump | Week 7 |
| Random forest | Week 7 |
| Feature randomness (`max_features`), decorrelation | Week 7 |
| Voting and probability averaging | Week 7 |
| Gradient boosting, `learning_rate` as shrinkage | Week 7 |
| `n_estimators` as a budget vs. as a capacity dial | Week 7 |
| XGBoost, and optional-dependency fallbacks | Week 7 |
| Feature importance (mean decrease in impurity) | Week 7 |
| Limitations of impurity-based importance | Week 7 |
| Permutation importance, SHAP | Week 7 (named), Week 8 (used) |
| Variance reduction by averaging | Week 6 (defined), Week 7 (exploited) |
| Bias-variance tradeoff | Week 5 (named), Week 6 (plotted), Week 7 (attacked from both ends) |

Each also appears in [`docs/ml_concepts.md`](../../ml_concepts.md) and
[`docs/glossary.md`](../../glossary.md).

## Connection to the previous week

Week 6 ended on an unresolved observation. Its unlimited decision tree scored a
perfect **100%** on the rows it was fitted on and **98.52%** on rows it had not
seen, and the notes named that gap: variance, a model flexible enough to have
memorised the accidents of one sample of 1,760 fields. The remedy offered there
was to make the tree simpler — trade variance back for bias by capping
`max_depth`.

Week 7 takes the other route. Keep the overfitting trees exactly as they are,
fit a hundred of them on different resamples with different features available,
and average the result: the variance cancels and the low bias survives. The
single tree Week 6 built is not discarded, it becomes a *component*.

Boosting then attacks the same tradeoff from the opposite end. Where the forest
starts from members that are too flexible and averages them down, boosting
starts from members that are far too rigid — depth-3 trees, or literal stumps —
and chains enough corrections together to become flexible. Both algorithms are
the bias-variance plot from Week 6 §11, read in opposite directions.

The comparison protocol is unchanged and that is again the point: the same 1,760
training rows, the same five stratified folds, the same seed, the same metric,
the same preprocessor. The two new models are appended to Week 6's six-row table
rather than measured in a new experiment.

## Connection to the ML lifecycle

```
Frame the problem -> Get the data -> Explore -> Prepare -> [Model]
    -> Evaluate -> Improve -> Productionize -> Deploy -> Monitor
```

Week 7 is the last week that stays purely in the **model** stage: it is the end
of "try another algorithm" as a strategy. The remaining ways to improve a result
are to tune what you already have (search, Part 2) and to understand what it is
actually doing (evaluation and explanation, Week 8) — which is why this week
introduces `feature_importances_` and immediately explains why it is not yet a
trustworthy answer.

## Expected student outcome

### The student CAN, after this week

* **Explain bagging versus boosting in their own words.** Bagging fits many
  strong, overfitting models in parallel — each on a bootstrap resample of the
  rows, each seeing only 2 of the 7 features at any split — and averages their
  votes, so that errors made in different places cancel; it reduces variance.
  Boosting fits many weak models in sequence, each one trained on what the
  ensemble still gets wrong, and adds up their shrunken contributions; it
  reduces bias, and it can overfit if the chain runs too long.
* **Read a feature-importance plot.** From the forest's plot: `rainfall` (0.23)
  and `humidity` (0.22) together account for nearly half the impurity removed,
  `K` (0.18) and `P` (0.15) follow, `ph` (0.05) is last — and "last" means "this
  model rarely needed it given the other six", not "soil pH does not matter".
  The seven numbers sum to 1, they are computed on training data, and
  duplicating a column halves its apparent importance without changing accuracy
  at all.
* **State whether ensembles beat the Week 5/6 models on this dataset.** Yes,
  except for one: the random forest (**99.26%** ±0.58) and gradient boosting
  (**99.09%** ±0.33 with XGBoost, 98.69% ±0.34 with the scikit-learn fallback)
  beat the decision tree (98.52%), the SVM (97.90%), logistic regression
  (96.82%) and KNN (96.53%) — but Gaussian naive Bayes still leads at **99.49%**,
  and its 0.23-point margin over the forest is inside the fold spread, so the
  honest verdict is a **draw at the top**.
* Explain why the forest halves the single tree's error rate (1.48% -> 0.74%)
  using the same kind of member, and why one tree *inside* a forest (94.83%) is
  worse than a plain decision tree.
* Say what bootstrap sampling does, and quote the ~63%/37% split between
  in-bag and out-of-bag rows.
* Say why `n_estimators` cannot overfit a forest but can overfit a booster.
* Run `pytest tests/test_ensemble_models.py` and execute
  `notebooks/06_model_selection.ipynb` end to end, with or without XGBoost
  installed.

### The student CANNOT yet

* **Tune hyperparameters systematically.** The `n_estimators` sweep and the
  boosting-rounds sweep are demonstrations of curves, not searches. Nothing from
  either is adopted: `n_estimators = 100`, `learning_rate = 0.1`,
  `max_depth = 3` and `max_features = "sqrt"` remain the defaults, because
  choosing one by eye from validation scores is exactly the mistake a stated
  search protocol exists to prevent. `GridSearchCV`, `RandomizedSearchCV`,
  search spaces and validation curves are Part 2 of the model-selection
  notebook.
* **Explain an individual prediction.** `feature_importances_` is a statement
  about the fitted model over the whole training set. "Why was *this* field
  labelled rice?" needs SHAP — **Week 8**.
* Quote a *held-out* feature importance. Permutation importance — shuffle a
  column in data the model did not train on and measure the accuracy it costs —
  is also Week 8.
* Build a voting or stacking classifier, or an ensemble of *different* model
  families.
* Interpret precision, recall, F1 or a confusion matrix — **Week 8**.
* Quote a test-set score. `data/processed/test.csv` stays unopened until Week 8.

## Deliverables for the week

* `src/models/ensemble_models.py` — `get_random_forest()` and
  `get_gradient_boosting()`, both returning unfitted estimators, plus
  `ENSEMBLE_MODEL_FACTORIES`, the `XGBOOST_AVAILABLE` /
  `GRADIENT_BOOSTING_BACKEND` flags and the `XGBoostStringLabelClassifier`
  adapter that lets XGBoost accept crop names as labels.
* `tests/test_ensemble_models.py` — 61 tests (292 in the whole suite), passing
  with **and** without XGBoost installed: factory defaults and validation, the
  shared training loop, measured variance reduction from bagging, measured
  decorrelation from feature randomness, measured error-correction from
  boosting, the meaning and the limitations of `feature_importances_`, and the
  ensembles' standing against the Week 5/6 models on the real data.
* `notebooks/06_model_selection.ipynb` — Part 1 (§0-§7), committed with executed
  output, ending in an eight-row results table and the forest's feature
  importances plotted beside the booster's.
* This week's four curriculum documents.
* Updated `requirements.txt` (optional `xgboost==2.1.3`),
  `src/models/__init__.py`, `docs/ml_concepts.md`, `docs/glossary.md` and the
  README progress table.
