# Week 6 — Margin-based and Tree-based Models

## Title

**Two more ways to draw a boundary, and the first honest look at overfitting**

## Learning objectives

By the end of this week a student should be able to:

1. Explain what a support vector machine optimises — the **widest margin**
   between the classes — and why only the **support vectors** decide where the
   boundary sits.
2. Explain what the **soft margin** buys, and which way `C` moves it: small `C`
   tolerates violations for a wider, smoother boundary; large `C` insists on the
   training rows and narrows it.
3. **Say what a kernel does in one sentence** — a kernel lets an SVM draw a
   curved boundary by measuring similarity between rows instead of positions in
   space — and name the difference between the `linear` and `rbf` kernels.
4. Say when an SVM is a sensible first choice on a small tabular dataset, and
   why the linear kernel is not beaten by the RBF one here.
5. Describe how a decision tree is fitted: a greedy search over
   `feature <= threshold` questions, keeping at each node the split that leaves
   the two sides **purest**, measured by Gini impurity or entropy.
6. State conceptually what Gini impurity and entropy measure, and that they
   almost always choose the same splits.
7. **Explain overfitting from the tree-depth plot they generated**: both curves
   low and together on the left (bias), the training curve reaching 100% while
   the validation curve stops at 98.52% on the right (variance), and the gap
   between them as the part of the training score that does not generalise.
8. Define **bias** and **variance**, and place `max_depth`, `min_samples_leaf`,
   `k` and `C` on the same axis of "how flexible is this model".
9. Explain how a decision boundary is drawn — classify a grid of points, colour
   the plane by the answers — and why a two-feature picture is an illustration
   and never a reported score.
10. Read the three boundary shapes: straight cuts from a linear model, curves
    from an RBF SVM, axis-aligned boxes from a tree, and say why the tree can
    only produce boxes.
11. **Name the best-performing model so far** — Gaussian naive Bayes at
    **99.49%** — and quote the two new entries, the decision tree at 98.52% and
    the SVM at 97.90%.

## Prerequisites

Weeks 1-5, in full. This week assumes and does **not** re-explain:

* the dataset contract and the 22-crop label set
  ([Week 1 notes](../week01/learning_notes.md));
* class separation — the reason every score here is so high
  ([Week 2 notes](../week02/learning_notes.md));
* the stratified 80/20 split, `random_state=42` and `data/processed/train.csv`
  ([Week 3 notes](../week03/learning_notes.md));
* `Pipeline`, `ColumnTransformer` and fitting the scaler inside each fold
  (Week 3 §5-§6);
* accuracy, the 4.55% baseline and `StratifiedKFold`
  ([Week 4 notes](../week04/learning_notes.md));
* the `fit`/`predict` loop, fair comparison on shared folds, and the four-row
  results table this week extends
  ([Week 5 notes](../week05/learning_notes.md)).

No new dependencies: `scikit-learn==1.6.1` and `matplotlib==3.10.0` were both
pinned in Week 1, and `SVC`, `DecisionTreeClassifier` and `export_text` all live
in scikit-learn.

## Concepts covered

| Concept | First introduced |
| --- | --- |
| Support vector machine (SVM) | Week 6 |
| Margin, margin maximisation | Week 6 |
| Support vectors | Week 6 |
| Soft margin and `C` for an SVM | Week 6 |
| Kernel, and the kernel trick | Week 6 |
| Linear vs. RBF kernel, and `gamma` | Week 6 |
| Decision tree | Week 6 |
| Split, threshold, node, leaf, depth | Week 6 |
| Node purity: Gini impurity and entropy | Week 6 |
| Information gain, greedy splitting | Week 6 |
| `max_depth`, `min_samples_leaf` | Week 6 |
| Model capacity / complexity | Week 6 |
| Bias-variance tradeoff | Week 5 (named), Week 6 (measured and plotted) |
| Generalisation gap (train minus validation) | Week 6 |
| Overfitting and underfitting, plotted | Week 1 (defined), Week 6 (shown) |
| Decision boundary, drawn | Week 5 (described), Week 6 (visualised) |
| Axis-aligned splits | Week 6 |
| Scale invariance of trees | Week 6 |

Each also appears in [`docs/ml_concepts.md`](../../ml_concepts.md) and
[`docs/glossary.md`](../../glossary.md).

## Connection to the previous week

Week 5 produced a table with three models in it and two loose ends.

* **Every boundary so far was flat, absent or bell-shaped.** Logistic regression
  cuts with hyperplanes, KNN draws no boundary of its own, and naive Bayes puts a
  Gaussian on each class. The SVM adds a principled reason to prefer one
  boundary over another, and a kernel that lets it curve; the tree adds a
  boundary made of rectangles.
* **Overfitting was asserted, never shown.** Week 5 saw `k = 1` score a perfect
  training accuracy and observed the effect of `k` on the smoothness of KNN's
  boundary, but produced no picture of a model's training score departing from
  its validation score. A decision tree is the ideal instrument for that,
  because `max_depth` is a single, interpretable dial from "cannot express
  anything" to "memorises everything".

The comparison protocol is unchanged and that is the point: the same 1,760
training rows, the same five stratified folds, the same seed, the same metric,
the same preprocessor. The two new models are appended to Week 5's table rather
than measured in a new experiment.

## Connection to the ML lifecycle

```
Frame the problem -> Get the data -> Explore -> Prepare -> [Model]
    -> Evaluate -> Improve -> Productionize -> Deploy -> Monitor
```

Week 6 is still the **model** stage — widening the field of candidates — but it
is the first week that looks hard at the *evaluate* arrow while doing so. The
train-versus-validation curve is the tool that decides whether a better training
score is a better model or merely a better memory, and every later week's
"improve" step depends on being able to tell those apart.

## Expected student outcome

### The student CAN, after this week

* **Explain overfitting using the tree-depth plot they generated.** Both curves
  at 9.09% for a depth-1 tree, rising together to ~98% by depth 10, and then the
  training curve alone continuing to a perfect 100% while validation sticks at
  98.52% — with the ~1.5-point gap named as the part of the training score that
  does not survive contact with unseen rows.
* **Explain what an SVM kernel does in one sentence.** "A kernel lets an SVM draw
  a curved boundary by measuring the similarity between rows instead of their
  positions, so it can fit a flat boundary in a higher-dimensional space it never
  actually computes."
* **Name the best-performing model so far.** Gaussian naive Bayes, **99.49%**
  (±0.42) under 5-fold stratified cross-validation, ahead of the new decision
  tree at 98.52% (±0.68) and the new SVM at 97.90% (±1.03), with logistic
  regression at 96.82% and KNN at 96.53% behind them and the baseline at 4.55%.
* Describe margin maximisation, support vectors and the direction `C` moves the
  margin, and read the support-vector count as evidence of it (943 of 1,760 rows
  at `C = 1`, 612 at `C = 100`).
* Describe how a tree chooses a split, in terms of purity rather than formulae,
  and read the tree's first questions (`rainfall <= 30.18`, then
  `humidity <= 27.98`) as ordinary sentences.
* Define bias and variance and identify which end of the depth curve each one
  occupies.
* Use `plot_decision_boundary(model, X_2d, y)` on two features, and state why the
  picture is an illustration rather than a result.
* Run `pytest tests/test_classical_models.py` and execute
  `notebooks/05_classification_models.ipynb` end to end.

### The student CANNOT yet

* **Combine models into an ensemble.** No random forest, no gradient boosting,
  no voting or stacking classifier. §11 of the notebook ends exactly where a
  random forest would begin — a crowd of deep trees whose variance cancels — and
  deliberately stops there; ensembles are a later week.
* **Search hyperparameters systematically.** The `C` sweep and the `max_depth`
  sweep are demonstrations of curves, not searches. Nothing from either is
  adopted: `C = 1.0` and `max_depth=None` remain the defaults, because choosing
  them by eye from validation scores is exactly the mistake a stated search
  protocol exists to prevent. Grid search, random search and validation curves
  are a later week.
* Decide between two models a fraction of a point apart — the SVM and the tree
  differ by 0.62 points with fold spreads of 1.03 and 0.68.
* Say which features drive an individual prediction — **Week 7**, which starts
  from the tree's readable rules.
* Interpret precision, recall, F1 or a confusion matrix — **Week 8**.
* Quote a test-set score. `data/processed/test.csv` stays unopened until Week 8.

## Deliverables for the week

* `src/models/classical_models.py` — `get_svm()` and `get_decision_tree()` added
  beside the Week 5 factories, both returning unfitted estimators, both
  registered in `CLASSICAL_MODEL_FACTORIES` (now five entries).
* `src/utils/visualization.py` — `plot_decision_boundary(model, X_2d, y)`, the
  first model-facing plotting helper, documented and tested.
* `tests/test_classical_models.py` — extended to 119 tests (231 in the whole
  suite): the new factories and their validation, SVM behaviour (support vectors
  decide the boundary; an RBF kernel solves a ring-shaped problem a linear one
  cannot; larger `C` means fewer support vectors), tree behaviour (memorisation
  when unlimited, underfitting at depth 1, a widening train/validation gap,
  monotone training accuracy in depth, scale invariance) and the boundary plot.
* `notebooks/05_classification_models.ipynb` — Part 2 (§8-§15) added to the same
  notebook, committed with executed output, ending in a six-row results table.
* This week's four curriculum documents.
* Updated `docs/ml_concepts.md`, `docs/glossary.md` and the README progress
  table.

No change to `requirements.txt`.
