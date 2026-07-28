# Week 4 — Baseline Models

## Title

**Establishing what "good" means, before building anything real**

## Learning objectives

By the end of this week a student should be able to:

1. Explain why a baseline is built *before* a "real" model, and why an accuracy
   figure quoted without one cannot be interpreted.
2. Describe what a `DummyClassifier` does, and the difference between its
   `most_frequent`, `prior`, `stratified` and `uniform` strategies.
3. Predict a naive baseline's accuracy from the class distribution alone — 1/k
   on balanced data, the majority share on imbalanced data — and check it.
4. State this project's baseline number, **4.55%**, from memory, and explain
   where the number comes from.
5. Explain why any real model that fails to beat the baseline is broken or
   trivial, and name the usual causes.
6. Explain why accuracy alone can mislead, even on a balanced dataset, and
   demonstrate it by reframing this data as an imbalanced two-class problem.
7. Describe k-fold cross-validation: how the folds are built, what is fitted on
   what, and why every row is validated exactly once.
8. Read `cross_val_score` output — an array with one score per fold — and say
   what the mean, the standard deviation and a single outlying fold each mean.
9. Explain why a single train/test split is not enough to trust a score, with
   the split-to-split spread measured in the notebook as evidence.
10. State that this dataset is unusually easy, that later models will reach
    98-99%+, and why that does not make Weeks 5-8 pointless.

## Prerequisites

Weeks 1-3, in full. This week assumes and does **not** re-explain:

* the dataset contract and the frozen 22-crop label set
  ([Week 1 notes §8](../week01/learning_notes.md));
* class balance — 100 rows per crop
  ([Week 2 notes §2](../week02/learning_notes.md));
* class separation: the features split the crops almost perfectly
  (Week 2 notes §5) — the reason later scores will be so high;
* data leakage and train-only fitting (Week 2 §6, Week 3 §5);
* the stratified 80/20 split, `random_state=42`, and the processed CSVs in
  `data/processed/` ([Week 3 notes](../week03/learning_notes.md));
* `fit` / `predict` / `transform` and the scikit-learn estimator API (Week 3 §5).

No new dependencies: `scikit-learn==1.6.1` was pinned in Week 1.

## Concepts covered

| Concept | First introduced |
| --- | --- |
| Baseline model | Week 4 |
| `DummyClassifier` and its strategies | Week 4 |
| Accuracy | Week 4 |
| The 1/k rule for a balanced dataset | Week 4 |
| Majority-class accuracy under imbalance | Week 4 |
| Classification report (per-class view) | Week 4 (glimpsed), Week 8 (taught) |
| Validation set (as distinct from the test set) | Week 4 |
| k-fold cross-validation | Week 4 |
| `StratifiedKFold` | Week 4 |
| `cross_val_score` and reading its output | Week 4 |
| Score variance / fold-to-fold spread | Week 4 |
| Evaluation protocol | Week 4 |
| Sanity check / smoke test of a pipeline | Week 4 |
| Dataset difficulty and performance ceilings | Week 4 |

Each also appears in [`docs/ml_concepts.md`](../../ml_concepts.md) and
[`docs/glossary.md`](../../glossary.md).

## Connection to the previous weeks

Week 3 ended with model-ready data and nothing fitted but a scaler. The obvious
next step is to train a classifier — and that is exactly the step this week
refuses to take, for one week, because a score with nothing to compare it
against teaches the wrong habit permanently.

Two Week 2 findings are cashed in here:

* **The classes are perfectly balanced** (100 rows each). That makes accuracy a
  fair metric *and* fixes the baseline at 1/22 by arithmetic, so the number can
  be predicted before it is measured.
* **The features separate the classes strongly.** That is why Weeks 5-8 will see
  98-99%+ from nearly everything, and why this syllabus says so out loud now.

Week 3's fixed seed also pays off: the folds, and therefore the baseline number,
are identical on every machine.

## Connection to the ML lifecycle

```
Frame the problem -> Get the data -> Explore -> Prepare -> Model
    -> [Evaluate] -> Improve -> Productionize -> Deploy -> Monitor
```

Week 4 is the **evaluate** stage, reached before the modelling stage on purpose.
Deciding how a result will be judged *before* producing any results is what
keeps the judgement honest; deciding afterwards is how a project ends up
choosing the metric its favourite model happens to win.

## Expected student outcome

### The student CAN, after this week

* **State the baseline accuracy from memory: 4.55%** — the 5-fold
  cross-validated accuracy of a `most_frequent` `DummyClassifier` on the 1,760
  training rows — and derive it as `1/22` from 22 balanced classes.
* **Explain why any real model that does not beat it is broken or trivial.** It
  has extracted nothing from the seven features; the usual causes are shuffled
  or misaligned labels, fitting on the wrong array, scoring against the wrong
  vector, or a target that never reached the model.
* Build and fit a `DummyClassifier`, choose between its strategies, and say what
  each predicts.
* Explain why accuracy alone can mislead, and produce the demonstration: framed
  as "is this field suited to rice?", the same baseline scores 95.45% while
  never predicting `rice` once.
* Describe k-fold cross-validation and read `cross_val_score`'s array — mean,
  standard deviation, and what a single bad fold indicates.
* Explain why a single train/test split is not enough, citing the notebook's
  ten-seed spread (2.27% to 6.82% for the same model).
* Run `pytest tests/test_baseline.py` and execute
  `notebooks/04_baseline_models.ipynb` end to end.

### The student CANNOT yet

* **Compare multiple real algorithms.** No logistic regression, KNN, decision
  tree or random forest is trained this week — that is **Week 5**, and comparing
  them rigorously is **Week 6**.
* **Tune a hyperparameter.** No grid search, no random search, no validation
  curve — **Week 6**.
* Interpret precision, recall, F1 or a confusion matrix. The classification
  report is *shown* this week to make one point; it is *taught* in **Week 8**.
* Say which features a model relies on — **Week 7**.
* Quote a test-set score. `data/processed/test.csv` is still unopened, and stays
  that way until Week 8.

## Deliverables for the week

* `src/models/baseline.py` — `get_baseline_model(strategy)`, returning an
  unfitted `DummyClassifier` restricted to four vetted strategies.
* `src/evaluation/metrics.py` — `evaluate_model(model, X, y)` returning accuracy
  plus a `classification_report` string, `cross_validated_accuracy()` and
  `build_cv()`. **Extended in later weeks, never replaced.**
* `tests/test_baseline.py` — 38 tests covering the factory, the evaluation
  helpers, the fold construction and the 1/22 result on the real data.
* `notebooks/04_baseline_models.ipynb` — committed with executed output, ending
  in the number every future model must beat.
* This week's four curriculum documents.
* Updated `docs/ml_concepts.md`, `docs/glossary.md` and the README progress
  table.

No change to `requirements.txt`.
