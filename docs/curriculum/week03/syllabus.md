# Week 3 — Data Preparation

## Title

**Turning raw data into model-ready data, correctly**

## Learning objectives

By the end of this week a student should be able to:

1. Explain why raw features usually cannot go straight into a model, using this
   dataset's own numbers: `K` spans 200 units while `ph` spans 6.4.
2. State what `StandardScaler` computes — `(x - mean) / std`, per column — and
   what it does *not* do (it does not remove skew, outliers or ordering).
3. Name which model families need scaling (KNN, SVM, logistic regression, neural
   networks, PCA) and which are indifferent to it (decision trees and their
   ensembles), and say why in terms of how each family uses a feature's value.
4. Encode a categorical target with `LabelEncoder`, explain why an estimator
   needs numeric labels, and explain why those integers must not be read as
   quantities.
5. Perform a **stratified** train/test split and justify stratification for a
   22-class problem.
6. Distinguish `fit`, `transform` and `fit_transform`, and say exactly which is
   allowed on which half of the data.
7. Explain, with a concrete example, why a scaler is never fitted on test data —
   turning Week 2's abstract data-leakage rule into a procedure.
8. Build a `ColumnTransformer` from scratch and wrap it in a `Pipeline`, and say
   what those objects will be doing in Weeks 6, 9 and 10.
9. Explain why the split uses a fixed `random_state`, and why choosing the seed
   that maximises a score is a form of cheating.

## Prerequisites

Weeks 1 and 2, in full. This week assumes and does **not** re-explain:

* `load_data()` and the dataset contract ([Week 1 notes §8](../week01/learning_notes.md));
* the frozen 22-crop label set (Week 1 notes §3, §8);
* mean, standard deviation, skew and feature scale
  ([Week 2 notes §1](../week02/learning_notes.md));
* class balance (Week 2 notes §2);
* the definition of data leakage and the train-only fitting rule (Week 2 notes
  §6) — stated there, **enforced here**;
* running `ruff check .` and `pytest`.

New this week: `scikit-learn`, pinned in Week 1 at `1.6.1` precisely so that
this week needed no environment change.

## Concepts covered

| Concept | First introduced |
| --- | --- |
| Data preparation / preprocessing | Week 3 |
| Feature scaling | Week 3 |
| Standardisation (`StandardScaler`, z-score) | Week 3 |
| Normalisation (min-max), and how it differs | Week 3 |
| Scale-sensitive vs. scale-invariant models | Week 3 |
| Label encoding (`LabelEncoder`) | Week 3 |
| One-hot encoding (contrasted, not yet needed) | Week 3 |
| Train/test split | Week 1 (defined), Week 3 (performed) |
| Stratified sampling | Week 3 |
| `fit` / `transform` / `fit_transform` | Week 3 |
| Estimator, transformer and the scikit-learn API | Week 3 |
| Fitted state (learned parameters, the trailing-underscore convention) | Week 3 |
| Random seed / `random_state` and reproducibility | Week 3 |
| `ColumnTransformer` | Week 3 |
| `Pipeline` | Week 3 |
| Data leakage, prevented in practice | Week 2 (defined), Week 3 (enforced) |
| Processed data artifacts | Week 3 |

Each also appears in [`docs/ml_concepts.md`](../../ml_concepts.md) and
[`docs/glossary.md`](../../glossary.md).

## Connection to the previous weeks

Week 1 asked *what is this data?* Week 2 asked *what does it look like?* — and
its two most consequential findings both come due now:

* **Features are on wildly different scales.** Week 2 measured it; Week 3 fixes
  it with a `StandardScaler`.
* **Data leakage exists.** Week 2 defined it while no split existed, so the rule
  could only be stated. Week 3 creates the split, which is the moment the rule
  becomes something you can actually violate — and the moment the code has to
  prevent it.

Week 2's decisions to *not* delete outliers and to *not* drop the correlated
`P`/`K` pair also stand: this week rescales columns and nothing else.

## Connection to the ML lifecycle

```
Frame the problem -> Get the data -> Explore -> [Prepare] -> Model
    -> Evaluate -> Improve -> Productionize -> Deploy -> Monitor
```

Week 3 is the **prepare** stage. It is also the stage that quietly determines
whether Weeks 4-8's numbers mean anything: an evaluation set contaminated here
cannot be decontaminated later, and every score computed afterwards is wrong in
a direction that looks like success.

## Expected student outcome

### The student CAN, after this week

* **Explain `fit` vs `transform` vs `fit_transform`.** `fit` learns parameters
  from data and stores them on the object; `transform` applies stored parameters
  to any data; `fit_transform` does both and is for training data only.
* **Explain why we never fit the scaler on test data.** Because the mean and
  standard deviation would then be computed partly from rows the model is meant
  never to have seen, so the test score would measure performance on data the
  pipeline already knew about — an optimistic estimate that production will not
  reproduce.
* **Build a `ColumnTransformer` from scratch**, i.e. write
  `ColumnTransformer([("numeric", StandardScaler(), FEATURES)], remainder="drop")`
  without copying it, and say what each of the three parts of the tuple does.
* Encode the target and decode it back, and say why the codes are not numbers.
* Split 2,200 rows into 1,760 train and 440 test, stratified, reproducibly, and
  prove stratification by comparing class proportions.
* Say which of KNN, SVM, logistic regression and random forest need the scaling
  and which do not.
* Run `pytest tests/test_preprocessing.py` and execute
  `notebooks/03_data_preparation.ipynb` end to end.

### The student CANNOT yet

* **Train or compare real classification models.** Nothing is fitted this week
  except a scaler. Baselines are **Week 4**; real classifiers are **Week 5**;
  comparing them properly is **Week 6**.
* Report accuracy, precision, recall or a confusion matrix — Week 4 onward.
* Tune a hyperparameter or cross-validate — **Week 6**. (Cross-validation is
  also where the `Pipeline` introduced this week starts to earn its keep.)
* Look at test-set *values*. The test rows are counted and their labels tallied;
  nothing has inspected or fitted on their contents, and nothing may until
  Week 8's final evaluation.
* Say which features matter most to a model — **Week 7**.
* Engineer new features, drop columns or delete rows. Week 3 rescales; it does
  not redesign.

## Deliverables for the week

* `src/data/split.py` — `stratified_split()` and `class_proportions()`, with the
  project-wide `DEFAULT_RANDOM_STATE = 42` and `DEFAULT_TEST_SIZE = 0.2`.
* `src/preprocessing/preprocessor.py` — `build_preprocessor()` returning the
  unfitted `ColumnTransformer`, and `build_preprocessing_pipeline()` wrapping it
  in a `Pipeline`.
* `tests/test_preprocessing.py` — 32 tests: scaled training features have ~0
  mean and ~1 std, the split is stratified, the split is reproducible, and the
  test rows are provably not part of the fit.
* `notebooks/03_data_preparation.ipynb` — the full preparation, committed with
  executed output, writing five files to `data/processed/`.
* This week's four curriculum documents.
* Updated `docs/ml_concepts.md`, `docs/glossary.md` and the README progress
  table.

No change to `requirements.txt`: `scikit-learn==1.6.1` was pinned in Week 1 for
exactly this week.
