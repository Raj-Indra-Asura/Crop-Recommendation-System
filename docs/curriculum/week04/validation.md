# Week 4 — Validation

Run these commands in order from the repository root, with the virtual
environment from Week 1 activated. Each block shows the command and the **real
output captured from an actual run** on this repository, so you can compare
against what you see.

The two commands this week must be able to run, verbatim, are:

```bash
pytest tests/test_baseline.py
jupyter nbconvert --to notebook --execute notebooks/04_baseline_models.ipynb
```

Everything below expands on those two, with the output each one produced.

---

## Step 0 — Environment

No new dependencies this week. `scikit-learn==1.6.1` was pinned in Week 1, and
`DummyClassifier`, `StratifiedKFold` and `cross_val_score` all live in it.

```bash
source venv/bin/activate       # Windows: venv\Scripts\activate
python -c "import sklearn; print(sklearn.__version__)"
```

Actual output:

```
1.6.1
```

If that fails, re-run `pip install -r requirements.txt` — see
[Week 1 validation, Step 2](../week01/validation.md).

This week also reads `data/processed/train.csv`, written by Week 3. If it is
missing, re-run Week 3's notebook.

---

## Step 1 — Lint the codebase

```bash
ruff check .
```

Actual output:

```
All checks passed!
```

This now also covers `src/models/baseline.py`, `src/evaluation/metrics.py`,
`tests/test_baseline.py` and the code cells of
`notebooks/04_baseline_models.ipynb`.

---

## Step 2 — Run the Week 4 test file

```bash
pytest tests/test_baseline.py
```

Actual output:

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-8.3.4, pluggy-1.6.0
rootdir: /.../Crop-Recommendation-System
configfile: pyproject.toml
plugins: anyio-4.14.2
collected 38 items

tests/test_baseline.py ......................................            [100%]

============================== 38 passed in 0.28s ==============================
```

**All 38 tests must pass, with zero failures.** Most run on a small synthetic
frame built inside the test file, so they pass with or without the CSV; the last
three need the real 2,200 rows and are skipped (not failed) if it is missing.

`pytest tests/test_baseline.py -v` names them:

```
tests/test_baseline.py::test_get_baseline_model_returns_an_unfitted_dummy_classifier PASSED [  2%]
tests/test_baseline.py::test_get_baseline_model_defaults_to_most_frequent PASSED [  5%]
tests/test_baseline.py::test_get_baseline_model_accepts_every_supported_strategy[most_frequent] PASSED [  7%]
tests/test_baseline.py::test_get_baseline_model_accepts_every_supported_strategy[stratified] PASSED [ 10%]
tests/test_baseline.py::test_get_baseline_model_accepts_every_supported_strategy[uniform] PASSED [ 13%]
tests/test_baseline.py::test_get_baseline_model_accepts_every_supported_strategy[prior] PASSED [ 15%]
tests/test_baseline.py::test_get_baseline_model_rejects_an_unsupported_strategy[constant] PASSED [ 18%]
tests/test_baseline.py::test_get_baseline_model_rejects_an_unsupported_strategy[MOST_FREQUENT] PASSED [ 21%]
tests/test_baseline.py::test_get_baseline_model_rejects_an_unsupported_strategy[best] PASSED [ 23%]
tests/test_baseline.py::test_get_baseline_model_rejects_an_unsupported_strategy[] PASSED [ 26%]
tests/test_baseline.py::test_get_baseline_model_uses_the_project_seed_by_default PASSED [ 28%]
tests/test_baseline.py::test_random_baselines_are_reproducible PASSED    [ 31%]
tests/test_baseline.py::test_most_frequent_baseline_predicts_a_single_class PASSED [ 34%]
tests/test_baseline.py::test_baseline_ignores_the_features PASSED        [ 36%]
tests/test_baseline.py::test_baseline_scores_one_over_k_on_balanced_data PASSED [ 39%]
tests/test_baseline.py::test_baseline_accuracy_is_misleadingly_high_on_imbalanced_data PASSED [ 42%]
tests/test_baseline.py::test_evaluate_model_returns_accuracy_report_and_sample_count PASSED [ 44%]
tests/test_baseline.py::test_evaluate_model_accuracy_matches_a_hand_computation PASSED [ 47%]
tests/test_baseline.py::test_evaluate_model_report_names_every_class PASSED [ 50%]
tests/test_baseline.py::test_evaluate_model_does_not_fit_the_model PASSED [ 52%]
tests/test_baseline.py::test_evaluate_model_rejects_mismatched_lengths PASSED [ 55%]
tests/test_baseline.py::test_evaluate_model_scores_a_perfect_model_at_one PASSED [ 57%]
tests/test_baseline.py::test_build_cv_is_stratified_shuffled_and_seeded PASSED [ 60%]
tests/test_baseline.py::test_build_cv_rejects_fewer_than_two_folds PASSED [ 63%]
tests/test_baseline.py::test_cv_folds_preserve_the_class_balance PASSED  [ 65%]
tests/test_baseline.py::test_cv_folds_are_disjoint_and_cover_every_row PASSED [ 68%]
tests/test_baseline.py::test_cross_validated_accuracy_returns_one_score_per_fold PASSED [ 71%]
tests/test_baseline.py::test_cross_validated_accuracy_honours_the_fold_count PASSED [ 73%]
tests/test_baseline.py::test_cross_validated_accuracy_is_reproducible PASSED [ 76%]
tests/test_baseline.py::test_cross_validated_accuracy_leaves_the_estimator_unfitted PASSED [ 78%]
tests/test_baseline.py::test_cross_validated_accuracy_rejects_mismatched_lengths PASSED [ 81%]
tests/test_baseline.py::test_cross_validated_accuracy_matches_a_balanced_baseline PASSED [ 84%]
tests/test_baseline.py::test_real_baseline_lands_on_one_over_twenty_two PASSED [ 86%]
tests/test_baseline.py::test_no_baseline_strategy_escapes_the_one_over_k_ceiling[most_frequent] PASSED [ 89%]
tests/test_baseline.py::test_no_baseline_strategy_escapes_the_one_over_k_ceiling[stratified] PASSED [ 92%]
tests/test_baseline.py::test_no_baseline_strategy_escapes_the_one_over_k_ceiling[uniform] PASSED [ 94%]
tests/test_baseline.py::test_no_baseline_strategy_escapes_the_one_over_k_ceiling[prior] PASSED [ 97%]
tests/test_baseline.py::test_real_most_frequent_baseline_predicts_exactly_one_crop PASSED [100%]

============================== 38 passed in 0.28s ==============================
```

The three the week is graded on are
`test_real_baseline_lands_on_one_over_twenty_two` (the baseline really is
1/22), `test_baseline_ignores_the_features` (scrambling the features cannot
change a prediction) and
`test_baseline_accuracy_is_misleadingly_high_on_imbalanced_data` (95% accuracy
while never predicting the minority class).

---

## Step 3 — Run the whole suite

```bash
pytest
```

Actual output:

```
======================= 112 passed, 26 warnings in 1.76s =======================
```

112 = Week 1's 20 + Week 2's 22 + Week 3's 32 + Week 4's 38. **Weeks 1-3 must
still pass unchanged**: this week adds files and changes none of the data or
preprocessing contracts.

---

## Step 4 — Reproduce the key numbers on the command line

Each of these underpins a claim in the notebook or the learning notes.

### The baseline number itself

```bash
python -c "
import pandas as pd
from src.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluation import cross_validated_accuracy
from src.models import get_baseline_model
train = pd.read_csv('data/processed/train.csv')
X, y = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]
result = cross_validated_accuracy(get_baseline_model('most_frequent'), X, y)
print('per-fold:', result['scores'].round(4))
print('mean    :', round(result['mean'], 4))
print('std     :', round(result['std'], 4))
print('1 / 22  :', round(1 / 22, 4))
"
```

Actual output:

```
per-fold: [0.0455 0.0455 0.0455 0.0455 0.0455]
mean    : 0.0455
std     : 0.0
1 / 22  : 0.0455
```

**This is the number of the week.** The measured mean equals `1/22` exactly, and
the standard deviation is zero because a constant prediction is right on exactly
1/22 of every stratified fold of perfectly balanced data.

### No naive strategy escapes 1/k

```bash
python -c "
import pandas as pd
from src.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluation import cross_validated_accuracy
from src.models import BASELINE_STRATEGIES, get_baseline_model
train = pd.read_csv('data/processed/train.csv')
X, y = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]
for strategy in BASELINE_STRATEGIES:
    r = cross_validated_accuracy(get_baseline_model(strategy), X, y)
    print(f'{strategy:14s} mean {r[\"mean\"]:.4f}  std {r[\"std\"]:.4f}')
"
```

Actual output:

```
most_frequent  mean 0.0455  std 0.0000
stratified     mean 0.0472  std 0.0064
uniform        mean 0.0466  std 0.0073
prior          mean 0.0455  std 0.0000
```

`stratified` edged above `most_frequent` here. That is luck, not skill — note
its standard deviation — and it is why the deterministic strategy is the one
quoted as the baseline.

### A single split gives a different answer every time

```bash
python -c "
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluation import evaluate_model
from src.models import get_baseline_model
train = pd.read_csv('data/processed/train.csv')
scores = []
for seed in range(10):
    a, b = train_test_split(train, test_size=0.2, random_state=seed,
                            stratify=train[TARGET_COLUMN])
    model = get_baseline_model('stratified', random_state=seed)
    model.fit(a[list(FEATURE_COLUMNS)], a[TARGET_COLUMN])
    scores.append(evaluate_model(model, b[list(FEATURE_COLUMNS)], b[TARGET_COLUMN])['accuracy'])
s = pd.Series(scores)
print('ten single-split scores:', [round(v, 4) for v in scores])
print('min', round(s.min(), 4), '| max', round(s.max(), 4), '| range', round(s.max() - s.min(), 4))
"
```

Actual output:

```
ten single-split scores: [0.0653, 0.0426, 0.0369, 0.0341, 0.0682, 0.0625, 0.0568, 0.0511, 0.0398, 0.0227]
min 0.0227 | max 0.0682 | range 0.0455
```

Same model, same data, ten legitimate splits, and a spread as wide as the
baseline itself. This is the argument for cross-validation in one command.

### The folds are stratified

```bash
python -c "
import pandas as pd
from src.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluation import build_cv
train = pd.read_csv('data/processed/train.csv')
X, y = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]
cv = build_cv()
print(cv)
for i, (fit_idx, val_idx) in enumerate(cv.split(X, y), start=1):
    counts = y.iloc[val_idx].value_counts()
    print(f'fold {i}: fit {len(fit_idx)} | validate {len(val_idx)}'
          f' | crops {counts.size} | per crop {counts.min()}-{counts.max()}')
"
```

Actual output:

```
StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
fold 1: fit 1408 | validate 352 | crops 22 | per crop 16-16
fold 2: fit 1408 | validate 352 | crops 22 | per crop 16-16
fold 3: fit 1408 | validate 352 | crops 22 | per crop 16-16
fold 4: fit 1408 | validate 352 | crops 22 | per crop 16-16
fold 5: fit 1408 | validate 352 | crops 22 | per crop 16-16
```

All 22 crops in every fold, 16 rows each. 1,408 + 352 = 1,760, and the five
validation folds together cover each training row exactly once.

### Accuracy on its own can be badly misleading

```bash
python -c "
import numpy as np, pandas as pd
from src.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluation import evaluate_model
from src.models import get_baseline_model
train = pd.read_csv('data/processed/train.csv')
X, y = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]
binary = np.where(y == 'rice', 'rice', 'not rice')
model = get_baseline_model().fit(X, binary)
out = evaluate_model(model, X, binary)
print('accuracy:', round(out['accuracy'], 4))
print(out['report'])
"
```

Actual output:

```
accuracy: 0.9545
              precision    recall  f1-score   support

    not rice       0.95      1.00      0.98      1680
        rice       0.00      0.00      0.00        80

    accuracy                           0.95      1760
   macro avg       0.48      0.50      0.49      1760
weighted avg       0.91      0.95      0.93      1760
```

**95.45% accuracy, from a model that never once predicts `rice`.** Same rows,
same features — only the framing of the question changed, and with it the class
balance. This is the concrete reason Week 8 exists.

---

## Step 5 — Execute the Week 4 notebook

Open it interactively — this is what a student runs:

```bash
jupyter notebook notebooks/04_baseline_models.ipynb
```

Run all cells (*Kernel → Restart & Run All*); every cell must execute without
error. Stop the server with `Ctrl-C`.

The non-interactive check, which is the command this week is graded on:

```bash
jupyter nbconvert --to notebook --execute notebooks/04_baseline_models.ipynb
```

Actual output:

```
[NbConvertApp] Converting notebook notebooks/04_baseline_models.ipynb to notebook
[NbConvertApp] Writing 37108 bytes to notebooks/04_baseline_models.nbconvert.ipynb
```

**Exit code 0 is the pass condition.** `nbconvert --execute` aborts on the first
cell that raises, and this notebook ends with `assert` statements — on the
baseline equalling 1/22, on no naive strategy exceeding 10%, and on the training
frame still holding 1,760 rows — so a clean exit is a real check.

Without `--inplace` the result goes to
`notebooks/04_baseline_models.nbconvert.ipynb`, which is gitignored and can be
deleted. To refresh the committed outputs, which is how this repository's version
was produced:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/04_baseline_models.ipynb
```

The committed notebook contains genuinely executed output — never hand-written
or fabricated results.

---

## Step 6 — Check what was and was not produced

No model artifacts yet; the baseline is rebuilt in milliseconds whenever it is
needed, and the first saved artifact lands in `models/` in Week 9.

```bash
ls models/
```

Actual output:

```
```

Empty.

The raw data is untouched, as always:

```bash
git status --short data/raw/
```

Actual output:

```
```

Empty. Week 4 only *reads*.

And the test set was never opened. The notebook reads exactly one file, and it
is the training split:

```bash
grep -n "read_csv" notebooks/04_baseline_models.ipynb
```

Actual output:

```
116:    "train = pd.read_csv(REPO_ROOT / \"data\" / \"processed\" / \"train.csv\")\n",
```

One line, one file. `data/processed/test.csv` is not loaded here, nor by
`src/models/baseline.py` or `src/evaluation/metrics.py` — it stays sealed until
Week 8.

---

## What "done" looks like this week

The student CAN:

* **State the baseline accuracy from memory: 4.55%.** In full: the 5-fold
  stratified cross-validated accuracy of a `most_frequent` `DummyClassifier`,
  seed 42, on the 1,760 rows of `data/processed/train.csv`. Equivalently `1/22`,
  because 22 classes hold equal numbers of rows, so always predicting one class
  is right on exactly one row in 22.
* **Explain why any real model that does not beat it is broken or trivial.** The
  baseline never looks at a feature, so its score is the value of the label
  distribution alone. A model at or below it has extracted *nothing* from `N`,
  `P`, `K`, temperature, humidity, pH and rainfall — features Week 2 showed
  separate the crops almost perfectly. That is a bug report, not a result, and
  the usual causes are shuffled or misaligned labels, fitting on the wrong
  array, scoring against the wrong vector, or a target that never reached the
  model. Debug the pipeline; do not tune the model.
* Fit a `DummyClassifier`, choose between `most_frequent`, `prior`, `stratified`
  and `uniform`, and say what each predicts and which are deterministic.
* Explain and demonstrate why accuracy alone misleads: Step 4's 95.45% on the
  `rice` / `not rice` framing, with recall 0.00 on `rice`.
* Describe k-fold cross-validation, explain why one split is not enough (Step
  4's 2.27%-6.82% spread), and read `cross_val_score`'s per-fold array — mean,
  standard deviation, and what a single outlying fold indicates.
* Say that this dataset is unusually easy, that Weeks 5-8 will see 98-99%+ from
  nearly every algorithm, and why the value of those weeks is in **how** models
  are compared, tuned and explained rather than in the accuracy reached.

The student CANNOT yet:

* **Compare multiple real algorithms.** No logistic regression, KNN, decision
  tree or random forest has been trained. That is **Week 5**; comparing them
  rigorously is **Week 6**.
* **Tune hyperparameters.** No grid search, no random search, no validation
  curves — **Week 6**. `DummyClassifier` has no hyperparameter worth tuning,
  which is part of why it is this week's only model.
* Interpret precision, recall, F1 or a confusion matrix. The classification
  report was *shown* to make one point; it is *taught* in **Week 8**.
* Say which features a model relies on — **Week 7**.
* Quote any test-set number. See Step 6.

---

## Common errors at this stage

**`FileNotFoundError: data/processed/train.csv`**
Week 3's notebook has not been run in this checkout, or you are running from
somewhere other than the repository root. Execute
`notebooks/03_data_preparation.ipynb`, or `cd` to the root.

**`ValueError: Unsupported baseline strategy 'constant'`**
`get_baseline_model` deliberately accepts only `most_frequent`, `prior`,
`stratified` and `uniform`. `constant` needs a second argument naming the class
to predict; instantiate `DummyClassifier` directly if you really want it.

**`sklearn.exceptions.NotFittedError` from `evaluate_model`**
`evaluate_model` scores an *already fitted* model — that is what lets it be
pointed at held-out data safely. Call `model.fit(X, y)` first, or use
`cross_validated_accuracy`, which fits inside each fold itself.

**Your baseline is not 4.55%**
Check three things, in order: that you are using the training rows (1,760, not
2,200 and not 440); that the strategy is `most_frequent`; and that the CSV is
unmodified (`git status --short data/`). A baseline far above 4.55% usually
means the classes are no longer balanced, i.e. the data has been filtered.

**Your `stratified` baseline changes on every run**
It is meant to vary between *seeds*, not between *runs*. If it varies with the
same seed, you are constructing `DummyClassifier` directly without a
`random_state`; use `get_baseline_model`, which supplies the project seed.

**Two models look different, but the difference is smaller than the fold spread**
Then they are not yet different. Report the mean *and* the standard deviation of
the per-fold scores, and treat a gap inside that spread as unproven — this is
the habit Week 6 depends on.

**Your cross-validated score is suspiciously high after adding a scaler**
You probably fitted the scaler outside the cross-validation loop, on all the
data. Pass a `Pipeline` to `cross_validated_accuracy` instead, so the scaler is
re-fitted inside each fold on that fold's training rows only — Week 3's
`build_preprocessing_pipeline()` exists for exactly this.

**`E501 Line too long` from `ruff` pointing at a notebook cell**
`ruff` lints notebook code cells as well as `.py` files. Wrap the line; do not
loosen the rule.
