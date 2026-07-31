# Week 5 — Validation

Run these commands in order from the repository root, with the virtual
environment from Week 1 activated. Each block shows the command and the **real
output captured from an actual run** on this repository, so you can compare
against what you see.

The two commands this week must be able to run, verbatim, are:

```bash
pytest tests/test_classical_models.py
jupyter nbconvert --to notebook --execute notebooks/05_classification_models.ipynb
```

Everything below expands on those two, with the output each one produced.

---

## Step 0 — Environment

No new dependencies this week. `scikit-learn==1.6.1` was pinned in Week 1, and
`LogisticRegression`, `KNeighborsClassifier` and `GaussianNB` all live in it.

```bash
source venv/bin/activate       # Windows: venv\Scripts\activate
python -c "import sklearn; print(sklearn.__version__)"
```

Actual output:

```
1.6.1
```

This week reads `data/processed/train.csv`, written by Week 3. If it is missing,
re-run Week 3's notebook. `data/processed/test.csv` is **not** read by anything
this week, and should not be.

---

## Step 1 — Lint the codebase

```bash
ruff check .
```

Actual output:

```
All checks passed!
```

This now also covers `src/models/classical_models.py`,
`tests/test_classical_models.py` and the code cells of
`notebooks/05_classification_models.ipynb`.

---

## Step 2 — Run the Week 5 test file

```bash
pytest tests/test_classical_models.py
```

Actual output:

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-8.3.4, pluggy-1.6.0
rootdir: /.../Crop-Recommendation-System
configfile: pyproject.toml
plugins: anyio-4.14.2
collected 57 items

tests/test_classical_models.py ......................................... [ 71%]
................                                                         [100%]

============================== 57 passed in 1.57s ==============================
```

**All 57 tests must pass, with zero failures.** Most run on a small synthetic
frame built inside the test file, so they pass with or without the CSV; the last
five need the real 2,200 rows and are skipped (not failed) if it is missing.

`pytest tests/test_classical_models.py -v` names them. The groups to look for:

```
tests/test_classical_models.py::test_get_logistic_regression_returns_a_logistic_regression PASSED [  1%]
tests/test_classical_models.py::test_get_knn_returns_a_kneighbors_classifier PASSED [  3%]
tests/test_classical_models.py::test_get_naive_bayes_returns_a_gaussian_nb PASSED [  5%]
tests/test_classical_models.py::test_factories_return_unfitted_models[get_logistic_regression] PASSED [  7%]
tests/test_classical_models.py::test_factories_return_unfitted_models[get_knn] PASSED [  8%]
tests/test_classical_models.py::test_factories_return_unfitted_models[get_naive_bayes] PASSED [ 10%]
tests/test_classical_models.py::test_logistic_regression_defaults PASSED [ 12%]
tests/test_classical_models.py::test_knn_defaults PASSED                 [ 14%]
tests/test_classical_models.py::test_naive_bayes_defaults PASSED         [ 15%]
...
tests/test_classical_models.py::test_every_model_supports_the_same_fit_predict_loop[get_logistic_regression] PASSED [ 45%]
tests/test_classical_models.py::test_every_model_supports_the_same_fit_predict_loop[get_knn] PASSED [ 47%]
tests/test_classical_models.py::test_every_model_supports_the_same_fit_predict_loop[get_naive_bayes] PASSED [ 49%]
...
tests/test_classical_models.py::test_one_nearest_neighbour_memorises_its_training_data PASSED [ 71%]
tests/test_classical_models.py::test_a_large_k_smooths_the_boundary_towards_ignorance PASSED [ 73%]
tests/test_classical_models.py::test_knn_is_sensitive_to_feature_scaling PASSED [ 75%]
tests/test_classical_models.py::test_naive_bayes_is_insensitive_to_standardisation PASSED [ 77%]
tests/test_classical_models.py::test_knn_degrades_when_meaningless_features_are_added PASSED [ 78%]
...
tests/test_classical_models.py::test_every_model_beats_the_real_baseline_by_a_wide_margin[logistic_regression] PASSED [ 92%]
tests/test_classical_models.py::test_every_model_beats_the_real_baseline_by_a_wide_margin[knn] PASSED [ 94%]
tests/test_classical_models.py::test_every_model_beats_the_real_baseline_by_a_wide_margin[naive_bayes] PASSED [ 96%]
tests/test_classical_models.py::test_naive_bayes_is_the_best_of_the_three_this_week PASSED [ 98%]
tests/test_classical_models.py::test_the_real_models_are_reproducible PASSED [100%]

============================== 57 passed in 1.58s ==============================
```

---

## Step 3 — Run the whole suite

```bash
pytest
```

Actual output (tail):

```
169 passed, 12 warnings in 3.39s
```

**169 = 20 (Week 1) + 22 (Week 2) + 32 (Week 3) + 38 (Week 4) + 57 (Week 5).**
Week 5 adds tests; it changes no earlier behaviour, so every earlier test must
still pass. The warnings come from seaborn and matplotlib inside the Week 2
tests and are unrelated to this week.

---

## Step 4 — Reproduce the key numbers on the command line

### The results table

```bash
python - <<'PY'
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluation import cross_validated_accuracy
from src.models import get_baseline_model, get_knn, get_logistic_regression, get_naive_bayes
from src.preprocessing import build_preprocessor

train = pd.read_csv("data/processed/train.csv")
X, y = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]

candidates = {
    "baseline": get_baseline_model(),
    "logistic": get_logistic_regression(),
    "knn (k=5)": get_knn(),
    "naive bayes": get_naive_bayes(),
}
for name, model in candidates.items():
    pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", model)])
    out = cross_validated_accuracy(pipeline, X, y)
    print(f"{name:<12} mean {out['mean']:.4f}  std {out['std']:.4f}")
PY
```

Actual output:

```
baseline     mean 0.0455  std 0.0000
logistic     mean 0.9682  std 0.0066
knn (k=5)    mean 0.9653  std 0.0121
naive bayes  mean 0.9949  std 0.0042
```

These are the week's headline numbers. If yours differ in the fourth decimal
place, check that you used the seed-42 folds (the default) and the preprocessor;
if they differ in the second, something is wrong with the data.

### The effect of `k`

```bash
python - <<'PY'
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluation import cross_validated_accuracy
from src.models import get_knn
from src.preprocessing import build_preprocessor

train = pd.read_csv("data/processed/train.csv")
X, y = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]
for k in [1, 5, 51, 401]:
    pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", get_knn(n_neighbors=k))])
    print(f"k = {k:>3}: {cross_validated_accuracy(pipeline, X, y)['mean']:.4f}")
PY
```

Actual output:

```
k =   1: 0.9665
k =   5: 0.9653
k =  51: 0.8705
k = 401: 0.5409
```

The curve falls as the neighbourhood grows: at `k = 401` each vote is drawn from
nearly a quarter of the training set. Push `k` far enough and it arrives at the
4.55% baseline.

### `k = 1` memorises its training data

```bash
python - <<'PY'
import pandas as pd

from src.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluation import cross_validated_accuracy, evaluate_model
from src.models import get_knn

train = pd.read_csv("data/processed/train.csv")
X, y = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]

fitted = get_knn(n_neighbors=1).fit(X, y)
print("training accuracy (k=1):", round(evaluate_model(fitted, X, y)["accuracy"], 4))
print("cross-validated  (k=1) :", round(cross_validated_accuracy(get_knn(n_neighbors=1), X, y)["mean"], 4))
PY
```

Actual output:

```
training accuracy (k=1): 1.0
cross-validated  (k=1) : 0.975
```

A perfect training accuracy that means nothing: every row's nearest neighbour is
itself. Only the second number was measured on rows the model had not seen.

### How much each model stores

```bash
python - <<'PY'
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.models import get_logistic_regression, get_naive_bayes
from src.preprocessing import build_preprocessor

train = pd.read_csv("data/processed/train.csv")
X, y = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]

logistic = Pipeline(
    [("preprocess", build_preprocessor()), ("model", get_logistic_regression())]
).fit(X, y).named_steps["model"]
bayes = get_naive_bayes().fit(X, y)

print("logistic coef_:", logistic.coef_.shape, "+ intercept_", logistic.intercept_.shape,
      "=", logistic.coef_.size + logistic.intercept_.size, "numbers")
print("naive bayes   :", bayes.theta_.shape, "means and", bayes.var_.shape, "variances =",
      bayes.theta_.size + bayes.var_.size, "numbers")
PY
```

Actual output:

```
logistic coef_: (22, 7) + intercept_ (22,) = 176 numbers
naive bayes   : (22, 7) means and (22, 7) variances = 308 numbers
```

KNN, by contrast, stores all 1,760 training rows.

### Why `max_iter` is 1,000

```bash
python - <<'PY'
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.models import get_logistic_regression
from src.preprocessing import build_preprocessor

train = pd.read_csv("data/processed/train.csv")
X, y = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]

scaled = Pipeline([("preprocess", build_preprocessor()), ("model", get_logistic_regression())])
print("iterations, standardised:", scaled.fit(X, y).named_steps["model"].n_iter_[0])
print("iterations, raw         :", get_logistic_regression().fit(X, y).n_iter_[0], "(the cap)")
PY
```

Actual output:

```
iterations, standardised: 52
iterations, raw         : 1000 (the cap)
```

The 1,000 is headroom, not a necessity: behind the preprocessor `lbfgs`
converges in 52 iterations, comfortably inside scikit-learn's own default of
100. On raw features it never converges and stops at whatever cap it is given —
which is what the `ConvergenceWarning` below reports.

### Scaling changes KNN's answers and not naive Bayes'

```bash
python - <<'PY'
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.models import get_knn, get_naive_bayes

train = pd.read_csv("data/processed/train.csv")
X, y = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]
standardised = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

print("KNN identical after standardising:", np.array_equal(
    get_knn().fit(X, y).predict(X), get_knn().fit(standardised, y).predict(standardised)))
print("NB  identical after standardising:", np.array_equal(
    get_naive_bayes().fit(X, y).predict(X),
    get_naive_bayes().fit(standardised, y).predict(standardised)))
PY
```

Actual output:

```
KNN identical after standardising: False
NB  identical after standardising: True
```

KNN's answer is a function of distances, and standardising changes every
distance. Naive Bayes models each feature separately, so a per-column rescale
moves every class's mean and variance for that column identically.

### The curse of dimensionality, staged

```bash
python - <<'PY'
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluation import cross_validated_accuracy
from src.models import get_knn
from src.preprocessing import build_preprocessor

train = pd.read_csv("data/processed/train.csv")
X, y = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]

rng = np.random.default_rng(42)
noise = pd.DataFrame(rng.normal(0, 1, (len(X), 100)), columns=[f"noise_{i}" for i in range(100)])
noisy = pd.concat([X.reset_index(drop=True), noise], axis=1)

clean = Pipeline([("preprocess", build_preprocessor()), ("model", get_knn())])
diluted = Pipeline([("preprocess", build_preprocessor(list(noisy.columns))), ("model", get_knn())])

print("KNN, 7 features :", round(cross_validated_accuracy(clean, X, y)["mean"], 4))
print("KNN, +100 noise :", round(cross_validated_accuracy(diluted, noisy, y)["mean"], 4))
PY
```

Actual output:

```
KNN, 7 features : 0.9653
KNN, +100 noise : 0.2233
```

The seven informative columns are still there. They have simply been buried in a
distance that is now mostly noise.

### The factories refuse nonsense

```bash
python - <<'PY'
from src.models import get_knn, get_logistic_regression, get_naive_bayes

for call in [
    lambda: get_logistic_regression(C=0),
    lambda: get_knn(n_neighbors=0),
    lambda: get_knn(weights="closest"),
    lambda: get_naive_bayes(var_smoothing=-1),
]:
    try:
        call()
    except ValueError as error:
        print("ValueError:", error)
PY
```

Actual output:

```
ValueError: `C` must be strictly positive, got 0.
ValueError: `n_neighbors` must be at least 1, got 0.
ValueError: Unsupported KNN weighting 'closest'. Choose one of: uniform, distance.
ValueError: `var_smoothing` must not be negative, got -1.
```

---

## Step 5 — Execute the Week 5 notebook

```bash
jupyter nbconvert --to notebook --execute notebooks/05_classification_models.ipynb
```

Actual output:

```
[NbConvertApp] Converting notebook notebooks/05_classification_models.ipynb to notebook
[NbConvertApp] Writing 44940 bytes to notebooks/05_classification_models.nbconvert.ipynb
```

It must exit 0 with no traceback; the run takes about seven seconds. The command
writes `notebooks/05_classification_models.nbconvert.ipynb`, which is gitignored
and can be deleted. To overwrite the committed notebook with a fresh run
instead, add `--inplace`.

The final code cell of the notebook asserts the conclusions, so a silent drift
between the prose and the numbers fails the execution rather than passing
unnoticed:

```python
assert baseline_accuracy < 0.06
assert summary.drop(index="baseline (most_frequent)")["mean"].min() > 0.90
assert best_model == "naive bayes"
assert len(train) == 1_760
```

The printed conclusion immediately above them:

```
BEST SO FAR : naive bayes at 0.9949 (99.49%)
BASELINE    : 0.0455 (4.55%)
protocol    : 5-fold stratified CV, seed 42, on data/processed/train.csv
```

---

## Step 6 — Check what was and was not produced

```bash
git status --short
ls docs/curriculum/week05/
```

Expected: `src/models/classical_models.py`, `tests/test_classical_models.py`,
`notebooks/05_classification_models.ipynb`, the four documents in
`docs/curriculum/week05/`, and the updated `src/models/__init__.py`,
`docs/ml_concepts.md`, `docs/glossary.md` and `README.md`.

Not expected, and a bug if present:

* any change to `data/raw/` or `data/processed/`;
* any file under `models/` — nothing is saved to disk until Week 9;
* any read of `data/processed/test.csv`. Confirm with:

```bash
grep -rn "read_csv" notebooks/05_classification_models.ipynb src/models/classical_models.py tests/test_classical_models.py
```

Actual output — one line, and it names `train.csv`:

```
notebooks/05_classification_models.ipynb:114:    "train = pd.read_csv(REPO_ROOT / \"data\" / \"processed\" / \"train.csv\")\n",
```

The notebook *mentions* `data/processed/test.csv` three times in prose, always to
say that it stays closed until Week 8.

---

## What "done" looks like this week

| Check | Command | Expected |
| --- | --- | --- |
| Lint clean | `ruff check .` | `All checks passed!` |
| Week 5 tests | `pytest tests/test_classical_models.py` | `57 passed` |
| Whole suite | `pytest` | `169 passed` |
| Notebook runs | `jupyter nbconvert --to notebook --execute notebooks/05_classification_models.ipynb` | exit 0, no traceback |
| Results reproduced | Step 4 | 0.9682 / 0.9653 / 0.9949 against 0.0455 |
| Test set untouched | `grep -rn "read_csv" ...` | only `train.csv` |

And, in words — the student can now:

* train and compare three classical models on identical folds;
* state that **Gaussian naive Bayes currently performs best at 99.49%**, that
  logistic regression reaches 96.82% and KNN 96.53%, and that all three beat the
  Week 4 baseline of 4.55% by more than 90 percentage points;
* explain when they would prefer KNN over logistic regression — curved or
  fragmented class boundaries, few features, plentiful data, no time to retrain
  — and when the reverse.

The student cannot yet tune hyperparameters systematically or use ensembles.
Both are a later week.

---

## Common errors at this stage

**`ModuleNotFoundError: No module named 'src'`**
You are not in the repository root, or the notebook's `sys.path` cell has not
been run. `pytest` handles this itself via `pythonpath = ["."]` in
`pyproject.toml`.

**`FileNotFoundError: data/processed/train.csv`**
Week 3's notebook has not been run in this clone. Re-run
`notebooks/03_data_preparation.ipynb`.

**`ConvergenceWarning: lbfgs failed to converge`**
Logistic regression was fitted on unscaled features, or with a smaller
`max_iter` than the project default of 1,000. Put `build_preprocessor()` in
front of it — this is the warning that motivates the pipeline.

**Scores that differ from this document by more than a rounding error**
Check, in order: the seed (42), the fold count (5), whether the preprocessor was
included, and whether `data/processed/train.csv` still has 1,760 rows.

**`NotFittedError` when calling `predict`**
The factories return unfitted estimators on purpose. Call `fit` first — or, if
this happened inside cross-validation, remember that `cross_val_score` clones
the estimator and leaves the original unfitted by design.

**A different model "wins" than the one documented**
Almost always an unfair comparison: different seeds per model, a missing
preprocessor on one of them, or a score taken from a single split rather than
cross-validation.
