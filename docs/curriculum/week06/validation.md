# Week 6 — Validation

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

No new dependencies this week. `scikit-learn==1.6.1` and `matplotlib==3.10.0`
were both pinned in Week 1, and `SVC`, `DecisionTreeClassifier` and `export_text`
all live in scikit-learn.

```bash
source venv/bin/activate       # Windows: venv\Scripts\activate
python -c "import sklearn, matplotlib; print(sklearn.__version__, matplotlib.__version__)"
```

Actual output:

```
1.6.1 3.10.0
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

This now also covers `src/utils/visualization.py`, the two new factories in
`src/models/classical_models.py`, and the Part 2 code cells of
`notebooks/05_classification_models.ipynb`.

---

## Step 2 — Run the test file

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
collected 119 items

tests/test_classical_models.py ......................................... [ 34%]
........................................................................ [ 94%]
......                                                                   [100%]

======================= 119 passed, 27 warnings in 3.49s =======================
```

**All 119 tests must pass, with zero failures.** The file held 57 after Week 5;
Week 6 adds 62. Most run on small synthetic frames built inside the test file, so
they pass with or without the CSV; the handful needing the real 2,200 rows are
skipped (not failed) if it is missing. The warnings come from scikit-learn's
`lbfgs` solver options and are unrelated to this week.

`pytest tests/test_classical_models.py -v` names them. The Week 6 groups to look
for:

```
tests/test_classical_models.py::test_get_svm_returns_an_svc PASSED       [ 22%]
tests/test_classical_models.py::test_get_decision_tree_returns_a_decision_tree_classifier PASSED [ 23%]
tests/test_classical_models.py::test_svm_defaults PASSED                 [ 24%]
tests/test_classical_models.py::test_decision_tree_defaults PASSED       [ 25%]
tests/test_classical_models.py::test_svm_rejects_an_unsupported_kernel[gaussian] PASSED [ 26%]
tests/test_classical_models.py::test_svm_accepts_every_supported_kernel[linear] PASSED [ 33%]
tests/test_classical_models.py::test_svm_accepts_every_supported_kernel[rbf] PASSED [ 34%]
tests/test_classical_models.py::test_decision_tree_accepts_every_supported_criterion[gini] PASSED [ 36%]
tests/test_classical_models.py::test_decision_tree_rejects_a_non_positive_depth[0] PASSED [ 39%]
...
tests/test_classical_models.py::test_the_svm_depends_only_on_its_support_vectors PASSED
tests/test_classical_models.py::test_an_rbf_kernel_solves_a_problem_a_linear_one_cannot PASSED
tests/test_classical_models.py::test_a_larger_svm_c_fits_the_training_data_harder PASSED
tests/test_classical_models.py::test_an_unlimited_decision_tree_memorises_its_training_data PASSED
tests/test_classical_models.py::test_a_stump_underfits PASSED
tests/test_classical_models.py::test_tree_depth_controls_the_train_validation_gap PASSED
tests/test_classical_models.py::test_the_decision_tree_is_insensitive_to_feature_scaling PASSED
...
tests/test_classical_models.py::test_plot_decision_boundary_returns_axes_labelled_with_the_two_features PASSED
tests/test_classical_models.py::test_plot_decision_boundary_needs_a_fitted_model PASSED
tests/test_classical_models.py::test_naive_bayes_is_still_the_best_model_so_far PASSED
tests/test_classical_models.py::test_the_tree_overfits_the_real_training_data_when_it_is_unlimited PASSED
```

---

## Step 3 — Run the whole suite

```bash
pytest
```

Actual output (tail):

```
231 passed, 39 warnings in 5.09s
```

**231 = 20 (Week 1) + 22 (Week 2) + 32 (Week 3) + 38 (Week 4) + 119 (Weeks 5-6,
in one file).** Week 6 adds tests and changes no earlier behaviour, so every
earlier test must still pass.

---

## Step 4 — Reproduce the key numbers on the command line

### The extended results table

```bash
python - <<'PY'
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluation import cross_validated_accuracy
from src.models import CLASSICAL_MODEL_FACTORIES, get_baseline_model
from src.preprocessing import build_preprocessor

train = pd.read_csv("data/processed/train.csv")
X, y = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]

candidates = {"baseline": get_baseline_model, **CLASSICAL_MODEL_FACTORIES}
for name, factory in candidates.items():
    pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", factory())])
    out = cross_validated_accuracy(pipeline, X, y)
    print(f"{name:<20} mean {out['mean']:.4f}  std {out['std']:.4f}")
PY
```

Actual output:

```
baseline             mean 0.0455  std 0.0000
logistic_regression  mean 0.9682  std 0.0066
knn                  mean 0.9653  std 0.0121
naive_bayes          mean 0.9949  std 0.0042
svm                  mean 0.9790  std 0.0103
decision_tree        mean 0.9852  std 0.0068
```

These are the week's headline numbers, and the answer to "which model performs
best so far" is the third line: **Gaussian naive Bayes, 99.49%**. The two new
models land second and third. If yours differ in the fourth decimal place, check
that you used the seed-42 folds (the default) and the preprocessor; if they
differ in the second, something is wrong with the data.

### Depth, overfitting, and the generalisation gap

```bash
python - <<'PY'
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluation import cross_validated_accuracy, evaluate_model
from src.models import get_decision_tree
from src.preprocessing import build_preprocessor

train = pd.read_csv("data/processed/train.csv")
X, y = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]

def pipeline(model):
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])

for depth in [1, 3, 5, 10, 15, None]:
    fitted = pipeline(get_decision_tree(max_depth=depth)).fit(X, y)
    trained = evaluate_model(fitted, X, y)["accuracy"]
    validated = cross_validated_accuracy(pipeline(get_decision_tree(max_depth=depth)), X, y)["mean"]
    print(f"depth {str(depth):>4}: train {trained:.4f}  validation {validated:.4f}"
          f"  gap {trained - validated:+.4f}")
PY
```

Actual output:

```
depth    1: train 0.0909  validation 0.0909  gap +0.0000
depth    3: train 0.2273  validation 0.2261  gap +0.0011
depth    5: train 0.4091  validation 0.4074  gap +0.0017
depth   10: train 0.9818  validation 0.9750  gap +0.0068
depth   15: train 0.9983  validation 0.9852  gap +0.0131
depth None: train 1.0000  validation 0.9852  gap +0.0148
```

This is the week's central table, and the plot in notebook §11 is a picture of
it. Read it from both ends:

* **Depth 1-5: high bias.** Terrible scores, and *identical* on both columns. The
  model is too simple to express the answer, so it fails equally on rows it has
  seen and rows it has not.
* **Depth 15 and unlimited: high variance.** Training accuracy reaches a perfect
  1.0000 while validation stops at 0.9852 and does not move again. That
  1.5-point gap is training-set accuracy that does not exist on new data.

### The two kernels, and what a support vector is

```bash
python - <<'PY'
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluation import cross_validated_accuracy
from src.models import get_svm
from src.preprocessing import build_preprocessor

train = pd.read_csv("data/processed/train.csv")
X, y = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]

def pipeline(model):
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])

for kernel in ["linear", "rbf"]:
    out = cross_validated_accuracy(pipeline(get_svm(kernel=kernel)), X, y)
    fitted = pipeline(get_svm(kernel=kernel)).fit(X, y).named_steps["model"]
    print(f"{kernel:>6}: cv {out['mean']:.4f} (+/- {out['std']:.4f}),"
          f" support vectors {fitted.n_support_.sum()}")
PY
```

Actual output:

```
linear: cv 0.9818 (+/- 0.0077), support vectors 615
   rbf: cv 0.9790 (+/- 0.0103), support vectors 943
```

The curved boundary is **not** better here, and the difference is well inside
either fold spread. Week 2 explains why: the crops occupy compact, well-separated
regions, which a flat boundary already separates. A kernel is worth paying for
when classes interleave or one encloses another.

### `C` and the margin

```bash
python - <<'PY'
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluation import cross_validated_accuracy
from src.models import get_svm
from src.preprocessing import build_preprocessor

train = pd.read_csv("data/processed/train.csv")
X, y = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]

def pipeline(model):
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])

for c_value in [0.01, 0.1, 1.0, 10.0, 100.0]:
    fitted = pipeline(get_svm(C=c_value)).fit(X, y).named_steps["model"]
    mean = cross_validated_accuracy(pipeline(get_svm(C=c_value)), X, y)["mean"]
    print(f"C = {c_value:>6}: support vectors {fitted.n_support_.sum():>4}  cv {mean:.4f}")
PY
```

Actual output:

```
C =   0.01: support vectors 1760  cv 0.8716
C =    0.1: support vectors 1608  cv 0.9352
C =    1.0: support vectors  943  cv 0.9790
C =   10.0: support vectors  640  cv 0.9824
C =  100.0: support vectors  612  cv 0.9818
```

At `C = 0.01` violations are almost free, the margin swallows every one of the
1,760 rows, and accuracy falls to 87.2% — underfitting. As `C` grows the margin
tightens and the support-vector count falls towards 600. **None of these values
is adopted:** `C = 1.0` remains the default, because picking the best row of this
table by eye is choosing a hyperparameter from validation scores without a
protocol. That is a later week's subject.

### The tree's first questions

```bash
python - <<'PY'
import pandas as pd
from sklearn.tree import export_text

from src.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.models import get_decision_tree

train = pd.read_csv("data/processed/train.csv")
X, y = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]

print(export_text(get_decision_tree(max_depth=2).fit(X, y), feature_names=list(FEATURE_COLUMNS)))
tree = get_decision_tree().fit(X, y)
print("unlimited tree: depth", tree.get_depth(), "| leaves", tree.get_n_leaves())
PY
```

Actual output:

```
|--- rainfall <= 30.18
|   |--- class: muskmelon
|--- rainfall >  30.18
|   |--- humidity <= 27.98
|   |   |--- class: chickpea
|   |--- humidity >  27.98
|   |   |--- class: apple

unlimited tree: depth 17 | leaves 38
```

Readable rules are what no other model in this project produces, and they are why
`humidity` and `rainfall` are the two features the notebook plots boundaries on.

### The decision boundary helper

```bash
python - <<'PY'
import matplotlib
matplotlib.use("Agg")

import pandas as pd

from src.models import get_decision_tree
from src.utils.visualization import plot_decision_boundary

train = pd.read_csv("data/processed/train.csv")
X_2d, y = train[["humidity", "rainfall"]], train["label"]
model = get_decision_tree(max_depth=4).fit(X_2d, y)

ax = plot_decision_boundary(model, X_2d, y, resolution=120)
print("axes title :", ax.get_title())
print("x label    :", ax.get_xlabel(), "| y label:", ax.get_ylabel())

try:
    plot_decision_boundary(model, train[["humidity", "rainfall", "ph"]], y)
except ValueError as error:
    print("ValueError:", error)
PY
```

Actual output:

```
axes title : Decision boundary on humidity and rainfall
x label    : humidity | y label: rainfall
ValueError: `X_2d` must have exactly two columns to be plottable, got 3.
```

The helper draws and returns axes; it never calls `plt.show()`, never writes a
file and never fits anything, which is what lets the tests run it head-lessly.
Passing a model that has not been fitted raises `NotFittedError` from
scikit-learn.

### The registry now lists five models

```bash
python -c "
from src.models import CLASSICAL_MODEL_FACTORIES
for name, factory in CLASSICAL_MODEL_FACTORIES.items():
    print(f'{name:<20} {factory()}')"
```

Actual output:

```
logistic_regression  LogisticRegression(max_iter=1000, random_state=42)
knn                  KNeighborsClassifier()
naive_bayes          GaussianNB()
svm                  SVC(random_state=42)
decision_tree        DecisionTreeClassifier(random_state=42)
```

### The new factories refuse nonsense

```bash
python - <<'PY'
from src.models import get_decision_tree, get_svm

for call in [
    lambda: get_svm(kernel="gaussian"),
    lambda: get_svm(C=0),
    lambda: get_svm(gamma=-1),
    lambda: get_decision_tree(max_depth=0),
    lambda: get_decision_tree(criterion="variance"),
    lambda: get_decision_tree(min_samples_leaf=0),
]:
    try:
        call()
    except ValueError as error:
        print("ValueError:", error)
PY
```

Actual output:

```
ValueError: Unsupported SVM kernel 'gaussian'. Choose one of: linear, rbf, poly, sigmoid.
ValueError: `C` must be strictly positive, got 0.
ValueError: `gamma` as a number must be strictly positive, got -1.
ValueError: `max_depth` must be at least 1 or None, got 0.
ValueError: Unsupported tree criterion 'variance'. Choose one of: gini, entropy, log_loss.
ValueError: `min_samples_leaf` must be at least 1, got 0.
```

---

## Step 5 — Execute the notebook

```bash
jupyter nbconvert --to notebook --execute notebooks/05_classification_models.ipynb
```

Actual output:

```
[NbConvertApp] Converting notebook notebooks/05_classification_models.ipynb to notebook
[NbConvertApp] Writing 507262 bytes to notebooks/05_classification_models.nbconvert.ipynb
```

It must exit 0 with no traceback; the run takes about fifteen seconds now that
Part 2 fits a `C` sweep, a depth sweep and three two-feature models. The command
writes `notebooks/05_classification_models.nbconvert.ipynb`, which is gitignored
and can be deleted. To overwrite the committed notebook with a fresh run instead,
add `--inplace`.

Part 2's final code cell asserts its own conclusions, so prose and numbers cannot
drift apart silently:

```python
assert best_model_v2 == "naive bayes"                       # still the leader
assert summary_v2.loc["decision tree (unlimited)", "mean"] > summary_v2.loc["knn (k=5)", "mean"]
assert summary_v2.loc["svm (rbf, C=1)", "mean"] > summary_v2.loc["knn (k=5)", "mean"]
assert tree_train_accuracy == 1.0                           # the unlimited tree memorises
assert depth_sweep.loc[1, "gap"] < depth_sweep["gap"].max()  # the gap opens with depth
assert len(train) == 1_760                                  # the test set was never touched
```

The printed conclusion immediately above them:

```
BEST SO FAR   : naive bayes at 0.9949 (99.49%)
NEW THIS WEEK : svm 0.9790, tree 0.9852
BASELINE      : 0.0455 (4.55%)
protocol      : 5-fold stratified CV, seed 42, on data/processed/train.csv
```

Two figures must appear in the executed notebook: the two-panel
training-versus-validation curve over `max_depth` (§11) and the three decision
boundaries on `humidity` and `rainfall` (§12).

---

## Step 6 — Check what was and was not produced

```bash
git status --short
ls docs/curriculum/week06/
```

Expected: the two new factories in `src/models/classical_models.py`, the new
`src/utils/visualization.py`, the extended `tests/test_classical_models.py`, Part
2 in `notebooks/05_classification_models.ipynb`, the four documents in
`docs/curriculum/week06/`, and the updated `src/models/__init__.py`,
`src/utils/__init__.py`, `tests/conftest.py`, `docs/ml_concepts.md`,
`docs/glossary.md` and `README.md`.

Not expected, and a bug if present:

* any change to `data/raw/` or `data/processed/`;
* any file under `models/` — nothing is saved to disk until Week 9;
* any ensemble estimator (`RandomForestClassifier`, `GradientBoostingClassifier`,
  `VotingClassifier`) or any search object (`GridSearchCV`, `RandomizedSearchCV`)
  — both are out of scope this week;
* any read of `data/processed/test.csv`. Confirm with:

```bash
grep -rn "read_csv" notebooks/05_classification_models.ipynb src/models/classical_models.py \
  src/utils/visualization.py tests/test_classical_models.py
```

Actual output — one line, and it names `train.csv`:

```
notebooks/05_classification_models.ipynb:118:    "train = pd.read_csv(REPO_ROOT / \"data\" / \"processed\" / \"train.csv\")\n",
```

---

## What "done" looks like this week

| Check | Command | Expected |
| --- | --- | --- |
| Lint clean | `ruff check .` | `All checks passed!` |
| Week 5-6 tests | `pytest tests/test_classical_models.py` | `119 passed` |
| Whole suite | `pytest` | `231 passed` |
| Notebook runs | `jupyter nbconvert --to notebook --execute notebooks/05_classification_models.ipynb` | exit 0, no traceback |
| Results reproduced | Step 4 | 0.9790 (SVM) / 0.9852 (tree) beside 0.9949 (naive Bayes) |
| Overfitting reproduced | Step 4 | train 1.0000 vs. validation 0.9852 for the unlimited tree |
| Test set untouched | `grep -rn "read_csv" ...` | only `train.csv` |

And, in words — the student can now:

* **explain overfitting using the tree-depth plot they generated**: the two
  curves sit together and low while the tree is too shallow to express the
  answer, rise together while it is learning real structure, and then separate —
  training to a perfect 100%, validation stuck at 98.52% — once the extra depth
  is only being spent on memorising this particular sample;
* **explain what an SVM kernel does in one sentence**: it lets the SVM draw a
  curved boundary by measuring the similarity between rows rather than their
  positions, fitting a flat boundary in a higher-dimensional space it never
  actually computes;
* **name the best-performing model so far**: Gaussian naive Bayes, 99.49% under
  5-fold stratified cross-validation, with the new decision tree second at 98.52%
  and the new SVM third at 97.90%.

The student cannot yet combine models into an ensemble or search hyperparameters
systematically. Both come later in the course.

---

## Common errors at this stage

**`ModuleNotFoundError: No module named 'src.utils.visualization'`**
The file is new this week. Check it exists and that you are in the repository
root; `pytest` handles the path itself via `pythonpath = ["."]` in
`pyproject.toml`.

**`NotFittedError` from `plot_decision_boundary`**
The helper never fits anything — that is deliberate, so it cannot quietly train a
model you then read a score from. Fit the model on the two columns first.

**`ValueError: X has 7 features, but DecisionTreeClassifier is expecting 2`**
The model passed to `plot_decision_boundary` was fitted on all seven features.
Fit a separate two-feature model for the picture, as notebook §12 does, and
remember that its accuracy is an illustration, not a result.

**`AttributeError: predict_proba is not available when probability=False`**
scikit-learn's SVM only reports probabilities when it has fitted the extra
internal calibration. Use `get_svm(probability=True)` if you genuinely need them,
and expect a slower fit.

**The tree gives a different structure on a re-run**
Ties between equally good splits are broken randomly. `get_decision_tree()`
passes `random_state=42` for exactly this reason; if you construct
`DecisionTreeClassifier()` yourself without a seed, expect drift.

**A depth sweep that shows no overfitting at all**
Check that the training accuracy is measured on a model fitted on *all* the
training rows and the validation accuracy comes from `cross_validated_accuracy`.
Scoring both with cross-validation gives two nearly identical curves and hides
the very effect the plot exists to show.

**Scores that differ from this document by more than a rounding error**
Check, in order: the seed (42), the fold count (5), whether the preprocessor was
included, and whether `data/processed/train.csv` still has 1,760 rows.
