# Week 7 — Validation

Run these commands in order from the repository root, with the virtual
environment from Week 1 activated. Each block shows the command and the **real
output captured from an actual run** on this repository, so you can compare
against what you see.

The two commands this week must be able to run, verbatim, are:

```bash
pytest tests/test_ensemble_models.py
jupyter nbconvert --to notebook --execute notebooks/06_model_selection.ipynb
```

Everything below expands on those two, with the output each one produced.

---

## Step 0 — Environment

`RandomForestClassifier` and `GradientBoostingClassifier` both ship with
`scikit-learn==1.6.1`, pinned in Week 1, so **the week has no required new
dependency**. `xgboost` is optional. It is listed at the end of
`requirements.txt` under its own comment, and if it is missing every factory,
test and notebook cell still works — scikit-learn's boosting takes over.

```bash
source venv/bin/activate       # Windows: venv\Scripts\activate
python -c "import sklearn, matplotlib; print(sklearn.__version__, matplotlib.__version__)"
python -c "from src.models import XGBOOST_AVAILABLE, GRADIENT_BOOSTING_BACKEND; print(XGBOOST_AVAILABLE, GRADIENT_BOOSTING_BACKEND)"
```

Actual output on the machine these notes were written on:

```
1.6.1 3.10.0
True xgboost
```

If `xgboost` did not install, you will see `False sklearn` instead, and
`get_gradient_boosting()` returns a `GradientBoostingClassifier`:

```
False sklearn
GradientBoostingClassifier(random_state=42)
```

**Both are a pass.** Nothing this week is blocked on an external package. The
only consequence is that the boosting row of the results table lands at
**0.9909** with XGBoost and **0.9869** with the scikit-learn fallback, and that
the fallback is roughly twenty times slower to cross-validate (about 33 s versus
about 1.5 s). Record which backend you are on before quoting any boosting
number — `GRADIENT_BOOSTING_BACKEND` exists so that you can.

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

This now also covers the new `src/models/ensemble_models.py`, the new
`tests/test_ensemble_models.py`, and the code cells of
`notebooks/06_model_selection.ipynb`.

---

## Step 2 — Run the test file

```bash
pytest tests/test_ensemble_models.py
```

Actual output:

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-8.3.4, pluggy-1.6.0
rootdir: /.../Crop-Recommendation-System
configfile: pyproject.toml
plugins: anyio-4.14.2
collected 61 items

tests/test_ensemble_models.py .......................................... [ 68%]
...................                                                      [100%]

======================= 61 passed, 70 warnings in 19.71s =======================
```

**All 61 tests must pass, with zero failures.** Most run on small synthetic
frames built inside the test file, so they pass with or without the CSV; the
handful needing the real 1,760 training rows are skipped (not failed) if it is
missing. The warnings come from scikit-learn's `lbfgs` solver and are unrelated
to this week.

The run is slower than earlier weeks because boosting fits are sequential by
construction. On the scikit-learn fallback expect roughly ninety seconds rather
than twenty; that is the backend, not a bug.

`pytest tests/test_ensemble_models.py -v` names them. The groups to look for:

```
tests/test_ensemble_models.py::test_get_random_forest_returns_a_random_forest PASSED [  1%]
tests/test_ensemble_models.py::test_get_gradient_boosting_returns_a_classifier_either_way PASSED [  3%]
tests/test_ensemble_models.py::test_gradient_boosting_backend_matches_the_availability_flag PASSED [  4%]
tests/test_ensemble_models.py::test_gradient_boosting_falls_back_to_sklearn_without_xgboost PASSED [  6%]
tests/test_ensemble_models.py::test_random_forest_defaults PASSED        [ 11%]
tests/test_ensemble_models.py::test_gradient_boosting_defaults PASSED    [ 13%]
...
tests/test_ensemble_models.py::test_a_forest_is_more_stable_across_resamples_than_a_single_tree PASSED
tests/test_ensemble_models.py::test_a_forest_beats_its_average_member PASSED
tests/test_ensemble_models.py::test_feature_randomness_produces_different_trees PASSED
tests/test_ensemble_models.py::test_more_trees_never_hurts_much PASSED
tests/test_ensemble_models.py::test_one_boosting_round_is_much_weaker_than_a_hundred PASSED
tests/test_ensemble_models.py::test_a_smaller_learning_rate_learns_more_slowly PASSED
tests/test_ensemble_models.py::test_feature_importances_split_credit_between_duplicated_columns PASSED
...
tests/test_ensemble_models.py::test_the_random_forest_beats_the_single_tree_it_is_made_of PASSED
tests/test_ensemble_models.py::test_the_ensembles_do_not_beat_naive_bayes_but_tie_with_it PASSED
tests/test_ensemble_models.py::test_the_ensembles_beat_every_single_model_except_naive_bayes[random_forest] PASSED
tests/test_ensemble_models.py::test_the_ensembles_beat_every_single_model_except_naive_bayes[gradient_boosting] PASSED
```

Those last three are the week's honesty tests. They encode the real result — the
ensembles beat every other single model but only *tie* with naive Bayes — so
that nobody can quietly upgrade the claim later without a test going red.

To prove the fallback really works, hide the package and run the file again:

```bash
mkdir -p /tmp/noxgb/xgboost
printf 'raise ImportError("hidden for this run")\n' > /tmp/noxgb/xgboost/__init__.py
PYTHONPATH=/tmp/noxgb pytest tests/test_ensemble_models.py
```

Actual output (tail):

```
61 passed, 70 warnings in 92.52s (0:01:32)
```

Same 61 tests, same result, different backend.

---

## Step 3 — Run the whole suite

```bash
pytest
```

Actual output (tail):

```
292 passed, 109 warnings in 24.97s
```

**292 = 20 (Week 1) + 22 (Week 2) + 32 (Week 3) + 38 (Week 4) + 119 (Weeks 5-6)
+ 61 (Week 7).** Week 7 adds a module and tests and changes no earlier
behaviour, so every earlier test must still pass.

---

## Step 4 — Reproduce the key numbers on the command line

### The extended results table

```bash
python - <<'PY'
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluation import cross_validated_accuracy
from src.models import (
    CLASSICAL_MODEL_FACTORIES,
    ENSEMBLE_MODEL_FACTORIES,
    GRADIENT_BOOSTING_BACKEND,
    get_baseline_model,
)
from src.preprocessing import build_preprocessor

train = pd.read_csv("data/processed/train.csv")
X, y = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]

print("boosting backend:", GRADIENT_BOOSTING_BACKEND)
candidates = {
    "baseline": get_baseline_model,
    **CLASSICAL_MODEL_FACTORIES,
    **ENSEMBLE_MODEL_FACTORIES,
}
for name, factory in candidates.items():
    pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", factory())])
    out = cross_validated_accuracy(pipeline, X, y)
    print(f"{name:<20} mean {out['mean']:.4f}  std {out['std']:.4f}")
PY
```

Actual output:

```
boosting backend: xgboost
baseline             mean 0.0455  std 0.0000
logistic_regression  mean 0.9682  std 0.0066
knn                  mean 0.9653  std 0.0121
naive_bayes          mean 0.9949  std 0.0042
svm                  mean 0.9790  std 0.0103
decision_tree        mean 0.9852  std 0.0068
random_forest        mean 0.9926  std 0.0058
gradient_boosting    mean 0.9909  std 0.0033
```

Read this carefully, because it is the answer to the week's headline question.
The two ensembles beat KNN, logistic regression, the RBF SVM **and** the
unlimited decision tree they are built from. They do **not** beat Gaussian naive
Bayes, which still leads at 0.9949. The forest is 0.0023 behind it — a quarter of
a percentage point, comfortably inside the ±0.0058 and ±0.0042 fold spreads. The
defensible sentence is "the ensembles are level with the leader and ahead of
everything else", not "the ensembles win".

On the scikit-learn fallback the last line reads `mean 0.9869  std 0.0034`
instead, and every statement above still holds.

### Averaging trees: the forest against the tree

```bash
python - <<'PY'
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluation import cross_validated_accuracy, evaluate_model
from src.models import get_decision_tree, get_random_forest
from src.preprocessing import build_preprocessor

train = pd.read_csv("data/processed/train.csv")
X, y = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]

def pipeline(model):
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])

for name, model in [("single tree", get_decision_tree()), ("random forest", get_random_forest())]:
    out = cross_validated_accuracy(pipeline(model), X, y)
    fitted = pipeline(model).fit(X, y)
    trained = evaluate_model(fitted, X, y)["accuracy"]
    print(f"{name:<14} cv {out['mean']:.4f} (+/- {out['std']:.4f})  train {trained:.4f}")

forest = get_random_forest().fit(X, y)
first = forest.estimators_samples_[0]
print("bootstrap draw", len(first), "| distinct rows", len(np.unique(first)),
      "| out-of-bag", len(X) - len(np.unique(first)))
roots = pd.Series(
    [FEATURE_COLUMNS[tree.tree_.feature[0]] for tree in forest.estimators_]
).value_counts()
print(roots.to_string())
print("single tree root:", FEATURE_COLUMNS[get_decision_tree().fit(X, y).tree_.feature[0]])
PY
```

Actual output:

```
single tree    cv 0.9852 (+/- 0.0068)  train 1.0000
random forest  cv 0.9926 (+/- 0.0058)  train 1.0000
bootstrap draw 1760 | distinct rows 1114 | out-of-bag 646
humidity       25
K              20
P              19
rainfall       17
N              16
temperature     3
single tree root: rainfall
```

Every mechanism from the notes is in those eight lines:

* **Bootstrap sampling.** The draw has 1,760 entries because it samples the 1,760
  training rows *with replacement*; only 1,114 of them are distinct (63.3%),
  leaving 646 rows out-of-bag for that tree.
* **Feature randomness.** One tree on its own always opens on `rainfall`. The
  forest's 100 trees open on six different features, because `max_features="sqrt"`
  offers only 2 of the 7 columns at each split and `rainfall` is often not among
  them.
* **Averaging.** Both models still score a perfect 1.0000 on their own training
  rows — the members did not stop overfitting. What changed is the validation
  column: 0.9852 to 0.9926, because averaging cancels the *uncorrelated* part of
  each tree's error.

### Forest size and boosting rounds

```bash
python - <<'PY'
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluation import cross_validated_accuracy, evaluate_model
from src.models import get_gradient_boosting, get_random_forest
from src.preprocessing import build_preprocessor

train = pd.read_csv("data/processed/train.csv")
X, y = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]

def pipeline(model):
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])

for size in [1, 3, 10, 30, 100, 300]:
    out = cross_validated_accuracy(pipeline(get_random_forest(n_estimators=size)), X, y)
    print(f"n_estimators {size:>3}: cv {out['mean']:.4f}")

print()
for rounds in [1, 2, 5, 20, 60]:
    model = get_gradient_boosting(n_estimators=rounds, max_depth=1)
    fitted = pipeline(model).fit(X, y)
    print(f"{rounds:>2} boosting rounds of stumps: train {evaluate_model(fitted, X, y)['accuracy']:.4f}")
PY
```

Actual output:

```
n_estimators   1: cv 0.9483
n_estimators   3: cv 0.9756
n_estimators  10: cv 0.9909
n_estimators  30: cv 0.9932
n_estimators 100: cv 0.9926
n_estimators 300: cv 0.9932

 1 boosting rounds of stumps: train 0.6193
 2 boosting rounds of stumps: train 0.6710
 5 boosting rounds of stumps: train 0.6733
20 boosting rounds of stumps: train 0.9307
60 boosting rounds of stumps: train 0.9881
```

The two blocks are the two families side by side. The forest curve climbs fast
and then **flattens** — after about thirty trees, more members buy noise
reduction that has already happened, which is why `n_estimators` is a compute
budget rather than a dial with a bad end. The boosting curve climbs from a
depth-1 stump that gets 61.9% of the training rows right to 98.8% after sixty
rounds, each round fitted on what the previous chain still got wrong.

Note what the second block measures: **training** accuracy. That is legitimate
for showing the mechanism and illegitimate as a result, which is exactly the
distinction Week 6 drew for tree depth. **Neither block chooses a setting.**
Picking 30 or 300 by eye from a validation curve is choosing a hyperparameter
without a protocol, and the protocol is a later week.

### Feature importances, and where they mislead

```bash
python - <<'PY'
import pandas as pd

from src.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.models import get_random_forest

train = pd.read_csv("data/processed/train.csv")
X, y = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]

forest = get_random_forest().fit(X, y)
importances = pd.Series(forest.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
print(importances.round(4).to_string())
print("sum:", round(float(importances.sum()), 6))

print()
X_duplicated = X.assign(humidity_copy=X["humidity"])
shared = pd.Series(get_random_forest().fit(X_duplicated, y).feature_importances_,
                   index=X_duplicated.columns)
print(f"humidity      {shared['humidity']:.4f}")
print(f"humidity_copy {shared['humidity_copy']:.4f}")
print(f"together      {shared['humidity'] + shared['humidity_copy']:.4f}"
      f"   (was {importances['humidity']:.4f})")
PY
```

Actual output:

```
rainfall       0.2302
humidity       0.2242
K              0.1754
P              0.1508
N              0.0964
temperature    0.0724
ph             0.0506
sum: 1.0

humidity      0.1485
humidity_copy 0.1370
together      0.2855   (was 0.2242)
```

The first block is what the notebook plots: seven non-negative numbers summing
to 1, with `rainfall` and `humidity` on top and `ph` last — the same two columns
Week 2's exploration singled out and Week 6's decision tree split on first.

The second block is the caveat, and it is worth more than the ranking. Adding an
exact copy of `humidity` does not give the forest one scrap of new information,
and training accuracy does not move. But `humidity` drops from 0.2242 to 0.1485
because the trees now sometimes reach for the copy instead. **Importance is
split between correlated columns**, so a low score can mean "this feature is
useless" *or* "a twin took the credit". `P` and `K` correlate at 0.74 on this
dataset, so this is not a hypothetical. That, plus the fact that these numbers
are computed on the training data and are biased towards features with many
possible split points, is why Week 8 replaces them with permutation importance
and SHAP.

### The registry, and the factories refusing nonsense

```bash
python - <<'PY'
from src.models import ENSEMBLE_MODEL_FACTORIES, get_gradient_boosting, get_random_forest

for name, factory in ENSEMBLE_MODEL_FACTORIES.items():
    print(f"{name:<18} {factory()}")

print()
for call in [
    lambda: get_random_forest(n_estimators=0),
    lambda: get_random_forest(max_depth=0),
    lambda: get_random_forest(max_features="all"),
    lambda: get_random_forest(max_features=0),
    lambda: get_gradient_boosting(n_estimators=0),
    lambda: get_gradient_boosting(learning_rate=0),
    lambda: get_gradient_boosting(max_depth=0),
]:
    try:
        call()
    except ValueError as error:
        print("ValueError:", error)
PY
```

Actual output:

```
random_forest      RandomForestClassifier(random_state=42)
gradient_boosting  XGBoostStringLabelClassifier()

ValueError: `n_estimators` must be at least 1, got 0.
ValueError: `max_depth` must be at least 1 or None, got 0.
ValueError: Unsupported `max_features` 'all'. Choose one of: sqrt, log2, an int, a float, or None.
ValueError: `max_features` as a number must be positive, got 0.
ValueError: `n_estimators` must be at least 1, got 0.
ValueError: `learning_rate` must be strictly positive, got 0.
ValueError: `max_depth` must be at least 1, got 0.
```

The second registry line reads `GradientBoostingClassifier(random_state=42)` on
the fallback. Both estimators are unfitted and seeded, exactly like the five
Week 5-6 factories, and both refuse bad arguments at construction time rather
than fifteen seconds into a cross-validation run.

---

## Step 5 — Execute the notebook

```bash
jupyter nbconvert --to notebook --execute notebooks/06_model_selection.ipynb
```

Actual output:

```
[NbConvertApp] Converting notebook notebooks/06_model_selection.ipynb to notebook
[NbConvertApp] Writing 109335 bytes to notebooks/06_model_selection.nbconvert.ipynb
```

It must exit 0 with no traceback. The run takes about nineteen seconds with
XGBoost and a few minutes on the scikit-learn fallback, because the forest-size
and boosting-round sweeps are all fitted live. The command writes
`notebooks/06_model_selection.nbconvert.ipynb`, which is gitignored and can be
deleted. To overwrite the committed notebook with a fresh run instead, add
`--inplace`.

The final code cell asserts its own conclusions, so the prose and the numbers
cannot drift apart silently:

```python
assert forest_mean > tree_mean                      # bagging beats its own member
assert boosting_mean > tree_mean                    # so does boosting
assert forest_mean > summary.loc["svm (rbf, C=1)", "mean"]
assert boosting_mean > summary.loc["knn (k=5)", "mean"]
assert naive_bayes_mean >= forest_mean              # the leader is unchanged
assert naive_bayes_mean - forest_mean < summary.loc["naive bayes", "std"] + summary.loc[
    "random forest (100 trees)", "std"
]                                                   # ...but only inside the fold spread
assert forest.score(X_train, y_train) == 1.0        # the members still memorise
assert round(importances.sum(), 6) == 1.0           # importances are normalised
assert len(train) == 1_760                          # the test set was never touched
```

The printed conclusion immediately above them:

```
RANDOM FOREST    : 0.9926 (99.26%)
GRADIENT BOOSTING: 0.9909 (99.09%)  [backend: xgboost]
BEST SO FAR      : naive bayes
protocol         : 5-fold stratified CV, seed 42, on data/processed/train.csv
```

Note that the assertions are written to hold on **either** backend, and that
they encode the tie rather than a win. Three figures must appear in the executed
notebook: the forest-size curve, the boosting-rounds curve, and the horizontal
bar chart of the random forest's `feature_importances_`.

---

## Step 6 — Check what was and was not produced

```bash
git status --short
ls docs/curriculum/week07/
```

Expected: the new `src/models/ensemble_models.py`, the new
`tests/test_ensemble_models.py`, the new `notebooks/06_model_selection.ipynb`,
the four documents in `docs/curriculum/week07/`, and the updated
`src/models/__init__.py`, `requirements.txt`, `docs/ml_concepts.md`,
`docs/glossary.md` and `README.md`.

Not expected, and a bug if present:

* any change to `data/raw/` or `data/processed/`;
* any file under `models/` — nothing is saved to disk until Week 9;
* any search object (`GridSearchCV`, `RandomizedSearchCV`) or any Week 8 tool
  (`permutation_importance`, `shap`, `confusion_matrix`) — all out of scope;
* any read of `data/processed/test.csv`. Confirm with:

```bash
grep -rn "read_csv" notebooks/06_model_selection.ipynb src/models/ensemble_models.py \
  tests/test_ensemble_models.py
```

Actual output — one line, and it names `train.csv`:

```
notebooks/06_model_selection.ipynb:137:    "train = pd.read_csv(REPO_ROOT / \"data\" / \"processed\" / \"train.csv\")\n",
```

---

## What "done" looks like this week

| Check | Command | Expected |
| --- | --- | --- |
| Lint clean | `ruff check .` | `All checks passed!` |
| Week 7 tests | `pytest tests/test_ensemble_models.py` | `61 passed` |
| Whole suite | `pytest` | `292 passed` |
| Backend recorded | `python -c "from src.models import GRADIENT_BOOSTING_BACKEND; print(GRADIENT_BOOSTING_BACKEND)"` | `xgboost` **or** `sklearn` — both pass |
| Fallback works | `PYTHONPATH=/tmp/noxgb pytest tests/test_ensemble_models.py` | `61 passed` |
| Notebook runs | `jupyter nbconvert --to notebook --execute notebooks/06_model_selection.ipynb` | exit 0, no traceback |
| Results reproduced | Step 4 | forest 0.9926, boosting 0.9909, naive Bayes still 0.9949 |
| Importances reproduced | Step 4 | seven non-negative numbers summing to 1, `rainfall` first |
| Test set untouched | `grep -rn "read_csv" ...` | only `train.csv` |

And, in words — the student can now:

* **explain bagging versus boosting in their own words**: bagging fits many deep,
  overfit trees in parallel on bootstrap samples of the rows and a random subset
  of the columns, then votes — the members' independent mistakes cancel, so it
  attacks *variance*; boosting fits shallow, deliberately weak trees in sequence,
  each one trained on what the chain so far still gets wrong and added with a
  small `learning_rate`, so it attacks *bias*;
* **read a feature-importance plot**: the seven bars are non-negative, sum to 1,
  and say how much each column reduced impurity across the fitted forest's
  splits — `rainfall` 0.23 and `humidity` 0.22 lead, `ph` 0.05 trails — while
  remembering the three caveats (training data, cardinality bias, credit split
  between correlated columns) that make this a description of *this fitted
  model*, not a scientific claim about the crops;
* **state whether the ensembles beat the Week 5 and 6 models**: yes against KNN,
  logistic regression, the RBF SVM and the unlimited tree; no against Gaussian
  naive Bayes, which still leads at 0.9949 to the forest's 0.9926 — a gap inside
  the fold spread, so the honest verdict is a tie at the top.

The student cannot yet tune hyperparameters systematically — every sweep here
was drawn to explain a mechanism, and none of them picked a value — nor explain
an individual prediction. `GridSearchCV` and permutation importance/SHAP both
come next.

---

## Common errors at this stage

**`ModuleNotFoundError: No module named 'src.models.ensemble_models'`**
The file is new this week. Check it exists and that you are in the repository
root; `pytest` handles the path itself via `pythonpath = ["."]` in
`pyproject.toml`.

**`ModuleNotFoundError: No module named 'xgboost'`**
Not an error. `xgboost` is optional; `get_gradient_boosting()` already fell back
to scikit-learn and `GRADIENT_BOOSTING_BACKEND` says `sklearn`. Install it with
`pip install xgboost==2.1.3` only if you want the faster backend.

**`ValueError: Invalid classes inferred from unique values of y`**
XGBoost's own `XGBClassifier` demands integer labels, and this dataset's target
is crop names. That is why `get_gradient_boosting()` returns
`XGBoostStringLabelClassifier`, a thin wrapper that label-encodes on `fit` and
decodes on `predict`. If you see this message you built a raw `XGBClassifier`
yourself — use the factory.

**Boosting numbers that do not match this document**
Check `GRADIENT_BOOSTING_BACKEND` first. XGBoost and scikit-learn implement
different boosting algorithms; 0.9909 and 0.9869 are both correct, for different
backends. Always quote the backend alongside the score.

**`pytest` seems to hang**
It is fitting boosting chains, which are sequential and cannot be parallelised
away. Twenty seconds with XGBoost, around ninety on the fallback. Use
`pytest tests/test_ensemble_models.py -v` to watch the progress.

**The forest scores 1.0 under cross-validation**
Almost certainly leakage: `data/processed/train.csv` also carries a
`label_encoded` column, and `train.drop(columns=["label"])` leaves the answer in
the features. Always select `train[list(FEATURE_COLUMNS)]`.

**"My ensembles beat naive Bayes"**
Check the seed (42), the fold count (5) and that the preprocessor was in the
pipeline for every row of the table. On these folds the forest lands 0.0023
behind. If a re-run flips the order, that is the point of the fold spread — the
difference is not resolvable at this sample size, which is a result worth
reporting rather than a problem to fix.

**Importances that disagree with the ranking above**
`feature_importances_` is computed on whatever frame you fitted. Refitting on a
different column set, a subset of rows, or with a different seed moves the
numbers; a duplicated or highly correlated column moves them a lot. Fit on all
1,760 rows and the seven `FEATURE_COLUMNS` to reproduce this table.
