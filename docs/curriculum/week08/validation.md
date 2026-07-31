# Week 8 — Validation

Run these in order from the repository root. Each step lists the exact command
and the output captured from a real run on 2026-07-31 (Python 3.12.3, the pinned
`requirements.txt`, plus the optional `shap==0.46.0`).

Numbers that come from a seeded computation should match yours exactly. Timings
will not.

---

## Step 0 — Environment

```bash
python --version
pip install -r requirements.txt
```

The pins that matter this week:

```
numpy==2.2.1
pandas==2.2.3
scikit-learn==1.6.1
matplotlib==3.10.0
xgboost==2.1.3        # optional, Week 7
shap==0.46.0          # optional, Week 8
```

### The optional SHAP install

`shap` is listed at the bottom of `requirements.txt`, commented as optional, for
the same reason `xgboost` was in Week 7: it must never be able to break the
environment.

```bash
pip install shap==0.46.0
python -c "import shap; print(shap.__version__)"
```

```
0.46.0
```

> **Do not install unpinned `shap`.** The current release force-upgrades
> `numpy`, `pandas` and `scikit-learn` past this project's pins (observed:
> numpy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0), which changes results
> everywhere else in the course. `0.46.0` is the version that resolves against
> the existing pins. If a resolver ever refuses, skip the install — everything
> below still passes.

Check which backend the code will use:

```bash
python -c "from src.evaluation import SHAP_AVAILABLE, EXPLAINER_BACKEND; print(SHAP_AVAILABLE, EXPLAINER_BACKEND)"
```

```
True shap
```

**This is the configuration Week 8 was written and executed in: SHAP was used,
not the fallback.** Step 6 verifies the fallback separately.

---

## Step 1 — Tests

```bash
pytest tests/test_tuning.py tests/test_explainability.py
```

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-8.3.4, pluggy-1.5.0
rootdir: /home/runner/work/Crop-Recommendation-System/Crop-Recommendation-System
configfile: pyproject.toml
testpaths: tests
collected 54 items

tests/test_tuning.py ......................                              [ 40%]
tests/test_explainability.py ...........................s....            [100%]

=============================== warnings summary ===============================
../../../.local/lib/python3.12/site-packages/shap/plots/colors/_colorconv.py:819
  909 warnings
  DeprecationWarning: Converting `np.inexact` or `np.floating` to a dtype is
  deprecated. The current result is `float64` which is not strictly correct.

================= 53 passed, 1 skipped, 909 warnings in 4.12s ==================
```

Three things to notice:

* **54 collected, 53 passed, 1 skipped.** The skip is
  `test_insisting_on_shap_without_shap_fails_clearly`, which only has anything to
  assert when `shap` is *absent*. With `shap` installed it is skipped; without,
  it runs and a different one is skipped instead. Either way the count is 53/1.
* **909 warnings, all from inside `shap`.** `shap 0.46.0`'s colour module uses a
  NumPy idiom deprecated in 2.x. They come from the library, not from this
  repository, and there is nothing to fix here.
* **~4 seconds.** These are smoke tests: they confirm the code runs, returns the
  documented keys and holds its invariants on small models. They do **not** check
  SHAP values against hand-computed Shapley values — that would be testing the
  library, not this project.

The whole suite:

```bash
pytest
```

```
================ 345 passed, 1 skipped, 1018 warnings in 27.56s ================
```

292 before this week, 54 added, 346 collected.

Lint:

```bash
ruff check .
```

```
All checks passed!
```

`ruff` lints the notebooks too, so a stray f-string or an unused import in a cell
fails this.

---

## Step 2 — The library, by hand

Every number below comes from `/tmp/demo.py`-style calls you can paste into a
REPL. Set up once:

```python
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from src.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.preprocessing import build_preprocessor
from src.evaluation import (
    confusion_frame, evaluate_model, explain_prediction,
    permutation_feature_importance, tune_model,
)

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")
X_train, y_train = train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN]
X_test, y_test = test[list(FEATURE_COLUMNS)], test[TARGET_COLUMN]

model = Pipeline([("preprocess", build_preprocessor()), ("model", GaussianNB())])
model.fit(X_train, y_train)
```

### 2a — The new metrics

```python
result = evaluate_model(model, X_test, y_test)
print(result["accuracy"], result["macro_f1"], result["weighted_f1"])
```

```
accuracy    0.9955
macro_f1    0.9954
weighted_f1 0.9954
```

Macro and weighted agree because the test set holds exactly 20 rows of every
crop. On an imbalanced set they would not, and the gap between them would be the
story.

### 2b — The confusion matrix

```python
matrix = confusion_frame(y_test, model.predict(X_test))
matrix.shape                                # (22, 22)
matrix.to_numpy().diagonal().sum()          # 438
```

```
shape (22, 22) diagonal 438
off-diagonal: [('rice', 'jute', 2)]
```

438 of 440 correct, and **every** mistake in one cell: two fields whose true crop
was rice were predicted as jute. 462 off-diagonal cells exist; 461 are zero.

### 2c — A search that changes nothing

```python
tuned = tune_model(model, {"model__var_smoothing": [1e-11, 1e-9, 1e-7]}, X_train, y_train)
```

```
best_params {'model__var_smoothing': 1e-11} best_score 0.994886 candidates 3 fits 15
distinct scores: 1
```

`n_fits = n_candidates x n_splits = 3 x 5`. All three candidates score
**identically** — `mean_test_score.nunique()` is 1 — so `best_params` is just the
first one encountered. The notebook runs the full twelve-value grid from `1e-11`
to `1e-6` and gets the same single score. `var_smoothing` does nothing on this
data.

### 2d — Permutation importance

```python
permutation_feature_importance(model, X_test, y_test, n_repeats=10)
```

```
             importance_mean  importance_std
feature
humidity              0.4507          0.0153
K                     0.4336          0.0125
rainfall              0.3589          0.0140
P                     0.1811          0.0163
N                     0.1530          0.0133
temperature           0.0832          0.0116
ph                    0.0518          0.0024
```

Units are *accuracy lost*: scrambling `humidity` costs 45 points, from 99.5% to
about 55%. `GaussianNB` has no `feature_importances_` at all, which is exactly
why this method is the one that generalises.

### 2e — One prediction explained

```python
explanation = explain_prediction(model, X_test.iloc[[378]],
                                 background=X_train.sample(100, random_state=42))
```

```
prediction jute probability 0.8354 method shap top rainfall
jute      0.8354
rice      0.1646
coffee    0.0000

rainfall       0.3318
K              0.1217
N              0.1203
temperature    0.0980
P              0.0971
humidity       0.0240
ph            -0.0225
```

`method: shap` — the SHAP path was taken. The true label of row 378 is `rice`, so
this is one of the two errors, and `rainfall` dominates the attribution: the
field measured 186.75 mm against training means of 237 mm (rice) and 176 mm
(jute).

---

## Step 3 — `notebooks/06_model_selection.ipynb`

```bash
jupyter nbconvert --to notebook --execute notebooks/06_model_selection.ipynb
```

```
[NbConvertApp] Converting notebook notebooks/06_model_selection.ipynb to notebook
[NbConvertApp] Writing 512843 bytes to notebooks/06_model_selection.nbconvert.ipynb

real    3m53.030s
```

**~4 minutes.** Almost all of it is the searches: 120 forest fits for the grid,
100 for the randomised one, each fitting a hundred trees.

The `.nbconvert.ipynb` output is a by-product; delete it or leave it, it is
git-ignored. A non-zero exit means a cell raised.

### What Part 2 (§8-§15) should show

**§9 — grid search on the random forest**

```
Best parameters : {'model__max_depth': 10, 'model__max_features': 'sqrt',
                   'model__min_samples_leaf': 1, 'model__n_estimators': 100}
Best CV accuracy: 0.9943 (+/- 0.0060)
Candidates      : 24
Model fits      : 120
```

Compare with Week 7's untuned forest: **0.9926 (± 0.0058)**. The search gained
**0.0017** against a fold-to-fold spread of **0.0060**. That is not an
improvement, and the notebook says so.

**§10 — randomised search**

```
Best CV accuracy: 0.9943 (+/- 0.0040)
Candidates      : 20 sampled from a space of 300
Model fits      : 100
```

Same score, different settings (300 trees, `entropy`, `min_samples_leaf=2`), a
third of the fits an exhaustive search of that space would have needed. Two
different settings reaching the same score is the strongest evidence available
that the hyperparameters do not matter here.

**§11 — naive Bayes**

Twelve `var_smoothing` values, `mean_test_score.nunique() == 1`, every score
`0.994886 (± 0.004175)`.

**§12 — the test set, opened once**

```
Tuned random forest : 0.9955
Gaussian naive Bayes: 0.9955
```

438/440 for both. This is the first and only time in the project that
`data/processed/test.csv` is used for a score.

**§13 — confusion matrices and reports**

The tuned forest, two errors:

```
              precision    recall  f1-score   support
   blackgram       1.00      0.95      0.97        20
        jute       0.95      1.00      0.98        20
       maize       0.95      1.00      0.98        20
        rice       1.00      0.95      0.97        20
   ...
    accuracy                           1.00       440
   macro avg       1.00      1.00      1.00       440
weighted avg       1.00      1.00      1.00       440
```

macro F1 **0.9955**, weighted F1 **0.9955**. Read the pairs: jute has perfect
recall and 0.95 precision because it absorbed a rice field; rice is the mirror
image.

Gaussian naive Bayes, two errors of the *same kind*: macro F1 **0.9954**,
weighted F1 **0.9954**, both `rice -> jute`.

**§14 — the decision**

The written conclusion selects **Gaussian naive Bayes**, with the tuned random
forest recorded as runner-up. The stated reasons are tied accuracy and macro F1,
~40x cheaper fitting, 308 stored numbers against a hundred trees, direct
interpretability, no hyperparameter that changes its behaviour, and one kind of
error rather than two.

**§15 — guard rails**

A cell of `assert`s pinning the accuracies, the error counts and the confusion
pairs. If the data or the seed ever changes, this cell fails loudly instead of
the prose quietly becoming false.

---

## Step 4 — `notebooks/07_model_explainability.ipynb`

```bash
jupyter nbconvert --to notebook --execute notebooks/07_model_explainability.ipynb
```

```
[NbConvertApp] Converting notebook notebooks/07_model_explainability.ipynb to notebook
[NbConvertApp] Writing 398217 bytes to notebooks/07_model_explainability.nbconvert.ipynb

real    0m12.064s
```

**~12 seconds** — no searching here, just one fit and some explaining.

### What it should show

**§1 — permutation importance, tuned forest, test set, 10 repeats**

```
humidity       0.3170
N              0.2020
rainfall       0.1689
K              0.1484
P              0.1114
temperature    0.0066
ph             0.0045
```

**§1 (continued) — MDI against permutation**

| Feature | Built-in MDI (train) | Permutation (test) |
| --- | --- | --- |
| humidity | 0.2185 | 0.3170 |
| rainfall | 0.2164 | 0.1689 |
| K | 0.1896 | 0.1484 |
| P | 0.1457 | 0.1114 |
| N | 0.1044 | 0.2020 |
| temperature | 0.0743 | 0.0066 |
| ph | 0.0511 | 0.0045 |

`N` and `rainfall` swap ranks; `temperature` and `ph` are credited by MDI and
found nearly free to destroy on held-out data.

**§2 — the correlation trap**

```
baseline               0.9955
shuffle P              0.8164   cost 0.1791
shuffle K              0.5630   cost 0.4325
shuffle P and K        0.4302   cost 0.5652
shuffle temperature    0.9105   cost 0.0850
shuffle ph             0.9475   cost 0.0480
shuffle temperature+ph 0.8502   cost 0.1452
```

`P` and `K` (correlated at 0.74 since Week 2) cost far more together than the sum
of their parts; the uncorrelated control pair does not. A low individual score
means "the model can manage without this column *alone*".

**§3 — SHAP plots**

A summary bar plot over all 22 classes and a per-class beeswarm. Read the
beeswarm for `rainfall` on the rice row: red (high rainfall) to the right, blue
to the left — the model saying *rice wants water*, in a form an agronomist can
check.

> **Implementation note.** `TreeExplainer` explains the estimator, not the
> pipeline wrapped round it, so the notebook transforms the sample through
> `tuned_forest.named_steps["preprocess"]` first. Feeding raw columns to an
> explainer for a model fitted on scaled ones produces attributions that are
> quietly wrong. `StandardScaler` is monotone, so the beeswarm's high/low colours
> still mean what they look like.

**§4 — one prediction, end to end**

```
prediction   jute
probability  0.8354
runner-up    rice 0.1646
method       shap
top feature  rainfall (0.3318)
```

Followed by the plain-language paragraph: *the model recommended jute at 84% with
rice second at 16%; the deciding measurement was rainfall at 186.75 mm, against
training averages of 237 mm for rice and 176 mm for jute, while all six other
measurements are nearly identical between the two crops.*

**§5 — the same row through the fallback**

```
                 SHAP    permutation
rainfall       0.3318         0.6053
temperature    0.0980         0.2749
P              0.0971         0.2485
N              0.1203         0.2391
K              0.1217         0.1966
ph            -0.0225         0.0385
humidity       0.0240        -0.0050
```

Both name `rainfall` first; the ordering below it differs because the two
quantities are different. SHAP values are additive on the model's output scale;
the fallback's numbers are probability drops when one feature is perturbed. The
notebook's markdown explains the divergence rather than hiding it.

---

## Step 5 — Test set discipline

```bash
grep -rn "test.csv" notebooks/*.ipynb | grep -v 07_model
```

`data/processed/test.csv` should appear for scoring **only** in
`06_model_selection.ipynb` §12 onwards. Notebook 07 loads it too, but after the
decision, and only to explain a model that was already chosen.

If you re-tune anything after reading Step 3's test numbers, the test measurement
is spent. The honest options are to say so in writing, or to go back to
cross-validating the training rows.

---

## Step 6 — The fallback, verified

The point of a documented fallback is that it is exercised, not just written
down. Hide the library:

```bash
mkdir -p /tmp/noshap && printf 'raise ImportError("hidden for validation")\n' > /tmp/noshap/shap.py
PYTHONPATH=/tmp/noshap python -c "from src.evaluation import SHAP_AVAILABLE, EXPLAINER_BACKEND; print(SHAP_AVAILABLE, EXPLAINER_BACKEND)"
```

```
False permutation
```

```bash
PYTHONPATH=/tmp/noshap pytest tests/test_tuning.py tests/test_explainability.py
```

```
tests/test_explainability.py ...........................s....            [100%]

======================== 53 passed, 1 skipped in 3.29s =========================
```

Identical counts, and 909 fewer warnings because the noisy library is gone. The
*skipped* test is a different one: with `shap` present,
`test_insisting_on_shap_without_shap_fails_clearly` has nothing to assert; with
`shap` absent, the test that checks a real SHAP base value does.

The same row explained by the fallback:

```
prediction   jute
probability  0.8354
method       permutation
top feature  rainfall (0.6053)
```

Same prediction, same probability — those come from the model, not the explainer
— same decisive feature, different numbers. Clean up with
`rm -rf /tmp/noshap`.

---

## Step 7 — The learning outcomes

Answer without looking anything up.

### CAN — read a confusion matrix

* What does cell `(i, j)` count? *Rows whose true class is `i` and whose
  predicted class is `j`.*
* Rows or columns for recall? *Rows. Columns give precision.*
* Name the confusions in this project. *`rice -> jute` (both models) and
  `blackgram -> maize` (the forest only).*
* Why `rice -> jute`? *Six of the seven features are nearly identical between
  those crops; only rainfall separates them (237 mm vs 176 mm), and the field
  measured 186.75 mm.*

### CAN — explain macro vs weighted F1

* Macro = plain mean of the per-class F1 scores; weighted = mean weighted by
  support.
* They agree here (0.9954 / 0.9954) because the stratified split gives every crop
  exactly 20 test rows.
* They diverge under imbalance, and the weighted number alone would then hide
  failure on the rare classes.

### CAN — explain why a specific prediction was made

Row 378, out loud, with: the predicted crop, the probability, the runner-up and
its probability, the deciding measurement and its value, and how that value
compares with the two crops' training averages. Then say which explainer produced
the contribution numbers.

### CAN — name and justify the final chosen model

**Gaussian naive Bayes.** Three reasons that are not accuracy: it is ~40x cheaper
to fit and stores 308 numbers instead of a hundred trees; it is directly
interpretable (a mean and a variance per crop per feature); it has no
hyperparameter that changes its behaviour on this data, so there is no tuning
decision that might fail to transfer. Its errors are also one kind rather than
two. The tuned random forest is the recorded runner-up.

### CANNOT — package this into a reusable production pipeline

Check honestly:

```bash
ls models/
```

Empty. Nothing is serialised, there is no `predict()` entry point outside a
notebook, no versioned artifact and no way for another program to import the
chosen model. Every model in this project still exists only inside a Jupyter
kernel that is about to be shut down. That is Week 9.

Also still out of reach: serving over HTTP (Week 10), a UI (Week 11), deployment
and monitoring (Week 12), nested cross-validation, probability calibration, and
any causal claim whatsoever.

---

## Definition of done

| Check | Command | Expected |
| --- | --- | --- |
| Week 8 tests pass | `pytest tests/test_tuning.py tests/test_explainability.py` | 53 passed, 1 skipped |
| Whole suite passes | `pytest` | 345 passed, 1 skipped |
| Lint clean | `ruff check .` | All checks passed! |
| Notebook 06 runs | `jupyter nbconvert --to notebook --execute notebooks/06_model_selection.ipynb` | exit 0, ~4 min |
| Notebook 07 runs | `jupyter nbconvert --to notebook --execute notebooks/07_model_explainability.ipynb` | exit 0, ~12 s |
| Fallback works | `PYTHONPATH=/tmp/noshap pytest tests/test_explainability.py` | 32 collected, 1 skipped |
| Backend recorded | `python -c "from src.evaluation import EXPLAINER_BACKEND; print(EXPLAINER_BACKEND)"` | `shap` (or `permutation`) |
| Docs exist | `ls docs/curriculum/week08/` | four markdown files |
| Test set opened once | inspection | scored only in notebook 06 §12+ |
