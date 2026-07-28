# Week 3 — Validation

Run these commands in order from the repository root, with the virtual
environment from Week 1 activated. Each block shows the command and the **real
output captured from an actual run** on this repository, so you can compare
against what you see.

The two commands this week must be able to run, verbatim, are:

```bash
pytest tests/test_preprocessing.py
jupyter nbconvert --to notebook --execute notebooks/03_data_preparation.ipynb
```

Everything below expands on those two, with the output each one produced.

---

## Step 0 — Environment

No new dependencies this week. `scikit-learn==1.6.1` was pinned in Week 1 for
exactly this purpose, so if Week 1's Step 2 succeeded there is nothing to
install.

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

---

## Step 1 — Lint the codebase

```bash
ruff check .
```

Actual output:

```
All checks passed!
```

This now also covers `src/data/split.py`,
`src/preprocessing/preprocessor.py`, `tests/test_preprocessing.py` and the code
cells of `notebooks/03_data_preparation.ipynb`.

---

## Step 2 — Run the Week 3 test file

```bash
pytest tests/test_preprocessing.py
```

Actual output:

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-8.3.4, pluggy-1.6.0
rootdir: /.../Crop-Recommendation-System
configfile: pyproject.toml
plugins: anyio-4.14.2
collected 32 items

tests/test_preprocessing.py ................................       [100%]

============================== 32 passed in 0.16s ==============================
```

**All 32 tests must pass, with zero failures.** Most run on a small synthetic
frame built inside the test file, so they pass with or without the CSV; the last
three need the real 2,200 rows and are skipped (not failed) if it is missing.

`pytest tests/test_preprocessing.py -v` names them:

```
tests/test_preprocessing.py::test_split_sizes_follow_test_size PASSED    [  3%]
tests/test_preprocessing.py::test_split_keeps_every_column PASSED        [  6%]
tests/test_preprocessing.py::test_split_is_stratified PASSED             [  9%]
tests/test_preprocessing.py::test_split_rows_are_disjoint_and_complete PASSED [ 12%]
tests/test_preprocessing.py::test_split_is_reproducible_with_the_default_seed PASSED [ 15%]
tests/test_preprocessing.py::test_split_differs_with_a_different_seed PASSED [ 18%]
tests/test_preprocessing.py::test_split_resets_the_index PASSED          [ 21%]
tests/test_preprocessing.py::test_split_rejects_a_missing_target PASSED  [ 25%]
tests/test_preprocessing.py::test_split_rejects_an_out_of_range_test_size[0.0] PASSED [ 28%]
tests/test_preprocessing.py::test_split_rejects_an_out_of_range_test_size[1.0] PASSED [ 31%]
tests/test_preprocessing.py::test_split_rejects_an_out_of_range_test_size[-0.1] PASSED [ 34%]
tests/test_preprocessing.py::test_split_rejects_an_out_of_range_test_size[1.5] PASSED [ 37%]
tests/test_preprocessing.py::test_split_rejects_a_class_with_a_single_row PASSED [ 40%]
tests/test_preprocessing.py::test_class_proportions_sum_to_one PASSED    [ 43%]
tests/test_preprocessing.py::test_class_proportions_rejects_a_missing_target PASSED [ 46%]
tests/test_preprocessing.py::test_build_preprocessor_returns_an_unfitted_column_transformer PASSED [ 50%]
tests/test_preprocessing.py::test_build_preprocessor_defaults_to_the_project_features PASSED [ 53%]
tests/test_preprocessing.py::test_build_preprocessor_rejects_empty_or_duplicated_columns PASSED [ 56%]
tests/test_preprocessing.py::test_scaled_training_features_have_zero_mean_and_unit_std PASSED [ 59%]
tests/test_preprocessing.py::test_scaling_is_fitted_on_train_only PASSED [ 62%]
tests/test_preprocessing.py::test_learned_statistics_come_from_the_training_rows PASSED [ 65%]
tests/test_preprocessing.py::test_transform_is_reversible PASSED         [ 68%]
tests/test_preprocessing.py::test_preprocessor_drops_columns_it_was_not_given PASSED [ 71%]
tests/test_preprocessing.py::test_preprocessor_output_column_order_matches_the_request PASSED [ 75%]
tests/test_preprocessing.py::test_fit_transform_equals_fit_then_transform PASSED [ 78%]
tests/test_preprocessing.py::test_transform_before_fit_raises PASSED     [ 81%]
tests/test_preprocessing.py::test_pipeline_wraps_the_preprocessor PASSED [ 84%]
tests/test_preprocessing.py::test_pipeline_produces_the_same_numbers_as_the_bare_preprocessor PASSED [ 87%]
tests/test_preprocessing.py::test_label_encoder_maps_classes_to_contiguous_integers PASSED [ 90%]
tests/test_preprocessing.py::test_real_split_is_stratified_across_all_22_crops PASSED [ 93%]
tests/test_preprocessing.py::test_real_split_holds_back_the_expected_number_of_rows PASSED [ 96%]
tests/test_preprocessing.py::test_real_scaled_training_features_are_standardised PASSED [100%]
```

The two the week is graded on are
`test_scaled_training_features_have_zero_mean_and_unit_std` (scaled train
features have ~0 mean and ~1 std) and `test_split_is_stratified` /
`test_real_split_is_stratified_across_all_22_crops` (class proportions match
between train and test).

---

## Step 3 — Run the whole suite

```bash
pytest
```

Actual output:

```
======================= 74 passed, 26 warnings in 1.53s ========================
```

74 = Week 1's 20 + Week 2's 22 + Week 3's 32. **Weeks 1 and 2 must still pass
unchanged**: this week adds files and changes none of the data contract.

---

## Step 4 — Reproduce the key numbers on the command line

Each of these underpins a claim in the notebook or the learning notes.

### The split has the expected shape

```bash
python -c "
from src.data import load_data, stratified_split
train, test = stratified_split(load_data())
print('train', train.shape, 'test', test.shape)
print('test rows per crop:', sorted(test['label'].value_counts().unique()))
"
```

Actual output:

```
train (1760, 8) test (440, 8)
test rows per crop: [np.int64(20)]
```

1,760 + 440 = 2,200, and every crop contributes exactly 20 test rows.

### The split is stratified

```bash
python -c "
from src.data import class_proportions, load_data, stratified_split
train, test = stratified_split(load_data())
diff = (class_proportions(train) - class_proportions(test)).abs()
print('classes:', len(diff))
print('largest difference in class share:', diff.max())
"
```

Actual output:

```
classes: 22
largest difference in class share: 0.0
```

Every crop holds 4.5454...% of both halves.

### An unstratified split does not manage that

```bash
python -c "
from sklearn.model_selection import train_test_split
from src.data import load_data
crops = load_data()
_, plain_test = train_test_split(crops, test_size=0.2, random_state=42)
counts = plain_test['label'].value_counts()
print('unstratified test rows per crop: min', counts.min(), 'max', counts.max())
"
```

Actual output:

```
unstratified test rows per crop: min 11 max 27
```

One crop would be evaluated on 11 rows and another on 27 — the reason
`stratify=` is not optional here.

### Scaling: exactly standardised on train, only nearly on test

```bash
python -c "
import numpy as np
from src.data import load_data, stratified_split
from src.preprocessing import build_preprocessor
train, test = stratified_split(load_data())
pre = build_preprocessor()
X_train = pre.fit_transform(train)
X_test = pre.transform(test)
print('train mean:', np.round(X_train.mean(axis=0), 12))
print('train std :', np.round(X_train.std(axis=0, ddof=0), 12))
print('test  mean:', np.round(X_test.mean(axis=0), 3))
print('test  std :', np.round(X_test.std(axis=0, ddof=0), 3))
"
```

Actual output:

```
train mean: [ 0.  0.  0. -0. -0. -0.  0.]
train std : [1. 1. 1. 1. 1. 1. 1.]
test  mean: [ 0.001  0.003  0.001  0.007  0.015 -0.028  0.001]
test  std : [1.008 1.006 0.994 0.984 0.996 0.939 0.997]
```

This is the single most important output of the week. The training columns are
exactly 0 and 1 because the scaler was fitted on them. The test columns are
close but not equal — and that is the *proof* the scaler never saw the test
rows. Exact zeros on the test line would mean leakage.

### The split is reproducible

```bash
python -c "
from src.data import load_data, stratified_split
crops = load_data()
a_train, a_test = stratified_split(crops)
b_train, b_test = stratified_split(crops)
print('same seed -> identical test set:', a_test.equals(b_test))
c_train, c_test = stratified_split(crops, random_state=7)
print('seed 7   -> identical test set:', c_test.equals(a_test))
"
```

Actual output:

```
same seed -> identical test set: True
seed 7   -> identical test set: False
```

### What the leak would have changed

```bash
python -c "
from src.data import FEATURE_COLUMNS, load_data, stratified_split
from src.preprocessing import build_preprocessor
crops = load_data()
train, test = stratified_split(crops)
honest = build_preprocessor().fit(train).named_transformers_['numeric']
leaky = build_preprocessor().fit(crops).named_transformers_['numeric']
for name, h, l in zip(FEATURE_COLUMNS, honest.mean_, leaky.mean_):
    print(f'{name:12s} train-only {h:9.4f}  all-data {l:9.4f}  diff {l - h:+.4f}')
"
```

Actual output:

```
N            train-only   50.5477  all-data   50.5518  diff +0.0041
P            train-only   53.3398  all-data   53.3627  diff +0.0230
K            train-only   48.1432  all-data   48.1491  diff +0.0059
temperature  train-only   25.6094  all-data   25.6162  diff +0.0068
humidity     train-only   71.4168  all-data   71.4818  diff +0.0650
ph           train-only    6.4738  all-data    6.4695  diff -0.0044
rainfall     train-only  103.4516  all-data  103.4637  diff +0.0121
```

Small — the largest drift, `humidity`, is 0.065 against a standard deviation of
22.3. That is *not* an argument for allowing the leak: the size of a leak cannot
be measured without first committing it, and the same mistake in an imputer, a
feature selector or a resampler is far from small. See
[learning notes §5](learning_notes.md).

---

## Step 5 — Execute the Week 3 notebook

Open it interactively — this is what a student runs:

```bash
jupyter notebook notebooks/03_data_preparation.ipynb
```

Run all cells (*Kernel → Restart & Run All*); every cell must execute without
error. Stop the server with `Ctrl-C`.

The non-interactive check, which is the command this week is graded on:

```bash
jupyter nbconvert --to notebook --execute notebooks/03_data_preparation.ipynb
```

Actual output:

```
[NbConvertApp] Converting notebook notebooks/03_data_preparation.ipynb to notebook
[NbConvertApp] Writing 79698 bytes to notebooks/03_data_preparation.nbconvert.ipynb
```

**Exit code 0 is the pass condition.** `nbconvert --execute` aborts on the first
cell that raises, and this notebook contains `assert` statements — on the label
set, on the round-trip of the encoding, and on the reloaded CSVs — so a clean
exit is a real check, not just "no syntax errors".

Without `--inplace` the result goes to `notebooks/03_data_preparation.nbconvert.ipynb`,
which is gitignored and can be deleted. To refresh the committed outputs, which
is how this repository's version was produced:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/03_data_preparation.ipynb
```

The committed notebook contains genuinely executed output — never hand-written
or fabricated results.

---

## Step 6 — Check the artifacts on disk

```bash
ls -l data/processed
```

Actual output:

```
total 464
-rw-rw-r-- 1 runner runner    245 Jul 28 08:01 label_classes.csv
-rw-rw-r-- 1 runner runner  30650 Jul 28 08:01 test.csv
-rw-rw-r-- 1 runner runner  61866 Jul 28 08:01 test_scaled.csv
-rw-rw-r-- 1 runner runner 122794 Jul 28 08:01 train.csv
-rw-rw-r-- 1 runner runner 247363 Jul 28 08:01 train_scaled.csv
```

```bash
head -3 data/processed/train.csv
head -4 data/processed/label_classes.csv
```

Actual output:

```
N,P,K,temperature,humidity,ph,rainfall,label,label_encoded
0,18,14,29.77149434,92.00719952,7.207991261,114.41617859999998,orange,16
9,122,201,29.58748357,80.91934392,5.570290539,68.06417307,grapes,7
```

```
code,crop
0,apple
1,banana
2,blackgram
```

And the raw data is untouched, as it must be:

```bash
git status --short data/raw/
```

Actual output:

```
```

Empty — `data/raw/Crop_recommendation.csv` is byte-identical to what Week 1
committed. Week 3 only *derives*; it never edits the source.

---

## What "done" looks like this week

The student CAN:

* **Explain `fit` vs `transform` vs `fit_transform`.** `fit` learns parameters
  from data and stores them on the estimator (`mean_`, `scale_`); `transform`
  applies stored parameters to any data; `fit_transform` does both in one call
  and belongs to training data only.
* **Explain why we never fit the scaler on test data.** Because the mean and
  standard deviation would then be computed partly from rows the model is
  supposed never to have seen, so the eventual test score would be measured on
  data the pipeline already knew about — an optimistic estimate that production
  will not reproduce. The evidence it was done correctly is Step 4's scaling
  output: train exactly 0/1, test only near 0/1.
* **Build a `ColumnTransformer` from scratch**, without copying:

  ```python
  ColumnTransformer(
      transformers=[("numeric", StandardScaler(), list(FEATURE_COLUMNS))],
      remainder="drop",
  )
  ```

  and say what each element of the triple does, and what `remainder` decides.
* Encode the 22 crop names to integers and decode them back losslessly, and
  explain why the codes must not be treated as quantities.
* Produce a stratified, reproducible 80/20 split — 1,760 / 440, 80 and 20 rows
  per crop — and prove it is stratified by comparing class proportions.
* Say which model families need scaling (KNN, SVM, logistic regression, neural
  networks, PCA) and which do not (decision trees and their ensembles).

The student CANNOT yet:

* **Train or compare real classification models.** Nothing has been fitted this
  week except a scaler. Baselines are **Week 4**, real classifiers **Week 5**,
  and comparing them properly **Week 6**.
* Report accuracy, precision, recall, or draw a confusion matrix — **Week 4**
  onward.
* Cross-validate or tune a hyperparameter — **Week 6**.
* Say which features a model relies on — **Week 7**.
* Look at test-set *values*. This week counted test rows and tallied their
  labels; nothing has inspected or fitted on their contents.

Evidence that no model exists yet:

```bash
ls models/
```

Actual output:

```
```

Empty. The first artifact lands there in Week 9.

---

## Common errors at this stage

**`ModuleNotFoundError: No module named 'src.preprocessing'`**
You are running Python from somewhere other than the repository root, or from an
older checkout. Change to the root; `pytest` sets `pythonpath = ["."]` itself,
and the notebook's first cell handles it with `sys.path.insert`.

**`sklearn.exceptions.NotFittedError: This ColumnTransformer instance is not fitted yet.`**
You called `transform` before `fit`. This is a guard rail, not a bug — see
[learning notes §5](learning_notes.md). Fit on the *training* frame first.

**Your scaled test columns come out at exactly 0 mean and 1 std**
You fitted the scaler on the test set. That is the leak this whole week exists to
prevent: the line should be `preprocessor.transform(test)`, never
`preprocessor.fit_transform(test)`.

**Your split has a different number of rows per crop than shown above**
Either you dropped `stratify` (see Step 4's unstratified comparison) or you
changed `test_size`. `stratified_split()` supplies both defaults; call it rather
than `train_test_split` directly.

**Your numbers differ from those pasted here even with the same code**
Check the seed first — `DEFAULT_RANDOM_STATE` is 42 in `src/data/split.py`. If
the seed matches, the CSV has been modified: restore it with
`git checkout data/raw/Crop_recommendation.csv`. Week 1's contract tests catch
this — run `pytest tests/test_data_loader.py`.

**`ValueError: The least populated class in y has only 1 member`**
Stratification needs at least two rows per class. `stratified_split()` raises its
own clearer error first; if you see scikit-learn's version you are calling
`train_test_split` directly on data with a singleton class.

**`E501 Line too long` from `ruff` pointing at a notebook cell**
`ruff` lints notebook code cells as well as `.py` files. Wrap the line; do not
loosen the rule.

**You "improved" the result by trying several seeds**
Undo it. Choosing the seed with the best downstream score is overfitting the test
set by hand — learning notes §4. If results move a lot between seeds, the answer
is cross-validation (Week 6), not a luckier number.
