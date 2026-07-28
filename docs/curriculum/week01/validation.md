# Week 1 — Validation

Run these commands in order from the repository root. Each block shows the
command and the **real output captured from an actual run** on this
repository, so you can compare against what you see.

The four commands this week must be able to run, verbatim, are:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pytest tests/test_data_loader.py
jupyter notebook notebooks/01_problem_definition.ipynb
```

Everything below expands on those four, with the output each one produced.

---

## Step 1 — Check your Python version

```bash
python3 --version
```

Expected: Python 3.11.x — the project's reference version — or Python 3.12.x,
which is also supported.

Actual output from the run that produced this document:

```
Python 3.12.3
```

Python 3.11 was not available in the environment used to record these
commands, so every command below was genuinely run on 3.12.3. This is
supported: nothing in Week 1 depends on version-specific behaviour, and `ruff`
is configured with `target-version = "py311"` so style rules stay pinned to the
reference version regardless of the interpreter you use. Either 3.11.x or
3.12.x is a correct result here.

---

## Step 2 — Create and populate the virtual environment

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

On Windows the activation line is `venv\Scripts\activate`. Both `venv/` and
`.venv/` are gitignored, so either name is safe.

Confirm the nine pinned packages are installed:

```bash
pip list | grep -E "^(numpy|pandas|matplotlib|seaborn|scikit-learn|jupyter|pytest|ruff) "
```

Actual output (exact versions must match `requirements.txt`):

```
jupyter                   1.1.1
matplotlib                3.10.0
numpy                     2.2.1
pandas                    2.2.3
pytest                    8.3.4
ruff                      0.8.4
scikit-learn              1.6.1
seaborn                   0.13.2
```

`pip list` on its own shows far more than nine packages — those are transitive
dependencies pulled in automatically (jupyter alone brings in several dozen).
Only direct dependencies are pinned in `requirements.txt`.

---

## Step 3 — Lint the codebase

```bash
ruff check .
```

Actual output:

```
All checks passed!
```

Any other output is a failure and must be fixed before the week is done. Most
findings can be auto-corrected with `ruff check . --fix`, but read what it
changed rather than trusting it blindly.

---

## Step 4 — Run the Week 1 test file

```bash
pytest tests/test_data_loader.py
```

Actual output:

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-8.3.4, pluggy-1.6.0
rootdir: /.../Crop-Recommendation-System
configfile: pyproject.toml
plugins: anyio-4.14.2
collected 20 items

tests/test_data_loader.py ....................                           [100%]

============================== 20 passed in 0.04s ==============================
```

**All 20 tests must pass, with zero failures and zero skips.** Run `pytest -v`
to see them named:

```
tests/test_data_loader.py::test_raw_dataset_file_is_committed PASSED     [  5%]
tests/test_data_loader.py::test_raw_dataset_has_expected_shape PASSED    [ 10%]
tests/test_data_loader.py::test_raw_dataset_has_expected_columns_in_order PASSED [ 15%]
tests/test_data_loader.py::test_raw_dataset_has_expected_number_of_crops PASSED [ 20%]
tests/test_data_loader.py::test_raw_dataset_label_set_matches_recorded_set PASSED [ 25%]
tests/test_data_loader.py::test_raw_dataset_has_no_missing_values PASSED [ 30%]
tests/test_data_loader.py::test_raw_dataset_features_are_numeric PASSED  [ 35%]
tests/test_data_loader.py::test_load_data_validates_by_default PASSED    [ 40%]
tests/test_data_loader.py::test_validate_accepts_a_frame_matching_the_contract PASSED [ 45%]
tests/test_data_loader.py::test_validate_rejects_wrong_columns PASSED    [ 50%]
tests/test_data_loader.py::test_validate_rejects_whitespace_in_column_names PASSED [ 55%]
tests/test_data_loader.py::test_validate_rejects_reordered_columns PASSED [ 60%]
tests/test_data_loader.py::test_validate_rejects_wrong_row_count PASSED  [ 65%]
tests/test_data_loader.py::test_validate_rejects_null_features PASSED    [ 70%]
tests/test_data_loader.py::test_validate_rejects_non_numeric_features PASSED [ 75%]
tests/test_data_loader.py::test_validate_rejects_null_labels PASSED      [ 80%]
tests/test_data_loader.py::test_validate_rejects_unexpected_label_values PASSED [ 85%]
tests/test_data_loader.py::test_validate_rejects_wrong_number_of_labels PASSED [ 90%]
tests/test_data_loader.py::test_load_data_reports_missing_file_clearly PASSED [ 95%]
tests/test_data_loader.py::test_load_data_raises_on_malformed_csv PASSED [100%]

============================== 20 passed in 0.04s ==============================
```

The first eight read the real committed CSV; the rest break a synthetic frame
one rule at a time and assert that `validate_dataset()` refuses it.

---

## Step 5 — Load the dataset

```bash
python -c "from src.data import load_data; print(load_data().shape)"
```

Actual output:

```
(2200, 8)
```

2,200 rows, 8 columns — seven features plus `label`. `load_data()` validated
the file before returning it, so this line succeeding is itself a check.

---

## Step 6 — Inspect the crop distribution

```bash
python -c "from src.data import load_data; print(load_data()['label'].value_counts())"
```

Actual output:

```
label
rice           100
maize          100
chickpea       100
kidneybeans    100
pigeonpeas     100
mothbeans      100
mungbean       100
blackgram      100
lentil         100
pomegranate    100
banana         100
mango          100
grapes         100
watermelon     100
muskmelon      100
apple          100
orange         100
papaya         100
coconut        100
cotton         100
jute           100
coffee         100
Name: count, dtype: int64
```

22 crops, exactly 100 rows each, summing to 2,200 — a perfectly balanced
dataset.

---

## Step 7 — The expected label set (recorded here, permanently)

```bash
python -c "from src.data import load_data; print(sorted(load_data()['label'].unique()))"
```

Actual output:

```
['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon']
```

**This is the expected label set for the whole project.** It is recorded here
in prose and in code as `EXPECTED_LABELS` in `src/data/validate_schema.py`:

| # | Label | # | Label |
| --- | --- | --- | --- |
| 1 | `apple` | 12 | `maize` |
| 2 | `banana` | 13 | `mango` |
| 3 | `blackgram` | 14 | `mothbeans` |
| 4 | `chickpea` | 15 | `mungbean` |
| 5 | `coconut` | 16 | `muskmelon` |
| 6 | `coffee` | 17 | `orange` |
| 7 | `cotton` | 18 | `papaya` |
| 8 | `grapes` | 19 | `pigeonpeas` |
| 9 | `jute` | 20 | `pomegranate` |
| 10 | `kidneybeans` | 21 | `rice` |
| 11 | `lentil` | 22 | `watermelon` |

All names are lowercase, with no spaces and no punctuation.

**The rule for every later week:** any code that encodes, filters, groups by or
predicts `label` must match against this exact set, and must fail loudly if it
differs — a new crop, a missing crop, a capitalised `Rice` or a padded
`'rice '` all mean the data is not what this course was written against.
`validate_dataset()` already enforces it at load time; later weeks must not
weaken that check to make their own code pass.

---

## Step 8 — Execute the Week 1 notebook

Open it interactively — this is the command a student runs:

```bash
jupyter notebook notebooks/01_problem_definition.ipynb
```

That starts a local Jupyter server and opens the notebook in a browser. Run all
cells (*Kernel → Restart & Run All*); every cell must execute without error.
Stop the server with `Ctrl-C` in the terminal.

To check the same thing non-interactively — which is how the committed output
was produced:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_problem_definition.ipynb
```

Actual output:

```
[NbConvertApp] Converting notebook notebooks/01_problem_definition.ipynb to notebook
[NbConvertApp] Writing 20776 bytes to notebooks/01_problem_definition.ipynb
```

Exit code 0, and the notebook is rewritten with fresh output cells. The
committed notebook contains genuinely executed output — never hand-written or
fabricated results.

---

## What "done" looks like this week

The student CAN:

* run the repository end-to-end from a fresh virtual environment;
* load the dataset through `load_data()` and get `(2200, 8)`;
* explain in one paragraph what problem is being solved, and why it is a
  supervised, multiclass classification problem.

The student CANNOT yet:

* explore the data statistically or visually (Week 2);
* preprocess it — split, scale or encode (Week 3);
* train or evaluate any model, including a baseline (Week 4 onwards).

---

## If the dataset is missing

`data/raw/Crop_recommendation.csv` is committed to this repository and must
stay committed. If it is ever absent:

1. Obtain it from the
   [Kaggle dataset page](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset).
2. Place it at `data/raw/Crop_recommendation.csv`.
3. Commit it. `data/` is deliberately **not** in `.gitignore` — see
   `learning_notes.md` §4 for why.
4. Re-run steps 3–8. `pytest` must report **20 passed, 0 skipped**.

While it is absent the eight contract tests report *skipped*, not *failed* — a
skip says "we did not check this", a failure says "we checked and it is
broken". Do not work around a missing dataset by generating random or synthetic
data: every later week builds on these numbers, and fabricated data would
silently invalidate all of it while still appearing to work.

---

## Common errors at this stage

**`ModuleNotFoundError: No module named 'src'`**
You are running Python from somewhere other than the repository root, so `src`
is not on the import path. Change to the repo root and retry. (`pytest` itself
is fine from anywhere below the root: `pyproject.toml` sets
`pythonpath = ["."]`.) Inside a notebook this is why the first cell adds the
repository root to `sys.path` — see `learning_notes.md` §8.

**`ModuleNotFoundError: No module named 'pandas'`**
The virtual environment is not activated, or `pip install -r requirements.txt`
has not been run. Check with `which python` — it should point inside `venv`.

**`command not found: ruff` / `pytest`**
Same cause as above: the environment is not active. Either activate it, or call
the tools by full path (`venv/bin/ruff`).

**`pytest` collects 0 items**
You are not in the repository root. `pyproject.toml` sets `testpaths = ["tests"]`,
which is resolved relative to the root.

**`DatasetValidationError: Expected 2200 rows, got ...`**
The CSV has been modified. Restore the committed version with
`git checkout data/raw/Crop_recommendation.csv`. Never "fix" this by loosening
the constant — the constant is the specification.

**`DatasetValidationError: Label set does not match ...`**
The `label` column contains a crop that is not in the recorded set above, or is
missing one that should be there. Same rule applies: fix the data, not the
contract.

**Permission or activation errors on Windows PowerShell**
`venv\Scripts\activate` may be blocked by execution policy. Run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` first.
