# Week 1 — Validation

Run these commands in order from the repository root. Each block shows the
command and the **real output captured from an actual run**, so you can compare
against what you see.

> ### ⚠️ Status of this week's validation
>
> The raw dataset `data/raw/Crop_recommendation.csv` was **not present** in the
> repository when these commands were run. It is supposed to be committed here.
>
> Consequently the checks below are split into two groups:
>
> * **Steps 1–4** are data-independent and their pasted output is a complete,
>   passing run.
> * **Steps 5–7** need the real file. Their output shows the *actual observed*
>   behaviour with the dataset absent, together with the output you should
>   expect once it is restored.
>
> This week is **not complete** until the CSV is restored and steps 5–7 produce
> the expected output. See "If the dataset is missing" at the bottom.

---

## Step 1 — Check your Python version

```bash
python3 --version
```

Expected: Python 3.11.x, which is this project's target version.

Actual output from the run that produced this document:

```
Python 3.12.3
```

**Note the discrepancy.** The recording environment provided 3.12.3, not 3.11.
Everything in Week 1 works identically on both, and `ruff` is configured with
`target-version = "py311"` so style rules stay pinned to 3.11 regardless. If
you see 3.11.x, that is correct and expected; if you see 3.12.x, Week 1 will
still pass.

---

## Step 2 — Create and populate the virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Confirm the four pinned packages are installed:

```bash
pip list | grep -E "^(pandas|jupyter|pytest|ruff) "
```

Expected output (exact versions must match `requirements.txt`):

```
jupyter                   1.1.1
pandas                    2.2.3
pytest                    8.3.4
ruff                      0.8.4
```

`pip list` on its own shows far more than four packages — those are transitive
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

## Step 4 — Run the test suite

```bash
pytest -v
```

Actual output from the run that produced this document:

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-8.3.4, pluggy-1.6.0
cachedir: .pytest_cache
rootdir: /.../Crop-Recommendation-System
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.14.2
collected 10 items

tests/test_data_loader.py::test_raw_dataset_file_is_committed SKIPPED    [ 10%]
tests/test_data_loader.py::test_raw_dataset_has_expected_shape SKIPPED   [ 20%]
tests/test_data_loader.py::test_raw_dataset_has_expected_columns_in_order SKIPPED [ 30%]
tests/test_data_loader.py::test_raw_dataset_has_expected_number_of_crops SKIPPED [ 40%]
tests/test_data_loader.py::test_raw_dataset_has_no_missing_values SKIPPED [ 50%]
tests/test_data_loader.py::test_raw_dataset_features_are_numeric SKIPPED [ 60%]
tests/test_data_loader.py::test_load_raw_data_validates_by_default SKIPPED [ 70%]
tests/test_data_loader.py::test_validate_rejects_wrong_columns PASSED    [ 80%]
tests/test_data_loader.py::test_validate_rejects_wrong_row_count PASSED  [ 90%]
tests/test_data_loader.py::test_load_raw_data_reports_missing_file_clearly PASSED [100%]

========================= 3 passed, 7 skipped in 0.02s =========================
```

The three behaviour tests pass — the loader's own logic is verified. The seven
skips are the contract tests, skipped because the dataset is absent. Run
`pytest -rs` to print the skip reason:

```
SKIPPED [7] tests/conftest.py: Raw dataset missing at .../data/raw/Crop_recommendation.csv.
It is meant to be committed to the repository; restore it before running the data tests.
```

**Once the dataset is restored, this must read `10 passed` with zero skips.**

---

## Step 5 — Load the dataset

```bash
python -c "from src.data import load_raw_data; print(load_raw_data().shape)"
```

Expected output once the dataset is present:

```
(2200, 8)
```

Actual output observed (dataset absent):

```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File ".../src/data/loader.py", line 115, in load_raw_data
    raise FileNotFoundError(
FileNotFoundError: Raw dataset not found at .../data/raw/Crop_recommendation.csv.
This file is version-controlled and should already be present. Download
'Crop_recommendation.csv' from
https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset and
place it at .../data/raw/Crop_recommendation.csv.
```

This is the fail-fast behaviour working as designed: the error names the exact
path and how to fix it, rather than surfacing as a confusing failure later.

---

## Step 6 — Inspect the crop distribution

```bash
python -c "from src.data import load_raw_data; print(load_raw_data()['label'].value_counts())"
```

Expected once the dataset is present: 22 crop names, **100 rows each**, summing
to 2,200. Cannot be run yet — blocked by step 5.

---

## Step 7 — Execute the Week 1 notebook

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_problem_definition.ipynb
```

Expected: exits 0, and the notebook is rewritten with fresh output cells.

**Not yet produced.** The notebook loads the real dataset, so executing it
without the CSV would either fail or require faking the data. Committing a
notebook with empty or fabricated output cells would violate this project's
rule that notebooks are committed with genuinely executed output, so the
notebook is deferred until the dataset is restored.

---

## If the dataset is missing

This is the situation the run above was in. To resolve it:

1. Obtain `Crop_recommendation.csv` from the
   [Kaggle dataset page](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset).
2. Place it at `data/raw/Crop_recommendation.csv`.
3. Commit it. `data/` is deliberately **not** in `.gitignore` — see
   `learning_notes.md` §4 for why.
4. Re-run steps 4–7. `pytest` must report **10 passed, 0 skipped**.

Do not work around a missing dataset by generating random or synthetic data.
Every later week builds on these numbers, and fabricated data would silently
invalidate all of it while still appearing to work.

---

## Common errors at this stage

**`ModuleNotFoundError: No module named 'src'`**
You are running Python from somewhere other than the repository root, so `src`
is not on the import path. Change to the repo root and retry. Inside a
notebook, this is why the notebook lives in `notebooks/` but the loader
computes an absolute path — see `learning_notes.md` §6.

**`ModuleNotFoundError: No module named 'pandas'`**
The virtual environment is not activated, or `pip install -r requirements.txt`
has not been run. Check with `which python` — it should point inside `.venv`.

**`command not found: ruff` / `pytest`**
Same cause as above: the environment is not active. Either activate it, or call
the tools by full path (`.venv/bin/ruff`).

**`pytest` collects 0 items**
You are not in the repository root. `pyproject.toml` sets `testpaths = ["tests"]`,
which is resolved relative to the root.

**`DatasetValidationError: Expected 2200 rows, got ...`**
The CSV has been modified. Restore the committed version with
`git checkout data/raw/Crop_recommendation.csv`. Never "fix" this by loosening
the constant — the constant is the specification.

**Permission or activation errors on Windows PowerShell**
`.venv\Scripts\activate` may be blocked by execution policy. Run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` first.
