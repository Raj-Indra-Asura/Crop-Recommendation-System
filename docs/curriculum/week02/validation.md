# Week 2 — Validation

Run these commands in order from the repository root, with the virtual
environment from Week 1 activated. Each block shows the command and the **real
output captured from an actual run** on this repository, so you can compare
against what you see.

The two commands this week must be able to run, verbatim, are:

```bash
pytest tests/test_eda.py
jupyter nbconvert --to notebook --execute notebooks/02_EDA.ipynb
```

The second confirms the notebook runs top-to-bottom without error. Everything
below expands on those two, with the output each one produced.

---

## Step 0 — Environment

No new dependencies this week. `matplotlib==3.10.0` and `seaborn==0.13.2` were
already pinned in Week 1 for exactly this purpose, so if Week 1's Step 2
succeeded there is nothing to install.

```bash
source venv/bin/activate       # Windows: venv\Scripts\activate
python -c "import matplotlib, seaborn; print(matplotlib.__version__, seaborn.__version__)"
```

Actual output:

```
3.10.0 0.13.2
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

This now covers `src/utils/eda.py`, `tests/test_eda.py` **and the code cells of
`notebooks/02_EDA.ipynb`** — `ruff` lints notebooks too, which is why the
notebook's cells obey the same 100-character line limit as the rest of the
project.

---

## Step 2 — Run the Week 2 test file

```bash
pytest tests/test_eda.py
```

Actual output (warning noise from matplotlib/seaborn omitted):

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-8.3.4, pluggy-1.6.0
rootdir: /.../Crop-Recommendation-System
configfile: pyproject.toml
plugins: anyio-4.14.2
collected 22 items

tests/test_eda.py ......................                                 [100%]

======================= 22 passed, 26 warnings in 2.13s ========================
```

**All 22 tests must pass, with zero failures and zero skips.** Unlike Week 1's
contract tests, none of these are skippable: they run on a small synthetic
dataframe built inside the test file, not on the real CSV. That is deliberate —
the helpers in `src/utils/eda.py` are general-purpose, and testing them against
synthetic data keeps them that way.

`pytest tests/test_eda.py -v` names them:

```
tests/test_eda.py::test_describe_features_returns_a_row_per_feature PASSED [  4%]
tests/test_eda.py::test_describe_features_does_not_mutate_its_input PASSED [  9%]
tests/test_eda.py::test_describe_features_rejects_an_unknown_column PASSED [ 13%]
tests/test_eda.py::test_class_balance_counts_every_class PASSED          [ 18%]
tests/test_eda.py::test_class_balance_rejects_a_missing_target PASSED    [ 22%]
tests/test_eda.py::test_count_outliers_iqr_returns_a_row_per_feature PASSED [ 27%]
tests/test_eda.py::test_count_outliers_iqr_flags_an_extreme_value PASSED [ 31%]
tests/test_eda.py::test_count_outliers_iqr_rejects_a_negative_whisker PASSED [ 36%]
tests/test_eda.py::test_separation_scores_rank_the_separating_feature_first PASSED [ 40%]
tests/test_eda.py::test_separation_scores_are_zero_for_a_constant_feature PASSED [ 45%]
tests/test_eda.py::test_plot_class_balance_returns_axes PASSED           [ 50%]
tests/test_eda.py::test_plot_class_balance_draws_on_supplied_axes PASSED [ 54%]
tests/test_eda.py::test_plot_feature_histograms_returns_one_subplot_per_feature PASSED [ 59%]
tests/test_eda.py::test_plot_feature_histograms_handles_a_single_feature PASSED [ 63%]
tests/test_eda.py::test_plot_feature_histograms_rejects_empty_columns PASSED [ 68%]
tests/test_eda.py::test_plot_feature_histograms_rejects_non_positive_bins PASSED [ 72%]
tests/test_eda.py::test_plot_correlation_heatmap_returns_axes PASSED     [ 77%]
tests/test_eda.py::test_plot_correlation_heatmap_accepts_spearman PASSED [ 81%]
tests/test_eda.py::test_plot_correlation_heatmap_rejects_an_unknown_column PASSED [ 86%]
tests/test_eda.py::test_plot_boxplot_by_label_returns_axes PASSED        [ 90%]
tests/test_eda.py::test_plot_boxplots_by_label_returns_one_panel_per_feature PASSED [ 95%]
tests/test_eda.py::test_plot_boxplots_by_label_rejects_a_missing_target PASSED [100%]
```

These are **smoke tests**: they assert that each helper runs, returns the
documented type, and refuses invalid input. They do not assert pixel-level plot
contents, which would be testing matplotlib rather than this project.

---

## Step 3 — Run the whole suite

```bash
pytest
```

Actual output:

```
======================= 42 passed, 26 warnings in 2.14s ========================
```

42 = Week 1's 20 + Week 2's 22. **Week 1's tests must still pass unchanged**:
Week 2 only reads the data, so nothing about the dataset contract may have
moved.

---

## Step 4 — Reproduce the key numbers on the command line

Each of these underpins a claim in the notebook's conclusions.

### Descriptive statistics

```bash
python -c "
from src.data import load_data, FEATURE_COLUMNS
from src.utils.eda import describe_features
print(describe_features(load_data(), list(FEATURE_COLUMNS))[['mean','median','std','min','max','skew']].round(2))
"
```

Actual output:

```
               mean  median    std    min     max  skew
N             50.55   37.00  36.92   0.00  140.00  0.51
P             53.36   51.00  32.99   5.00  145.00  1.01
K             48.15   32.00  50.65   5.00  205.00  2.38
temperature   25.62   25.60   5.06   8.83   43.68  0.18
humidity      71.48   80.47  22.26  14.26   99.98 -1.09
ph             6.47    6.43   0.77   3.50    9.94  0.28
rainfall     103.46   94.87  54.96  20.21  298.56  0.97
```

Note `K`: mean 48.15 against a median of 32.00, skew +2.38. Note also the scale
gap between `ph` (range ≈ 6) and `K` (range 200) — the evidence for Week 3's
feature scaling.

### Class balance

```bash
python -c "
from src.data import load_data
from src.utils.eda import class_balance
b = class_balance(load_data())
print(b['count'].min(), b['count'].max(), len(b))
print(b['proportion'].round(4).unique())
"
```

Actual output:

```
100 100 22
[0.0455]
```

22 classes, 100 rows each, 4.55% apiece — **perfectly balanced**.

### Correlation, and why the one strong pair is a class artifact

```bash
python -c "
from src.data import load_data
c = load_data()
rest = c[~c['label'].isin(['apple','grapes'])]
print('all 22 crops      :', round(c['P'].corr(c['K']), 3))
print('minus apple/grapes:', round(rest['P'].corr(rest['K']), 3))
"
```

Actual output:

```
all 22 crops      : 0.736
minus apple/grapes: 0.043
```

### Outliers resolve to crops, not errors

```bash
python -c "
from src.data import load_data
c = load_data()
print('K > 92.5        :', c.loc[c['K'] > 92.5, 'label'].value_counts().to_dict())
print('humidity < 15.73:', c.loc[c['humidity'] < 15.73, 'label'].value_counts().to_dict())
print('rainfall > 213.8:', c.loc[c['rainfall'] > 213.84, 'label'].value_counts().to_dict())
"
```

Actual output:

```
K > 92.5        : {'grapes': 100, 'apple': 100}
humidity < 15.73: {'chickpea': 30}
rainfall > 213.8: {'rice': 68, 'papaya': 17, 'coconut': 15}
```

Every flagged group is a crop's own population, not a corrupt reading — which is
why Week 2 deletes nothing.

### Which features separate the crops

```bash
python -c "
from src.data import load_data, FEATURE_COLUMNS
from src.utils.eda import separation_scores
print(separation_scores(load_data(), list(FEATURE_COLUMNS)).round(3))
"
```

Actual output:

```
K              0.996
humidity       0.968
P              0.948
N              0.896
rainfall       0.854
temperature    0.496
ph             0.368
Name: eta_squared, dtype: float64
```

### The three separation claims, checked

```bash
python -c "
from src.data import load_data
c = load_data()
print('apple/grapes K range:', c[c['label'].isin(['apple','grapes'])]['K'].min(), '-', c[c['label'].isin(['apple','grapes'])]['K'].max())
print('max K, all other crops:', c[~c['label'].isin(['apple','grapes'])]['K'].max())
print('rice rainfall min:', round(c[c['label']=='rice']['rainfall'].min(), 1))
print('muskmelon rainfall max:', round(c[c['label']=='muskmelon']['rainfall'].max(), 1))
print('chickpea humidity range:', round(c[c['label']=='chickpea']['humidity'].min(), 1), '-', round(c[c['label']=='chickpea']['humidity'].max(), 1))
"
```

Actual output:

```
apple/grapes K range: 195 - 205
max K, all other crops: 85
rice rainfall min: 182.6
muskmelon rainfall max: 29.9
chickpea humidity range: 14.3 - 20.0
```

---

## Step 5 — Execute the Week 2 notebook

Open it interactively — this is what a student runs:

```bash
jupyter notebook notebooks/02_EDA.ipynb
```

Run all cells (*Kernel → Restart & Run All*); every cell must execute without
error. Stop the server with `Ctrl-C`.

The non-interactive check, which is the command this week is graded on:

```bash
jupyter nbconvert --to notebook --execute notebooks/02_EDA.ipynb
```

Actual output:

```
[NbConvertApp] Converting notebook notebooks/02_EDA.ipynb to notebook
[NbConvertApp] Writing 717070 bytes to notebooks/02_EDA.nbconvert.ipynb
```

**Exit code 0 is the pass condition.** `nbconvert --execute` aborts on the first
cell that raises, so a clean exit means the notebook genuinely runs
top-to-bottom.

Without `--inplace`, the result is written to a *new* file,
`notebooks/02_EDA.nbconvert.ipynb`, leaving the committed notebook untouched.
That scratch file is gitignored (`*.nbconvert.ipynb`) and can be deleted. To
refresh the committed outputs instead — which is how this repository's version
was produced — add `--inplace`:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/02_EDA.ipynb
```

```
[NbConvertApp] Converting notebook notebooks/02_EDA.ipynb to notebook
[NbConvertApp] Writing 717070 bytes to notebooks/02_EDA.ipynb
```

The committed notebook contains genuinely executed output — never hand-written
or fabricated results.

---

## What "done" looks like this week

The student CAN:

* **Describe the dataset's shape and balance** — 2,200 × 8; 22 crops with
  exactly 100 rows each (4.55% apiece), i.e. perfectly balanced.
* **Name at least three features that visually separate certain crops**, with
  numbers:
  1. `K` — apple and grapes sit at 195–205; no other crop exceeds 85.
  2. `rainfall` — rice never below 182.6 mm; muskmelon never above 29.9 mm.
  3. `humidity` — chickpea spans 14.3–20.0% while most crops sit above 80%.
  (`P` and `N` also separate strongly — eta-squared 0.948 and 0.896.)
* **Explain data leakage in one sentence** — for example: *"Data leakage is when
  a model is trained using information it would not have at prediction time, so
  its test score flatters it and production disappoints."* Follow-up they should
  also be able to answer: why were this week's full-dataset statistics safe?
  Because they were read by a human to understand the problem, not used to
  transform data or fit a model.

The student CANNOT yet:

* **Preprocess the data** — no scaling, no encoding, no dropping of rows or
  columns, no train/test split. All of that is **Week 3**, and the rule stated
  this week (fit on training data only) is what governs it.
* **Train or evaluate a model**, including a baseline — **Week 4** onward.
* Say which crops a *model* confuses — that needs a fitted model (Week 5) and a
  confusion matrix (Week 8). This week only shows which crops separate visually
  on single features.
* Declare any feature unimportant. Eta-squared ranks features in isolation;
  importance in combination is **Week 7**.

Evidence that nothing was preprocessed:

```bash
git status --short data/
```

Actual output:

```
```

Empty — the raw CSV is byte-identical to what Week 1 committed, and
`data/processed/` is still empty apart from its `.gitkeep`.

---

## Common errors at this stage

**`ModuleNotFoundError: No module named 'src.utils'`**
You are running Python from somewhere other than the repository root. Change to
the root, or rely on `pytest`, which sets `pythonpath = ["."]` itself. Inside
the notebook this is what the first cell's `sys.path.insert` handles.

**Plots do not appear in Jupyter**
The helpers deliberately never call `plt.show()` — the caller decides when to
render. Either end the cell with `plt.show()`, as the notebook does, or let the
returned `Axes`/`Figure` be the cell's last expression.

**`nbconvert` hangs or reports "Kernel died"**
Usually a missing kernel. Confirm with `jupyter kernelspec list` that `python3`
points inside your virtual environment, and that `ipykernel` is installed (it
arrives with `jupyter==1.1.1`).

**`RuntimeError: main thread is not in main loop`, or a display-related crash**
matplotlib is trying to open a window on a machine with no display. Force the
non-interactive backend: `export MPLBACKEND=Agg`, or `matplotlib.use("Agg")`
before importing `pyplot` — which is exactly what `tests/test_eda.py` does at
the top of the file.

**`E501 Line too long` from `ruff` pointing at a notebook cell**
`ruff` lints notebook code cells as well as `.py` files. Wrap the line; do not
loosen the rule.

**Statistics in your run differ from those pasted above**
The CSV has been modified. Restore it with
`git checkout data/raw/Crop_recommendation.csv`. Week 1's contract tests should
have caught this first — run `pytest tests/test_data_loader.py` to confirm.

**You removed some outlier rows and now the numbers look "cleaner"**
Undo it. Section 5 of `learning_notes.md` explains why: the flagged rows are
whole crop populations, and any removal decision belongs after Week 3's split,
using training data only.
