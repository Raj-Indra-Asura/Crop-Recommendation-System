# Week 1 — Learning Notes

> How to read these notes: every new idea is introduced with four questions —
> *what is it, why do we need it, where is it used, how does it actually work*
> — followed by the mistakes people commonly make with it. Code blocks are
> always explained, either just before or just after they appear.

---

## 1. What machine learning actually is

### Artificial intelligence, machine learning — where does this project sit?

Two words get used interchangeably and should not be.

**Artificial intelligence** is the broad ambition: software that performs tasks
we would call intelligent if a person did them — planning a route, translating
a sentence, recommending a crop. It includes approaches that involve no
learning at all, such as hand-written expert systems and search algorithms.

**Machine learning** is one family of techniques within AI: instead of encoding
the knowledge by hand, the system derives it from data. (**Deep learning**, in
turn, is one family within machine learning, using many-layered neural
networks. This course does not need it: with 2,200 rows and 7 numeric
features, classical algorithms are both stronger and easier to explain.)

This project is **machine learning, not deep learning**, applied to a tabular
dataset. That is the most common shape of real industrial ML work, and it is
where the transferable skills are.

### What is it?

Machine learning is a way of building software where you do not write the rules
yourself. Instead you supply examples of the correct behaviour, and a learning
algorithm searches for rules that reproduce those examples — and, crucially,
that keep working on examples it has never seen.

Compare two ways of deciding what crop to plant:

**Approach A — rules written by a human.**

```python
if rainfall > 200 and humidity > 80:
    return "rice"
elif rainfall < 60 and temperature > 30:
    return "millet"
# ...and so on, for 22 crops and 7 interacting variables
```

**Approach B — machine learning.** Collect thousands of records of the form
"these were the soil and weather conditions, and this crop grew well here", and
let an algorithm work out the boundaries.

### Why do we need it here?

Approach A collapses under its own weight in this problem, for three reasons:

1. **The rules interact.** Rice tolerates lower rainfall if humidity is high;
   potassium requirements shift with soil pH. Writing conditions for seven
   variables that all influence one another means an explosion of nested
   branches.
2. **Nobody can state the thresholds exactly.** An agronomist will say "rice
   likes a lot of water" — not "rice requires rainfall above 182.4 mm". The
   precise numbers exist in the data, not in anyone's head.
3. **The rules would need constant hand-editing.** Add a new crop, or get data
   from a new region, and a human must revisit every branch. A learning
   algorithm just retrains.

The general principle, which is worth memorising:

> Use machine learning when the rules are too numerous, too interacting, or too
> unknown to be written down by hand — and when you have examples of the right
> answer.

The corollary matters just as much: **when a short, stable rule works, use the
rule.** Machine learning adds a training pipeline, a model artifact, a
monitoring burden and a source of silent wrongness. Do not pay that cost to
compute a tax rate.

### Where is it used in industry?

Spam filtering, fraud detection, product recommendations, demand forecasting,
medical triage, speech recognition. In every one of these the same condition
holds: abundant examples, no crisp hand-writable rule.

### Common mistakes

* **Reaching for ML first.** A regular expression or a lookup table is often
  the correct engineering answer, and is far cheaper to maintain.
* **Assuming ML infers causation.** Our model will learn that certain
  conditions *co-occur* with certain crops. It does not prove those conditions
  *cause* good yield. That distinction becomes serious when someone acts on the
  output.

---

## 2. Naming the kind of problem

Before writing any code, an ML problem gets classified along a few axes. This
is not academic box-ticking: the answers determine which algorithms are even
applicable and which evaluation metrics are meaningful.

### Supervised or unsupervised?

* **Supervised learning** — every training example comes with the correct
  answer attached. The algorithm learns the mapping from inputs to that answer.
* **Unsupervised learning** — no answers are provided. The algorithm looks for
  structure on its own, for example grouping similar records together.

Our dataset has a `label` column stating which crop belongs to each row. The
answers are given, so this is **supervised learning**.

### Classification or regression?

Within supervised learning, the split is decided by *what kind of thing* you
are predicting:

* **Regression** — predict a number on a continuous scale. "How many kilograms
  of rice will this field yield?" Being off by a little is a small error.
* **Classification** — predict which category something belongs to. "Which crop
  suits this field?" There is no notion of being *slightly* wrong: you either
  named the right crop or you did not.

We predict one of 22 named crops, so this is **classification**. Because there
are more than two categories, it is specifically **multiclass classification**
(as opposed to *binary*, which has exactly two).

A subtlety worth noticing now: the crops have **no meaningful order**. Rice is
not "greater than" maize. If we naively encoded the crops as the numbers 0–21
and fed them to a regression algorithm, the algorithm would assume that crop 5
sits between crop 4 and crop 6, and that predicting 4.5 is a reasonable hedge.
It is not. Handling this correctly is Week 3's work; for now just record that
the target is *categorical and unordered*.

### Batch or online?

* **Batch learning** — the model is trained once on the whole dataset, then
  deployed as a fixed artifact, and retrained on a schedule.
* **Online learning** — the model updates continuously as new data streams in.

Growing conditions and crop suitability change over seasons, not seconds, and
our dataset is a fixed file. We use **batch learning**.

### Common mistakes

* **Skipping this step.** Teams that never wrote down "this is multiclass
  classification" end up choosing a metric that does not fit, then arguing
  about results that were never comparable.
* **Confusing "numeric target" with regression.** If the numbers are category
  codes, it is still classification.

---

## 3. The problem statement for this project

Here is the framing, written out properly. Everything in the rest of the course
is judged against it.

**Task.** Given the measurable growing conditions of a plot of land, recommend
the crop most suited to it.

**Inputs (features), seven of them:**

| Feature | Meaning | Unit |
| --- | --- | --- |
| `N` | Nitrogen content of the soil | ratio |
| `P` | Phosphorus content of the soil | ratio |
| `K` | Potassium content of the soil | ratio |
| `temperature` | Average temperature | °C |
| `humidity` | Relative humidity | % |
| `ph` | Soil acidity/alkalinity | pH scale, 0–14 |
| `rainfall` | Rainfall | mm |

**Output (target):** `label` — one of 22 crop names.

**Worked example of the intended behaviour:**

```
Input:  N=90, P=42, K=43, temperature=25, humidity=80, ph=6.5, rainfall=200
Output: rice
```

**Type:** supervised, multiclass classification, batch-trained.

**How success is measured:** left deliberately open until Week 4. Choosing a
metric is a substantial topic and deserves its own treatment; guessing at one
now would be worse than admitting we have not decided.

**What this system is not.** It does not predict yield, does not account for
market prices, seed availability, local practice or crop rotation, and its
recommendation is advisory. Writing down these limits early prevents the
project from quietly being sold as something it is not.

### Terminology introduced here

* **Feature** — one input variable. `humidity` is a feature.
* **Target** (or *label*, or *class*) — the value being predicted. Here,
  `label`.
* **Instance** (or *sample*, *observation*, *row*) — one complete example: a
  set of seven feature values together with its crop.
* **Dataset** — the whole collection of instances.

---

## 4. The dataset

The file is `data/raw/Crop_recommendation.csv`, taken from the public
[Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
on Kaggle. It holds **2,200 rows across 22 crop labels** and eight columns —
the seven features plus `label`.

Two conventions in this repository deserve explanation.

**The CSV is committed to git, not ignored.** Large datasets normally do not
belong in a git repository, and you will often see `data/` in a `.gitignore`.
This one is about 150 KB — smaller than a typical photograph — and committing
it buys something valuable: every student, every CI run and every future reader
gets *byte-identical* data. Results become reproducible. If the file were
downloaded on demand instead, an upstream change would silently alter everyone's
results with no trace in the git history.

**`data/raw/` is treated as read-only.** Nothing in this project ever writes
back to it. Anything derived — cleaned data, splits, encodings — is written to
`data/processed/`. This means the raw input is always recoverable, and a broken
preprocessing step can never destroy the source of truth. It is why the two
directories exist separately.

---

## 5. Features, labels, and what a dataset is for

The four words that the rest of the course leans on, stated once, with this
project as the running example.

A **dataset** is a collection of *instances*. An **instance** (also called a
row, sample or observation) is one complete example. In our file, one instance
is one line of the CSV:

```
N=90, P=42, K=43, temperature=20.88, humidity=82.00, ph=6.50, rainfall=202.94, label=rice
```

The first seven values are the **features** — the inputs, the things we get to
measure about a plot of land. The last is the **label** (or target) — the
answer we want the model to produce. Splitting each row into `X` (features) and
`y` (label) is the very first thing any supervised learning code does:

```
X = the 7 numeric columns        (2200 rows x 7 columns)
y = the label column             (2200 values, each one of 22 crops)
```

The features are all numeric here, which is convenient; the label is text,
which is not, and Week 3 has to encode it. What makes the pair *supervised* is
simply that `y` exists at all.

### Train and test: why we deliberately hide data from ourselves

A model that reproduces the answers it was shown has proved nothing — a lookup
table does that. What we actually care about is **generalisation**: performance
on plots of land the model has never seen.

The standard way to measure that is to split the dataset before training:

```
 2,200 instances
 ├── training set  (~80%)  -> the model is fitted on these
 └── test set      (~20%)  -> hidden away, used once, to estimate real performance
```

The test set is not a formality. It is the only honest estimate of how the
system will behave in the field, and it is honest **only** while it stays
unseen. Every decision made after looking at the test set — choosing an
algorithm, tuning a setting, dropping a feature — leaks information from it and
inflates the score.

Two more terms follow from that:

* **Overfitting** — the model has learned the training examples themselves,
  including their noise, rather than the pattern behind them. It scores
  brilliantly on training data and poorly on the test set.
* **Underfitting** — the model is too simple to capture the pattern, and scores
  poorly on both.

This is **concept only for now**. No splitting happens in Week 1: the
implementation, the reason the split must be *stratified* for a 22-class
target, and the subtler failure mode of *data leakage* are all Week 3.

### Training versus inference

These are two different activities on two different clocks, and confusing them
is the source of a lot of muddled ML systems.

| | **Training** | **Inference** (prediction) |
| --- | --- | --- |
| Input | Many labelled instances (`X` and `y`) | One unlabelled instance (`X` only) |
| Output | A fitted model | A predicted label |
| Runs | Occasionally — offline, on a schedule | Constantly — online, per request |
| Timescale | Seconds to hours | Milliseconds |
| Needs the labels? | Yes | No — that is the whole point |

In this project, training happens from Week 4 onwards on the training set;
inference is what the API in Week 10 and the Streamlit app in Week 11 do —
a farmer supplies seven numbers and gets back a crop name, with no `label`
anywhere in sight.

---

## 6. The ML lifecycle, and where the 12 weeks sit on it

Real machine learning projects follow the same loop regardless of domain. This
course walks it once, in order, without skipping:

```
   ┌──────────────────────────────────────────────────────────────┐
   │                                                              │
   v                                                              │
Frame the problem -> Get the data -> Explore -> Prepare -> Model   │
                                                            |     │
                                                            v     │
        Monitor  <-  Deploy  <-  Productionize  <-  Evaluate ------┘
                                                    (and improve)
```

Mapped onto the plan:

| Stage of the lifecycle | Weeks | What happens |
| --- | --- | --- |
| Frame the problem | 1 | Decide what is being predicted, from what, and why ML at all |
| Get the data | 1 | Load it, and pin down a contract it must satisfy |
| Explore | 2 | Distributions, relationships, class balance, anomalies |
| Prepare | 3 | Train/test split, scaling, encoding the target |
| Model | 4–5 | A baseline first, then real classifiers |
| Evaluate & improve | 6–7 | Model selection, tuning, and explaining the predictions |
| Productionize | 8–9 | Pipelines, packaging, testing the whole path |
| Deploy | 10–11 | An HTTP API, then a user-facing application |
| Monitor & iterate | 12 | Containers, CI, deployment, and watching it in the wild |

Two things to notice.

**It is a loop, not a line.** The arrow back from monitoring to framing is the
real shape of the work: production behaviour reveals that the problem was
framed slightly wrong, or that the data has drifted, and the cycle restarts.

**Earlier stages are cheaper to fix.** A mistake in framing (Week 1) that
survives to deployment (Week 11) invalidates everything built on top of it. A
mistake in a hyperparameter costs a re-run. That asymmetry is why this course
spends a whole week on framing and validation before touching a model, and why
Week 1's dataset contract is written before anything depends on it.

---

## 7. Setting up a reproducible environment

### What is a virtual environment?

A virtual environment is a private folder holding its own copy of the Python
interpreter's package directory. Packages installed while it is active go
there, not into the system-wide Python.

### Why do we need it?

Without one, every project on your machine shares one set of installed
packages. Project A needs pandas 1.5, project B needs pandas 2.2, and only one
of them can win. Worse, installing into the system Python can break OS tools
that depend on it. A virtual environment gives each project an isolated,
disposable dependency set — if it becomes tangled, delete the folder and
rebuild it.

```bash
python -m venv venv               # create it
source venv/bin/activate          # use it (Windows: venv\Scripts\activate)
pip install -r requirements.txt   # install this project's pinned packages
```

Both `venv/` and `.venv/` are listed in `.gitignore`: the folder holds hundreds
of megabytes of files that are rebuildable from `requirements.txt` in seconds. **Commit the recipe, not
the meal.**

### Why every version is pinned

`requirements.txt` says `pandas==2.2.3`, not `pandas`. An unpinned requirement
means "whatever is newest today", so two people installing a week apart get
different code, and a bug appears for one of them and not the other. Pinning
makes the environment reproducible and turns an upgrade into a deliberate,
reviewable commit.

Note the difference between the two files that will eventually exist:
`requirements.txt` at the repo root is the *development* environment and grows
each week; `deployment/requirements.txt`, introduced much later, will hold only
the narrower set needed to serve predictions in production.

### This week's dependencies

Nine packages are pinned. Five are needed to run the Week 1 code; four are
installed now so that the environment is built **once**, from one file, rather
than being rebuilt every week — a student who runs `pip install -r
requirements.txt` in Week 1 should not hit a `ModuleNotFoundError` in Week 2.
Each one is listed with what it does and when this project first needs it.

**`numpy==2.2.1`** — the numeric array library that pandas, scikit-learn,
matplotlib and seaborn are all built on. It provides the fixed-type,
contiguous `ndarray` and vectorised operations on it, which is why a
whole-column calculation in pandas is one fast C loop rather than 2,200 Python
iterations. Used directly this week only in the tests, to build synthetic
frames; used constantly from Week 3 onwards, where feature matrices are numpy
arrays.

**`pandas==2.2.3`** — the library for tabular data in Python. It provides the
`DataFrame`: a table of rows and named, typed columns, with operations for
selecting, filtering, grouping and summarising. Plain Python could hold the CSV
in a list of dictionaries, but every subsequent question ("how many rows per
crop?", "any missing values?") would become a hand-written loop. Needed this
week to read and inspect the CSV.

**`matplotlib==3.10.0`** — the foundational plotting library; everything else
in the Python plotting world either wraps it or imitates it. First genuinely
used in Week 2's exploratory data analysis, where distributions and
relationships have to be *seen* rather than described, and again whenever a
result needs a figure (confusion matrices in Week 4, feature importances in
Week 7).

**`seaborn==0.13.2`** — a statistical plotting layer on top of matplotlib. It
turns multi-line matplotlib code into single calls for the plots that
exploratory analysis needs most (`histplot`, `boxplot`, `pairplot`,
`heatmap`) and it accepts a `DataFrame` with column names directly. Also first
used in Week 2. It does not replace matplotlib — it produces matplotlib
figures, which are then tweaked with matplotlib.

**`scikit-learn==1.6.1`** — the classical machine learning library: consistent
`fit`/`predict`/`transform` interfaces over dozens of algorithms, plus the
splitting, scaling, encoding, cross-validation, metric and pipeline utilities
that surround them. First used in Week 3 for the train/test split and
preprocessing, then for every model from Week 4 onward.

**`jupyter==1.1.1`** — runs notebooks: documents interleaving code, its output
and prose. Notebooks suit the exploratory half of ML work, where you run a
snippet, look at the result, and adjust. Needed this week for
`notebooks/01_problem_definition.ipynb`.

*A caution that applies all course long:* notebooks are for exploring, not for
shipping. Their cells can be executed out of order, producing results nobody
can reproduce, and they resist testing and reuse. This project's rule is that
**logic lives in `src/` and is imported by notebooks.** That is exactly why the
loader is a module rather than a cell — the notebook and the test suite then
exercise the same code path.

**`pytest==8.3.4`** — the test runner. It discovers functions named `test_*`,
runs them, and reports failures with useful context. Needed this week to
enforce the dataset contract automatically instead of by memory.

**`ruff==0.8.4`** — a fast linter and style checker. It catches unused imports,
undefined names, inconsistent import order and missing docstrings. Needed from
week one because "passes lint checks" appears in this project's Definition of
Done, and a standard that is never machine-checked is not a standard.

## 8. Loading the data, and why the loader is a module

Reading a CSV with pandas is one line:

```python
import pandas as pd
frame = pd.read_csv("data/raw/Crop_recommendation.csv")
```

So why does `src/data/data_loader.py` exist at all? Three reasons, each of which
would eventually bite a project that skipped it.

**1. One place knows the path.** That relative path only works if you happen to
launch Python from the repository root — from `notebooks/` it fails. The loader
computes an absolute path from its own location instead:

```python
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
RAW_DATA_PATH: Path = PROJECT_ROOT / "data" / "raw" / "Crop_recommendation.csv"
```

`__file__` is the path of this source file; `.resolve()` makes it absolute; and
`.parents[2]` climbs three levels (`data_loader.py` → `data/` → `src/` → repo root).
The result is correct no matter which directory you started from. If the
dataset ever moves, exactly one line changes.

**2. One place knows the expected shape.** The column names, the row count, the
class count and the exact set of crop names are named constants in
`src/data/validate_schema.py`, so notebooks and tests agree by construction
rather than by copy-paste:

```python
EXPECTED_ROW_COUNT: int = 2_200
EXPECTED_LABEL_COUNT: int = 22
```

(The underscore in `2_200` is Python's digit separator — purely for
readability.)

**3. Failures are loud.** This is the important one, discussed next.

### Fail-fast validation and the dataset contract

The contract lives in `src/data/validate_schema.py` and `load_data()` calls it
immediately after `read_csv`, so **every** future week receives data that has
already been checked. `validate_dataset()` raises `DatasetValidationError`
unless all of the following hold:

| Check | What it catches |
| --- | --- |
| Columns are exactly `N, P, K, temperature, humidity, ph, rainfall, label`, in that order | Renamed columns, stray whitespace in a header (`' ph'`), extra or missing columns, a re-ordered export |
| Row count is exactly 2,200 | A truncated file, a duplicated file, rows appended by accident |
| The seven features are numeric | A stray text cell, a `'N/A'` string, or European decimal commas — any of which make pandas read the whole column as text |
| The seven features have no nulls | Blank cells, which would propagate as `NaN` through every later computation |
| `label` has no nulls | Rows with no answer, which would break training |
| `label`'s unique values are exactly the 22 crops recorded in `validation.md` | A misspelled, renamed, capitalised or brand-new crop |

Why be so strict about a file we control?

**Because the file will not always be the file we think it is.** Datasets get
swapped by hand: someone re-downloads it, re-exports it from a spreadsheet,
appends a few rows from a new region, or "tidies" a header. None of those
edits announce themselves. What arrives afterwards can silently have different
column names, stray whitespace, wrong dtypes or extra label categories — and
`pd.read_csv` will accept all of it without a murmur.

The cost of catching that late is what makes the Week 1 investment worth it:

* A renamed or whitespace-padded column surfaces in **Week 3** as
  `KeyError: 'ph'` in the middle of preprocessing — an error that says nothing
  about the real cause, three weeks and several files away from it.
* A column read as text instead of float surfaces as a scaler failing on
  `'6.5'`, or worse, silently producing nonsense.
* An extra or misspelled crop surfaces in **Week 5** as a label mismatch: the
  encoder learned 22 classes, the data now has 23, and either the code crashes
  in a confusing place or a model is trained on categories that do not mean
  what the report says they mean.
* A truncated file does not surface *at all*. `read_csv` succeeds, training
  succeeds, a model is produced, accuracy is reported — and every number is
  quietly wrong, in a way that looks completely normal. Days can disappear into
  debugging that.

Catching all of these in Week 1, at the single point where data enters the
project, costs one function and a handful of tests. Catching them in Week 3 or
Week 5 costs a debugging session that starts in the wrong place. This is the
**fixed contract** the rest of the repository is built on: from here on, every
week loads data through `load_data()`, and therefore through validation.

The rule this expresses is:

> Corrupt input should stop the program at the moment it is read, not produce a
> plausible-looking wrong answer three stages later.

Note that the missing-file case raises `FileNotFoundError` with a message
saying where the file should be and where to obtain it. Compare that to
returning an empty dataframe — which would fail somewhere much later with an
error mentioning neither the file nor the reason.

The checks are **exact**, not approximate (`== 2200`, not `> 2000`). This is
appropriate here precisely because the dataset is a fixed, version-controlled
artifact. A loose bound would accept a truncated file, which defeats the point.
Data arriving from a live source would need different, statistical checks —
a topic for a later week.

The expected label set is not only in the code: it is written out in
[`validation.md`](validation.md) so a human can read it without opening a
module. Every later week that touches `label` — encoding it in Week 3,
reporting per-class scores in Week 5, returning a prediction from the API in
Week 10 — must match against that recorded set and fail loudly if it does not.

Defining a custom exception type rather than raising bare `ValueError` lets
callers catch *this* problem specifically:

```python
class DatasetValidationError(ValueError):
    ...
```

It subclasses `ValueError` so that code catching the broader type still works.

### Why validation is optional in tests

`load_data(validate=False)` exists for one reason: the test fixture needs
to load the data *without* validating it, so that each test can assert its own
expectation and report a precise failure. If the fixture validated, a single
bad row would make every test fail with the same generic message.

---

## 9. Tests and lint as guard rails

`tests/test_data_loader.py` encodes the dataset contract as executable checks.
Writing them now, in week one, is deliberate: from here on, any change that
damages data loading is caught by `pytest` rather than by a confusing result
weeks later.

The tests are in two groups, and the distinction is worth understanding.

**Contract tests** read the real committed CSV and assert its exact shape.
They carry a skip marker:

```python
requires_raw_dataset = pytest.mark.skipif(
    not RAW_DATA_PATH.is_file(),
    reason=f"Raw dataset missing at {RAW_DATA_PATH}. ...",
)
```

If the dataset is absent, these tests report **skipped with a stated reason**
rather than failing with a confusing error. A skip says "we did not check
this"; a failure says "we checked this and it is broken". Conflating the two
trains people to ignore red test runs.

**Behaviour tests** build a synthetic in-memory frame that satisfies the whole
contract, then break it one way at a time — rename a column, pad a header with
a space, reorder the columns, truncate the rows, insert a null, turn a numeric
column into text, null a label, misspell a crop — and assert that each one is
rejected. A last pair checks that a missing file produces a clear
`FileNotFoundError` and that a malformed CSV on disk is refused by
`load_data()` itself. These need no data file and therefore always run.

That second group is the one that protects every later week. The contract tests
say "today's file is right"; the behaviour tests say "if tomorrow's file is
wrong, we will be told immediately, and told which rule it broke".

Running them:

```bash
ruff check .    # style and correctness
pytest          # behaviour
```

Both must pass before a week is considered complete.

### Common mistakes

* **Testing that the code runs, not that it is right.** `assert frame is not
  None` passes even when the data is garbage. Assert the specific facts.
* **Deleting or weakening a test to make the suite green.** The test is the
  specification; if it fails, either the code or the specification is wrong,
  and you must decide which.

---

## 10. Where this leaves us

We can now state the problem precisely, and load the data with confidence that
it is the data we think it is. We have said nothing yet about what the numbers
*look like*: how the crops are distributed, whether features overlap, whether
anything is anomalous. That is exploratory data analysis — **Week 2**.

## Recap of new terms

Full definitions live in [`docs/glossary.md`](../../glossary.md); the concept
index is [`docs/ml_concepts.md`](../../ml_concepts.md).

Artificial intelligence · machine learning · deep learning · supervised
learning · unsupervised learning · classification · regression · multiclass
classification · feature · target · label · instance · dataset · training set ·
test set · generalisation · overfitting · underfitting · training · inference ·
ML lifecycle · batch learning · dataframe · virtual environment · pinned
dependency · dataset contract · expected label set · fail-fast validation ·
linting.
