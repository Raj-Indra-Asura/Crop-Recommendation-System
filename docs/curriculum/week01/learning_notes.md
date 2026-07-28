# Week 1 — Learning Notes

> How to read these notes: every new idea is introduced with four questions —
> *what is it, why do we need it, where is it used, how does it actually work*
> — followed by the mistakes people commonly make with it. Code blocks are
> always explained, either just before or just after they appear.

---

## 1. What machine learning actually is

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

## 5. Setting up a reproducible environment

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
python3.11 -m venv .venv          # create it
source .venv/bin/activate         # use it (Windows: .venv\Scripts\activate)
pip install -r requirements.txt   # install this project's pinned packages
```

`.venv` is listed in `.gitignore`: it holds hundreds of megabytes of files that
are rebuildable from `requirements.txt` in seconds. **Commit the recipe, not
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

### This week's four dependencies

Each is introduced because it is needed *now*, not because it might be useful
later.

**`pandas==2.2.3`** — the library for tabular data in Python. It provides the
`DataFrame`: a table of rows and named, typed columns, with operations for
selecting, filtering, grouping and summarising. Plain Python could hold the CSV
in a list of dictionaries, but every subsequent question ("how many rows per
crop?", "any missing values?") would become a hand-written loop. Needed this
week to read and inspect the CSV.

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

---

## 6. Loading the data, and why the loader is a module

Reading a CSV with pandas is one line:

```python
import pandas as pd
frame = pd.read_csv("data/raw/Crop_recommendation.csv")
```

So why does `src/data/loader.py` exist at all? Three reasons, each of which
would eventually bite a project that skipped it.

**1. One place knows the path.** That relative path only works if you happen to
launch Python from the repository root — from `notebooks/` it fails. The loader
computes an absolute path from its own location instead:

```python
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
RAW_DATA_PATH: Path = PROJECT_ROOT / "data" / "raw" / "Crop_recommendation.csv"
```

`__file__` is the path of this source file; `.resolve()` makes it absolute; and
`.parents[2]` climbs three levels (`loader.py` → `data/` → `src/` → repo root).
The result is correct no matter which directory you started from. If the
dataset ever moves, exactly one line changes.

**2. One place knows the expected shape.** The column names, the row count and
the class count are named constants, so notebooks and tests agree by
construction rather than by copy-paste:

```python
EXPECTED_ROW_COUNT: int = 2_200
EXPECTED_LABEL_COUNT: int = 22
```

(The underscore in `2_200` is Python's digit separator — purely for
readability.)

**3. Failures are loud.** This is the important one, discussed next.

### Fail-fast validation and the dataset contract

`load_raw_data()` validates by default, and `validate_raw_data()` raises
`DatasetValidationError` if anything deviates: wrong columns, wrong column
*order*, wrong row count, any missing value, wrong number of crops.

Why be so strict about a file we control?

Because of the failure mode this prevents. Imagine the CSV is accidentally
truncated to 1,500 rows. Nothing crashes. `read_csv` succeeds, training
succeeds, a model is produced, accuracy is reported — and every number is
quietly wrong, in a way that looks completely normal. Days can disappear into
debugging that.

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

Defining a custom exception type rather than raising bare `ValueError` lets
callers catch *this* problem specifically:

```python
class DatasetValidationError(ValueError):
    ...
```

It subclasses `ValueError` so that code catching the broader type still works.

### Why validation is optional in tests

`load_raw_data(validate=False)` exists for one reason: the test fixture needs
to load the data *without* validating it, so that each test can assert its own
expectation and report a precise failure. If the fixture validated, a single
bad row would make every test fail with the same generic message.

---

## 7. Tests and lint as guard rails

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

**Behaviour tests** build tiny in-memory dataframes to check the loader's own
logic — that a renamed column is rejected, that a wrong row count is rejected,
that a missing file produces a clear `FileNotFoundError`. These need no data
file and therefore always run.

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

## 8. Where this leaves us

We can now state the problem precisely, and load the data with confidence that
it is the data we think it is. We have said nothing yet about what the numbers
*look like*: how the crops are distributed, whether features overlap, whether
anything is anomalous. That is exploratory data analysis — **Week 2**.

## Recap of new terms

Full definitions live in [`docs/glossary.md`](../../glossary.md); the concept
index is [`docs/ml_concepts.md`](../../ml_concepts.md).

Machine learning · supervised learning · unsupervised learning ·
classification · regression · multiclass classification · feature · target ·
label · instance · dataset · batch learning · dataframe · virtual environment ·
pinned dependency · dataset contract · fail-fast validation · linting.
