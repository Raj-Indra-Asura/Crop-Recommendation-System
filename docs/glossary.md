# Glossary

Alphabetical quick reference for terms used across the project. For the
teaching order and fuller treatment, see
[`docs/ml_concepts.md`](ml_concepts.md) and the weekly learning notes.

Terms are added as they are introduced. Entries marked *(W1)* were introduced
in Week 1.

---

**Artificial intelligence** *(W1)* — The broad field of building software that
performs tasks we would call intelligent. Machine learning is one family of
techniques within it; deep learning is one family within machine learning.

**Batch learning** *(W1)* — Training a model once on a complete dataset to
produce a fixed artifact, which is then deployed and periodically retrained.
Contrast with online learning, where the model updates continuously as data
arrives.

**Classification** *(W1)* — A supervised learning task whose output is a
category rather than a number. Predicting which of 22 crops suits a field is
classification.

**Dataframe** *(W1)* — pandas' table structure: rows and named, typed columns,
with built-in operations for filtering, grouping and summarising.

**Dataset** *(W1)* — The complete collection of instances available for
training and evaluation.

**Dataset contract** *(W1)* — The properties input data is required to satisfy
— here: exact column names and order, exactly 2,200 rows, numeric features, no
missing values, and exactly the 22 recorded crop names. Enforced by
`validate_dataset()` in `src/data/validate_schema.py`.

**Deep learning** *(W1)* — Machine learning using many-layered neural networks.
Not used in this project: with 2,200 rows and seven numeric features, classical
algorithms are both stronger and easier to explain.

**Expected label set** *(W1)* — The exact set of 22 crop names recorded in Week
1 (`EXPECTED_LABELS`, and written out in
`docs/curriculum/week01/validation.md`). Every later week that touches `label`
must match against it and fail loudly if it differs.

**Generalisation** *(W1)* — Performing well on instances never seen during
training. The actual goal of machine learning, as opposed to reproducing the
training data.

**Inference** *(W1)* — Using a trained model to predict the label of a new,
unlabelled instance. Happens per request, in milliseconds. Contrast with
training.

**`DatasetValidationError`** *(W1)* — This project's custom exception, raised
when loaded data violates the dataset contract. Subclasses `ValueError`.

**Fail-fast validation** *(W1)* — Design principle: detect invalid input at the
moment it is read and stop, rather than continuing and producing a
plausible-looking but wrong result later.

**Feature** *(W1)* — One input variable used to make a prediction. This project
has seven: `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`.

**Instance** *(W1)* — One complete example: a set of feature values together
with its target. Also called a sample, observation or row.

**Label** *(W1)* — The target value attached to an instance. Here, the crop
name; also the literal name of the target column.

**Linting** *(W1)* — Automatic inspection of source code for style violations
and likely errors. Performed here by `ruff`.

**Machine learning** *(W1)* — Building software by supplying examples of the
desired behaviour and letting an algorithm infer rules that reproduce them and
generalise to unseen cases.

**ML lifecycle** *(W1)* — The loop a project travels: frame the problem, get
the data, explore, prepare, model, evaluate and improve, productionize, deploy,
monitor — and back to framing.

**Multiclass classification** *(W1)* — Classification with more than two
possible categories. Contrast with binary classification, which has exactly
two.

**Overfitting** *(W1)* — When a model learns the training examples and their
noise instead of the pattern behind them: strong on training data, weak on
unseen data. Its opposite, underfitting, is being too simple to capture the
pattern at all.

**Pinned dependency** *(W1)* — A dependency specified at an exact version
(`pandas==2.2.3`) rather than loosely (`pandas`), so that every install
produces an identical environment.

**Processed data** *(W1)* — Data derived from the raw input by cleaning,
splitting or transformation. Written to `data/processed/`; never written back
over the raw data.

**Raw data** *(W1)* — The original, unmodified dataset in `data/raw/`. Treated
as strictly read-only so it remains the recoverable source of truth.

**Regression** *(W1)* — A supervised learning task whose output is a number on
a continuous scale, such as predicting yield in kilograms.

**Reproducibility** *(W1)* — The property that the same code and data yield the
same results for any person at any time. The reason this project commits its
dataset and pins its dependency versions.

**Supervised learning** *(W1)* — Learning from examples in which the correct
answer is provided alongside each input.

**Target** *(W1)* — The value a model is trained to predict. In this project,
the `label` column.

**Test set** *(W1)* — The portion of the dataset held back from training and
used, ideally once, to estimate performance on unseen data. Implemented in
Week 3.

**Training** *(W1)* — The offline process of fitting a model to labelled
examples. Contrast with inference.

**Training set** *(W1)* — The portion of the dataset a model is fitted on.

**Unsupervised learning** *(W1)* — Learning patterns or structure from data
where no correct answers are supplied.

**Virtual environment** *(W1)* — An isolated, per-project Python package
directory created with `venv`, preventing dependency conflicts between
projects. Stored in `venv/` (or `.venv/`) and never committed.
