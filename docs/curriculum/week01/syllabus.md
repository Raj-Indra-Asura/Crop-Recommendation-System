# Week 1 — Framing the Problem and Meeting the Data

> 🗺 [Roadmap](../README.md) › [Part I — Foundations (Weeks 1-3)](../README.md#part-i--foundations-weeks-1-3) › [Chapter 1 — Framing the Problem and Meeting the Data](README.md) › **§1.1 Syllabus**

## Title

**From a vague wish to a machine learning problem statement**

## Learning objectives

By the end of this week a student should be able to:

1. Explain what machine learning is, and describe the specific situations in
   which it beats writing ordinary rules by hand.
2. Classify a problem along the standard axes: supervised vs. unsupervised,
   classification vs. regression, batch vs. online.
3. Turn an informal request ("tell farmers what to plant") into a written
   problem statement with named inputs, a named output, and a stated success
   measure.
4. Set up a reproducible Python 3.11 working environment using `venv` and a
   pinned `requirements.txt`.
5. Load a CSV into a pandas dataframe and inspect its shape, columns, data
   types and missing values.
6. Explain the difference between training and inference, and describe (not yet
   implement) what a train/test split is for.
7. Explain why a project validates its input data at load time, and write an
   automated test that enforces a dataset contract.

## Prerequisites

* Basic Python: variables, functions, imports, running a script from a terminal.
* Comfort with a command line: changing directory, running a command, reading
  an error message.
* No prior machine learning, statistics or pandas knowledge is assumed. This is
  the first week; everything ML-specific starts here.

## Concepts covered

| Concept | First introduced |
| --- | --- |
| What machine learning is, and when to use it | Week 1 |
| Supervised vs. unsupervised learning | Week 1 |
| Classification vs. regression | Week 1 |
| Multiclass classification | Week 1 |
| Features and target | Week 1 |
| Training set / test set (concept only) | Week 1 |
| Generalisation, overfitting, underfitting | Week 1 |
| Training vs. inference | Week 1 |
| Instance (row / sample / observation) | Week 1 |
| Label | Week 1 |
| The ML project lifecycle | Week 1 |
| Reproducible environments (`venv`, pinned dependencies) | Week 1 |
| Dataframe | Week 1 |
| Dataset contract and fail-fast validation | Week 1 |
| Linting and automated testing as guard rails | Week 1 |

Every one of these also has a short definition in
[`docs/ml_concepts.md`](../../ml_concepts.md) and
[`docs/glossary.md`](../../glossary.md).

## Connection to the previous week

None — this is the starting point of the course. Later weeks will keep
back-references short and point here rather than re-explaining the basics.

## Connection to the ML lifecycle

The lifecycle we follow for the whole course is:

```
Frame the problem -> Get the data -> Explore -> Prepare -> Model
    -> Evaluate -> Improve -> Productionize -> Deploy -> Monitor
```

Week 1 covers the **first two stages only**: framing the problem and getting
the data into memory in a trustworthy way. It is tempting to skip straight to
model training; the reason we do not is that every later stage inherits the
mistakes made here. A misframed problem produces a model that is accurate at
answering the wrong question.

## Expected student outcome

### The student CAN, after this week

* State the crop recommendation task as a supervised, multiclass
  classification problem, and justify that classification.
* Name the seven input features and the target column without looking them up.
* Create and activate a virtual environment, and install pinned dependencies.
* Load the raw dataset with `load_data()` and describe its shape.
* Explain what each pinned Week 1 dependency contributes, and which are used
  now (`numpy`, `pandas`, `jupyter`, `pytest`, `ruff`) versus installed now for
  later weeks (`matplotlib`, `seaborn`, `scikit-learn`).
* Say what a train/test split is *for*, and distinguish training from
  inference — without having implemented either.
* Run `ruff check .` and `pytest` and interpret the result.
* Explain why the loader raises an exception rather than returning an empty
  dataframe when the CSV is absent.

### The student CANNOT yet

* Visualise or statistically summarise the data — that is Week 2 (EDA).
* Actually split data into training and test sets, choose a split ratio,
  stratify it, or explain data leakage — Week 3.
* Scale features or encode the target — Week 3.
* Train any model at all, including a baseline — Week 4.
* Say anything about which crops are hard to tell apart — that needs Week 2's
  exploration and Week 5's models.
* Measure accuracy, precision or recall — Week 4 onwards.
* Serve predictions over HTTP or through a UI — Weeks 9–11.

## Deliverables for the week

* `src/data/data_loader.py` — `load_data()`, the single entry point to the data.
* `src/data/validate_schema.py` — `validate_dataset()`, the dataset contract.
* `tests/test_data_loader.py` — automated enforcement of that contract.
* `notebooks/01_problem_definition.ipynb` — the written problem framing plus a
  first look at the loaded dataframe.
* `requirements.txt` — nine pinned packages.
* This week's four curriculum documents, including the recorded expected label
  set in `validation.md`.

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| ◀ [Chapter 1 — Framing the Problem and Meeting the Data](README.md) | [Chapter 1 — Framing the Problem and Meeting the Data](README.md) · 🗺 [Roadmap](../README.md) | [§1.2 Learning notes](learning_notes.md) ▶ |

<!-- nav:end -->
