# Week 1 — Exercises

Work through these in order. Beginner exercises check that you can reproduce
what the notes covered; intermediate exercises make you apply it somewhere new;
challenge exercises deliberately go slightly beyond the notes.

Keep scratch work in a new notebook or a scratch script — do not edit
`src/data/data_loader.py`, `src/data/validate_schema.py` or the existing tests
unless an exercise says so.

---

## Beginner

**B1 — Build the environment.**
Create a virtual environment, activate it, install `requirements.txt`, then run
`pip list`. Confirm the nine Week 1 packages appear at exactly the pinned
versions. Why does `pip list` show far more than nine packages?

**B2 — Load the data.**
In a Python shell, load the dataset and print its shape:

```python
from src.data import load_data
frame = load_data()
print(frame.shape)
```

Then print `frame.head()`, `frame.columns` and `frame.dtypes`. Write one
sentence describing what each of those three tells you.

**B3 — Count the crops.**
Print `frame["label"].value_counts()`. How many rows does each crop have? What
does that even distribution suggest about how the dataset was assembled?

**B4 — Classify the problem.**
Without re-reading the notes, write down four answers: supervised or
unsupervised, classification or regression, binary or multiclass, batch or
online. Justify each in one sentence, then check yourself against
`syllabus.md`.

**B5 — Read the failure message.**
Call `load_data(path="does_not_exist.csv")`. Read the resulting
`FileNotFoundError` carefully. Name two pieces of information the message gives
you that a bare `FileNotFoundError: does_not_exist.csv` would not.

**B6 — Run the guard rails.**
Run `ruff check .` and `pytest -v`. How many tests ran? All of them should
pass. Then rename `data/raw/Crop_recommendation.csv` temporarily, re-run, and
explain in your own words why the contract tests now report *skipped* rather
than *failed*. Restore the file afterwards.

**B7 — Break the contract on purpose.**
In a scratch script, load the data with `load_data()`, rename the `ph` column
to `' ph'` (note the leading space), and pass the result to
`validate_dataset()`. Read the error. Why is a stray space in a header exactly
the kind of thing that is cheap to catch now and expensive to catch in Week 3?

---

## Intermediate

**I1 — Make a test fail on purpose.**
Temporarily change `EXPECTED_ROW_COUNT` in `src/data/validate_schema.py` to
`2199` and run `pytest`. Which tests fail, and what do the messages say? **Revert the
change** and confirm the suite is green again. What does this exercise
demonstrate about the value of exact rather than approximate checks?

**I2 — Break the lint.**
Add `import os` at the top of `src/data/data_loader.py` without using it, and
run `ruff check .`. Note the rule code reported. Now remove it, and instead delete
a function's docstring and re-run. What rule code fires this time? Restore the
file afterwards.

**I3 — Extend the contract.**
`ph` is a pH value, so it must lie between 0 and 14, and `humidity` is a
percentage, so it must lie between 0 and 100. Add these range checks to
`validate_dataset()`, and add corresponding tests. Make sure `pytest` and
`ruff check .` both pass afterwards.

**I4 — Frame a different problem.**
Suppose the request changed to "predict how many kilograms per hectare this
field will produce". Write a problem statement for it in the same style as the
notes: inputs, output, problem type. Which of the four classification axes
changes, and why?

**I5 — Why not a lookup table?**
Estimate how many rows a lookup table would need in order to answer any query
by exact match, if each of the seven features were bucketed into just 10 ranges.
Compare that number to 2,200. What does the comparison tell you about why the
model must *generalise* rather than memorise?

---

## Challenge

**C1 — Path independence.**
`RAW_DATA_PATH` is built with `Path(__file__).resolve().parents[2]`. Verify
that the loader works from at least three different working directories (repo
root, `notebooks/`, and `/tmp`). Then explain precisely what would break if
`data_loader.py` were moved to `src/data_loader.py` without changing that line.

**C2 — Design a contract for streaming data.**
Our exact checks (`== 2200`) suit a fixed file. Imagine the data now arrives
daily from field sensors. Design a validation strategy for that setting: what
would you check, what would you refuse to check, and what should happen when a
check fails — reject the batch, warn, or something else? Write half a page.

**C3 — Argue against machine learning.**
Build the strongest case you can that this problem should be solved with
hand-written agronomic rules instead of ML. What would you need to know about
the domain and the users for that case to win? This is not a rhetorical
exercise — knowing when *not* to apply ML is part of the skill.

**C4 — Investigate the data's origin.**
The dataset is suspiciously tidy: no missing values, and exactly 100 rows per
crop. Real agricultural measurements are never like this. What does that imply
about how the file was produced, and what does it imply about the confidence
you should place in a model trained on it? Note your hypothesis — Week 2's
exploration will give you evidence to test it against.
