# Week 4 — Exercises

Work through these in order. Beginner exercises check that you can reproduce
what the notes covered; intermediate exercises make you apply it somewhere new;
challenge exercises deliberately go slightly beyond the notes.

Keep scratch work in a new notebook or a scratch script. Do not edit
`notebooks/04_baseline_models.ipynb`, `src/models/baseline.py`,
`src/evaluation/metrics.py` or `tests/test_baseline.py` unless an exercise says
so — and **never** modify `data/raw/Crop_recommendation.csv`.

Everything this week happens on the **training rows only**.
`data/processed/test.csv` stays closed until Week 8; an exercise that needs
held-out data uses a cross-validation fold or an inner split of the training
set.

Most exercises start from the same lines:

```python
import pandas as pd
from src.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluation import build_cv, cross_validated_accuracy, evaluate_model
from src.models import get_baseline_model

train = pd.read_csv("data/processed/train.csv")
FEATURES = list(FEATURE_COLUMNS)
X, y = train[FEATURES], train[TARGET_COLUMN]
```

---

## Beginner

**B1 — Predict the baseline before measuring it.**
Without running anything: how many classes does `y` have, how many rows does
each hold, and what accuracy will a `most_frequent` `DummyClassifier` get?
Write the number down *first*, then fit the model and check. If your prediction
was wrong, work out which of the two assumptions — the class count or the
balance — was the mistake.

**B2 — Fit the baseline and look at what it predicts.**
Fit `get_baseline_model("most_frequent")` on `X, y` and print
`set(model.predict(X))`. How many distinct crops does it ever name? Which one,
and what decided that? Would the answer change if the dataset had 101 rows of
`rice`?

**B3 — Prove the features are ignored.**
Shuffle the rows of `X` (`X.sample(frac=1.0, random_state=0)`) and predict
again. Are the predictions identical? Now try predicting from a frame of all
zeros with the same column names. Explain in one sentence why neither changes
anything, and why that is the defining property of a baseline.

**B4 — Read the classification report.**
Call `evaluate_model(model, X, y)` and print `result["report"]`. What is the
recall for `apple`? For `banana`? What is the precision for `apple`, and why is
it 0.05 rather than 1.00? Write one sentence explaining what the accuracy figure
hid.

**B5 — Compare the four strategies.**
Cross-validate all four names in `BASELINE_STRATEGIES` with
`cross_validated_accuracy` and tabulate mean and std. Which two have a standard
deviation of exactly zero, and why? Which strategy scored highest, and why is
that *not* a reason to quote it as the baseline?

**B6 — Read the fold structure.**
Loop over `build_cv().split(X, y)` and print, for each fold, how many rows are
fitted on, how many validated on, and how many distinct crops appear in the
validation part. Then confirm that every row index appears in exactly one
validation fold across the five.

**B7 — One split, ten answers.**
Split the training rows into an inner train/validation pair ten times with ten
seeds, score `get_baseline_model("stratified")` each time, and report the min,
max and range. Then answer: if a colleague showed you the best of those ten
numbers as "the accuracy", what exactly would be wrong with it?

**B8 — Break the pipeline on purpose.**
Shuffle `y` (`y.sample(frac=1.0, random_state=0).reset_index(drop=True)`) so the
labels no longer match their rows, then cross-validate the baseline against the
shuffled labels. Does the score change? Explain why, and why this means the
baseline can catch some pipeline bugs but not this one.

**B9 — State the number from memory.**
Close the notebook. Write down, without looking: the baseline accuracy, the
strategy that produced it, the number of folds, the seed, and which rows it was
computed on. Then check yourself against
[the learning notes §4](learning_notes.md).

**B10 — Run the guard rails.**
Run `pytest tests/test_baseline.py` and `ruff check .`. How many tests ran? Then
execute the notebook end to end with
`jupyter nbconvert --to notebook --execute notebooks/04_baseline_models.ipynb`
and confirm it exits 0.

---

## Intermediate

**I1 — Derive 1/k, then falsify it.**
Take a stratified subsample of the training rows containing only 5 crops, and
predict the `most_frequent` baseline's accuracy before measuring it. Repeat with
10 and with 22. Plot predicted against measured. Now make the 5-crop subset
imbalanced — 200 rows of one crop, 20 of each of the others — and predict again.
Which formula applies to which case, and why is "1/k" the wrong rule for the
second?

**I2 — Build the misleading-accuracy example yourself.**
Reframe the training rows as a two-class problem for a crop of your choice
(`y == "banana"` versus everything else). What accuracy does the
`most_frequent` baseline get? What are precision and recall for the positive
class? Now repeat the reframing for a *group* of crops — say all cereals versus
the rest — so the split is closer to 50/50. How does the baseline's accuracy
move, and what does that tell you about what accuracy is really measuring?

**I3 — How much does k matter?**
Cross-validate the `stratified` baseline with `n_splits` of 2, 3, 5, 10 and 20.
Tabulate the mean and standard deviation for each. Which way does the standard
deviation move as k grows, and why? How much longer does k = 20 take? State the
trade-off in one sentence.

**I4 — Stratified folds versus plain folds.**
Cross-validate the baseline twice, once with `StratifiedKFold` and once with
plain `KFold(n_splits=5, shuffle=True, random_state=42)`. Compare the means and
the per-fold spreads. Then construct a case where the difference is dramatic:
sort the training frame by `label` before splitting with `KFold(shuffle=False)`
and look at what each fold contains. What score do you get, and why?

**I5 — Seeds and the illusion of tuning.**
Cross-validate the `stratified` baseline with `random_state` running from 0 to
19 and record the 20 means. What are the min and max? Now imagine a colleague
reports the maximum as their model's accuracy. Quantify how much "improvement"
they appear to have made over the mean, and relate this to Week 3's rule about
never choosing a seed to flatter a result.

**I6 — A first strong reference point.**
Cross-validate `DummyClassifier(strategy="constant", constant="rice")`. What is
its accuracy, and how does it compare to `most_frequent`? Then explain why
`get_baseline_model` deliberately does not expose `constant`, and what argument
a factory would need to support it.

**I7 — The leak cross-validation is designed to prevent.**
Build two versions of a scaled baseline: (a) fit a `StandardScaler` on all of
`X` first, then cross-validate the dummy on the scaled array; (b) cross-validate
a `Pipeline([("preprocess", build_preprocessor()), ("model",
get_baseline_model())])` on the raw frame. Do the scores differ here? Explain why
they do not for a *dummy* — and why (a) is nevertheless wrong and would matter
for a real model in Week 5.

**I8 — Extend the metrics module (in a copy).**
Copy `evaluate_model` into a scratch file and add a `macro_f1` key using
`sklearn.metrics.f1_score(..., average="macro")`. Compute it for the baseline on
the balanced 22-class problem and on your I2 binary framing. Which of accuracy
and macro-F1 changes more between the two, and why does that make macro-F1 the
more informative number under imbalance?

**I9 — A single bad fold.**
Construct a training frame where one fold scores far worse than the others: for
example, cross-validate with `KFold(shuffle=False)` on data sorted by crop. Look
at the resulting score array. Write two sentences on why averaging that array
would be the wrong response, and what you would investigate instead.

---

## Challenge

**C1 — Reimplement k-fold cross-validation.**
Using only `numpy` and `pandas`, write `my_cross_val_score(model, X, y, k,
random_state)` that builds stratified folds, clones the model for each fold
(`sklearn.base.clone`), fits, predicts and returns an array of accuracies. Match
`cross_val_score`'s output on the baseline exactly. Then answer: why must the
model be cloned rather than re-fitted in place, and what would break if you
forgot?

**C2 — Derive the distribution of the `stratified` baseline.**
The `stratified` strategy guesses class `c` with probability equal to class
`c`'s training share. Derive, on paper, its *expected* accuracy on a balanced
k-class problem, and the standard deviation of its accuracy on a validation fold
of n rows. Then run it 1,000 times with different seeds and compare your formula
to the empirical mean and spread. How many folds would you need for the observed
`stratified` mean of 4.72% to be distinguishable from `most_frequent`'s 4.55%?

**C3 — Argue for a different baseline.**
This project quotes `most_frequent`. Make the strongest case that `stratified`
or `uniform` should be quoted instead, then the strongest case against. Then
consider a genuinely different kind of baseline: a single rule such as "predict
`rice` if rainfall > 200 else `maize`". What would it score, and what does its
existence say about how demanding a "real" model's bar should be?

**C4 — Design the evaluation protocol for a harder problem.**
Suppose the dataset were 10× larger, imbalanced (some crops with 30 rows, some
with 30,000), and collected over five years. Write half a page specifying the
evaluation protocol you would use: which metric, which splitter, how many folds,
what stratification, what you would do about the time dimension, and what your
baseline would be. Justify each choice against a specific failure it prevents.

**C5 — When 99% is a bug.**
The learning notes claim this dataset is easy and later models will hit
98-99%+. Write out the checklist you would run through, on a *real* project,
before believing a 99% first result. For each item, say what evidence would
confirm or refute it, and point to where in Weeks 1-4 of this repository that
evidence already exists for this dataset.

**C6 — Nested cross-validation, on paper.**
Week 6 will tune hyperparameters using cross-validation. Explain, in prose, why
tuning and estimating performance with the *same* cross-validation loop produces
an optimistic number, what nested cross-validation does about it, and what it
costs. Then say which of the two loops this project's Week 8 test set is playing
the role of.
