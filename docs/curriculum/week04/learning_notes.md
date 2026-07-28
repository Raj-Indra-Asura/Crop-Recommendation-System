# Week 4 — Learning Notes

**Baselines and cross-validation: deciding what "good" means before there is
anything good to measure.**

Weeks 1-3 got the data into shape. The natural next move is to train a
classifier and see what it scores — and that move is postponed by exactly one
week, because a score has no meaning on its own and a student who learns to
report bare accuracies learns a habit that is very hard to unlearn.

This week therefore builds a model designed to be *stupid*, measures it
carefully, and writes the result down. Everything Weeks 5-8 claim will be
claimed relative to that number.

Code produced this week: `src/models/baseline.py`,
`src/evaluation/metrics.py`, `tests/test_baseline.py`,
`notebooks/04_baseline_models.ipynb`.

---

## 0. What "evaluation" is, and why it comes before modelling

**Evaluation** is measuring how well a model performs on data it was not fitted
on, using a metric chosen in advance and a procedure that can be repeated.

Three parts, and all three have to be fixed before the first model is trained:

* **the metric** — what counts as a correct answer, and what a mistake costs;
* **the protocol** — what the model is fitted on and what it is scored on;
* **the reference point** — the number that says whether the score is good.

Fixing them afterwards is how projects fool themselves. If the metric is chosen
after the results are in, it will be the metric the favourite model wins on. If
the protocol is chosen afterwards, it will be the split that flatters. And if
there is no reference point at all, any number can be presented as a success,
because nobody in the room knows what an untrained guess would have scored.

This week fixes all three:

| Part | This project's choice | Where |
| --- | --- | --- |
| Metric | Accuracy, with the per-class report always printed beside it | §1, §5 |
| Protocol | 5-fold stratified cross-validation on the training rows, seed 42 | §7 |
| Reference point | 4.55% — a `most_frequent` `DummyClassifier` | §4 |

### Common mistakes

* Training a model first and deciding how to judge it second.
* Reporting a single number with no baseline, no sample size and no spread.
* Treating the *test* set as the thing you tune against. It is not; §8.

---

## 1. Accuracy

### What it is

**Accuracy** is the share of predictions that were correct:

```
accuracy = number of correct predictions / total number of predictions
```

Nothing more. It ranges from 0 to 1 (often quoted as a percentage), it is
defined for any number of classes, and it is what `evaluate_model` returns as
its `accuracy` key.

### Why it is a fair first metric *here*

Accuracy weights every row equally, which means it is dominated by whichever
class holds the most rows. In this dataset each of the 22 crops holds exactly
100 rows (Week 2 §2), so no class can dominate the average: getting `rice`
completely wrong costs exactly as much as getting `maize` completely wrong.
Accuracy on balanced data with equal error costs is exactly the situation
accuracy is designed for.

### Why it is not the last metric

Two things accuracy cannot express, even here:

* **Which** classes the mistakes fall on. One number cannot distinguish "wrong
  on 5% of every crop" from "perfect on 21 crops and useless on the 22nd".
* **What a mistake costs.** Recommending a crop that grows a little worse and
  recommending one that cannot grow at all are both "one error".

Both are taken up in §5, and properly in Week 8.

### Common mistakes

* Quoting accuracy without the number of rows it was computed on. `92%` on 25
  rows is 23 correct answers, and one row is worth four percentage points.
* Quoting accuracy on the data the model was fitted on. For the dummy in §3 that
  happens to be harmless — it has nothing to memorise with — but for any real
  model it measures memory, not learning.
* Assuming an accuracy is comparable across datasets. It is not: 80% on a
  two-class problem is barely above guessing, 80% on this 22-class problem is a
  very strong result.

---

## 2. The baseline, and why it always comes first

### What a baseline is

A **baseline** is a model built to be as unintelligent as possible while still
being a legal model. It is fitted on the same training rows, obeys the same
`fit`/`predict` API, and is scored with the same metric and the same protocol as
any real candidate — but it does not look at the features at all.

Its score is therefore the value of *knowing the label distribution and nothing
else*. Whatever a real model achieves above that line is the part attributable
to `N`, `P`, `K`, temperature, humidity, pH and rainfall.

### Why you always build one

**An accuracy is a comparison, and the baseline is its other side.** "97%" is
uninterpretable in isolation. On a dataset where 96% of rows share one label,
97% is almost worthless. On this dataset, where a guess scores 4.5%, 97% is an
enormous amount of learned structure. Same number, opposite verdicts, and only
the baseline distinguishes them.

**It defines "broken".** A model that does not beat the baseline has learned
nothing from the features. Not "underperformed" — *learned nothing*. That is a
bug report, not a result, and the usual causes are mundane:

* labels shuffled relative to the features, or a misaligned join or index;
* the model fitted on the wrong array (all-zeros, one column, the wrong split);
* the score computed against the wrong vector;
* a target accidentally constant after a filter.

Without a floor, all of these produce numbers that look like modest results and
get reported as such.

**It is the cheapest sanity check available.** Fitting a dummy exercises the
whole path — load, split, fit, predict, score — in milliseconds, before any
modelling choice can hide a plumbing bug behind a plausible score. If the
baseline pipeline runs, the shapes line up, the label encoding survived and the
metric is wired to the right vectors.

**It sets the unit of improvement.** "4.55% → 99%" is a result. "99%" is a
number.

### What a baseline is not

It is not a *lower bound on what is achievable*, and it is not a target. It is a
floor below which a model is certainly broken. Beating it is necessary, not
sufficient: a model at 6% has beaten the 4.55% baseline and is still useless.

Beyond the naive baseline there are stronger reference points a mature project
also keeps: the score of the simplest sensible real model (Week 5's logistic
regression will serve), the current production model's score, and the accuracy
of a human expert doing the same task. Each raises the bar. The naive baseline
is simply the first and the cheapest.

### Common mistakes

* Building the baseline *after* the real model, "to have something to compare
  against". By then the comparison has already been made informally, and the
  bar has been set by whatever the real model happened to do.
* Skipping it because the problem is "obviously hard". The obviousness is the
  problem: a 96%-majority dataset also looks hard until a constant guess scores
  96%.
* Treating a baseline that scores well as a reason to stop. It is a reason to
  ask why — usually imbalance (§5).

---

## 3. `DummyClassifier` and its strategies

scikit-learn's baseline is `sklearn.dummy.DummyClassifier`. `fit(X, y)` accepts
`X` and ignores it entirely; only `y` is examined. `predict(X)` then answers
from the recorded label distribution, using one of several **strategies**.

| Strategy | Predicts | Deterministic? | Accuracy on balanced data |
| --- | --- | --- | --- |
| `most_frequent` | Always the most common training class | Yes | 1/k |
| `prior` | Same labels as `most_frequent`; differs in `predict_proba` | Yes | 1/k |
| `stratified` | A random class, drawn in proportion to training frequencies | No | ≈1/k |
| `uniform` | A random class, every class equally likely | No | ≈1/k |
| `constant` | A class you name | Yes | that class's share |

The two that matter conceptually are the first and third.

**`most_frequent`** is the "always answer the same thing" baseline. It is the
right default because it is deterministic — the number it produces is the same
for every student on every run — and because on imbalanced data it is exactly
the trap accuracy sets (§5). On perfectly balanced data there is no true
majority, so scikit-learn breaks the tie by class order: `apple`, alphabetically
first, wins. The tie-break is arbitrary and carries no meaning.

**`stratified`** is the "monkey with a weighted die" baseline: it guesses
randomly but respects how common each class was. It is a slightly *stronger*
opponent than `most_frequent` on imbalanced data in the sense of being less
degenerate, and its score wobbles from run to run, which is why
`get_baseline_model` gives it the project seed.

**`uniform`** ignores the class frequencies too, so it differs from `stratified`
only when the classes are imbalanced. Here the two are statistically
indistinguishable.

`prior` and `most_frequent` produce identical *predictions*; they differ only in
`predict_proba`, where `prior` returns the observed class distribution
(`[0.045, 0.045, ...]`) and `most_frequent` returns a hard `1.0` on the winner.
Nothing this week depends on that difference; it matters when a downstream step
consumes probabilities.

`constant` is deliberately not exposed by `get_baseline_model`, because it needs
a second argument — the class to predict — that a strategy-name-only factory
cannot supply.

### Common mistakes

* Passing features to a dummy and expecting them to matter. Scrambling the
  feature matrix cannot change a single prediction; the notebook checks this.
* Reading anything into *which* class `most_frequent` picks on balanced data.
  It is an alphabetical tie-break.
* Comparing a `stratified` run against another `stratified` run without a fixed
  seed and calling the difference meaningful.

---

## 4. This project's baseline: 4.55%

### The arithmetic, before the measurement

A model that always predicts one fixed class is correct exactly when the true
label is that class. So its accuracy is that class's share of the rows:

```
accuracy of a constant guess = (rows in the predicted class) / (total rows)
```

With 22 crops holding equal numbers of rows, every class's share is `1/22`:

```
1 / 22 = 0.045454... = 4.55%
```

That is the whole derivation, and it is worth being able to reproduce on paper:
**on a balanced dataset with k classes, the naive baseline is 1/k.** For 2
classes it is 50%, for 10 classes 10%, for 22 classes 4.55%.

### The measurement

`notebooks/04_baseline_models.ipynb` fits all four strategies on the 1,760
training rows and cross-validates each over five folds:

| Strategy | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | mean | std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `most_frequent` | 0.0455 | 0.0455 | 0.0455 | 0.0455 | 0.0455 | **0.0455** | 0.0000 |
| `stratified` | 0.0398 | 0.0540 | 0.0540 | 0.0398 | 0.0483 | 0.0472 | 0.0064 |
| `uniform` | 0.0540 | 0.0341 | 0.0540 | 0.0455 | 0.0455 | 0.0466 | 0.0073 |
| `prior` | 0.0455 | 0.0455 | 0.0455 | 0.0455 | 0.0455 | 0.0455 | 0.0000 |

Two observations.

**`most_frequent` has a standard deviation of exactly zero.** On stratified folds
of balanced data, one class is always exactly 1/22 of the validation rows, so
there is nothing left to vary. A perfectly stable number is a property of this
particular model and this particular data — not something to expect from a real
model.

**The random strategies sometimes beat it.** `stratified` averaged 4.72% here,
above `most_frequent`'s 4.55%. That is luck, and it is the reason the *quoted*
baseline is the deterministic strategy: quoting the luckiest naive score would
make the bar depend on a random seed.

### The number, stated

> **Baseline: 4.55%** — the 5-fold cross-validated accuracy of a
> `most_frequent` `DummyClassifier` on the 1,760 training rows of
> `data/processed/train.csv`, with `random_state=42`. Equivalently, `1/22`.

Memorise it, and memorise its consequence:

> **Any model scoring at or below 4.55% is broken or trivial.** It has extracted
> nothing from seven features that Week 2 showed separate the crops almost
> perfectly. Debug the pipeline; do not tune the model.

### Common mistakes

* Quoting the baseline without its protocol. "4.55%" is only checkable as
  "4.55%, 5-fold stratified CV, seed 42, training rows".
* Recomputing the baseline on the test set. The test set is not open yet (§8),
  and the baseline's job is to be the floor for training-time comparisons.
* Assuming 1/k transfers to an imbalanced dataset. It does not: there the naive
  baseline is the *majority class share*, which can be 90% or more.

---

## 5. Why accuracy alone can mislead

### The general failure

Accuracy is an average over rows, and an average hides its terms. Two models
with identical accuracy can behave completely differently, and a single number
cannot tell them apart.

The dummy makes it concrete. At 4.55% accuracy, its per-class report reads:

```
              precision    recall  f1-score   support

       apple       0.05      1.00      0.09        80
      banana       0.00      0.00      0.00        80
   blackgram       0.00      0.00      0.00        80
        ...
```

Recall 1.00 on one crop, 0.00 on the other 21. "4.55% accuracy" compressed 22
very different behaviours into one number. Note the *shape* rather than the
values: a real model can fail in the same shape at 95% accuracy, and only the
per-class view will show it.

`evaluate_model` therefore returns the report alongside the accuracy from the
same call, so the two cannot drift apart in a report or a notebook.

### The imbalance failure

The dataset is balanced, so the point has to be made on a modified version of
it — which is exactly what the notebook does. Reframe the same rows as a
two-class question, "is this field suited to rice?":

```
not rice    1680
rice          80
```

The same `most_frequent` baseline now scores:

```
accuracy: 0.9545

              precision    recall  f1-score   support

    not rice       0.95      1.00      0.98      1680
        rice       0.00      0.00      0.00        80
```

**95.45% accuracy, from a model that never once predicts `rice`.** Reported
alone, it looks like a strong result. It is a model that cannot answer the
question it was asked.

Nothing about the data changed — the same 1,760 rows, the same features. Only
the *framing* changed, and with it the class balance. Balance is a property of
how a dataset was constructed and how the question was posed, not a law of the
domain: a real agricultural dataset would not contain exactly 100 examples of
each crop.

### What comes next

Week 8 introduces the metrics that describe what accuracy cannot:

* **precision** — of the fields I called `rice`, how many really were?
* **recall** — of the fields that really were `rice`, how many did I find?
* **F1** — their harmonic mean, one number when you must have one;
* the **confusion matrix** — the full table of what was predicted for what.

In the 95.45% example, precision and recall for `rice` are both 0.00. The
failure that accuracy hid is the first thing they report.

### Common mistakes

* Concluding "accuracy is bad, use F1". Accuracy is the correct headline metric
  for this balanced problem; the mistake is using it *alone*.
* Believing balance makes accuracy sufficient. It makes accuracy *fair*, not
  *complete*: it still cannot show which classes fail or what errors cost.
* Comparing accuracies across differently-framed problems (22-class vs. binary)
  as if they were on the same scale.

---

## 6. Why one train/test split is not enough

### The problem

A train/test split is a random draw. Score a model on it and you get one number
sampled from a distribution — and one sample tells you nothing about the width
of that distribution.

The notebook measures it. It splits the 1,760 training rows into an inner
train/validation pair ten times, with ten different shuffles, and scores the
same `stratified` baseline each time:

```
seed 0    0.0653      seed 5    0.0625
seed 1    0.0426      seed 6    0.0568
seed 2    0.0369      seed 7    0.0511
seed 3    0.0341      seed 8    0.0398
seed 4    0.0682      seed 9    0.0227

min 0.0227 | max 0.0682 | range 0.0455
```

Same model, same data, ten legitimate splits, and the answers range from 2.27%
to 6.82% — a spread as large as the baseline itself. Any one of them could have
been reported as "the" accuracy.

### Why it matters even when the spread looks small

Two consequences, and the second is the one that bites in Week 6:

* **A single score can be optimistic or pessimistic**, depending on whether the
  held-out rows happened to be easy or hard.
* **A single score cannot support a comparison.** If model A beats model B by
  0.4 points on one split, and the split-to-split spread is 2 points, the
  ordering is noise. Choosing A would be choosing the shuffle.

Two further failure modes of the one-split habit:

* **Wasted data.** 20% of the rows are used only for scoring and never for
  fitting. On 1,760 rows that is a real loss.
* **Silent test-set tuning.** If you look at the test score, adjust something,
  and look again, the test set has become part of training — through you. §8.

### Common mistakes

* Re-running with different seeds until the number is nice. That is choosing the
  shuffle, and it is the same sin as picking a `random_state` for the split
  (Week 3 §4).
* Reporting the best of several splits. Report the mean and the spread of all of
  them — which is exactly what cross-validation does.

---

## 7. k-fold cross-validation

### What it is

**k-fold cross-validation** replaces one split with `k` of them, arranged so
every row is used for validation exactly once.

1. Cut the data into `k` equal parts, called **folds**.
2. For each fold in turn: fit the model on the other `k - 1` folds, score it on
   the held-out fold.
3. You now have `k` scores. Report their mean and their standard deviation.

With `k = 5`:

```
fold 1:  [ VALID ][ train ][ train ][ train ][ train ]
fold 2:  [ train ][ VALID ][ train ][ train ][ train ]
fold 3:  [ train ][ train ][ VALID ][ train ][ train ]
fold 4:  [ train ][ train ][ train ][ VALID ][ train ]
fold 5:  [ train ][ train ][ train ][ train ][ VALID ]
```

On this project's 1,760 training rows that means five fits, each on 1,408 rows
and scored on the 352 left out — 16 rows of each of the 22 crops.

### Why it is worth k times the compute

**Every row contributes to both training and evaluation**, across different
fits. Nothing is wasted, and no row is ever scored by a model that saw it.

**The estimate is more stable.** Averaging five held-out scores reduces the
influence of one unlucky partition.

**You get a spread, not just a point.** The standard deviation is the part a
single split cannot give you, and it is what makes "model A beats model B"
checkable: if the gap is smaller than the fold-to-fold spread, there is no gap
yet.

**It is the honest way to compare and tune.** Week 6's grid search is
cross-validation run once per hyperparameter combination, precisely so the test
set is never consulted while choosing.

The cost is that the model is fitted `k` times instead of once. For a dummy this
is free; for a large model it is the reason `k = 5` rather than `k = 20`.

### Choosing k

* **5 or 10** are the standard choices. Both keep a decent-sized validation fold
  and a training set close to the full data.
* **Larger k** means more training data per fit (less pessimistic bias) but more
  fits and more correlated training sets.
* **k = n**, one row per fold, is *leave-one-out* cross-validation: maximally
  data-efficient, expensive, and high-variance in its estimate.
* This project uses **5**, fixed in `DEFAULT_CV_FOLDS`. Each fold holds 352 rows,
  16 per crop, which is enough for a per-class number not to hinge on one row.

### Stratified folds

`build_cv()` returns `StratifiedKFold(n_splits=5, shuffle=True,
random_state=42)`. Each part is chosen *within* each class, so all 22 crops
appear in every fold in the same proportion — the same reasoning as Week 3's
stratified split. Plain `KFold` on 22 classes can leave a crop out of a fold
entirely, making that fold's score partly a statement about the shuffle. The
notebook prints the fold composition to prove it: 22 crops present in every
fold, 16 rows each.

`shuffle=True` matters too. Without it, `KFold` cuts the data in the order it
arrives, so any ordering in the CSV — sorted by crop, say — becomes structure in
the folds. With shuffling and a fixed seed the folds are arbitrary but
reproducible.

### Reading `cross_val_score`'s output

`cross_val_score(model, X, y, cv=..., scoring="accuracy")` returns a plain NumPy
array with **one score per fold, in fold order**:

```
array([0.0398, 0.0540, 0.0540, 0.0398, 0.0483])
```

How to read it:

* **The mean** is the headline estimate of performance on unseen data.
* **The standard deviation** is how much that estimate depends on which rows it
  was measured on. Report it; a mean without it invites false comparisons.
* **A single outlying fold** — four folds at 0.95 and one at 0.70 — is a
  finding, not noise to average away. It usually means a subgroup of rows the
  model handles badly, and it deserves investigation.
* **`nan` in the array** means a fold raised an error; scikit-learn will have
  warned.

`cross_validated_accuracy()` wraps all of this and returns `scores`, `mean`,
`std` and `n_splits` together, so a result cannot be reported without its
spread.

One subtlety worth knowing now: `cross_val_score` **clones** the estimator before
each fold and fits the clone, so the object you passed in comes back unfitted.
That is deliberate — every fold must start from an untrained model — and it is
also why the estimator handed to it should be a `Pipeline` (Week 3 §6) whenever
preprocessing is involved: cloning a pipeline re-fits the scaler inside each
fold, on that fold's training rows only. Fitting a scaler once outside the loop
would leak every validation fold's statistics into training.

### Common mistakes

* Cross-validating a *fitted* model, or a model whose preprocessing was fitted
  on all the data. The second is the leak just described, and it is the single
  most common cross-validation error.
* Using plain `KFold` on many-class or imbalanced data.
* Reporting only the mean.
* Cross-validating on the test set. Cross-validation happens *inside* the
  training data; the test set is separate and untouched.

---

## 8. Validation set vs. test set

Three roles, and confusing them is what quietly invalidates results:

| Set | Used for | How often it may be looked at |
| --- | --- | --- |
| **Training set** | Fitting model parameters | Constantly |
| **Validation set** (or the CV folds) | Choosing between models, tuning hyperparameters, deciding anything | As often as you like |
| **Test set** | One final, honest estimate of performance | Once, at the end |

Cross-validation is a way of *manufacturing* validation sets from the training
data, which is why this week never opens `data/processed/test.csv`. Every number
in the Week 4 notebook comes from the 1,760 training rows.

The rule for the test set is behavioural: every decision you make after seeing a
test score — a different model, a tweaked hyperparameter, an extra feature —
transfers a little information from the test set into the model, through you.
Do it often enough and the test score becomes an optimistic training score
wearing a disguise. That is why this project's test set stays sealed until
Week 8.

### Common mistakes

* Using the words "validation set" and "test set" interchangeably.
* "Just peeking" at the test score to see how things are going.
* Reporting the best of several test-set evaluations. The best of many is not an
  unbiased estimate of anything.

---

## 9. What to expect in Weeks 5-8 — and why this dataset is easy

Set expectations now, so nothing later is misread.

**The dummy scores ~4.5%. Real models will score 98-99%+.** That is not a
promise about machine learning in general; it is a property of *this dataset*.
The Crop Recommendation dataset is famously easy: Week 2's class-separation
measurements showed the seven features splitting the 22 crops almost perfectly,
with humidity, rainfall and the NPK values carving out near-disjoint regions per
crop. The classes are balanced, there are no missing values, no messy
categoricals, no temporal drift and essentially no label noise. Almost any
reasonable algorithm will land somewhere around 98-99%+, and the gaps between
them will be fractions of a percent.

**Do not conclude that Weeks 5-8 are pointless.** The value of those weeks is
not in chasing a hard accuracy ceiling — there isn't one to chase here. It is in
**how** models are compared, tuned and explained:

* **Week 5** — what different algorithm families actually do, and why two models
  that agree on 99% of rows can disagree completely about *why*.
* **Week 6** — comparing candidates with a protocol whose differences are
  believable, and tuning hyperparameters without letting the test set into the
  decision. When every model is at 99%, recognising that a 0.2-point gap is
  smaller than the fold-to-fold spread — and therefore not a gap — is the skill,
  and it is far more transferable than the winning model's name.
* **Week 7** — explaining a model: which features it leans on, and whether that
  agrees with the agronomy.
* **Week 8** — looking past accuracy to per-class behaviour and the confusion
  matrix, where even a 99% model has a story to tell about which two crops it
  confuses.

Those skills are exactly the ones that carry over to a dataset whose ceiling is
72%. Squeezing the last 0.3% out of this one carries over to nothing.

There is also a healthy suspicion to learn here. When a first model returns 99%
on a real project, the correct first reaction is *"what have I leaked?"*, not
*"we're done"*. Here the answer is genuinely "the dataset is separable" — Week 3
enforced the split discipline that lets us say that with a straight face — but
the reflex is worth keeping.

### Common mistakes

* Reading a 99% result as evidence of skill rather than of an easy dataset.
* Deciding that because everything scores 99%, the choice between models does
  not matter. Speed, interpretability, stability and calibration all still
  differ, and Week 6 chooses on more than one axis.
* Assuming this accuracy would survive on real farm data. It would not: real
  fields are not sampled 100 per crop, and their labels are not noise-free.

---

## 10. The code this week

### `src/models/baseline.py`

```python
get_baseline_model(strategy="most_frequent", random_state=42) -> DummyClassifier
```

A factory, not a class. It returns an **unfitted** estimator — matching
`build_preprocessor()` from Week 3, and for the same reason: what an estimator
is fitted on is a decision the caller must make explicitly, and
cross-validation needs to fit it itself, per fold.

It restricts `strategy` to four vetted names and raises a `ValueError` naming
the alternatives otherwise, so a typo fails loudly instead of silently
producing a differently-defined baseline.

### `src/evaluation/metrics.py`

```python
evaluate_model(model, X, y) -> {"accuracy": float, "report": str, "n_samples": int}
build_cv(n_splits=5, random_state=42) -> StratifiedKFold
cross_validated_accuracy(model, X, y, ...) -> {"scores", "mean", "std", "n_splits"}
```

`evaluate_model` scores an **already fitted** model; it never calls `fit`, so it
can be pointed at held-out data with no risk of training on it. It returns the
accuracy and the per-class report *together* on purpose — §5's whole argument is
that separating them is how people mislead themselves.

`cross_validated_accuracy` takes an **unfitted** model and does the fitting
itself, five times.

This module is designed to be **extended, not replaced**. Week 5 compares real
classifiers through the same two functions; Week 6 tunes them; Week 8 adds
precision, recall, F1 and the confusion matrix as further keys and further
functions. The existing keys keep their names, so a Week 4 notebook does not
break in Week 8.

### `tests/test_baseline.py`

38 tests, grouped by what they protect:

* the factory returns an unfitted dummy, honours every supported strategy,
  rejects the rest, and is reproducible under a fixed seed;
* the baseline really is uninformed — scrambling the features cannot change a
  prediction — and it scores exactly 1/k on balanced data;
* `evaluate_model` matches a hand-computed accuracy, names every class in its
  report, refuses mismatched lengths, and does not fit;
* the folds are stratified, disjoint and complete, and `cross_val_score` leaves
  the estimator unfitted;
* on the real 2,200 rows, the cross-validated baseline lands on 1/22 and no
  naive strategy escapes it.

---

## 11. What this week produced, and what it deliberately did not

**Produced**

* A definition of "good" for this project: 5-fold stratified cross-validated
  accuracy on the training rows, quoted with its spread and always accompanied
  by a per-class report.
* A baseline of **4.55%**, derived from `1/22` before it was measured, and the
  reasoning that makes it the floor for every later week.
* Reusable code: `get_baseline_model`, `evaluate_model`,
  `cross_validated_accuracy`, `build_cv`.
* A worked demonstration that the same baseline scores 95.45% on an imbalanced
  framing of the same data while never predicting the minority class.

**Not produced, on purpose**

* No real classifier — **Week 5**.
* No hyperparameter tuning — **Week 6**.
* No feature importances — **Week 7**.
* No precision/recall analysis beyond the glimpse in §5 — **Week 8**.
* No test-set number at all. `data/processed/test.csv` has still never been
  opened.
