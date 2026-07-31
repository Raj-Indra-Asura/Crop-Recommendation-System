# Week 3 — Learning Notes

**Data preparation: turning raw data into model-ready data, correctly.**

Week 1 secured the data and wrote down what it must look like. Week 2 looked at
it and found, among other things, two facts that this week has to act on: the
seven features live on wildly different scales, and there is a way of preparing
data that quietly destroys the honesty of every score you will ever compute
afterwards.

This week does four things — encode the target, split the data, scale the
features, save the result — and spends most of its effort on the *order* in
which they happen and on *what is fitted on what*. That is not pedantry: it is
the difference between a project whose reported accuracy survives contact with
reality and one whose does not.

Code produced this week: `src/data/split.py`,
`src/preprocessing/preprocessor.py`, `tests/test_preprocessing.py`,
`notebooks/03_data_preparation.ipynb`.

---

## 0. What data preparation is, and why it is a stage of its own

**Data preparation** (or preprocessing) is everything that happens between "the
data as it was recorded" and "the array a model is fitted on". Typically:

* handling missing values,
* encoding non-numeric columns into numbers,
* rescaling numeric columns to comparable units,
* splitting off data to evaluate on,
* occasionally, constructing new features from existing ones.

On this dataset the first is unnecessary (Week 1 proved there are no nulls) and
the last is deliberately postponed. What remains is encoding, splitting and
scaling.

It is a stage of its own for two reasons.

**It is where models get their inputs' meaning.** A model cannot compensate for
badly presented inputs; it can only fit what it is given. Scaling is not
cosmetic tidying, it changes what a distance or a gradient *means*.

**It is where evaluation is most easily corrupted.** Every other stage's
mistakes announce themselves — a broken model scores badly. A preparation
mistake does the opposite: it makes the score *better*, and the project only
finds out in production. That asymmetry is why this week's rules are procedural
rather than judgement calls.

### Common mistakes

* Treating preparation as a chore to get past, and writing it as loose lines in
  a notebook cell. It must be an object you can re-apply, because at serving
  time (Week 10) a single incoming request has to be prepared in *exactly* the
  same way as the training rows.
* Preparing the whole dataset and splitting afterwards. §5.
* Preparing "until the numbers look nice", which usually means until the model
  scores well — a decision made using information from the test set.

---

## 1. Why raw data usually cannot go straight into a model

### What is the problem?

Here are the seven features as Week 2 measured them, with the ranges made
explicit (`notebooks/03_data_preparation.ipynb`, §1):

| Feature | min | max | range | mean | std | range ÷ `ph` range |
| --- | --- | --- | --- | --- | --- | --- |
| `N` | 0.00 | 140.00 | 140.00 | 50.55 | 36.92 | 21.8 |
| `P` | 5.00 | 145.00 | 140.00 | 53.36 | 32.99 | 21.8 |
| `K` | 5.00 | 205.00 | 200.00 | 48.15 | 50.65 | 31.1 |
| `temperature` | 8.83 | 43.68 | 34.85 | 25.62 | 5.06 | 5.4 |
| `humidity` | 14.26 | 99.98 | 85.72 | 71.48 | 22.26 | 13.3 |
| `ph` | 3.50 | 9.94 | 6.43 | 6.47 | 0.77 | 1.0 |
| `rainfall` | 20.21 | 298.56 | 278.35 | 103.46 | 54.96 | 43.3 |

These columns are not merely different, they are **incomparable**: `K` is
milligrams per kilogram of soil, `rainfall` is millimetres, `ph` is a
logarithmic acidity index, `humidity` is a percentage. The only thing they share
is that a computer stores them all as floats — and that is exactly the problem,
because several important algorithms treat "1" as "1" no matter which column it
came from.

### Why does it break models?

**Distance-based models.** k-nearest neighbours and support vector machines
decide by measuring how far apart two examples are, usually with Euclidean
distance:

```
d(a, b) = sqrt( (a_N - b_N)^2 + (a_P - b_P)^2 + ... + (a_rainfall - b_rainfall)^2 )
```

Every term is squared and summed as-is. A 50-unit difference in `K` contributes
2,500 to that sum; the *largest possible* difference in `ph` — 6.4 — contributes
41. `ph` is therefore, arithmetically, almost invisible next to `K`. The model
has not decided potassium is more informative than acidity; the units decided it
before the model saw anything.

**Gradient-based models.** Logistic regression and neural networks fit weights
by gradient descent. The gradient with respect to a weight is proportional to
its feature's values, so a feature ranging over hundreds produces gradients
hundreds of times larger than one ranging over units. The loss surface becomes a
long narrow valley, and a single shared learning rate is either too large for
the wide directions (divergence) or too small for the narrow ones (crawling).
Scaled features make the valley round, and convergence fast and stable.

**Regularised models.** L1/L2 penalties shrink coefficients by their *size*.
Since a coefficient's size depends on its feature's units, penalising unscaled
features penalises them unequally — arbitrarily.

**Variance-based methods.** PCA finds directions of maximum variance; on
unscaled data those directions are dominated by whichever feature happens to be
measured in the biggest numbers.

### Which models genuinely do not care

Decision trees, random forests and gradient-boosted trees. A tree only ever asks
*"is `K` ≤ 92.5?"*, and it chooses that threshold from the data itself. Rescale
`K` however you like — as long as the transformation preserves the order of the
values, the same rows fall on the same side of the equivalent split, and the
tree is identical. Standardisation is exactly such a monotone transformation, so
tree-based models are **scale-invariant**.

That is a genuine practical advantage of trees, and Week 5 will exploit it. It
is *not* a reason to skip scaling in this project: the same preprocessor is used
for every model so that Week 6's comparison measures the algorithms rather than
their inputs, and scaling costs a tree nothing but a few microseconds.

### Where is it used in industry?

Anywhere features come from different instruments or systems: a fraud model
mixing "transaction amount in euros" with "seconds since last login"; a medical
model mixing blood pressure with age; a recommender mixing "minutes watched"
with "number of clicks". The failure is always the same shape — the model
appears to have decided a feature is unimportant when in fact its units made it
small.

### Common mistakes

* Concluding from "trees don't need scaling" that scaling is optional. It is
  optional *for trees*.
* Believing scaling improves a model's information. It does not add or remove
  information; it removes an arbitrary weighting.
* Expecting scaling to fix skew. It does not — see §2.

---

## 2. Feature scaling

### What is it?

**Feature scaling** re-expresses each column so that all columns occupy a
comparable numeric range. This project uses **standardisation**, implemented by
scikit-learn's `StandardScaler`:

```
z = (x - mean) / std
```

where `mean` and `std` are that column's mean and standard deviation, *learned
from the training data*. The result is called a **z-score**: it says how many
standard deviations a value sits above (positive) or below (negative) its
column's mean. A field with `K` at +1.8 and `ph` at -0.4 has unusually high
potassium and slightly acidic soil, and those two numbers are now on the same
footing.

Applied to the data it was computed from, standardisation always yields a column
with mean exactly 0 and standard deviation exactly 1. That is not a coincidence
or a property of this dataset; it is arithmetic:

```
mean(z) = mean((x - mean(x)) / std(x)) = (mean(x) - mean(x)) / std(x) = 0
std(z)  = std(x - mean(x)) / std(x)    = std(x) / std(x)              = 1
```

The test set, transformed with the *training* mean and standard deviation, comes
out near 0 and 1 but not exactly — and §5 explains why that difference is the
single most reassuring number in the notebook.

### What standardisation does *not* do

It is a **linear** transformation: subtract a constant, divide by a constant. So
it moves and stretches a column but never reorders it. Consequently it does
**not**:

* remove skew — `K`'s skew of +2.38 is unchanged afterwards;
* remove outliers — a point 4 standard deviations out is still 4 out;
* make a distribution normal, despite the unfortunate nickname "normalising";
* change any model's answer if that model is scale-invariant (§1).

To change a distribution's *shape* you need a non-linear transform (a log, a
square root, a quantile transform). This project does not use one; that is
feature engineering, and it is out of scope until it is justified by a model
result.

### The alternatives, and why standardisation here

| Scaler | Formula | Output | Suits |
| --- | --- | --- | --- |
| `StandardScaler` | `(x - mean) / std` | mean 0, std 1, unbounded | The general default; assumes no fixed bounds |
| `MinMaxScaler` (normalisation) | `(x - min) / (max - min)` | exactly [0, 1] | When a bounded range is required, e.g. image pixels; very sensitive to a single extreme value, which compresses everything else |
| `RobustScaler` | `(x - median) / IQR` | median 0 | Data with genuine, heavy outliers, since median and IQR barely move |

`StandardScaler` is chosen because these features have no natural bounds, no
corrupt extreme values (Week 2 established the flagged points are real crop
populations), and because the downstream models that care about scale — SVM,
logistic regression, KNN — are conventionally fitted on standardised inputs.

Note that "normalisation" is used loosely in the wild: sometimes for min-max
scaling, sometimes for standardisation, sometimes for scaling a *row* to unit
length (`Normalizer`, a genuinely different operation applied per row rather
than per column). Say which one you mean.

### Where is it used in industry?

Effectively universally for non-tree models: any scikit-learn pipeline, any
production feature store that materialises "standardised" versions of numeric
features, any deep learning input layer (batch normalisation is the same idea
moved inside the network). Where it becomes operationally interesting is that
the *training* mean and standard deviation have to be stored and shipped
alongside the model, because the server must apply the identical transformation
to a single incoming request. Week 9 and Week 10 make that concrete; this week
is where those numbers first come into existence.

### Common mistakes

* Fitting the scaler on all the data (§5).
* Fitting a *new* scaler on the test set — the same error wearing a disguise. It
  standardises test rows by test statistics, which is not what training taught
  the model to expect.
* Recomputing statistics at serving time from whatever batch of requests happens
  to be in flight. The transformation must be a constant of the deployed model,
  not a function of today's traffic.
* Scaling the target column. `y` is a class code here; scaling it is meaningless.
* Scaling one-hot or binary columns as if they were continuous. Harmless-ish,
  but it makes coefficients hard to read.

---

## 3. Encoding the target label

### What is it?

The `label` column contains strings: `"rice"`, `"maize"`, `"apple"`. Estimators
fit on numeric arrays, so the 22 names have to become numbers.
`LabelEncoder` does exactly one thing: it takes the distinct values, sorts them
alphabetically, and assigns `0, 1, 2, ...` in that order.

```
apple 0, banana 1, blackgram 2, chickpea 3, coconut 4, coffee 5, cotton 6,
grapes 7, jute 8, kidneybeans 9, lentil 10, maize 11, mango 12, mothbeans 13,
mungbean 14, muskmelon 15, orange 16, papaya 17, pigeonpeas 18, pomegranate 19,
rice 20, watermelon 21
```

The mapping is stored on the fitted encoder as `classes_`, and it is
**lossless**: `inverse_transform` returns the original names exactly. That
matters, because everything a user ever sees must be a crop name, not a code.
The notebook asserts both that the class set equals Week 1's frozen
`EXPECTED_LABELS` and that decoding round-trips.

### Why do models need numeric labels?

Because internally a classifier represents classes as positions: index into a
coefficient matrix, a column of predicted probabilities, a row of a confusion
matrix. `predict_proba` returns an array whose *k*-th column is the probability
of the *k*-th class — that ordering has to be defined, and `classes_` defines it.
scikit-learn will in fact accept raw strings and encode them internally; doing it
explicitly keeps the mapping visible, saved to disk, and identical across weeks
rather than reconstructed inside each estimator.

### Why the codes are not quantities

`apple` is 0 and `banana` is 1, but banana is not "one more apple", and the
midpoint of `apple` (0) and `cotton` (6) is not `cotton`'s neighbour `coffee`
(5) in any meaningful sense. The integers are names written in digits, and the
alphabetical ordering they inherit is an artifact of spelling.

A classifier treats them correctly, as bare identifiers. Two things would not:

* a **regressor** fitted on the codes, which would try to predict 12.4 and be
  penalised more for confusing apple with watermelon (|0-21|) than with banana
  (|0-1|);
* any **distance** computed on the codes, e.g. using them as an input feature to
  KNN.

That second case is why a categorical *input* column is normally **one-hot
encoded** instead: 22 categories become 22 columns of 0/1, so no ordering can be
implied. This dataset has no categorical inputs — all seven features are numeric
— so one-hot encoding is introduced here only as the contrast. The
`ColumnTransformer` built in §6 is the object that would carry it if a
categorical feature ever appeared.

### Common mistakes

* Fitting a *second* `LabelEncoder` on the test labels. The alphabetical order
  happens to be stable here only because all 22 crops occur on both sides; on a
  split where a class is missing, the codes would silently shift and every
  prediction would be mislabelled.
* Losing the encoder. Codes without the mapping are unreadable output.
* Using `LabelEncoder` on input features. It is documented for targets; for
  inputs use `OrdinalEncoder` (when the categories genuinely have an order) or
  `OneHotEncoder` (when they do not).

---

## 4. The stratified train/test split

### What is it?

Week 1 defined the split conceptually: fit on the training set, keep the test set
back to estimate performance on unseen data. This week performs it — 80% train,
20% test — and adds one refinement.

A **stratified** split draws the test rows *within each class* rather than from
the pool as a whole. With 22 classes at 100 rows each, that means every crop
contributes exactly 20 rows to the test set and 80 to the training set: 440 and
1,760 in total.

### Why stratify?

Because a plain shuffle is random, and randomness at this scale is coarse. The
notebook runs both splits on the same data with the same seed:

```
stratified   test rows per crop: min 20 max 20
unstratified test rows per crop: min 11 max 27
```

Under the unstratified split one crop is evaluated on 11 rows and another on 27.
Two consequences:

* **Per-class scores become unstable.** Recall for the 11-row crop moves in
  steps of 9 percentage points — one row. A difference between two crops would
  then partly measure how the shuffle fell.
* **The training set is subtly reweighted.** The crop with 27 test rows has only
  73 training rows, so the model sees less of it, for no reason connected to the
  problem.

Neither is catastrophic on a perfectly balanced 2,200-row dataset, which is why
this week can *demonstrate* the effect rather than suffer from it. Stratifying
matters far more when classes are imbalanced: a class with 12 rows can be absent
from the test set entirely under a plain shuffle, making its recall undefined and
the model's weakest area unmeasurable. Stratification costs nothing and removes
the failure mode, so it is the default here and everywhere after.

The check that it worked is a comparison of class proportions, which
`class_proportions()` in `src/data/split.py` exists to make easy: every crop
holds 4.5454...% of both halves, and the largest difference between the two is
0.0.

### Why 20%?

A trade-off with no universally right answer. Too small a test set gives a noisy
estimate; too large a one starves training. 20% of 2,200 is 440 rows — 20 per
crop, enough that a per-class score is not decided by one or two rows — while
leaving 1,760 rows to learn from. Conventional choices sit between 10% and 30%;
very large datasets use much smaller fractions, because 1% of ten million rows
is already plenty.

### Why a fixed random seed?

`train_test_split` shuffles before cutting, and shuffling needs a source of
randomness. Left unseeded, it produces a different split on every run. That would
mean:

* two students running identical code get different accuracies;
* the same student cannot tell whether Tuesday's improvement is a better model or
  a luckier split;
* a bug that only appears for certain splits is unreproducible.

`random_state=42` fixes the pseudo-random sequence, so the shuffle — and
therefore the split — is identical forever. The notebook demonstrates both
halves of this: calling `stratified_split` twice returns identical frames, while
`random_state=7` produces a test set sharing only 84 of its 440 rows with the
default one.

The value 42 is a convention and means nothing. What matters is that it is fixed
and written down — in `src/data/split.py` as `DEFAULT_RANDOM_STATE`, once, so
that no two places in the course can disagree.

**Seeds must never be tuned.** Trying several and keeping the one with the best
test score is overfitting the test set by hand: the resulting number describes
that particular shuffle, not the model, and will not reproduce on real fields. If
a result changes a lot between seeds, that is information — the estimate is
unstable and needs cross-validation (Week 6), not a nicer seed.

### Common mistakes

* Splitting after preprocessing (§5).
* Splitting time-ordered data at random. For anything with a time dimension the
  test set must be *later* than the training set, otherwise the model is
  predicting the past from the future. This dataset has no time column, so a
  random split is appropriate.
* Splitting data with groups — several rows from the same field, patient or user
  — at random, so near-duplicates of a training row sit in the test set. That
  needs `GroupShuffleSplit`.
* Looking at the test set. Peeking at it to choose a preprocessing step, a
  model or a threshold makes it a second training set, and its estimate stops
  being an estimate of anything.

---

## 5. `fit`, `transform`, `fit_transform` — and where leakage actually enters

### What is it?

Every scikit-learn transformer has the same three methods, and telling them apart
is the central skill of this week:

| Call | What it does | Allowed on |
| --- | --- | --- |
| `fit(X)` | **Learns** parameters from `X` and stores them on the object | Training data only |
| `transform(X)` | **Applies** the stored parameters to `X` | Anything: train, test, one row from an API |
| `fit_transform(X)` | Both, in one call — a convenience, not a third operation | Training data only |

For `StandardScaler` the learned parameters are one mean and one standard
deviation per column. scikit-learn's convention is that attributes created by
fitting end in an underscore, so they are easy to spot: `mean_`, `scale_`,
`classes_`, later `coef_`. An object with no trailing-underscore attributes has
not been fitted, and calling `transform` on it raises `NotFittedError` — a
deliberate guard rail, tested in `tests/test_preprocessing.py`.

The 14 numbers this project's preprocessor learns (from
`notebooks/03_data_preparation.ipynb`, §4):

| Feature | `mean_` | `scale_` (std) |
| --- | --- | --- |
| `N` | 50.548 | 36.852 |
| `P` | 53.340 | 32.938 |
| `K` | 48.143 | 50.694 |
| `temperature` | 25.609 | 5.079 |
| `humidity` | 71.417 | 22.274 |
| `ph` | 6.474 | 0.783 |
| `rainfall` | 103.452 | 54.977 |

Every one of them was computed from the 1,760 **training** rows.

### The rule, and why it is the whole point

> **Fit on train. Transform both. Never fit on anything you intend to evaluate
> on.**

Week 2 defined data leakage as *training a model with information it would not
have at prediction time*. Here is that abstraction made arithmetic: if the
scaler's mean is computed over all 2,200 rows, then every training row is
standardised using a number that partly encodes the held-out rows. The model is
then fitted on inputs that contain a trace of the test set, and the test score it
eventually earns is not a clean estimate of performance on unseen data — because
the data was not entirely unseen.

The proof that the rule was followed is visible in the output:

| | train mean | train std | test mean | test std |
| --- | --- | --- | --- | --- |
| `N` | 0.000 | 1.000 | 0.001 | 1.008 |
| `P` | 0.000 | 1.000 | 0.003 | 1.006 |
| `K` | 0.000 | 1.000 | 0.001 | 0.994 |
| `temperature` | -0.000 | 1.000 | 0.007 | 0.984 |
| `humidity` | -0.000 | 1.000 | 0.015 | 0.996 |
| `ph` | -0.000 | 1.000 | -0.028 | 0.939 |
| `rainfall` | 0.000 | 1.000 | 0.001 | 0.997 |

The training columns are exactly 0 and 1 — that is what fitting means. The test
columns are *close but not equal*, and that is the correct and desirable
outcome: the test rows were shifted by the training mean, which is not quite
their own mean. If the test row of that table also read 0.000/1.000, the scaler
would have seen the test data.

The residual is not an error to be minimised. It is an honest measurement of how
much held-out data differs from training data — the same difference production
will show.

### "But the difference is tiny"

It is, on this dataset. The notebook fits a second, deliberately leaky scaler on
all 2,200 rows and compares: the largest shift in any feature mean is 0.065
(`humidity`), about a quarter of one percent of that column's standard
deviation. Standardising with the leaky numbers would change no conclusion here.

That is *why* the rule must be procedural rather than judged case by case:

* the size of the leak depends on the data, and you cannot measure it without
  first having leaked;
* leaks compound — one that is invisible in a scaler becomes serious in an
  imputer (a mean filled in from test rows), decisive in a feature selector (a
  column chosen because it correlates with test labels), and fatal in a resampler
  applied before a split (a synthetic training row interpolated from a test row);
* the habit is what protects you on the dataset where it *does* matter, and by
  then you will not be reasoning about it any more.

### Three forms leakage takes at this stage

1. **Preprocessing before splitting** — scaling, imputing, encoding or selecting
   features on the full dataset. The commonest, and the one this week forbids
   structurally.
2. **Fitting a transformer inside cross-validation incorrectly** — scaling once
   outside the loop, so every validation fold is transformed with statistics
   that include it. The fix is to put the transformer in a `Pipeline` and pass
   the pipeline to `cross_val_score`, which is exactly why §6 exists now and not
   in Week 6.
3. **Repeated peeking** — trying things until the test score improves. No single
   step is wrong; the accumulation is. The defence is a validation set or
   cross-validation (Week 6), with the test set touched once, at the end
   (Week 8).

### Common mistakes

* `fit_transform` on the test set. This is the leak in its purest form and it
  reads almost identically to the correct line — a one-word difference.
* `transform` before `fit`, which raises rather than silently misbehaving.
* Fitting on train, then *re-fitting* on the full data "now that we're done".
  Any score computed afterwards is no longer comparable.
* Assuming a leak-free split protects you if the raw data already contains
  duplicated rows across the split. It does not — that is a different leak, and
  it is checked at the data level, not here.

---

## 6. `ColumnTransformer` and `Pipeline`

### What are they?

Both are scikit-learn **composite estimators**: objects that hold other
estimators and present the same `fit`/`transform`/`predict` interface as a single
one.

**`ColumnTransformer` composes across columns.** It maps *sets of columns* to
*transformers*, so different columns can be prepared differently in one object.
Its `remainder` setting controls unnamed columns; `"drop"` removes them:

```python
ColumnTransformer(
    transformers=[("numeric", StandardScaler(), FEATURE_COLUMNS)],
    remainder="drop",
)
```

Each entry of `transformers` is a triple: a **name** (for later lookup and for
error messages), a **transformer instance**, and the **columns** it applies to.
`remainder` decides the fate of every unnamed column — `"drop"` (the choice
here) or `"passthrough"`. Dropping is deliberate: the target must never travel
through the feature preprocessor, and an unexpected extra column should disappear
rather than reach the model unexamined.

This project has one entry because all seven features are numeric and want the
same treatment. The structure earns its keep the moment that stops being true —
add a categorical soil-type column and it becomes a second triple with a
`OneHotEncoder`, with nothing else in the codebase changing.

**`Pipeline` composes across steps.** It chains named steps so the output of each
feeds the next:

```python
Pipeline([("preprocess", build_preprocessor()), ("model", LogisticRegression())])
```

`fit` fits every step in order; `predict` pushes new data through all of them.
This week's pipeline has a single step and changes nothing functionally — it is
introduced now because from Week 4 onward models are *appended* to it rather
than bolted on beside it.

### Why do we need them?

**They make the leakage rule structural.** `pipeline.fit(X_train, y_train)` fits
the scaler on the training data and only the training data, because that is the
only data the call receives. There is no line to write incorrectly.

**They make cross-validation correct by construction.** Passing a `Pipeline` to
`cross_val_score` re-fits the scaler inside every fold, on that fold's training
part alone. Scaling once outside the loop instead is the second form of leakage
from §5, and it is the reason this object is introduced a full three weeks
before Week 6 needs it.

**They make deployment a single artifact.** The fitted pipeline holds the
transformation *and* the model. Week 9 saves that one object; Week 10's API loads
it and calls `predict` on a raw request. Nothing at serving time has to remember
"and also subtract 50.548 from N first" — the arithmetic travels with the model.
**Training/serving skew** happens when a model is served with preparation that
differs, even slightly, from the preparation used in training. It is one of the
most common production ML failures, and this is the standard defence against it.

**They make hyperparameter search cover preparation too.** Week 6 can search
`preprocess__numeric__with_mean` alongside the model's own parameters, using the
double-underscore path syntax that composite estimators expose.

### Where is it used in industry?

The `Pipeline` (or its equivalent in other frameworks) *is* the deployed model in
most tabular ML systems: what is versioned, tested, saved, loaded and monitored.
Teams that instead ship a bare model plus a preprocessing script written twice —
once for training, once for the server — reliably discover the two have drifted
apart.

### Common mistakes

* Fitting the preprocessor separately and *then* putting it in a pipeline. The
  pipeline should own the fitting.
* Using `remainder="passthrough"` without thinking, and shipping the target or an
  identifier column into the model as a feature.
* Forgetting that a `ColumnTransformer` outputs a NumPy array, not a dataframe —
  column names are available via `get_feature_names_out()`, which is how Week 7
  will label feature importances.
* Reordering the transformer's columns between training and serving. The array
  is positional; the `ColumnTransformer` is what guarantees the order, provided
  it is the same object.

---

## 7. What this week produced

* **A target encoded as 22 integers**, with the mapping saved to
  `data/processed/label_classes.csv` so the codes are always readable.
* **A stratified 80/20 split**: 1,760 training rows and 440 test rows, exactly
  80 and 20 per crop, reproducible from `random_state=42`.
* **A fitted `ColumnTransformer`** holding seven means and seven standard
  deviations, all computed from training rows only, leaving the training features
  at mean 0 / std 1 and the test features near but deliberately not equal to it.
* **Five files in `data/processed/`** — `train.csv`, `test.csv`,
  `train_scaled.csv`, `test_scaled.csv`, `label_classes.csv` — so Week 4 starts
  from model-ready data.

And, just as importantly, what it did *not* do: no model, no metric, no new
feature, no deleted row, and no look at a single test-set value.

## 8. Where this leaves us

The data is ready and the evaluation set is clean. The obvious question — *how
well can anything predict a crop from these seven numbers?* — still has no
answer, and it cannot be answered by a good model alone: a score means nothing
without something to compare it against.

Week 4 therefore fits **baselines** first: a model that always predicts the most
frequent class, and a simple rule-based one. On a perfectly balanced 22-class
problem, "always guess the same crop" scores about 4.5%, and that number is the
floor every later model has to beat by a margin worth caring about.

---

## Recap of new terms

| Term | One-line meaning |
| --- | --- |
| **Data preparation / preprocessing** | Everything between raw recorded data and the array a model is fitted on. |
| **Feature scaling** | Re-expressing columns so their numeric ranges are comparable. |
| **Standardisation / z-score** | `(x - mean) / std`; the fitted data ends with mean 0 and std 1. |
| **Normalisation (min-max)** | `(x - min) / (max - min)`; output bounded to [0, 1]. |
| **Scale-invariant model** | A model whose output is unchanged by monotone rescaling — trees and their ensembles. |
| **Label encoding** | Mapping class names to integers `0..k-1`, alphabetically, losslessly. |
| **One-hot encoding** | Expanding a categorical column into one 0/1 column per category, so no ordering is implied. |
| **Stratified split** | Splitting within each class, so class proportions are preserved on both sides. |
| **`fit`** | Learn parameters from data and store them on the estimator. Training data only. |
| **`transform`** | Apply stored parameters to any data. |
| **`fit_transform`** | Both at once; training data only. |
| **Fitted state** | The attributes an estimator gains by being fitted; by convention they end in `_`. |
| **`random_state` / seed** | The fixed number that makes a shuffle deterministic and a result reproducible. |
| **`ColumnTransformer`** | An object mapping sets of columns to the transformers applied to them. |
| **`Pipeline`** | An object chaining named steps into one estimator with a single `fit`/`predict`. |
| **Training/serving skew** | Preparing data differently at serving time than at training time; the standard defence is shipping the pipeline. |
