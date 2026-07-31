# Week 5 — Learning Notes

> 🗺 [Roadmap](../README.md) › [Part II — Modelling (Weeks 4-7)](../README.md#part-ii--modelling-weeks-4-7) › [Chapter 5 — Classification Models](README.md) › **§5.2 Learning notes**

**Three classical classifiers, one training loop, and a comparison that is
allowed to mean something.**

Week 4 built a model designed to be stupid and measured it carefully: 4.55%,
the accuracy of guessing on 22 balanced classes. Nothing else was trained,
deliberately, so that the first real score would have something to be measured
against.

This week spends that number. Three algorithms from three different families are
trained on the same 1,760 training rows, through the same pipeline, and scored
on the same five cross-validation folds:

* **logistic regression** — linear, readable, and *discriminative*;
* **k-nearest neighbours** — distance-based, *lazy*, shape-agnostic;
* **Gaussian naive Bayes** — probabilistic, *generative*, built on an assumption
  that is provably false here.

The three italicised words are the jargon this week introduces: "lazy" is
defined in §3, "discriminative" and "generative" in §4. Nothing below assumes
you already know them.

The result is a four-row table (three models plus the baseline) which Weeks 6-8
**extend rather than replace**.

Code produced this week: `src/models/classical_models.py`,
`tests/test_classical_models.py`,
`notebooks/05_classification_models.ipynb`.

---

## 0. What changes this week, and what does not

Almost nothing about the *process* changes. The data is the same, the split is
the same, the metric is the same, the folds are the same, and the floor is the
same. The only new thing is that the final step of the pipeline now contains a
model that looks at the features.

That is not an accident of pacing: it is the shape of real work. Swapping the
algorithm should be the cheapest change in a project, because the expensive
parts — the data contract, the leakage discipline, the evaluation protocol —
were built first and are shared by every candidate.

### Common mistakes

* **Starting a new evaluation for every model.** A new seed, a new split or a
  new metric per model turns a comparison into a collection of anecdotes.
* **Believing the algorithm is the interesting part.** On most projects the
  choice between two reasonable algorithms is worth a fraction of what the data
  preparation and evaluation are worth. This week's leader wins by 2.7 points;
  Week 3's leakage discipline is worth much more than that in production.

---

## 1. The training loop, named once and for all

Every supervised model in scikit-learn — this week's three, a later week's random
forest, Week 9's saved artifact — is driven by two method calls:

```python
model.fit(X_train, y_train)        # learn parameters from labelled rows
predictions = model.predict(X_test)  # apply them to rows it has not seen
```

That is **the training loop**, and it is worth naming explicitly because it does
not change again for the rest of the course. Learning a new algorithm from here
on means learning what it does *inside* `fit`, not learning a new way to call
it.

What each call means:

* **`fit(X, y)`** looks at the features *and* the labels and stores whatever the
  algorithm needs on the object. By convention the stored attributes end in an
  underscore: `coef_`, `classes_`, `theta_`. Calling `fit` again starts over — it
  does not continue training.
* **`predict(X)`** takes features only and returns one label per row, in the same
  order. It never learns anything; it can be called any number of times.
* **`predict_proba(X)`** returns, for classifiers that support it, one
  probability per class per row, each row summing to 1.

Two rules travel with the loop, both inherited:

1. **`fit` sees training rows only.** Anything learned from a row that the model
   is later scored on is leakage (Weeks 2-3).
2. **Factories return unfitted objects.** `get_logistic_regression()` hands back
   an estimator that has never seen data, so the caller decides what it is fitted
   on — and `cross_val_score` can clone it and fit a fresh copy inside every
   fold.

### Models travel inside the pipeline

Because two of this week's three models care about feature units, each is
wrapped with the Week 3 preprocessor:

```python
Pipeline([("preprocess", build_preprocessor()), ("model", get_knn())])
```

A `Pipeline` *is* an estimator: it has `fit` and `predict` and can be handed to
`cross_val_score` like anything else. Inside cross-validation this is not a
convenience but a correctness requirement — the scaler is re-fitted on each
fold's training portion, so the validation fold's mean and standard deviation
never reach the model that is about to be scored on it. Scaling the whole
training set once, up front, would leak exactly that.

### Common mistakes

* **`fit(X_test, y_test)`.** Fitting on the data you are about to score is the
  fastest way to a meaningless 100%.
* **Reading a training-set accuracy as a result.** The notebook prints one
  (99.49% for naive Bayes) purely to show the loop running. A model that
  memorises scores 100% there and can still be worthless.
* **Passing `y` to `predict`.** `predict` takes features only; the labels are
  what you compare its output against.
* **Re-using a fitted estimator as if it were fresh.** Call the factory again
  (or let scikit-learn clone it); a second `fit` silently discards the first.

---

## 2. Logistic regression — a linear decision boundary

### What it computes

Despite its name, logistic regression is a **classifier**. For each class it
computes a weighted sum of the features plus an intercept:

```
score(crop) = w1*N + w2*P + w3*K + w4*temperature
            + w5*humidity + w6*ph + w7*rainfall + b
```

With 22 crops that is 22 scores per row, which the **softmax** function turns
into probabilities:

```
P(crop_i) = exp(score_i) / sum_j exp(score_j)
```

Exponentiating makes every value positive; dividing by the total makes them sum
to 1. The predicted crop is the one with the largest probability — and since
`exp` is increasing, that is also the one with the largest raw score.

Training chooses the weights that make the observed training labels as probable
as possible (maximum likelihood), by gradient-based optimisation — the `lbfgs`
solver by default. There is no closed-form answer, so the solver iterates until
it converges or runs out of budget, and `max_iter` is that budget. This project
sets it to 1,000 rather than scikit-learn's 100 purely as headroom: behind the
Week 3 preprocessor `lbfgs` converges in about 50 iterations, well inside either
figure, but on *unscaled* features it never converges at all and stops at
whichever cap it is given, emitting a `ConvergenceWarning` (§2, common
mistakes).

The whole fitted model on this dataset is **176 numbers**: a 22 x 7 coefficient
matrix plus 22 intercepts. The training rows are discarded once fitting ends.

### Why the boundary is linear

Two crops swap places exactly where their scores are equal:

```
w_a . x + b_a  =  w_b . x + b_b     =>    (w_a - w_b) . x + (b_a - b_b) = 0
```

That is the equation of a flat surface — a line in two dimensions, a plane in
three, a hyperplane in seven. **Logistic regression can only make straight
cuts.** If two crops occupy interleaved or ring-shaped regions, no amount of
data will let it separate them; a different model family will.

"Linear" refers to the features as given, so it is not quite a life sentence:
adding engineered columns (`humidity^2`, `N*K`) buys curved boundaries in the
original space while keeping the model linear in its inputs. That is feature
engineering, and it is not needed here.

### Softmax vs. one-vs-rest

Two ways to make an inherently two-class method handle 22 classes:

* **Multinomial / softmax** — one model over all classes at once, with the
  probabilities normalised across them jointly. This is what scikit-learn 1.6
  fits here, because `lbfgs` supports it.
* **One-vs-rest (OvR)** — train 22 separate "is it rice or not?" classifiers and
  return whichever is most confident. Simple, parallel, and usable with any
  binary method, but the 22 confidences are not *calibrated* against one another
  — that is, a 0.9 from one of them does not mean the same thing as a 0.9 from
  another, because each was trained on a different, badly imbalanced problem.

OvR is available as `OneVsRestClassifier`. Knowing the distinction matters more
than choosing between them here: several algorithms you will meet are binary
underneath, and the wrapper is how they reach a 22-class problem.

### `C`, the regularisation dial

`C` is the **inverse** regularisation strength:

* **small `C`** — strong regularisation. Weights are pulled towards zero, the
  boundary is flatter and simpler, the model may underfit.
* **large `C`** — weak regularisation. Weights are free to grow to fit the
  training data closely, and may overfit.

It is left at the default 1.0 this week on principle: choosing it by trying
values and keeping the best is a *search*, searches need a protocol, and that
protocol is a later week's subject. Adjusting it now by watching cross-validated
scores would be tuning without admitting it.

### Common mistakes

* **Thinking "regression" means it predicts numbers.** It predicts classes; the
  weighted sum it computes is a *log-odds* — the logarithm of "how many times
  more likely this class is than the alternatives" — and that quantity, not the
  label, is what the model is linear in.
* **Reading `C` backwards.** Larger `C` means *less* regularisation.
* **Skipping the scaler and then blaming the algorithm.** Unscaled features make
  the solver crawl and hit `max_iter`; the resulting `ConvergenceWarning` is a
  message about the data's units, not about the model's ability.
* **Interpreting coefficients before Week 7.** Their sizes only compare across
  features once the features share a scale, and even then the correlations
  between `P` and `K` complicate the story.

---

## 3. k-nearest neighbours — prediction by similarity

### What it computes

KNN barely trains. `fit` stores the training rows; that is the whole of its
learning, which is why it is called a **lazy** (or instance-based) learner. The
work happens at prediction time: for a new field, compute the distance to every
stored field, keep the `k` closest, and return the most common label among them.

Distance is Euclidean by default — the straight-line distance in
seven-dimensional feature space:

```
d(a, b) = sqrt( (a_N - b_N)^2 + (a_P - b_P)^2 + ... + (a_rain - b_rain)^2 )
```

There is no equation for the boundary. It is whatever shape the data implies,
which lets KNN follow curved and fragmented class regions that logistic
regression cannot — and means it can never justify an answer beyond "the
neighbours voted this way".

The cost profile is inverted compared with every other model here: fitting is
instant, while every single prediction has to touch every stored row and every
column — 1,760 x 7 arithmetic operations per query here, growing in proportion
to the training set — and the "model" that must be shipped is the entire
training set (1,760 rows x 7 features, and the labels).

### The effect of `k`

`k` is a smoothness dial, and it is the clearest illustration of the
overfitting/underfitting axis you will meet:

* **`k = 1`** — the prediction is the single nearest row's label. The boundary
  wraps every training point exactly, training accuracy is a perfect and
  completely uninformative 100%, and one mislabelled row owns its whole
  neighbourhood. Maximum variance: the model changes a lot if the data changes a
  little.
* **moderate `k`** — votes average over a small region, so isolated odd rows are
  outvoted and the boundary smooths.
* **very large `k`** — the neighbourhood grows until it spans most of the data
  and every query returns the overall majority class. That is precisely the Week
  4 baseline, reached from the other direction. Maximum bias: the model barely
  changes whatever the data says, because it has stopped listening to it.

The notebook's sweep on this dataset shows the curve flat and high for small `k`
and falling away as `k` grows:

| `k` | CV accuracy |
| --- | --- |
| 1 | 0.9665 |
| 3 | 0.9642 |
| 5 | 0.9653 |
| 11 | 0.9534 |
| 25 | 0.9267 |
| 51 | 0.8705 |
| 101 | 0.7710 |
| 201 | 0.6602 |
| 401 | 0.5409 |

It is a *demonstration of a shape*, not a search: nothing from it is adopted,
because "run several values and keep the winner" is a procedure with pitfalls
(a later week) rather than a free lunch.

### Why scaling matters most here

KNN's answer depends entirely on distances, and a distance sums all seven
columns into one number. `K` spans roughly 200 units, `ph` roughly 6; on raw
data a one-unit difference in `ph` is invisible next to potassium, whatever it
means agronomically. Standardising first (Week 3) gives each feature the same
say, which is the principled default.

Principled is not the same as better on one dataset. On this data:

```
KNN, standardised features: 0.9653
KNN, raw features         : 0.9767
```

The raw version scores about a point higher, because this dataset's raw units
happen to weight its most discriminative features (rainfall, humidity, `K`)
most heavily — an accident, not a method. The gap is also within about two fold
standard deviations, which by Week 4's rule is not yet a gap. Keeping the scaler
keeps the comparison between *algorithms* rather than between accidental
weightings, and it is the choice that transfers.

### The curse of dimensionality, briefly

As the number of features grows, distance-based reasoning degrades:

* volume grows exponentially with dimensions, so any fixed number of rows
  becomes sparse and the "nearest" neighbour is not actually near;
* distances **concentrate** — the gap between the nearest and the farthest point
  shrinks relative to their size, so ranking by distance carries less and less
  information;
* every uninformative column adds noise to the sum, diluting the informative
  ones.

Seven features and 1,760 rows are comfortably safe. The effect is easy to stage,
though: adding 100 columns of pure noise to the seven real ones, all
standardised, collapses the same model from 96.5% to about 22%. The information
did not go anywhere — it was buried in a distance that is now mostly noise.

This is why KNN is a poor first choice on text, images or any wide dataset,
and why dimensionality reduction is usually its prerequisite there.

### Common mistakes

* **Quoting `k = 1` training accuracy.** It is always 100% and always
  meaningless.
* **Forgetting the scaler.** Then the model silently ranks the features by their
  units.
* **Assuming odd `k` avoids ties.** It does for two classes; with 22 classes ties
  are perfectly possible and are broken by class order.
* **Treating the `k` sweep as tuning.** Picking the best value off a curve you
  also scored on is how a later week's cautions get earned.

---

## 4. Gaussian naive Bayes — a wrong assumption that works

### Bayes' rule, applied to crops

What we want is `P(crop | field)`. What training data makes easy is
`P(field | crop)` — for each crop, what its fields look like. Bayes' rule turns
one into the other:

```
P(crop | field)  =  P(field | crop) * P(crop) / P(field)
```

`P(field)` is the same for all 22 crops and cannot change which is largest, so
it is ignored. `P(crop)` is the class prior — 1/22 here, since the classes are
balanced. That leaves `P(field | crop)`, a seven-dimensional joint
distribution, which is the hard part.

### The naive assumption

**Naive** Bayes assumes the features are independent *given the class*:

```
P(N, P, K, ..., rainfall | crop)
    = P(N | crop) * P(P | crop) * ... * P(rainfall | crop)
```

A seven-dimensional problem becomes seven one-dimensional ones. **Gaussian**
naive Bayes then models each of those as a normal distribution, so fitting
reduces to computing a mean and a variance per feature per class: 22 x 7 x 2 =
308 numbers, in a single pass over the data. No iteration, no distances, no
solver. It is by far the cheapest model here.

`var_smoothing` (default `1e-9`) adds a small floor to every variance so that a
feature which happens to be constant within a class cannot produce a division by
zero.

### The assumption is false here — and it still wins

Week 2 measured a correlation of **0.74 between `P` and `K`**. Knowing a field's
phosphorus tells you a lot about its potassium, so the features are plainly not
independent, and the model's factorised probability is wrong.

It still scores 99.49%, the best of the three. Two reasons:

* **Classification needs a ranking, not a calibration.** Only which of the 22
  scores is largest decides the answer. Correlated features are effectively
  counted twice, which makes the winning probability wildly overconfident (often
  0.999-something) while usually leaving the winner unchanged.
* **Few parameters, low variance.** 308 numbers estimated from 1,760 rows are all
  estimated well. A model with a more faithful assumption and far more parameters
  can easily do worse at this sample size — the bias/variance trade-off, seen
  from the bias side.

There is a third reason specific to this dataset: Week 2 showed each crop
occupying a compact, well-separated blob in feature space. "One Gaussian per
crop per feature" is an almost exact description of that shape, so the model's
assumptions and the data's structure happen to match unusually well.

The moral is not "naive Bayes is best". It is that **model complexity is not a
ranking**: what matters is the fit between a model's assumptions and the data's
shape, and the only way to find out is to measure.

### Generative vs. discriminative, briefly

Naive Bayes models `P(features | class)` — how each class's data is *generated*
— and derives the decision from it. Logistic regression models the boundary
`P(class | features)` directly and never describes what a class's data looks
like. Generative models can generate plausible new rows and cope with missing
features more gracefully; discriminative models usually classify better when
data is plentiful, because they spend all their capacity on the only question
being asked.

### Practical notes

* **Standardisation does not affect it.** A per-column linear rescale moves every
  class's mean and variance for that column identically, so predictions are
  unchanged — verified on this dataset in
  [validation.md Step 4](validation.md#scaling-changes-knns-answers-and-not-naive-bayes),
  where raw and standardised features give identical predictions for all 1,760
  rows. (The invariance is exact only up to `var_smoothing`, which scikit-learn
  scales by the *largest* feature variance in the data, so an extreme rescale of
  one column can still nudge a handful of rows: multiplying `K` by 1,000 moves
  7 of the 1,760, which is exercise B8.) The preprocessor stays in front of it
  only so all three models receive identical inputs.
* **Trust its predictions more than its probabilities.** If you need calibrated
  confidence — "recommend a crop only above 90% certainty" — this is the wrong
  model, or it needs a calibration step.
* **It is the natural second baseline.** Fast, assumption-driven, and a bar that
  a well-chosen algorithm ought to clear. On this dataset that bar turns out to
  be demanding.

### Common mistakes

* **Rejecting it because the assumption is violated.** Almost every real dataset
  violates it. Measure, then decide.
* **Quoting its probabilities as confidence.** They are systematically
  overconfident when features are correlated.
* **Confusing "naive" with "simple to the point of useless".** It is a genuine
  probabilistic model — just one with a strong, explicit and usually wrong
  assumption.

---

## 5. Comparing models fairly

"Fairly" is not a feeling. It means four things held constant across every
candidate:

1. **The same rows** — all four are scored on the same 1,760 training rows.
2. **The same folds** — `StratifiedKFold(n_splits=5, shuffle=True,
   random_state=42)` produces identical partitions for every model, so each is
   asked the same five questions.
3. **The same metric** — accuracy, chosen in Week 4, before any candidate
   existed.
4. **The same preparation** — the same preprocessor in front of each model, so
   the experiment varies the algorithm and nothing else.

Vary any of these and the difference you measure includes the difference between
protocols. Comparing model A cross-validated at seed 42 with model B at seed 7 is
partly measuring the seeds.

### Report the spread, not just the mean

Week 4's rule applies with force here: **a gap smaller than the fold-to-fold
standard deviation is not yet a gap.** Logistic regression's 96.82% and KNN's
96.53% differ by 0.29 points while KNN alone wobbles by 1.21 points across
folds. Nothing in this experiment ranks them. Naive Bayes' 2.67-point lead over
logistic regression, against spreads of 0.42 and 0.66, is a real difference by
the same standard.

Distinguishing genuinely close models — repeated cross-validation, paired tests
over the same folds — is a later week.

### Keep the baseline in the table

Every row is quoted next to 4.55%. Without it, "96.5%" is a number without a
scale; with it, "96.5% against a 4.55% floor" is a claim about how much the
seven features are worth.

### The table is extended, not replaced

Weeks 6, 7 and 8 add rows and columns — tuned models, ensembles, precision and
recall — to *this* table. Keeping one running record makes the course's progress
visible and makes regressions obvious; rebuilding the comparison every week
would hide both.

### Common mistakes

* **Comparing means without spreads.** The single most common way to declare a
  false winner.
* **Comparing scores from different protocols.** Different seeds, different fold
  counts, or one model scored on a single split and another cross-validated.
* **Peeking at the test set to break a tie.** That converts the final honest
  estimate into another training signal. `data/processed/test.csv` stays closed
  until Week 8.
* **Choosing on accuracy alone.** Training time, prediction time, model size,
  explainability and probability quality are all legitimate criteria; this week
  reports accuracy because Week 8 is where the metric conversation broadens.

---

## 6. This week's result

5-fold stratified cross-validation, seed 42, on the 1,760 training rows, each
model behind the Week 3 preprocessor:

| Model | CV accuracy | Std across folds | vs. baseline | Error rate |
| --- | --- | --- | --- | --- |
| **Gaussian naive Bayes** | **0.9949** | 0.0042 | +0.9494 | 0.0051 |
| Logistic regression | 0.9682 | 0.0066 | +0.9227 | 0.0318 |
| KNN (`k = 5`) | 0.9653 | 0.0121 | +0.9198 | 0.0347 |
| `most_frequent` baseline | 0.0455 | 0.0000 | — | 0.9545 |

How to read it:

* **All three beat the baseline by more than 90 percentage points.** That was the
  only thing genuinely in doubt, and it is now settled: the seven features carry
  real information about which crop suits a field.
* **Gaussian naive Bayes leads at 99.49%.** The simplest and most obviously
  misspecified model of the three is the most accurate, for the reasons in §4.
* **Logistic regression and KNN are not separated by this experiment.**
* **Error rates are the sharper comparison.** 0.51% versus 3.18% is a six-fold
  difference in mistakes that "99.5% versus 96.8%" understates. On 440 test rows
  that is roughly 2 errors versus 14.

### When would you prefer KNN over logistic regression?

Prefer **KNN** when:

* the class regions are **curved, fragmented or oddly shaped** — KNN follows any
  boundary the data implies, logistic regression only cuts straight;
* the feature count is **small** and rows are plentiful, so distances stay
  meaningful;
* training must be **instantaneous**, or labelled rows arrive continuously —
  adding data to KNN is appending to a list, with no refit;
* you want a model with essentially no assumptions about the data's shape as a
  sanity check on a more structured one.

Prefer **logistic regression** when:

* you must **explain** a prediction (Week 7), or hand a regulator a set of
  weights;
* prediction must be **fast** or the artifact **small** — 176 numbers versus
  1,760 stored rows, and the gap grows linearly with the dataset;
* there are **many features**, where distances stop discriminating;
* you need **well-behaved probabilities** rather than the coarse fractions
  `k = 5` can produce (only 0, 0.2, 0.4, 0.6, 0.8 or 1 are possible);
* the data may contain many irrelevant columns, which logistic regression can
  down-weight and KNN cannot.

And prefer **naive Bayes** when speed matters most, data is scarce, or you need
a strong baseline in seconds — as here.

---

## 7. The code this week

### `src/models/classical_models.py`

Three factories, mirroring `get_baseline_model` from Week 4:

```python
get_logistic_regression(C=1.0, max_iter=1000, random_state=42)  -> LogisticRegression
get_knn(n_neighbors=5, weights="uniform")                       -> KNeighborsClassifier
get_naive_bayes(var_smoothing=1e-9)                             -> GaussianNB
```

Design decisions worth noticing:

* **They return unfitted estimators**, like the preprocessor and the baseline
  before them, so cross-validation can clone and re-fit them per fold.
* **They validate their arguments** — a non-positive `C`, `k < 1`, an unknown
  weighting or a negative `var_smoothing` raise `ValueError` immediately, rather
  than surfacing as an obscure failure inside `fit`.
* **`max_iter` defaults to 1,000, not scikit-learn's 100**, which is the one
  place the project overrides a library default. It is headroom rather than a
  necessity: behind the preprocessor the solver finishes in about 50 iterations,
  but on unscaled features it exhausts any budget it is given.
* **`CLASSICAL_MODEL_FACTORIES`** maps names to factories so the notebook can
  loop over the candidates instead of repeating itself, and so Weeks 6-8 can add
  entries rather than rewrite the loop.

### `tests/test_classical_models.py`

57 tests in four groups:

* **The factories** — types, defaults, argument pass-through, rejection of
  nonsense arguments, and the unfitted contract.
* **The shared loop** — `fit` then `predict` works identically for all three,
  alone and inside a pipeline; results are reproducible; cross-validation leaves
  the original estimator unfitted; unlike the baseline, scrambling the features
  changes the answers.
* **Algorithm-specific behaviour** — `k = 1` memorises; a huge `k` collapses
  towards ignorance; KNN's predictions change under rescaling and naive Bayes'
  do not; 100 noise columns wreck KNN; the fitted parameter shapes are what the
  theory says (22 x 7 coefficients, 22 x 7 means and variances).
* **The comparison** — every model beats the baseline, on synthetic data and on
  the real 1,760 rows, with naive Bayes the leader at ~99.5%.

### `notebooks/05_classification_models.ipynb`

Part 1 of the modelling notebook: the training loop named, the three algorithms
explained and fitted, the `k` sweep, the noise demonstration, and the results
table with guard-rail assertions at the end so the narrative cannot silently
drift away from the numbers.

---

## 8. What this week produced, and what it deliberately did not

**Produced**

* Three model factories, and the `fit`/`predict` loop named as the pattern every
  later model reuses.
* A fair comparison — identical rows, folds, metric and preparation — of three
  algorithms against the Week 4 baseline.
* The current leader: **Gaussian naive Bayes at 99.49%**, with logistic
  regression at 96.82% and KNN at 96.53%, all far above 4.55%.
* A results table that Weeks 6-8 extend.

**Not produced, on purpose**

* No tuned hyperparameter. `C = 1.0`, `k = 5`, `var_smoothing = 1e-9` are
  defaults; the `k` sweep is a demonstration, not a search — a later week.
* No ensemble: no random forest, gradient boosting, voting or stacking —
  a later week.
* No verdict on logistic regression versus KNN; their gap is smaller than the
  noise — a later week.
* No statement about which features drive a prediction — **Week 7**.
* No precision, recall, F1 or confusion matrix — **Week 8**.
* No test-set score. `data/processed/test.csv` remains unopened.

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| ◀ [§5.1 Syllabus](syllabus.md) | [Chapter 5 — Classification Models](README.md) · 🗺 [Roadmap](../README.md) | [§5.3 Exercises](exercises.md) ▶ |

<!-- nav:end -->
