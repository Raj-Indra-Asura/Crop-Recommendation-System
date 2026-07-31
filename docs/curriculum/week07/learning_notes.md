# Week 7 — Learning Notes

**The week the answer stops being "a better model" and starts being "more
models".**

Six models are in the running table, and the strongest of them is the one with
the fewest moving parts: Gaussian naive Bayes at 99.49%, ahead of the decision
tree at 98.52%, the SVM at 97.90%, logistic regression at 96.82% and KNN at
96.53%, all against a 4.55% floor. Week 6 also left an unresolved observation:
its unlimited tree scored a perfect **100%** on the rows it was fitted on and
**98.52%** on rows it had not seen, and that gap has a name — *variance*.

The obvious response is to make the tree simpler. This week takes the opposite
route:

* **Random forest** — keep the overfitting trees, fit a hundred of them on
  different random resamples with different features available at each split,
  and average their votes. This is **bagging**, and it removes variance.
* **Gradient boosting** — go to the other extreme: fit deliberately feeble trees
  one after another, each trained on what the ensemble so far still gets wrong,
  and add up their shrunken contributions. This is **boosting**, and it removes
  bias.

Then the first answer this course gives to "which of the seven measurements
actually matters?" — `feature_importances_`, plotted — followed immediately by
the three reasons not to over-read it.

The jargon introduced this week is: *ensemble* (§1), *bagging*, *bootstrap
sample* and *out-of-bag* (§2), *decorrelation* and *feature randomness* (§2),
*boosting*, *weak learner* and *shrinkage* (§3), and *mean decrease in impurity*
(§4). Nothing below assumes you already know them.

Code produced this week: `src/models/ensemble_models.py`,
`tests/test_ensemble_models.py`, and Part 1 of
`notebooks/06_model_selection.ipynb`.

---

## 0. What changes this week, and what does not

The protocol does not change: the same 1,760 training rows, the same stratified
5-fold split with seed 42, accuracy as the metric, the same Week 3 preprocessor
in front of every candidate, and the same running table — now eight rows instead
of six. `data/processed/test.csv` stays closed until Week 8.

What changes is the *unit of modelling*. Every week so far has fitted **one**
model and asked how good it is. From here on, a "model" can be a committee of
hundreds, and the interesting question becomes how the members differ from one
another rather than how good any of them is.

One practical change: an optional dependency. `xgboost==2.1.3` is now listed in
`requirements.txt`, and `get_gradient_boosting()` uses it **when it imports** and
falls back to scikit-learn's `GradientBoostingClassifier` when it does not
(§3.4). No part of this week is blocked on that install.

### Common mistakes

* **Believing an ensemble is automatically better.** It is better only when its
  members disagree; §1 makes that precise. Combining a hundred copies of the
  same model is a hundred times the compute for exactly nothing.
* **Assuming this week produces a new leader.** It does not, on this dataset.
  Naive Bayes still wins, narrowly, and §5 explains why that is a fact about the
  data rather than a failure of the ensembles.
* **Reading `feature_importances_` as a causal statement.** It describes what
  the fitted model used, not what makes crops grow (§4).

---

## 1. What an ensemble is, and the one condition it needs

An **ensemble** fits many models and combines their answers. The reason it can
beat any of its members is easier to see in numbers than in prose.

Imagine a committee of voters who each answer a yes/no question correctly 70% of
the time, and — this is the load-bearing word — whose mistakes are
**independent**. A majority vote of

| Members | Majority is right |
| --- | --- |
| 1 | 70.6% |
| 3 | 77.7% |
| 15 | 94.8% |
| 101 | 100.0% (on all 10,000 simulated questions) |

Nobody got smarter. A majority is wrong only when *most* members are wrong **at
the same time**, and independent errors rarely line up that way. Notebook §1
runs this simulation; it takes four lines and is worth running yourself, because
everything in the rest of the week is a way of manufacturing that independence
out of a single algorithm.

> **The one condition.** Averaging cancels only the errors the members do **not**
> share.

Fit 100 decision trees on the same 1,760 rows with the same settings and you get
100 *identical* trees. Their average is one tree, and nothing has been gained.
So every ensemble method is, at bottom, a scheme for forcing its members apart —
and the two families do that in opposite ways.

| | **Bagging** (random forest) | **Boosting** (gradient boosting) |
| --- | --- | --- |
| Members fitted | in parallel, independently | in sequence, each on the last ones' errors |
| Each member sees | a random resample of rows; a random subset of features per split | all rows, focused on what is still wrong |
| Members are | **strong** and overfit (deep trees) | **weak** (shallow trees, often stumps) |
| Combined by | voting / averaging probabilities | adding up shrunken contributions |
| Mainly reduces | **variance** | **bias** |
| More members overfits? | **no** — just more compute | **yes** — later rounds fit the leftover noise |
| Parallelisable | trivially | not across rounds |

In one sentence, and this is the sentence to be able to produce on demand:
**bagging averages independent opinions to cancel their noise; boosting builds a
chain of specialists, each hired to fix what the previous ones got wrong.**

### Common mistakes

* **"Ensemble" meaning "several different algorithms".** That is one kind
  (voting and stacking, not covered until later); both models this week are
  ensembles of *one* algorithm — decision trees — differing only in how they are
  fitted.
* **Expecting the 70% -> 95% jump on real data.** Real members are nowhere near
  independent. The table above is the *ceiling* the ensemble is reaching for,
  not a prediction.

---

## 2. Random forest — bagging plus feature randomness

A random forest is three ideas stacked, and each one has a job.

### 2.1 Bootstrap sampling

Each tree gets its own training set, drawn from the original **with
replacement** and of the same size: 1,760 rows drawn from 1,760 rows, so some
appear twice or three times and some not at all. That resample is a **bootstrap
sample**, and it is what "bootstrap aggregating" — *bagging* — is named after.

The arithmetic is fixed and worth remembering: the chance a given row is missed
by a draw of size *n* is `(1 - 1/n)^n`, which converges to `1/e ≈ 0.368`. So each
tree sees about **63%** of the distinct rows and about **37%** are left over —
its **out-of-bag** rows. The notebook confirms it on the real data: the first
tree of the forest was fitted on 1,114 distinct rows (63.3%), leaving 646
out-of-bag.

Those out-of-bag rows are held-out data that came free with the resampling, and
scikit-learn can score against them with `oob_score=True`. This project does not
use that, on purpose: every number in the running table comes from the same
five stratified folds, and swapping in a differently-computed score for one row
would break the comparison.

### 2.2 Feature randomness

Bootstrap sampling alone is not enough, because of §1's condition. Trees fitted
on resamples of the same data still agree with each other far too much: if one
feature is strongly predictive, nearly every tree splits on it first and they
all go wrong in the same places.

So a random forest adds a second, sharper source of randomness. **At every
node**, a tree may only choose its split from `max_features` randomly drawn
features — `"sqrt"` by default, which scikit-learn floors, so **2 of this
project's 7**. A tree that would have opened on `rainfall` is often not offered
`rainfall` at all, and has to find another way.

The effect is measurable. Across the 100 trees, the first question asked is:

```
humidity       25
K              20
P              19
rainfall       17
N              16
temperature     3
```

Six different opening questions, where the single Week 6 tree always opens on
`rainfall`. That spread **is** the decorrelation, and it is exactly what makes
the average worth more than its members.

### 2.3 Voting

Predictions are combined by averaging the trees' predicted probabilities — a
smoother version of "take a majority vote", and the reason a forest reports
sensible `predict_proba` output even though each individual tree's probabilities
are crude.

### 2.4 Why the members are left unpruned

`get_random_forest()` defaults to `max_depth=None`, which is the **opposite** of
the advice Week 6 gave for a single tree. That is deliberate. Bagging's job is
to remove variance; the members are therefore supposed to supply *low bias*,
even at the cost of high variance, and let the averaging clean up after them.
Pruning them would trade away accuracy the averaging was going to recover for
free.

The evidence, on the same folds as every other row in the table:

| Model | CV accuracy | Error rate |
| --- | --- | --- |
| One unlimited tree | 98.52% (±0.68) | 1.48% |
| 100 of them, averaged | **99.26%** (±0.58) | **0.74%** |

The forest **halves the error rate** of the very model it is built from, and its
fold-to-fold spread shrinks too — variance reduction showing up directly in the
measurement. Note also that a fitted forest still scores a perfect 1.0 on its
own training data: its members memorise exactly as before. Memorisation was
never the problem; *unshared* memorisation is the cure.

### 2.5 `n_estimators` is a budget, not a dial

This is the one hyperparameter in the project that cannot overfit. Adding more
members to an average cannot increase its variance — it can only cost time. The
curve rises and flattens:

| Trees | CV accuracy |
| --- | --- |
| 1 | 0.9483 |
| 3 | 0.9756 |
| 10 | 0.9909 |
| 30 | 0.9932 |
| 100 | 0.9926 |
| 300 | 0.9932 |

Two things to read off it. First, **one tree in a "forest" is worse than a plain
decision tree** (94.83% against 98.52%) — it is a deliberately handicapped
member, fitted on 63% of the rows with 2 features per split. Ten of them already
pass it. Second, past ~30 trees the differences (0.9932, 0.9926, 0.9932) are
smaller than the fold spread; the curve has flattened and more trees buy only
compute. Contrast that with Week 6's depth sweep, where *both* ends of the range
were bad. Here there is no bad end on the right.

### Common mistakes

* **Pruning the forest's trees by habit.** `max_depth=5` in a forest usually
  makes it worse; the members are meant to overfit.
* **Tuning `n_estimators` for accuracy.** Set it as high as your patience
  allows. It is a compute decision.
* **Thinking `max_features="sqrt"` is about speed.** It is about
  decorrelation. Setting `max_features=None` gives plain bagged trees, which are
  more correlated and usually slightly worse.
* **Quoting the forest's out-of-bag score alongside cross-validated scores from
  other models.** Different protocol, not comparable.

---

## 3. Gradient boosting — a chain of small corrections

Boosting inverts every choice a forest makes.

### 3.1 The procedure

1. Start from a constant prediction.
2. Look at the error the ensemble currently makes — formally, the gradient of
   the loss with respect to the current predictions, which is where *gradient*
   boosting gets its name.
3. Fit a small tree to **that error**.
4. Add `learning_rate ×` (the new tree) to the running prediction.
5. Repeat, `n_estimators` times.

That is the whole idea, and the full derivation is not needed to use it. The
useful mental picture is a student correcting a draft: each pass fixes what the
previous passes left wrong, and no pass tries to rewrite the whole thing.

### 3.2 Weak learners, on purpose

`max_depth=3` is the default here: a member that can combine at most three
features. Strength comes from the chain, not the link. Notebook §3 shows it with
literal stumps (`max_depth=1`), reporting **training** accuracy — used to show
the mechanism, never as a result:

| Rounds | Training accuracy |
| --- | --- |
| 1 | 0.6193 |
| 2 | 0.6710 |
| 5 | 0.6733 |
| 20 | 0.9307 |
| 60 | 0.9881 |

One round of stumps manages 61.9% — well above the 4.55% floor, because a single
threshold per crop already isolates the most extreme classes, and nowhere near a
usable model. Five rounds barely move it. Then the corrections compound: 93.1%
by round 20, 98.8% by round 60. Nothing about the individual learner changed
between those rows; only how many corrections were chained.

Compare with the forest sweep in §2.5, which *starts* high because one member is
already a complete model. That difference is the difference between bagging and
boosting, in one pair of tables.

### 3.3 `learning_rate` and `n_estimators` are one dial, not two

`learning_rate` shrinks each round's contribution. Small values take smaller
steps, so more rounds are needed, and the result usually generalises better —
this is **shrinkage**, and it is boosting's main regulariser. Halve the rate and
you need roughly twice the rounds for the same fit.

Unlike a forest, a booster **can** overfit as members are added: round 400 will
happily fit whatever error remains, including the part that is noise. That is
why `n_estimators` here is a real capacity dial and in a forest it is not. Tuned
properly, the two are searched together — Part 2's subject, not this week's.

### 3.4 XGBoost, and the fallback

`get_gradient_boosting()` returns an **XGBoost** model when the optional
`xgboost` package is importable, and scikit-learn's
`GradientBoostingClassifier` when it is not. `GRADIENT_BOOSTING_BACKEND` reports
which, and the notebook prints it beside every result rather than assuming.

Both implement the same algorithm and behave identically through
`fit` / `predict` / `predict_proba` / `feature_importances_`. XGBoost is
considerably faster — it bins the features and grows all 22 classes' trees per
round together — and adds L1/L2 penalties on the leaf values, which is why its
score differs slightly:

| Backend | CV accuracy |
| --- | --- |
| XGBoost 2.1.3 | 99.09% (±0.33) |
| scikit-learn `GradientBoostingClassifier` | 98.69% (±0.34) |

Both beat every Week 5/6 single model; both sit below the forest. **Nothing in
this week's conclusions depends on which one you have**, which is the point of
writing the fallback rather than making the package mandatory: an optional
dependency that fails to build on someone's machine must never be able to stop
them learning what boosting is.

One wrinkle is worth knowing, because it is a common source of confusion when
people first reach for XGBoost. Its scikit-learn wrapper is *almost*
API-compatible, with one gap: `XGBClassifier` requires the target to be the
integers `0..n_classes-1` and raises `ValueError` on anything else. This
project's target is the crop name. `src/models/ensemble_models.py` therefore
wraps it in a small `XGBoostStringLabelClassifier` that label-encodes `y` going
into `fit` and decodes it coming out of `predict` — closing that gap and nothing
else, so the model drops into the same pipeline, the same folds and the same
table as everything else.

### Common mistakes

* **Raising `n_estimators` until the training score is perfect.** That is what
  boosting will always do. Only held-out scores can tell you when to stop.
* **Comparing a boosting score across backends.** Quote which one produced it.
* **Expecting boosting to be as fast as a forest.** The rounds are sequential by
  construction; only the work *inside* a round parallelises.
* **Treating `max_depth=3` as a limitation to be fixed.** Deep boosted trees
  usually overfit faster and help less.

---

## 4. `feature_importances_` — the first answer, and its three holes

Both ensembles expose an array of one number per feature, non-negative and
summing to 1. For tree ensembles it is **mean decrease in impurity** (MDI):
every time a feature is chosen for a split, the drop in Gini impurity that split
achieved is weighted by how many rows passed through that node; the totals are
summed over every node of every tree and normalised.

The forest's, fitted on all 1,760 training rows:

| Feature | Importance |
| --- | --- |
| `rainfall` | 0.2302 |
| `humidity` | 0.2242 |
| `K` | 0.1754 |
| `P` | 0.1508 |
| `N` | 0.0964 |
| `temperature` | 0.0724 |
| `ph` | 0.0506 |

### How to read the plot

Read it as a **ranking with rough magnitudes**, not as a set of precise
quantities.

* **`rainfall` and `humidity` lead**, together accounting for nearly half the
  impurity the forest removed. That agrees with Week 2's boxplots and with Week
  6's single tree, whose first two questions were `rainfall <= 30.18` and
  `humidity <= 27.98`.
* **`K` and `P` follow**: the nutrient profile separates crops that the weather
  does not.
* **`ph` is last**, at 0.05. That does *not* mean soil pH is agronomically
  unimportant. It means that, **given the other six columns**, this forest
  rarely needed it.
* The forest's and the booster's plots **agree on the ranking and disagree on
  the magnitudes**, because they are different algorithms scoring splits
  differently. Another reason to read the order rather than the numbers.

### The three limitations

1. **It is computed on the training data.** There is no held-out set anywhere in
   this number, so a feature the model overfitted on scores highly whether or
   not it helps on unseen rows.
2. **It is biased towards features with many distinct values.** A continuous
   column offers hundreds of candidate thresholds and therefore many chances to
   win a split; a binary column offers one. Importance partly measures
   *opportunity*.
3. **Correlated features split the credit arbitrarily.** Week 2 measured a 0.74
   correlation between `P` and `K`. Whichever a tree happens to split on first
   absorbs the importance, and the other looks redundant.

The third is the trap that catches people most, so the notebook stages it:
duplicate the `humidity` column under a new name and change nothing else.

| | Importance |
| --- | --- |
| `humidity`, alone | 0.2242 |
| `humidity`, with a copy present | 0.1485 |
| `humidity_copy` | 0.1370 |
| the two together | 0.2855 |

Training accuracy is 1.0 in both cases. The information is unchanged, the model
is exactly as accurate, and yet `humidity` now looks about **half** as important
— because half the time a tree that wanted it drew the copy instead. Anyone
dropping "unimportant" features off such a plot would delete a column the model
depends on.

> **Feature importance describes the fitted model, not the world.** It says
> which columns *this* model used. It never says which measurements *cause* a
> crop to suit a field, and never why one individual field was classified as it
> was.

### What Week 8 does about it

* **Permutation importance** — shuffle one column in data the model did **not**
  train on and measure how far accuracy falls. This fixes limitation 1 (it is
  held-out) and partly 2 (it does not care how many thresholds a column offers).
* **SHAP** — attribute a *single* prediction across the features that produced
  it, so "why was this field labelled rice?" becomes answerable.

Until then, treat the plot as a hypothesis, not a finding.

### Common mistakes

* **Feature selection straight off an MDI plot.** Drop the bottom two features
  here and you may be dropping one that only looks small because its correlated
  twin took the credit.
* **Reading importance as effect size or direction.** It has no sign: it cannot
  tell you whether more rainfall makes rice *more* or *less* likely.
* **Comparing importances across models as if they were the same quantity.**
  They are computed differently by each algorithm.

---

## 5. The extended results table

Two rows appended to Week 6's six. Same rows, same folds, same seed, same
metric, same preprocessor.

> ### Week 7 result — 5-fold stratified CV on the 1,760 training rows
>
> | Model | CV accuracy | Error rate | Added |
> | --- | --- | --- | --- |
> | Gaussian naive Bayes | **99.49%** (±0.42) | 0.51% | Week 5 |
> | Random forest (100 trees) | 99.26% (±0.58) | 0.74% | **Week 7** |
> | Gradient boosting (XGBoost) | 99.09% (±0.33) | 0.91% | **Week 7** |
> | Decision tree (unlimited) | 98.52% (±0.68) | 1.48% | Week 6 |
> | SVM (rbf, `C = 1`) | 97.90% (±1.03) | 2.10% | Week 6 |
> | Logistic regression | 96.82% (±0.66) | 3.18% | Week 5 |
> | KNN, `k = 5` | 96.53% (±1.21) | 3.47% | Week 5 |
> | `most_frequent` baseline | 4.55% (±0.00) | 95.45% | Week 4 |
>
> With the scikit-learn fallback, gradient boosting scores 98.69% (±0.34) —
> still above every Week 5/6 model, still below the forest.

**So: do the ensembles beat the Week 5 and Week 6 models on this dataset?**

Yes, with exactly one exception. Both ensembles beat the decision tree, the SVM,
logistic regression and KNN. Both fall just short of **Gaussian naive Bayes**,
which has led since Week 5 and still does. The forest's shortfall is 0.23
points against fold standard deviations of ±0.42 and ±0.58 — comfortably inside
the noise — so the correct statement is that the top of the table is a **draw**,
not that naive Bayes wins or that the forest was beaten.

That is the answer to give, and giving it precisely is the skill: "the ensembles
beat every single model except naive Bayes, which they are level with."

### Why "ensembles usually win" and "naive Bayes wins here" are both true

Both statements are correct, and the reconciliation is a fact about *this
dataset*.

* Week 2 showed the 22 crops sit in **compact, well-separated, roughly
  bell-shaped blobs**. "One Gaussian per crop per feature" is an almost exact
  description of that shape, so naive Bayes has very little bias to remove — and
  with only 308 parameters, very little variance to average away either. The
  ensembles' machinery has nothing to fix.
* There is almost **no headroom**. Above 99.5%, a model has fewer than nine
  mistakes left in 1,760 rows; the difference between the top three rows of the
  table is a handful of individual fields.
* On messier data — overlapping classes, noisy labels, interacting features,
  hundreds of columns — that headroom is large and the ranking reverses.
  Gradient-boosted trees are the usual winner on real tabular problems for
  precisely that reason.

Two further things the table earns:

* **Ensembles are not free.** The forest fits 100 trees; the booster fits 100
  rounds × 22 classes. Naive Bayes computes 308 numbers in one pass over the
  data. When two models tie, the cheaper and more inspectable one is the better
  default — a point that returns in Week 9 when something has to be saved and
  served.
* **Some gaps in this table are still noise.** The forest and the booster are
  0.17 points apart with spreads of 0.58 and 0.33. That is not a decided
  contest, and no amount of staring at the means will decide it.

---

## 6. The code this week

### `src/models/ensemble_models.py`

Two factories, in the shape every earlier week established: explicit defaults,
argument validation that raises `ValueError` immediately rather than at `fit`
time, and an **unfitted** estimator returned, so cross-validation can fit a
fresh clone per fold.

```python
get_random_forest(n_estimators=100, max_depth=None, max_features="sqrt", random_state=42)
get_gradient_boosting(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
```

Alongside them:

* `ENSEMBLE_MODEL_FACTORIES` — the registry, kept separate from
  `CLASSICAL_MODEL_FACTORIES` so "single models" and "ensembles" stay
  distinguishable in the table;
* `XGBOOST_AVAILABLE` and `GRADIENT_BOOSTING_BACKEND` — which implementation the
  environment has;
* `XGBoostStringLabelClassifier` — the label-encoding adapter described in §3.4.

### `tests/test_ensemble_models.py`

61 tests (292 in the whole suite), and they pass **with and without** XGBoost
installed — the fallback branch is exercised explicitly by monkeypatching the
availability flag. Beyond the usual factory and validation checks, they measure
the claims this week makes rather than restating them: that a forest's
predictions change less than a single tree's across bootstrap resamples, that
feature randomness produces trees with different root splits, that a chain of
boosting rounds beats a single round, that importances sum to 1 and rank signal
above noise, that duplicating a column splits its importance, and that both
ensembles beat every Week 5/6 model except naive Bayes, with which they tie.

### `notebooks/06_model_selection.ipynb`, Part 1

§0 setup and backend report, §1 the committee simulation and the bagging/boosting
contrast, §2 the forest (bootstrap counts, root-split spread, the tree-versus-
forest comparison, the `n_estimators` sweep), §3 boosting (the chain, the rounds
sweep), §4 the feature-importance plot and the duplicated-column demonstration,
§5 the eight-row table, §6 the result with self-asserting guard rails, §7 what
was and was not produced. Part 2 — hyperparameter search — is added to the same
notebook in a later week, exactly as Week 6 extended Week 5's.

---

## 7. What this week produced, and what it deliberately did not

**Produced**

* Two new factories and a fallback that keeps an optional dependency optional.
* Bagging and boosting as a contrast rather than two definitions.
* The measured case for bagging: six different opening questions across 100
  trees, and half the single tree's error rate.
* The measured case for boosting: one stump is useless, sixty chained stumps are
  nearly perfect.
* The first feature-importance plot in the course, and the first demonstration
  of why it must not be trusted naively.
* An eight-row running table and a precise answer to "do ensembles win here?".

**Not produced, on purpose**

* No tuned hyperparameter. Both sweeps are demonstrations of curves; nothing
  from either is adopted, because choosing settings by eye from validation
  scores is what a stated search protocol exists to prevent. `GridSearchCV`,
  `RandomizedSearchCV` and validation curves come in Part 2.
* No explanation of an individual prediction, and no held-out importance —
  Week 8 (SHAP, permutation importance).
* No voting or stacking classifier, and no ensemble across model families.
* No precision, recall, F1 or confusion matrix — Week 8.
* No test-set score. `data/processed/test.csv` remains unopened.
