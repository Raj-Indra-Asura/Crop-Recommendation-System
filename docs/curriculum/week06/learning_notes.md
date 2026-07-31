# Week 6 — Learning Notes

**Two more algorithms, and the moment model complexity stops being a virtue.**

Week 5 put three classifiers on top of the Week 4 floor and read the result
honestly: Gaussian naive Bayes at 99.49%, logistic regression at 96.82%, KNN at
96.53%, all measured against a `most_frequent` baseline of 4.55% on 22 balanced
classes. The table it produced was described, repeatedly, as something Weeks 6-8
**extend rather than replace**. This week is the first extension.

Two algorithms join the comparison, from two families the course has not yet
touched:

* **the support vector machine (SVM)** — a *margin-based* model that places its
  boundary as far as it can from both classes, and can bend that boundary with a
  *kernel*;
* **the decision tree** — a chain of yes/no questions whose depth is a dial you
  can turn from useless to overfitted and watch the damage happen.

The tree is here for a second reason beyond its score. It is the clearest device
in the whole course for showing **overfitting**, the **bias-variance tradeoff**,
and why a training-set number is never a result. Turning one dial — `max_depth`
— and plotting two curves makes all three concrete on this dataset's own numbers.

The jargon introduced this week is: *margin* and *support vector* (§2), *kernel*
and *the kernel trick* (§3), *Gini impurity*, *entropy* and *greedy search*
(§4), and *bias* and *variance* (§5). Nothing below assumes you already know
them.

Code produced this week: two new factories in
`src/models/classical_models.py`, a plotting helper in
`src/utils/visualization.py`, more tests in `tests/test_classical_models.py`,
and Part 2 of `notebooks/05_classification_models.ipynb`.

---

## 0. What changes this week, and what does not

As in Week 5, almost nothing about the *process* changes. The 1,760 training
rows are the same, the split is the same, the five stratified folds are the
same, the metric is accuracy, and the floor is still 4.55%. The two new models
are wrapped in the same Week 3 preprocessor and appended to the same running
table. That is deliberate: a comparison only means something if every candidate
faces identical rows, identical folds, identical preparation and an identical
metric (the four constants named in [Week 5 §5](../week05/learning_notes.md)).

What is genuinely new is the *purpose*. Week 5 asked "do the seven features
carry information?" and answered yes by more than ninety points. That question
is settled and does not need re-asking. This week asks a harder one: **when does
adding capacity to a model start to hurt?** The SVM and the tree both have a
flexibility dial — `C` (§2) and `gamma` (§3) for the SVM, `max_depth` (§5) for
the tree — and the point of the week is to watch what those dials do rather than
to chase a new best score.

### Common mistakes

* **Assuming the newer, more flexible model must win.** It does not, here. The
  simplest model from last week — naive Bayes — still leads, and the linear SVM
  matches the RBF one. "Use the more powerful algorithm" is not a rule; matching
  the model to the shape of the data is.
* **Reading the sweeps in §3 and §5 as tuning.** They draw curves to explain a
  mechanism. Nothing from them is adopted, for the reason spelled out in §7.

---

## 1. Two more factories, the same loop

Every model in this project is still driven by the two method calls named once
and for all in [Week 5 §1](../week05/learning_notes.md):

```python
model.fit(X_train, y_train)          # learn parameters from labelled rows
predictions = model.predict(X_test)  # apply them to rows it has not seen
```

Learning a new algorithm means learning what happens *inside* `fit`, not a new
way to call it. So the whole of this week's new interface is two factories that
mirror the three from Week 4-5:

```python
get_svm(kernel="rbf", C=1.0, gamma="scale", probability=False, random_state=42)
get_decision_tree(max_depth=None, criterion="gini", min_samples_leaf=1,
                  random_state=42)
```

Both return **unfitted** estimators, exactly like the preprocessor, the baseline
and the Week 5 models before them, so cross-validation can clone them and fit a
fresh copy inside every fold. Both **validate their arguments** and raise
`ValueError` immediately on nonsense — a non-positive `C`, a `gamma` that is
neither `"scale"`, `"auto"` nor a positive number, a `max_depth` below 1, an
unknown `criterion` — rather than letting the mistake surface as an obscure
failure deep inside `fit`.

The registry that the notebook loops over now holds five entries:

```python
CLASSICAL_MODEL_FACTORIES = {
    "logistic_regression": get_logistic_regression,
    "knn": get_knn,
    "naive_bayes": get_naive_bayes,
    "svm": get_svm,             # new this week
    "decision_tree": get_decision_tree,  # new this week
}
```

Adding a model is adding a line to this mapping, not rewriting the comparison
loop. That is the whole point of having built the protocol first.

### Both models still travel inside the pipeline

Each new model is wrapped with the Week 3 preprocessor, for reasons that differ
between the two:

```python
Pipeline([("preprocess", build_preprocessor()), ("model", get_svm())])
```

* the **SVM genuinely needs it**. Both `C` and `gamma` are expressed in terms of
  distances between rows, so on raw features a single wide-ranging column — `K`
  spans roughly 200 units, `ph` roughly 6 — would define the kernel almost by
  itself. This is the same argument that made scaling matter most for KNN.
* the **tree does not need it at all** (§4), but keeps it anyway, so that every
  model in the comparison receives byte-for-byte identical inputs. Fairness is
  worth one redundant transform.

---

## 2. Support vector machines — the widest gap

### The idea

Suppose two classes can be separated by a straight line. Then they can usually be
separated by *infinitely many* straight lines, and logistic regression's choice
among them falls out of its loss function rather than from any geometric
argument. An SVM makes the choice explicit and geometric: **put the boundary
where the gap to the nearest training row of either class is as wide as
possible.**

That empty corridor either side of the boundary is the **margin**, and training
maximises its width. The rows that touch the edges of the corridor — the ones
the boundary is pressed up against — are the **support vectors**. Everything
else could be deleted and the fitted model would be identical: an SVM is defined
by the hardest, most ambiguous rows, not by the comfortable bulk.

Two consequences follow, and both matter on a dataset this size:

* **Only the support vectors decide the answer.** Move a row further from the
  boundary anywhere on its own side and nothing changes. The model spends its
  attention on the genuinely difficult cases.
* **A wide margin is a form of caution.** A boundary jammed against the training
  points is easily nudged by one new point; a boundary sitting in the middle of
  a wide corridor is not. This is why SVMs tend to do well on small tabular
  datasets — they are, by construction, reluctant to commit to a boundary the
  data does not force.

### Soft margins and `C`

Real data is rarely perfectly separable, and insisting on a corridor with no
rows inside it would often be impossible or, worse, achievable only by a
grotesquely contorted boundary. So scikit-learn's SVM maximises a **soft**
margin: rows are allowed to sit inside the corridor, or even on the wrong side of
the boundary, at a price. `C` sets that price.

* **small `C`** — violations are cheap. The model tolerates many rows inside the
  margin in exchange for a wider, smoother corridor. More bias, less variance
  (both terms are defined properly in §5; for now read them as "too rigid" and
  "too impressionable").
* **large `C`** — violations are expensive. The boundary bends to get the
  training rows right, and the margin narrows around them. Less bias, more
  variance.

This is the same `C`, playing the same role, as logistic regression's inverse
regularisation dial in [Week 5 §2](../week05/learning_notes.md): larger means
*less* regularisation, not more.

The number of support vectors is a visible trace of where `C` has put you. When
the margin is wide it swallows many rows, so the support-vector count is high;
as `C` tightens the margin, that count falls. The notebook's sweep on this
dataset, at the default RBF kernel, shows both moving together:

| `C` | Support vectors | CV accuracy |
| --- | --- | --- |
| 0.01 | 1760 (all) | 0.8716 |
| 0.1 | 1608 | 0.9352 |
| 1 | 943 | 0.9790 |
| 10 | 640 | 0.9824 |
| 100 | 612 | 0.9818 |

At `C = 0.01` violations are so cheap that the margin swallows the **entire**
training set — all 1,760 rows are support vectors — and accuracy collapses to
87.2%: underfitting, from the soft side. As `C` grows the margin tightens, the
count drops towards 600, and accuracy settles around 98%. `C = 1.0`, the default,
uses **943 of 1,760 rows (53.6%)** as support vectors. That share is high mostly
because there are 22 classes: every pair of neighbouring crops has a boundary,
and every boundary needs its own supporting rows. With two well-separated classes
the share would be a few per cent.

None of these numbers is adopted. `C = 1.0` stays; choosing it from this table
would be picking a hyperparameter by looking at validation scores, which is what
a proper search does — under a protocol this week does not build (§7).

### `predict_proba` is off by default

An SVM produces a decision, not a probability. To make `predict_proba` work,
scikit-learn fits an **extra internal calibration** (Platt scaling: a small
logistic model over the SVM's scores, cross-validated inside the fit). That
multiplies the training cost and can occasionally disagree with `predict`, so
`get_svm` leaves `probability=False`. Ask for it only when you actually need
probabilities, and expect fitting to slow down when you do.

### Common mistakes

* **Reading `C` backwards.** Larger `C` means *less* regularisation and a
  tighter margin, the same as in logistic regression.
* **Skipping the scaler.** On raw features the widest-ranging column defines the
  kernel by itself, and the "SVM" is really a one-feature model.
* **Quoting `predict` confidence without `probability=True`.** A bare SVM has no
  calibrated probability to quote.
* **Expecting the RBF kernel to always beat the linear one.** On this data it
  does not — see §3.

---

## 3. Kernels — a curved boundary from similarity

### What a kernel is

A **kernel** is a function that measures the similarity of two rows. Its quiet
magic is that the similarity it returns is exactly the inner product those two
rows *would have* in some higher-dimensional space — a space you never build and
never visit. An SVM only ever needs inner products between rows, so it can fit a
perfectly flat boundary up in that high-dimensional space without computing a
single coordinate there. A flat boundary up there folds back down into a
**curved** boundary in the original seven features.

That shortcut — getting the geometry of a high-dimensional space through a
similarity function alone — is the **kernel trick**. In one sentence: **a kernel
lets an SVM draw a curved boundary by measuring similarity instead of
position.**

The two kernels this week uses (`get_svm` also accepts scikit-learn's `poly` and
`sigmoid`, but neither is fitted anywhere in the course):

* **`linear`** — plain inner product. The boundary stays flat, like logistic
  regression's, but positioned by the margin rather than by likelihood.
* **`rbf`** (radial basis function) — similarity that decays with distance, so
  each training row's influence is a bubble around itself. Because the bubbles
  can be added up to enclose regions, the boundary can wrap around a class rather
  than merely slicing past it. `gamma` sets how fast the bubbles decay: large
  `gamma` means small bubbles and a wiggly boundary that can encircle single
  points (overfitting); small `gamma` means large bubbles and a boundary that
  flattens towards a line (underfitting). `gamma="scale"`, the default, sets it
  from the feature variances so it is sensible without tuning.

### Linear is not beaten by RBF here

The natural expectation is that the more flexible RBF kernel should win. On this
dataset it does not:

```
SVM, rbf kernel, C = 1 : 0.9790  +/- 0.0103   (943 support vectors)
SVM, linear kernel, C = 1: 0.9818  +/- 0.0077   (615 support vectors)
```

The linear kernel scores *slightly higher*, and the gap is comfortably inside
the fold spread of either — by Week 4's rule, not a gap at all. This is
informative rather than disappointing. [Week 2](../week02/learning_notes.md)
showed the 22 crops occupying compact, well-separated blobs in feature space, and
separating compact blobs is a job a flat boundary already does well. A curved
boundary earns its keep when classes are interleaved, or when one encloses
another — and this dataset simply is not shaped like that. The linear kernel also
needs fewer support vectors (615 against 943) to describe its simpler boundary,
which is a small independent sign that the extra flexibility is not being used.

The moral repeats [Week 5 §4](../week05/learning_notes.md)'s: model complexity is
not a ranking. What matters is the fit between a model's assumptions and the
data's shape, and the only way to find out is to measure.

### Common mistakes

* **Believing the kernel trick "adds features".** It computes the *inner
  products* of a richer feature space without ever forming the features. That is
  the whole efficiency of it.
* **Turning `gamma` up because the score looks low.** Large `gamma` overfits;
  and adjusting it by watching validation scores is, again, a search without a
  protocol.
* **Concluding RBF is pointless because it lost here.** It lost *on compact
  blobs*. On interleaved or ring-shaped classes it would be the linear kernel
  that fails — as [Week 5 exercise I2](../week05/exercises.md) stages for
  logistic regression.

---

## 4. Decision trees — a chain of questions

### What it computes

A decision tree asks one kind of question, and only one: `feature <= threshold`.
Starting from all 1,760 rows at the root, it tries **every feature and every
candidate threshold**, scores the two groups each split would produce, keeps the
single best split, and then repeats the whole procedure independently inside each
child group. This is a **greedy** search: it takes the locally best split at
every node without reconsidering earlier ones, because searching all possible
trees at once is computationally hopeless.

"Best" is measured by how **pure** the resulting groups are, and purity is set by
the `criterion`:

* **Gini impurity** — the probability of mislabelling a randomly chosen row in a
  node if you guessed labels at the node's own class frequencies. It is 0 when a
  node holds a single class and largest when the classes are evenly mixed.
* **Entropy** — the number of bits needed to encode the node's labels. Same 0,
  same maximum, a slightly different curve between them.

The split chosen at a node is the one that reduces the weighted-average impurity
of its two children the most; that reduction is the **information gain**. None of
this is deep mathematics for our purposes, and — importantly — the two criteria
almost always pick the **same splits** on this data. Gini is the default only
because it avoids computing a logarithm.

### The rules are readable

The first two questions the fitted tree asks on the crop data are:

```
rainfall <= 30.18   ?
    then humidity <= 27.98 ?  ...
```

"Is rainfall under about 30 mm? If so, is humidity under about 28%?" is a
sentence a farmer could read and check. No other model in this project produces
one: logistic regression hands you 176 numbers (154 weights and 22 intercepts),
KNN hands you the entire training set, naive Bayes hands you 308 means and
variances. A tree hands you a flowchart.
That legibility is why **Week 7 returns to trees for explainability**, and why
`humidity` and `rainfall` — the tree's own first two splits — are the two
features used for the boundary pictures in §6.

### Trees are invariant to feature scaling

A split at `rainfall <= 30.18` selects exactly the same rows however the
`rainfall` column is centred, divided or otherwise linearly rescaled — the order
of the values, which is all a threshold cares about, does not move. So a decision
tree is the one model in the comparison that is **completely indifferent to the
preprocessor**. It is kept behind it anyway, purely so that every model receives
identical inputs (§0); the scaling has no effect on what the tree learns.

### The unlimited tree memorises

Nothing in the greedy procedure knows when to stop. Left alone
(`max_depth=None`), the tree keeps splitting until every leaf is pure — in the
limit, one leaf per training row. On this data the unlimited tree grows to
**depth 17 with 38 leaves** (the two numbers the notebook prints — and since a
binary tree with 38 leaves has 37 internal splits, 75 nodes in all), and scores
a **perfect 100% on its own training data**.

That 100% is not a result. It is the *definition* of the algorithm's stopping
rule: it splits until it cannot be wrong on the rows it was shown. Its
cross-validated score — measured on rows it never saw — is 98.52%, which is the
number that means something. The gap between those two is the subject of §5, and
the clearest example of overfitting the course offers.

### Common mistakes

* **Scaling "to be safe" and expecting it to change the tree.** It cannot; a
  monotone rescale leaves every threshold's selection identical.
* **Quoting the training accuracy.** For an unlimited tree it is always close to
  or exactly 100% and always uninformative.
* **Assuming Gini and entropy give materially different trees.** Here they pick
  essentially the same splits; the choice is a tie-break, not a strategy.
* **Treating one deep tree as a strong model.** Its variance is high (§5); the
  device that fixes that — a forest of them — is a later week.

---

## 5. Depth, overfitting, and the bias-variance tradeoff

This is the picture the whole week is built around. For a range of `max_depth`
values, fit the tree on all 1,760 training rows and record two numbers side by
side:

* **training accuracy** — measured on the rows it was fitted on;
* **validation accuracy** — the mean of the five stratified cross-validation
  folds, each scored on rows that fold's model never saw.

The test set is not involved. "Test" in the phrase *train-versus-test curve*
means "data held out from fitting", and cross-validation supplies exactly that
while `data/processed/test.csv` stays closed until Week 8.

Here is the sweep on this dataset:

| `max_depth` | Leaves | Train accuracy | Validation accuracy |
| --- | --- | --- | --- |
| 1 | 2 | 0.0909 | 0.0909 |
| 2 | 3 | 0.1364 | 0.1364 |
| 3 | 5 | 0.2273 | 0.2261 |
| 4 | 7 | 0.3182 | 0.3170 |
| 5 | 9 | 0.4091 | 0.4074 |
| 6 | 13 | 0.5830 | 0.5636 |
| 7 | 17 | 0.7608 | 0.7381 |
| 8 | 21 | 0.8949 | 0.8716 |
| 9 | 25 | 0.9443 | 0.9477 |
| 10 | 28 | 0.9818 | 0.9750 |
| 12 | 32 | 0.9943 | 0.9847 |
| 15 | 36 | 0.9983 | 0.9852 |
| None (17) | 38 | 1.0000 | 0.9852 |

Read it in three stretches.

**Shallow trees — high bias (underfitting).** A depth-1 tree asks a single
question and can therefore name at most two crops out of 22, so it scores 9.09%
on *both* curves. Depth 2 gives 13.64%, depth 5 gives 40.91%. Through this whole
region the two lines sit exactly on top of one another and both are dreadful.
That is the signature of underfitting: the model is not complex enough to express
the answer, so it does equally badly on rows it has seen and rows it has not —
and, crucially, more training data would not help. The problem is the model, not
the sample.

**The middle — the model is learning.** Between depths 6 and 10 both curves climb
steeply together, from 58% to 98%. Each extra question buys real structure. Small
gaps flicker in and out here — about 2 points at depths 7 and 8 — but they come
and go with the folds rather than growing, and at depth 9 the validation score
(94.77%) is actually *above* the training score (94.43%), which a stable gap
never does. These are sampling wobble, not yet memorisation.

**Deep trees — high variance (overfitting).** From depth 12 the validation curve
essentially stops moving: 98.47%, then 98.52%, then 98.52% again for the
unlimited tree. The training curve, however, keeps climbing — 99.43%, 99.83%, and
finally a perfect 100% at depth 17, where all 38 leaves are pure. Every point of
that last stretch is bought by memorising this particular sample: a persistent
gap of roughly **1.5 points** that is training-set accuracy with no counterpart
on new data. Only the held-out curve can reveal it, which is precisely why a
training score is never quoted as a result.

### The tradeoff, named

The reason *both* ends of the plot are bad has a name:

* **Bias** — error from the model being too simple to represent the truth. A
  shallow tree has high bias: whatever data you feed it, its answer is roughly
  the same and roughly wrong, on training and validation rows alike.
* **Variance** — error from the model being so flexible that it changes a great
  deal when the training sample changes. A deep tree has high variance: fit it on
  a different 1,760 rows and you get a visibly different tree, with different
  thresholds and different leaves.

Total error is, loosely, bias plus variance (plus irreducible noise), and *every*
"how flexible" dial the course has met — `max_depth` and `min_samples_leaf`
here, `k` in KNN, `C` and `gamma` in the SVM, `C` in logistic regression — moves
a model along this same curve. In one sentence: **a shallow tree is wrong in the
same way every time; a deep tree is wrong in a different way every time.**

### An honest caveat about this dataset

The textbook train-versus-test curve turns *downwards* past the best depth: the
validation score peaks, then the model starts fitting noise and generalisation
gets actively worse. That is not quite what happens here. This dataset's
overfitting penalty is unusually **mild** — 100% training against 98.52%
validation — because the crops are almost separable and there is very little
noise for the tree to memorise. So the validation curve **flattens** instead of
falling. On messier, more realistic data the curve would visibly turn down past
the best depth, and the gap would be tens of points rather than one and a half.
The mechanism is identical; this dataset just makes it gentle, and it would be
dishonest to draw a dramatic downturn that the numbers do not show.

### Common mistakes

* **Picking the depth with the best validation score off this table.** That is
  choosing a hyperparameter by eye from validation numbers — a search without a
  protocol (§7).
* **Expecting a downturn and inventing one.** Report the flattening you see.
* **Treating the middle-region wobble as overfitting.** A gap that appears and
  vanishes with the folds, and that can go negative, is noise, not memorisation.

---

## 6. Seeing the boundary

"Decision boundary" has been a phrase since [Week 5
§2](../week05/learning_notes.md). This week it becomes a picture, drawn by
`plot_decision_boundary(model, X_2d, y)` in `src/utils/visualization.py`.

### The method is brute force

There is no mathematics in the drawing. Take two features, cover that plane with
a fine grid (200 points per axis by default), ask the fitted model to classify
**every** grid point, and colour each point by the answer. The blocks of colour
are the regions the model assigns to each class; the seams between colours are
the boundary. The helper then scatters the real rows on top so you can see which
ones land in the right region.

That brute-force grid is why the boundary can be *shown* even when it has no
closed-form equation, and it is also why the method scales badly — a 200x200 grid
is 40,000 predictions, and a finer grid or a third feature multiplies that fast.
For a picture, that cost is fine.

### It is an illustration, never a score

The plots below follow one honesty condition: each model is **fitted on exactly
the two columns being plotted**. So every panel shows a genuine *two-feature*
model — not a slice through the seven-feature model whose accuracy the results
table reports. (The helper does not — and cannot — check how a model was
fitted; break the rule by passing a seven-feature model and the mismatch
surfaces as scikit-learn's own error about the wrong number of features, which
is exactly what exercise I2 stages.) The accuracies you
could compute from these plots are lower than the table's for the obvious reason
that five features are missing. **No number in a boundary plot is ever a
reported result.** `humidity` and `rainfall` are chosen because the unlimited
tree splits on them first, so they are the two the data itself calls most
separating.

### Three algorithms, three shapes

Drawn on the same two features, the three model families give three
unmistakable shapes:

* **Logistic regression** cuts the plane with straight lines only. Every region
  is a polygon, because every boundary is `w . x + b = 0`.
* **The RBF SVM** encloses regions with smooth curves. The kernel is what buys
  the curvature, and `gamma` sets how tight it is.
* **The decision tree** produces rectangles with horizontal and vertical edges
  only, because every question it can ask is about *one* feature at a time. A
  diagonal boundary can only be approximated by a staircase of axis-aligned
  steps — which is exactly what a deep tree spends its depth doing, and one more
  reason deep trees overfit.

The stray points that fall in the "wrong" colour are the crops these two
features alone cannot separate; the full seven-feature models in the table below
resolve almost all of them.

The helper also validates its inputs: it rejects an `X_2d` that does not have
exactly two columns, a `y`
whose length does not match `X_2d`, a `resolution` below 2, and a negative
`padding`, each with a clear `ValueError` rather than a confusing plot.

---

## 7. The extended results table

Same rows, same five folds, same seed, same metric, same preprocessor — the two
new models are simply appended to the five Week 5 candidates. That is what makes
this a running record rather than two unrelated experiments.

5-fold stratified cross-validation, seed 42, on the 1,760 training rows, each
model behind the Week 3 preprocessor:

| Model | CV accuracy | Std across folds | Error rate | Added |
| --- | --- | --- | --- | --- |
| **Gaussian naive Bayes** | **0.9949** | 0.0042 | 0.0051 | Week 5 |
| Decision tree (unlimited) | 0.9852 | 0.0068 | 0.0148 | Week 6 |
| SVM (rbf, `C = 1`) | 0.9790 | 0.0103 | 0.0210 | Week 6 |
| Logistic regression | 0.9682 | 0.0066 | 0.0318 | Week 5 |
| KNN (`k = 5`) | 0.9653 | 0.0121 | 0.0347 | Week 5 |
| `most_frequent` baseline | 0.0455 | 0.0000 | 0.9545 | Week 4 |

How to read it:

* **Both new models beat the two Week 5 runners-up.** The tree reaches 98.52%
  and the SVM 97.90%, against logistic regression's 96.82% and KNN's 96.53%.
* **Naive Bayes still leads, at 99.49%.** Two more sophisticated algorithms did
  not displace the simplest one, and the reason is unchanged from [Week 5
  §4](../week05/learning_notes.md): the crops sit in compact, well-separated,
  roughly bell-shaped blobs, and "one Gaussian per crop per feature" describes
  exactly that shape.
* **The tree's 98.52% comes from a model that scores 100% on its training
  data.** Both facts are true at once, and only the second is a result. This is
  the concrete reason to distrust training scores.
* **Error rates make the ranking legible.** 0.51%, 1.48%, 2.10%, 3.18%, 3.47% —
  a nearly seven-fold spread in mistakes that the accuracy column compresses into
  three percentage points. On the 352 rows each cross-validation fold holds out,
  that is roughly 2 errors versus 12.
* **Some gaps are still inside the noise.** The SVM and the tree are 0.62 points
  apart with fold standard deviations of about 1.0 and 0.7; by [Week 4's
  rule](../week04/learning_notes.md) — a gap smaller than the fold-to-fold
  spread is not yet a gap — that contest is not decided. Naive Bayes' lead over
  the tree (0.97 points against spreads of 0.42 and 0.68) is real by the same
  standard.

The best-performing model so far is still **Gaussian naive Bayes at 99.49%**.

---

## 8. The code this week

### `src/models/classical_models.py`

Two factories join the three from Week 5:

```python
get_svm(kernel="rbf", C=1.0, gamma="scale", probability=False,
        random_state=42)                                       -> SVC
get_decision_tree(max_depth=None, criterion="gini",
                  min_samples_leaf=1, random_state=42)         -> DecisionTreeClassifier
```

Both follow the module's established conventions: they return unfitted
estimators, they validate every argument and raise `ValueError` on nonsense
before `fit` is ever called, and they are registered in
`CLASSICAL_MODEL_FACTORIES` (now five entries) so the notebook loops over them
instead of repeating itself. Their docstrings carry the conceptual explanations
of margins, kernels, purity and depth in full, so the module is readable on its
own.

### `src/utils/visualization.py`

`plot_decision_boundary(model, X_2d, y, ax=None, resolution=200, padding=0.05,
title=None)` draws the brute-force grid described in §6. It requires a model
already fitted on exactly the two plotted columns and raises `ValueError` for the
wrong column count, a mismatched `y`, `resolution < 2`, or negative `padding`. It
returns the axes it drew on, so panels can be composed into a figure.

### `tests/test_classical_models.py`

The suite grows to **119 tests** for the model factories, and the whole project
suite to **231 passing**. They check the same contracts as before — types,
defaults, argument pass-through, rejection of bad arguments, the unfitted
promise, `fit`/`predict` inside a pipeline — now extended to the SVM and the
tree, plus algorithm-specific facts: that an unlimited tree memorises its
training data, that a tree ignores feature scaling, that the SVM sits behind the
preprocessor, and that both new models beat the baseline on the real rows.

### `notebooks/05_classification_models.ipynb`, Part 2

Part 2 (§8-§15 of the notebook) sets up the two new factories, explains and fits
the SVM with its `C` and kernel sweeps, explains and fits the tree, draws the
depth train-versus-validation curve in two panels, draws the three decision
boundaries on `humidity` and `rainfall`, extends the results table to six rows,
and — as in Part 1 — closes with guard-rail assertions so the prose and the
numbers cannot drift apart silently.

---

## 9. What this week produced, and what it deliberately did not

**Produced**

* Two model factories — `get_svm` and `get_decision_tree` — returning unfitted
  estimators like every factory before them, registered in
  `CLASSICAL_MODEL_FACTORIES`.
* `plot_decision_boundary`, and with it the difference between a *linear*,
  *kernelled* and *tree* boundary shown rather than asserted.
* The train-versus-validation curve over `max_depth`: overfitting and
  underfitting drawn from this dataset's own numbers, with the bias-variance
  tradeoff attached to a concrete dial.
* A six-row results table extending Week 5's, with the tree at 98.52% and the SVM
  at 97.90%.
* The current answer to "which model is best so far": **Gaussian naive Bayes,
  99.49%.**

**Not produced, on purpose**

* **No ensemble.** No random forest, no gradient boosting, no voting or stacking.
  A random forest is a crowd of deep trees whose high variance largely cancels —
  the natural next step from §5, and deliberately not taken here. That is **a
  later week** (its position in the course is not yet fixed).
* **No systematic hyperparameter search.** The `C` sweep in §2 and the depth
  sweep in §5 draw curves to explain a mechanism. Nothing from either is adopted:
  `C = 1.0` and `max_depth=None` remain the defaults, because choosing them from
  those tables would be selecting a hyperparameter by eye from validation scores,
  with no search space, no protocol and no nested split to keep the choice
  honest. Grid and random search are also **a later week**.
* **No claim from the two-feature boundary plots.** They are illustrations; every
  quoted score comes from the seven-feature cross-validated table.
* **No statement about which features drive a prediction** — **Week 7**, which
  starts from the tree's readable rules.
* **No precision, recall, F1 or confusion matrix** — **Week 8**.
* **No test-set score.** `data/processed/test.csv` remains unopened.
