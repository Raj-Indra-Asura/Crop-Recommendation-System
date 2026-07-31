# Week 2 — Learning Notes

> 🗺 [Roadmap](../README.md) › [Part I — Foundations (Weeks 1-3)](../README.md#part-i--foundations-weeks-1-3) › [Chapter 2 — Exploratory Data Analysis](README.md) › **§2.2 Learning notes**

> How to read these notes: every new idea is introduced with four questions —
> *what is it, why do we need it, where is it used, how does it actually work* —
> followed by the mistakes people commonly make with it.
>
> These notes build directly on Week 1 and do not repeat it. Dataframes,
> `load_data()`, the seven features, the dataset contract and the `ruff`/`pytest`
> guard rails are all explained in
> [`week01/learning_notes.md`](../week01/learning_notes.md); follow the links
> rather than expecting a recap.

---

## 0. What exploratory data analysis is, and why it comes before modelling

**Exploratory data analysis** (EDA) is the stage where you look at the data
itself — its statistics and its shapes — before deciding how to prepare it or
what to train on it.

Week 1 verified that the file *is what we think it is*: right columns, right
row count, right 22 crop names, nothing missing. That is a check on structure.
It says nothing about the values. A file can satisfy every structural rule and
still contain humidity readings of 400%, a crop whose rows are all identical, or
two features that are the same number twice under different names.

Why bother, when a model could be trained today?

1. **Every later decision depends on it.** Whether features need scaling (Week
   3), whether accuracy is a fair metric (Week 4), which model family is
   sensible (Week 5), whether a feature-importance chart is interpretable (Week
   7) — each of those is answered by something visible in this week's plots.
2. **A model will not tell you the data is odd.** It will fit whatever it is
   given and report a number. Nothing in a training run says "two of your crops
   are separated by a single feature, so this problem is easier than you think".
3. **It is the last moment you may look at everything.** From Week 3 a portion
   of the data is held back and must stay unseen. Understanding gathered now is
   free; the same statistics computed after the split are a bug (§6).

The whole of this week's exploration is in
[`notebooks/02_EDA.ipynb`](../../../notebooks/02_EDA.ipynb), which calls the
helpers in `src/utils/eda.py`. As in Week 1, logic lives in `src/` so it can be
linted and tested; the notebook is the narrative around it.

### Common mistakes

* **Skipping EDA because the data "looks clean".** Clean structure and sensible
  values are different properties. Week 1 checked the first.
* **Exploring forever.** EDA has a stopping condition: you stop when you can
  state, in a few sentences, what the data contains and what that implies for
  the next stage. Section 8's conclusions are what "done" looks like.

---

## 1. Descriptive statistics

### What is it?

A **descriptive statistic** compresses a column of numbers into a single number
that describes some property of it. `DataFrame.describe()` computes eight of
them at once; this project wraps it in `describe_features()`, which transposes
the result (one row per feature reads better when there are seven) and adds
`median` and `skew`.

The ones that matter, in three groups:

**Centre — where do the values sit?**

* **Mean** — the arithmetic average. Sensitive to extreme values: one reading of
  10,000 in a column of tens will drag it noticeably.
* **Median** — the middle value when sorted; half the rows are below it. Not
  sensitive to extremes at all.

**Spread — how far apart are they?**

* **Standard deviation** (`std`) — roughly the typical distance of a value from
  the mean, expressed in the column's own units. Its square is the **variance**;
  variance is what the mathematics uses, standard deviation is what humans read,
  because it is in the original units.
* **Min** and **max** — the extremes, and the cheapest sanity check available. A
  humidity of 120% or a negative rainfall would be visible here instantly.

**Shape — how are they arranged?**

* **Quartiles** — the 25th, 50th and 75th **percentiles**. The *n*th percentile
  is the value below which *n*% of the rows fall, so the 50th percentile is the
  median. The gap between the 25th and 75th, `Q3 - Q1`, is the **interquartile
  range (IQR)**: the width of the middle half of the data. It returns in §4.
* **Skewness** — how lopsided the column is. Zero means symmetric; positive
  means a long right tail (a few large values); negative means a long left tail.

### Why do we need it?

Two decisions come straight out of this table on our data.

**Feature scale.** The Week 2 notebook produces:

| feature | mean | std | min | max |
| --- | --- | --- | --- | --- |
| `N` | 50.55 | 36.92 | 0.00 | 140.00 |
| `P` | 53.36 | 32.99 | 5.00 | 145.00 |
| `K` | 48.15 | 50.65 | 5.00 | 205.00 |
| `temperature` | 25.62 | 5.06 | 8.83 | 43.68 |
| `humidity` | 71.48 | 22.26 | 14.26 | 99.98 |
| `ph` | 6.47 | 0.77 | 3.50 | 9.94 |
| `rainfall` | 103.46 | 54.96 | 20.21 | 298.56 |

`K` ranges over 200 units; `ph` ranges over 6.4. Any algorithm that measures
distance between rows — k-nearest neighbours, support vector machines, anything
using gradient descent — implicitly treats "1 unit of `K`" and "1 unit of `ph`"
as equally important, so `K` would dominate purely because its numbers are
bigger. That is the argument for **feature scaling**, which is Week 3's job.
This table is the evidence for it.

**Where mean and median disagree.** For `K` the mean is 48.15 and the median is
32.00 — a large gap, matching a skew of +2.38. When those two numbers separate,
the mean is being pulled by a tail, and quoting the mean alone would misdescribe
a typical row. `humidity` shows the mirror image: skew -1.09, most rows high,
a thin tail reaching down to 14%.

### Where is it used in industry?

It is the first cell of essentially every data notebook ever written, and the
first thing a data-quality monitor computes in production: if next month's
inbound `mean` and `std` differ sharply from the training set's, the input
distribution has drifted and the model's predictions are no longer trustworthy.
That is Week 12's monitoring, in embryo.

### How does it actually work?

```python
from src.utils.eda import describe_features
describe_features(crops, FEATURES).round(2)
```

`describe_features()` calls `frame[columns].describe().T`, then appends
`median()` and `skew()` as extra columns. It never modifies the frame it is
given — a rule every helper in `src/utils/eda.py` follows, and one that
`tests/test_eda.py` asserts.

### Common mistakes

* **Quoting the mean of a skewed column as "typical".** For `K`, "the typical
  value is 48" is wrong; more than half the rows are below 32.
* **Confusing standard deviation with error.** It describes the spread of the
  data, not the uncertainty of a measurement.
* **Treating a wide range as a defect.** `K` spanning 5–205 is not a problem to
  fix; it is a fact to accommodate when scaling.

---

## 2. Class balance

### What is it?

**Class balance** is how many rows each class has. `value_counts()` on the label
column answers it in one line; `class_balance()` wraps that and adds the
proportion of the dataset each class occupies.

```python
crops["label"].value_counts()
```

On our data every one of the 22 crops has **exactly 100 rows**, 4.55% each. The
dataset is perfectly balanced, and the bar chart from `plot_class_balance()` is
a row of identical bars.

A dataset is **imbalanced** when the counts differ substantially — fraud
detection, where 0.1% of transactions are fraudulent, is the standard example.

### Why do we need it?

Balance changes two things that arrive later, which is why it is worth measuring
before either of them.

**It changes what a model learns.** Learning algorithms minimise total error
across the training set. If 95% of rows are one class, a model that always
predicts that class is already 95% correct, and there is little pressure to
learn anything about the remaining 5%. The model is not broken; it is optimising
what it was asked to optimise.

**It changes which metric is honest.** In that same 95/5 dataset, "95% accuracy"
describes a model that has learned nothing. This is why Week 8 introduces
precision, recall and the confusion matrix — metrics that report per-class
behaviour instead of one pooled number. **Foreshadowing Week 8:** accuracy is
only a fair summary when classes are balanced, which is exactly the condition
this section verifies. Because our dataset *is* balanced, Week 4 may use
accuracy for its baseline without apology — but that permission comes from this
measurement, not from habit.

There is a third consequence for Week 3: when a balanced dataset is split
randomly, an unlucky split can still leave a class thin in the test set.
**Stratified** splitting preserves the class proportions on both sides. That is
Week 3's mechanism; this section is why it exists.

### Where is it used in industry?

Everywhere a rare event is being predicted: fraud, machine failure, disease
screening, churn. In those settings the class balance is often the single most
important fact about the dataset, and it drives resampling, class weighting and
the choice of metric.

### How does it actually work?

```python
from src.utils.eda import class_balance, plot_class_balance
class_balance(crops, "label")      # counts + proportions
plot_class_balance(crops, "label") # the same thing as a bar chart
```

### A caution about *this* dataset

Exactly 100 rows per crop and no missing values is not what field data looks
like. Real records are dominated by
whatever the collecting region grows. This dataset is curated, and probably
partly synthetic (Week 1, exercise C4, asked you to form this hypothesis; this
is the evidence). Two consequences:

1. Results in this course are about *this file*, not about world agriculture.
2. Any claim that a model "handles rare crops well" is unsupported here, because
   there are no rare crops to handle.

### Common mistakes

* **Assuming balance and forgetting to check.** It costs one line.
* **Reading balance as a quality signal.** Perfect balance is a sign of
  curation, not of quality.
* **"Fixing" imbalance reflexively.** Resampling changes the base rates the
  model sees, which changes what its probability outputs mean. Not our problem
  this week — but do not learn the reflex.

---

## 3. Distributions and skew

### What is it?

A **distribution** is the pattern of which values a feature takes and how often.
A **histogram** shows it: the range is sliced into **bins** of equal width, and
the height of each bar is the number of rows falling in that bin.

Three shapes to be able to name on sight:

* **Symmetric** — the two sides mirror each other; mean ≈ median; skew ≈ 0.
* **Right-skewed** (positive skew) — most values low, a long tail of high ones;
  mean > median.
* **Left-skewed** (negative skew) — the mirror image; mean < median.

A fourth thing to notice is **multimodality**: more than one peak. It normally
means two different populations have been mixed into one column.

### Why do we need it?

The summary table of §1 can hide the shape entirely. Two columns with identical
mean and standard deviation can look completely different: one a single hump,
the other two humps with a gap between them. Only the picture distinguishes
them, and the difference matters — a gap between humps is a boundary a
classifier can exploit.

Skew specifically implies:

* **Mean-based summaries mislead**, as in §1.
* **Some models are sensitive to it.** Linear and distance-based models assume,
  loosely, that a unit of change means the same thing across a feature's range;
  a long tail breaks that. Transformations such as a log can help — Week 3's
  territory. Tree-based models split on order, not magnitude, and are largely
  indifferent (Week 5).
* **Extreme values in the tail will be flagged as outliers** by the rule in §4,
  which is a fact about the rule, not about the data.

### How does it actually work?

```python
from src.utils.eda import plot_feature_histograms
plot_feature_histograms(crops, FEATURES, bins=30)
```

**Bin count matters.** Too few bins and separate humps merge into one; too many
and every bar becomes noise. 30 is a reasonable default for 2,200 rows;
exercise I1 asks you to change it and watch the story change.

### What our data shows

* `K` (skew +2.38) and `P` (+1.01) are strongly right-skewed, each with a
  *detached* cluster far to the right — a gap, then a second group. Sections 4
  and 5 identify that group as apple and grapes.
* `humidity` is left-skewed (-1.09): a dense mass above 80%, a thin tail down
  toward 15%.
* `temperature` (+0.18) and `ph` (+0.28) are close to symmetric, centred near
  25 °C and pH 6.5 — agronomically plausible.

The crucial interpretive point: **these are histograms of all 22 crops pooled
together.** In a labelled dataset, a multi-peaked pooled histogram is expected —
each crop contributes its own hump. So the pooled skew of `K` is not a defect of
the `K` measurement; it is class structure showing through. The question worth
asking is not "is this column skewed" but "do the per-crop humps sit in
different places", which needs the grouped plots of §4.

### Common mistakes

* **Trusting one bin count.** Change it and see whether the story survives.
* **Transforming away a skew that is class structure.** Log-transforming `K`
  here would compress exactly the gap that makes apple and grapes recognisable.
* **Confusing a histogram with a probability distribution.** A histogram is a
  count of the sample you happen to have.

---

## 4. Correlation between features

### What is it?

**Correlation** measures how strongly two numeric features move together, on a
scale from -1 to +1:

* **+1** — perfectly proportional: when one rises, the other rises in step.
* **0** — no linear relationship.
* **-1** — perfectly opposed.

The default, **Pearson** correlation, measures *linear* association only. This
is the single most important caveat in this section: a Pearson correlation of 0
means "no straight-line relationship", **not** "no relationship". A perfect
U-shaped relationship scores approximately 0 while being entirely predictable.
(`plot_correlation_heatmap(..., method="spearman")` switches to a rank-based
measure, which catches any monotonic relationship, straight or not.)

A **correlation heatmap** is the whole feature-by-feature matrix drawn as a
colour grid, so a strong pair is visible instantly instead of being hunted for
in a table of 49 numbers.

### Why do we need it?

**Strongly correlated features matter differently to different models** — which
is exactly why this is worth knowing before choosing one in Week 5:

* **Linear models** (logistic regression) suffer from **multicollinearity**.
  When two inputs carry nearly the same information, many different coefficient
  combinations fit the data about equally well, so the fitted coefficients
  become unstable — large, opposite in sign, and swinging wildly on small data
  changes. The predictions can still be fine; the *explanation* becomes
  worthless.
* **Distance-based models** (k-NN) effectively double-count the shared
  information, silently weighting that aspect of the data twice.
* **Tree-based models** (decision trees, random forests) are largely untroubled:
  a tree simply picks one of the two correlated features at each split. This
  robustness is one reason they are a strong default on tabular data (Week 5).
  The cost surfaces in Week 7 instead: importance gets split arbitrarily between
  the correlated pair, so neither looks as important as it is.

### What our data shows, and why it is interesting

Every pair sits between -0.23 and +0.21 — essentially unrelated — **except**
`P` and `K` at **+0.74**. The notebook then asks why:

```python
without_high_k = crops[~crops["label"].isin(["apple", "grapes"])]
crops["P"].corr(crops["K"])                      # 0.736
without_high_k["P"].corr(without_high_k["K"])    # 0.043
```

The correlation is almost entirely produced by **two of the 22 crops**. Apple
and grapes both need high phosphorus *and* high potassium; drop those 200 rows
and the association collapses from 0.74 to 0.04.

Two lessons follow.

1. **A population-level correlation can be an artifact of class structure.**
   Within nearly every crop, `P` and `K` are unrelated. Pooling classes created
   the appearance of a relationship that does not exist inside any of them. (The
   general form of this is Simpson's paradox: an association at the aggregate
   level that reverses or vanishes within groups.)
2. **Correlated does not mean redundant.** `P` and `K` are not two names for one
   thing; their *combination* is precisely what identifies apple and grapes.
   Dropping either would throw away the signal. Neither is dropped.

And the older caution still applies: correlation is not causation. Nothing here
shows that potassium *causes* grapes to thrive — only that in this dataset the
two co-occur.

### How does it actually work?

```python
from src.utils.eda import plot_correlation_heatmap
plot_correlation_heatmap(crops, FEATURES)   # annotated, fixed -1..+1 colour scale
crops[FEATURES].corr().round(2)             # the same numbers as a table
```

The colour scale is pinned to -1..+1 rather than to the data's own range, so
that a matrix of weak correlations looks weak instead of being auto-scaled into
dramatic colours.

### Common mistakes

* **Reading a heatmap without its scale.** An auto-scaled heatmap makes 0.2 look
  like 0.9.
* **Dropping one of every correlated pair by reflex.** Ask what produced the
  correlation first; here the answer changes the decision.
* **Correlating the label.** `label` is text and has no correlation with
  anything. The feature-to-class relationship needs a different statistic —
  §5's eta-squared, or Week 7's model-based importances.

---

## 5. Outliers, boxplots, and how features separate the crops

### What is a boxplot?

A **boxplot** summarises a distribution with five numbers plus flagged points:

* the **box** spans `Q1` to `Q3` — the middle 50% of values, i.e. the IQR;
* the **line inside the box** is the median;
* the **whiskers** extend to the furthest value within `1.5 × IQR` of the box;
* everything beyond the whiskers is drawn as an **individual point**.

Those individual points are what people mean by **outliers** in a boxplot. Note
what that definition is: the output of an arithmetic rule about quartiles.
Nothing in it knows whether a value is a typing error, a broken sensor, or a
completely legitimate rare case.

### Why look at outliers at all?

Because they have three completely different causes, requiring three different
responses:

1. **Data errors** — a humidity of 400%, a negative rainfall, a decimal point in
   the wrong place. These should be corrected or removed.
2. **Legitimate rare cases** — a genuinely unusual field. These must be kept:
   they are the hardest cases and the ones a real user is most likely to be
   confused by.
3. **Class structure** — a value that is extreme *for the pooled dataset* but
   entirely ordinary *for its own class*. Neither error nor rare: an artifact of
   pooling.

You cannot tell which you have without looking, and you cannot tell by looking
at pooled data alone. So the notebook does both: it counts flagged values across
the whole dataset with `count_outliers_iqr()`, then draws one boxplot **per
feature, grouped by crop**.

### What our data shows

Pooled, the IQR rule flags: `K` 200 values (9.09%), `P` 138 (6.27%), `rainfall`
100 (4.55%), `temperature` 86, `ph` 57, `humidity` 30, `N` none. Then:

```
K above the upper whisker (92.5)  -> {'grapes': 100, 'apple': 100}
humidity below the lower whisker  -> {'chickpea': 30}
rainfall above the upper whisker  -> {'rice': 68, 'papaya': 17, 'coconut': 15}
```

Every flagged group resolves to a crop. All 200 high-`K` rows are apple and
grapes — their *entire* populations, 100 each. All 30 very-dry rows are
chickpea. This is cause (3) above, unambiguously: **class structure, not
corruption**. Removing "the outliers" would delete two crops' worth of signal
and make apple and grapes impossible to learn.

### The rule this week follows

> Look, count, explain — remove nothing.

Two reasons. First, as just shown, the flagged values here are legitimate.
Second, and more generally: **removal is a preprocessing decision, and
preprocessing decisions must be made after the split, using training data only**
— see §6. Deciding today which rows to delete, using bounds computed from all
2,200 rows, would be leakage before the split has even happened.

### Which features separate the crops?

Read side by side, the grouped boxplots answer the question that actually
matters for classification: does this feature *separate* the crops, or do all 22
boxes sit on top of one another? `separation_scores()` turns that visual
impression into a number — **eta-squared**, the share of a feature's total
variance that lies *between* crops rather than within them (0 = the crops are
indistinguishable on this feature; 1 = perfectly separated by it alone).

| feature | eta-squared |
| --- | --- |
| `K` | 0.996 |
| `humidity` | 0.968 |
| `P` | 0.948 |
| `N` | 0.896 |
| `rainfall` | 0.854 |
| `temperature` | 0.496 |
| `ph` | 0.368 |

Three concrete separations, visible in the boxplots and confirmed by the
per-crop means:

* **`K` isolates apple and grapes.** Both sit at 195–205; no other crop exceeds
  85. A single threshold at `K > 90` picks out those two crops and nothing else.
* **`rainfall` separates rice from the dry crops.** Rice averages 236 mm and
  never drops below 182.6 mm; muskmelon averages 24.7 mm and never exceeds
  29.9 mm. Over 150 mm of gap, with nothing in it.
* **`humidity` isolates chickpea.** Chickpea spans 14.3–20.0% while most crops
  sit above 80%.

**This is not feature selection.** A low eta-squared says a feature is weak *on
its own*; it says nothing about its value in combination with others. `ph`
scoring 0.368 is a reason to expect it to rank low in Week 7's importance
analysis — not a reason to drop it now. Nothing is dropped this week.

### How does it actually work?

```python
from src.utils.eda import count_outliers_iqr, plot_boxplot_by_label, separation_scores

count_outliers_iqr(crops, FEATURES)              # bounds + counts per feature
plot_boxplot_by_label(crops, "rainfall", "label")  # one box per crop
separation_scores(crops, FEATURES, "label")      # eta-squared, ranked
```

### Common mistakes

* **Deleting flagged points automatically.** The commonest destructive habit in
  applied ML. "Outlier" is a description, not a diagnosis.
* **Looking only at pooled boxplots.** Grouping by the label is what turns
  "200 weird `K` values" into "apple and grapes".
* **Reading a single-feature separation score as importance.** Features work in
  combination; that is the entire point of using a model rather than a
  threshold.

---

## 6. Data leakage — introduced now, enforced from Week 3

### What is it?

**Data leakage** is when information that would not be available at prediction
time influences the training process, making the model's measured performance
better than its real performance.

One sentence, worth memorising:

> Data leakage is when a model is trained using information it would not have
> when making a real prediction — so its test score flatters it, and production
> disappoints.

### Why introduce it now, before any split exists?

Because leakage is caused by habits, and the habit is formed here. Look at what
this week's notebook computed:

* `mean` and `std` for every feature — over all 2,200 rows;
* quartiles and IQR outlier bounds — over all 2,200 rows;
* the correlation matrix and eta-squared scores — over all 2,200 rows.

Each of those is a statistic **of the whole dataset**, including the rows that
Week 3 will hold back as a test set. Right now that is entirely legitimate,
because those numbers are being *read by a human in order to understand the
problem*. They are not stored, not passed to a model, and not used to transform
anything. Understanding is the one thing that safely crosses a split.

The danger is continuing the same habit one week later. The rule, from Week 3
onward:

> **Fit any preprocessing on the training set only, then apply the fitted
> transformation to the test set. Never fit on the full dataset.**

### How does leakage actually happen? Three concrete forms

1. **Scaling before splitting.** You compute a feature's mean and standard
   deviation over all 2,200 rows, standardise, then split. The scaler's
   parameters now encode information about the test rows, so the training
   process has seen a summary of data it was supposed to be blind to. The test
   score is optimistic.
2. **Removing outliers before splitting.** You compute IQR bounds on the full
   dataset and delete the rows outside them — from *both* halves. The test set
   is now artificially easy: you have deleted precisely the hard cases it
   existed to measure. (This is exactly what §5 declined to do.)
3. **Selecting features before splitting.** You check which features relate most
   strongly to the label across all 2,200 rows and keep the top few. That choice
   was made using test-set labels, so any evaluation of it is circular.

A fourth form, not about splitting at all, is worth naming because it is the
most expensive in practice: **a feature that could not exist at prediction
time.** Predicting crop from a `fertiliser_applied_after_planting` column would
score beautifully and be useless, because at recommendation time nothing has
been planted yet. Our seven features are all measurable before planting, so this
project is safe — but the check is one every real project must make explicitly.

### Where does it bite in industry?

It is one of the most common causes of a model that tests at 95% and performs at
70% after deployment. It is also silent: nothing errors, nothing warns, and the
numbers look *better* than they should. That is why scikit-learn's `Pipeline`
exists — it makes fitting-on-train-only the structurally easy option, which is
what Week 8 is about.

### Common mistakes

* **Thinking leakage is only about copying the label into a feature.** That is
  the most obvious form and the rarest.
* **Believing that "the test set is only used at the end" is enough.** If a
  transformation *fitted* on test rows is applied to training rows, the leak has
  already happened.
* **Re-running EDA on the full dataset after splitting, then acting on it.**
  Looking is fine. Changing preprocessing because of what you saw is not.

---

## 7. What the exploration concluded

The full write-up is in the notebook's final cell. In brief:

1. **The dataset is perfectly balanced** — 22 crops, exactly 100 rows each.
   Accuracy is a fair headline metric for Week 4, and there is no minority class
   to rescue. It is also strong evidence that the file is curated rather than
   observed.
2. **Potassium alone almost completely isolates apple and grapes** (~200 vs.
   ≤85 for every other crop; eta-squared 0.996).
3. **Rainfall and humidity separate crops sharply at the extremes** — rice
   ≥182.6 mm vs. muskmelon ≤29.9 mm; chickpea at 14.3–20.0% humidity against
   crops sitting above 80%.
4. **The features are near-independent, and the one exception is a class
   artifact** — every pair between -0.23 and +0.21 except `P`–`K` at +0.74,
   which falls to +0.04 once apple and grapes are removed. No feature is
   redundant; all seven go forward.

Nothing was dropped, deleted, scaled or encoded. The file on disk is
byte-identical to what Week 1 committed.

---

## 8. Where this leaves us

We can now say what the data contains, which features carry class information,
where the extreme values come from, and why none of them should be deleted yet.
We have also stated the rule that governs everything from here: preprocessing is
fitted on training data only.

**Week 3** takes the first irreversible step — splitting the data — and then
scales the features and encodes the label, honouring that rule.

## Recap of new terms

Full definitions live in [`docs/glossary.md`](../../glossary.md); the concept
index is [`docs/ml_concepts.md`](../../ml_concepts.md).

Exploratory data analysis · descriptive statistic · mean · median · standard
deviation · variance · percentile · quartile · interquartile range ·
skewness · histogram · bin · distribution · multimodality · class balance ·
class imbalance · correlation · Pearson correlation · multicollinearity ·
boxplot · whisker · outlier · 1.5 IQR rule · feature scale · class separation ·
eta-squared · data leakage.

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| ◀ [§2.1 Syllabus](syllabus.md) | [Chapter 2 — Exploratory Data Analysis](README.md) · 🗺 [Roadmap](../README.md) | [§2.3 Exercises](exercises.md) ▶ |

<!-- nav:end -->
