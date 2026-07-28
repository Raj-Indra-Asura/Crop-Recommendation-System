# Week 2 — Exploratory Data Analysis

## Title

**Looking before leaping: understanding the data statistically and visually**

## Learning objectives

By the end of this week a student should be able to:

1. Summarise a numeric column with descriptive statistics, and explain what
   `mean`, `median`, `std`, the quartiles and `skew` each tell you that the
   others do not.
2. Measure the **class balance** of a labelled dataset with `value_counts()`,
   and explain why balance changes which evaluation metric is trustworthy.
3. Read a **histogram**, name a distribution as symmetric, right-skewed or
   left-skewed, and say what a skew implies for later modelling.
4. Read a **correlation heatmap**, distinguish "uncorrelated" from "unrelated",
   and explain why strongly correlated features matter for some model families
   and not for others.
5. Read a **boxplot**, state the IQR rule that produces its outlier points, and
   argue why an outlier is a thing to investigate rather than to delete.
6. Define **data leakage** in one sentence, and name three concrete ways a
   preprocessing step can cause it once a train/test split exists.
7. Write a short, evidence-backed EDA conclusion: findings stated as specific
   numbers about specific crops, not vague impressions.

## Prerequisites

Week 1, in full. In particular this week assumes, and does **not** re-explain:

* what a dataframe is, and how to load one with `load_data()`
  ([Week 1 notes §7–8](../week01/learning_notes.md));
* the seven feature columns and the `label` target (Week 1 notes §3);
* the dataset contract, and why the loader validates on every read (Week 1
  notes §8);
* running `ruff check .` and `pytest` (Week 1 notes §9).

New this week: `matplotlib` and `seaborn`, which were installed in Week 1
precisely so this week could use them without changing the environment.

## Concepts covered

| Concept | First introduced |
| --- | --- |
| Exploratory data analysis (EDA) | Week 2 |
| Descriptive statistics | Week 2 |
| Mean, median and why they differ | Week 2 |
| Standard deviation and variance | Week 2 |
| Quartile, percentile, interquartile range (IQR) | Week 2 |
| Skewness | Week 2 |
| Histogram and bin width | Week 2 |
| Class balance / imbalance | Week 2 |
| Correlation (Pearson) | Week 2 |
| Multicollinearity | Week 2 |
| Correlation vs. causation, in feature terms | Week 2 |
| Boxplot | Week 2 |
| Outlier, and the 1.5 IQR rule | Week 2 |
| Feature scale, and why it varies | Week 2 |
| Class separation (eta-squared / correlation ratio) | Week 2 |
| Data leakage | Week 2 |
| Train-only fitting of preprocessing | Week 2 (rule stated; enforced Week 3) |

Each also appears in [`docs/ml_concepts.md`](../../ml_concepts.md) and
[`docs/glossary.md`](../../glossary.md).

## Connection to the previous week

Week 1 answered *what the data is*: 2,200 validated rows, seven numeric
features, 22 crop labels. It deliberately refused to look at the values, so that
the framing was not contaminated by whatever the numbers happened to suggest.

Week 2 answers *what the data looks like*, using the same `load_data()` entry
point and the same frozen label set. Nothing about the Week 1 contract is
relaxed; this week only reads.

## Connection to the ML lifecycle

```
Frame the problem -> Get the data -> [Explore] -> Prepare -> Model
    -> Evaluate -> Improve -> Productionize -> Deploy -> Monitor
```

Week 2 is the **explore** stage, and it is the last stage in which it is safe to
look at every row. From Week 3 onward part of the data is held back and must
stay unseen — which is why data leakage is introduced *now*, one week before the
split that makes it possible.

## Expected student outcome

### The student CAN, after this week

* Describe the dataset's shape (2,200 × 8) and its balance (22 crops × exactly
  100 rows, 4.55% each) without looking it up.
* Name at least three features that visually separate certain crops, with
  evidence: `K` isolates apple and grapes (~200 vs. ≤85 everywhere else);
  `rainfall` separates rice (≥182.6 mm) from muskmelon (≤29.9 mm); `humidity`
  isolates chickpea (14.3–20.0%) from the crops sitting above 80%.
* Explain data leakage in one sentence, and say why the statistics in this
  week's notebook are safe while the same statistics computed after Week 3's
  split would not be.
* Produce and read a histogram, a correlation heatmap and a grouped boxplot,
  using the helpers in `src/utils/eda.py`.
* Say why the `P`–`K` correlation of +0.74 is not a reason to drop a feature.
* Run `pytest tests/test_eda.py` and execute `notebooks/02_EDA.ipynb`
  top-to-bottom.

### The student CANNOT yet

* Preprocess the data in any way — no scaling, no encoding, no row or column
  removal. That is **Week 3**.
* Split into training and test sets, stratified or otherwise — **Week 3**.
* Train any model, including a baseline — **Week 4**.
* Say which crops a *model* confuses. This week shows which crops are *visually*
  separable on single features; confusion between crops is a property of a
  fitted model and needs Week 5 and Week 8.
* Claim that any feature is unimportant. Eta-squared ranks single features in
  isolation; importance in combination is Week 7.
* Report accuracy, precision or recall — Week 4 onward.

## Deliverables for the week

* `src/utils/eda.py` — nine documented, tested helpers: `describe_features`,
  `class_balance`, `plot_class_balance`, `plot_feature_histograms`,
  `plot_correlation_heatmap`, `plot_boxplot_by_label`, `plot_boxplots_by_label`,
  `count_outliers_iqr`, `separation_scores`.
* `tests/test_eda.py` — smoke tests running every helper on a small synthetic
  dataframe, plus checks that invalid input is refused.
* `notebooks/02_EDA.ipynb` — the full exploration, committed with executed
  output, ending in a written conclusion of four concrete findings.
* This week's four curriculum documents.
* Updated `docs/ml_concepts.md`, `docs/glossary.md` and the README progress
  table.

No change to `requirements.txt`: `matplotlib` and `seaborn` were already pinned
in Week 1 for exactly this purpose.
