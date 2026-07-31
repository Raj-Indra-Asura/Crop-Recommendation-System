# Week 2 — Exercises

> 🗺 [Roadmap](../README.md) › [Part I — Foundations (Weeks 1-3)](../README.md#part-i--foundations-weeks-1-3) › [Chapter 2 — Exploratory Data Analysis](README.md) › **§2.3 Exercises**

Work through these in order. Beginner exercises check that you can reproduce
what the notes covered; intermediate exercises make you apply it somewhere new;
challenge exercises deliberately go slightly beyond the notes.

Keep scratch work in a new notebook or a scratch script. Do not edit
`notebooks/02_EDA.ipynb`, `src/utils/eda.py` or `tests/test_eda.py` unless an
exercise says so — and **never** modify `data/raw/Crop_recommendation.csv`.

Every exercise starts from the same two lines:

```python
from src.data import FEATURE_COLUMNS, TARGET_COLUMN, load_data
crops = load_data()
```

---

## Beginner

**B1 — Read the summary table.**
Print `describe_features(crops, list(FEATURE_COLUMNS))`. For `K`, write down the
mean, median and skew, and explain in one sentence why the mean is so much
higher than the median. Then name the feature with the smallest spread and the
one with the largest, and say what that difference will force Week 3 to do.

**B2 — Confirm the balance.**
Print `crops["label"].value_counts()`. How many rows per crop? What proportion
of the dataset is each crop? Then answer: if 95% of the rows were rice, what
accuracy could a model that always answers "rice" achieve, and what would that
tell you about the model?

**B3 — Draw the histograms.**
Call `plot_feature_histograms(crops, list(FEATURE_COLUMNS))`. Which two features
show a clearly detached cluster on the right? Which feature is left-skewed?
Check your answers against the `skew` column from B1.

**B4 — Draw the heatmap.**
Call `plot_correlation_heatmap(crops, list(FEATURE_COLUMNS))`. Which single pair
of features stands out? Write down its coefficient. Then state, in one sentence,
why a correlation of 0 between `ph` and `humidity` does *not* prove they are
unrelated.

**B5 — Group a boxplot.**
Call `plot_boxplot_by_label(crops, "rainfall")`. Name the crop with the highest
median rainfall and the one with the lowest. Then do the same for `humidity` and
for `K`. You have now named three features that separate specific crops — write
each as a one-line claim with numbers in it.

**B6 — Count the flagged values.**
Call `count_outliers_iqr(crops, list(FEATURE_COLUMNS))`. Which feature has the
most flagged values, and which has none at all? Now print
`crops.loc[crops["K"] > 92.5, "label"].value_counts()`. What do you conclude
about whether those values are errors?

**B7 — Say it in one sentence.**
Without looking at the notes, write a one-sentence definition of data leakage.
Then check it against `learning_notes.md` §6 and correct it if needed.

**B8 — Run the guard rails.**
Run `pytest tests/test_eda.py` and `ruff check .`. How many tests ran? Then
execute the notebook end to end with
`jupyter nbconvert --to notebook --execute notebooks/02_EDA.ipynb` and confirm
it exits 0.

---

## Intermediate

**I1 — Change the bin count.**
Draw the `K` histogram with `bins=5`, then `bins=30`, then `bins=200`. Describe
what each one hides or invents. Which bin count would have made you miss the
apple/grapes cluster entirely, and what does that say about reporting a single
histogram as evidence?

**I2 — Break the P–K correlation apart yourself.**
Compute the `P`–`K` correlation for the full dataset, then for each crop
separately (`crops.groupby("label").apply(...)`). How many individual crops show
a correlation above 0.5? Explain in your own words why the pooled coefficient is
0.74 when almost every within-crop coefficient is near zero.

**I3 — Pearson vs. Spearman.**
Call `plot_correlation_heatmap(..., method="spearman")` and compare it to the
Pearson version. Which pairs change the most? Construct a small synthetic
example (say `y = x ** 2` for `x` in -10..10) where Pearson is near zero and
Spearman is not — or where the reverse holds — and explain what each is
measuring.

**I4 — Quantify a separation claim.**
Pick any two crops and any one feature. Compute both crops' min, max, mean and
median for that feature, and state whether their ranges overlap at all. Then
find a `(feature, crop-pair)` combination with *complete* separation — no
overlap — and one with *total* overlap. What does the second case tell you about
relying on one feature?

**I5 — Add a helper and a test.**
Add a function `plot_feature_pair(frame, x, y, target)` to `src/utils/eda.py`
that draws a scatter plot of two features coloured by class, with a docstring in
the same style as its neighbours. Add a smoke test for it in
`tests/test_eda.py`. Then use it on `P` vs `K` and describe what the apple and
grapes points look like. `ruff check .` and `pytest` must both pass.

**I6 — Design a leak.**
Write (but do **not** commit) a short script that would leak: compute the mean
and standard deviation of every feature over all 2,200 rows, standardise, then
split 80/20. Then write the corrected version in comments — the same steps in
the order that avoids the leak. State precisely which number differs between the
two versions and why it matters.

**I7 — Re-check the Week 1 hypothesis.**
Week 1's exercise C4 asked you to hypothesise how such a tidy dataset was
produced. Using this week's evidence — exactly 100 rows per crop, the tight
per-crop ranges visible in the boxplots, the near-zero within-crop feature
correlations — write half a page assessing that hypothesis. What would you
expect to see instead if these were genuine field measurements?

---

## Challenge

**C1 — Rank the crop pairs by difficulty.**
For every pair of crops, compute how many of the seven features have completely
non-overlapping ranges. Which pairs share overlapping ranges on *all seven*?
Those are the pairs a model is most likely to confuse. Record your prediction
now, in writing — Week 8's confusion matrix will let you check it.

**C2 — Build a one-rule classifier without training anything.**
Using only thresholds you can read off this week's boxplots, hand-write a
function that predicts a crop from the seven features, and measure its accuracy
on the full dataset. (This is not machine learning: you are the learning
algorithm.) How far can you get? What does the result suggest about the
difficulty of the problem, and why is measuring it on all 2,200 rows not a
legitimate estimate of how it would perform on new fields?

**C3 — Argue for keeping an outlier.**
Choose one crop whose values the IQR rule flags heavily. Write the strongest
case for deleting those rows, then the strongest case against, then state which
you would do and what evidence would change your mind. Being able to argue both
sides is the point.

**C4 — Where else could leakage enter this project?**
The notes list three split-related forms plus one impossible-feature form. Think
past the split: if this system were deployed and retrained monthly on data that
included its own past recommendations, what new leakage (or feedback loop) could
appear? Write half a page. Week 12's monitoring section is where this becomes
concrete.

**C5 — Reproduce a helper from scratch.**
Without reading `src/utils/eda.py`, implement `count_outliers_iqr` yourself from
the definition in `learning_notes.md` §5, then compare your output to the
library version on the real dataset. If they differ, work out which is right
before looking at the source.

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| ◀ [§2.2 Learning notes](learning_notes.md) | [Chapter 2 — Exploratory Data Analysis](README.md) · 🗺 [Roadmap](../README.md) | [§2.4 Validation](validation.md) ▶ |

<!-- nav:end -->
