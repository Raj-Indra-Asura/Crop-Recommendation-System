# Week 3 — Exercises

> 🗺 [Roadmap](../README.md) › [Part I — Foundations (Weeks 1-3)](../README.md#part-i--foundations-weeks-1-3) › [Chapter 3 — Data Preparation](README.md) › **§3.3 Exercises**

Work through these in order. Beginner exercises check that you can reproduce
what the notes covered; intermediate exercises make you apply it somewhere new;
challenge exercises deliberately go slightly beyond the notes.

Keep scratch work in a new notebook or a scratch script. Do not edit
`notebooks/03_data_preparation.ipynb`, `src/data/split.py`,
`src/preprocessing/preprocessor.py` or `tests/test_preprocessing.py` unless an
exercise says so — and **never** modify `data/raw/Crop_recommendation.csv`.

Most exercises start from the same lines:

```python
from src.data import FEATURE_COLUMNS, TARGET_COLUMN, load_data, stratified_split
crops = load_data()
FEATURES = list(FEATURE_COLUMNS)
```

---

## Beginner

**B1 — Measure the scale problem.**
Print `crops[FEATURES].max() - crops[FEATURES].min()`. Which feature has the
largest range, which the smallest, and what is the ratio between them? Then
answer in one sentence: if a k-nearest-neighbours model used these columns
unchanged, which feature would dominate its distance calculation, and why is
that not a decision about agronomy?

**B2 — Encode the labels.**
Fit a `LabelEncoder` on `crops["label"]` and print `encoder.classes_`. What
integer does `rice` get? What decides that number? Now decode the codes back
with `inverse_transform` and confirm you recover the original column exactly.

**B3 — Argue the codes are not numbers.**
`apple` is 0 and `watermelon` is 21. Write two sentences: one saying what would
go wrong if these codes were fed to a *regressor*, and one saying why a
*classifier* is unaffected.

**B4 — Split it.**
Call `stratified_split(crops)`. How many rows in each half? How many rows does
each crop contribute to the test set? Confirm the two halves together contain
every original row exactly once. **Hint:** the helper resets both indexes, so
compare the combined row count and crop counts rather than the index values.

**B5 — Prove the split is stratified.**
Use `class_proportions()` from `src.data` on both halves and subtract one from
the other. What is the largest absolute difference, and what would you expect
that number to be for an unstratified split? (B6 answers the second half.)

**B6 — Break it on purpose.**
Repeat B4 with plain `train_test_split(crops, test_size=0.2, random_state=42)`,
without `stratify`. Print the per-crop test counts. What is the smallest and
largest? Name the crop that is worst served, and say what that does to its
recall in Week 8.

**B7 — Fit, transform, and read the state.**
Build a preprocessor with `build_preprocessor()`, call `fit(train)`, and print
`preprocessor.named_transformers_["numeric"].mean_` and `.scale_`. How many
numbers has the object learned in total? Which rows did each of them come from?
Then call `transform` on a *fresh, unfitted* preprocessor and record the exact
exception you get.

**B8 — Check the scaling worked.**
Compute the mean and standard deviation of each column of the transformed
training array (use `ddof=0`) and then of the transformed test array. Which one
is exactly 0/1, which is only close, and why is "only close" the correct answer
rather than a bug?

**B9 — Run the guard rails.**
Run `pytest tests/test_preprocessing.py` and `ruff check .`. How many tests ran?
Then execute the notebook end to end with
`jupyter nbconvert --to notebook --execute notebooks/03_data_preparation.ipynb`
and confirm it exits 0.

---

## Intermediate

**I1 — Do the leak, then measure it.**
Fit one scaler on all 2,200 rows and another on the training rows only. Print
the difference between their `mean_` and `scale_` vectors. Which feature drifts
most? Express that drift as a percentage of the feature's standard deviation.
Then write two sentences on why a small measured drift is not an argument for
allowing the leak.

**I2 — Make the leak matter.**
Construct a version of the dataset where the leak is *not* small: for example,
append 50 rows with a `rainfall` of 5,000, or drop 90% of the rows for 10 of the
crops so the classes become imbalanced. Repeat I1 on it. How large does the
drift get? What property of your modified data caused it?

**I3 — Seed sensitivity.**
Run `stratified_split(crops, random_state=s)` for `s` in `range(10)` and record,
for each split, the mean `rainfall` of the test set. How much does it vary? Now
explain why picking the seed whose test set gives the most flattering later
accuracy is a form of overfitting, and what you should do instead when results
vary a lot between seeds.

**I4 — Min-max instead.**
Build a second `ColumnTransformer` using `MinMaxScaler` rather than
`StandardScaler`, fit it on the training rows and transform both halves. What
are the min and max of each transformed *test* column, and why are some of them
outside [0, 1]? Which scaler would you choose here, and why?

**I5 — Standardisation does not remove skew.**
Compute the skew of `K` before scaling and of the standardised `K` column after.
Compare them. Then apply `np.log1p` to `K` and compute the skew again. Explain,
in terms of linear versus non-linear transformations, why only one of the two
changed the number.

**I6 — Add a categorical feature and extend the preprocessor.**
Invent a categorical column — say `season`, filled with `"kharif"`/`"rabi"` at
random — and extend a *copy* of `build_preprocessor()` so that the numeric
columns are standardised and `season` is one-hot encoded, in one
`ColumnTransformer`. Print `get_feature_names_out()`. How many output columns
are there, and why is one-hot encoding correct here where label encoding would
not be? Start with `from sklearn.preprocessing import OneHotEncoder`; its
`OneHotEncoder()` instance replaces the `StandardScaler()` for the `season`
entry in the transformer's list.

**I7 — Pipeline with a model attached (a preview).**
Build `Pipeline([("preprocess", build_preprocessor()), ("model",
KNeighborsClassifier())])`, fit it on the training rows and their encoded
labels, and call `predict` on the *first five training rows only*. Do **not**
touch the test set, and do not report an accuracy — Week 4 exists for that. Then
answer: how many times was the scaler fitted, and on what? This is only a
pipeline preview: import the not-yet-explained classifier with
`from sklearn.neighbors import KNeighborsClassifier` and use it exactly as
shown; Week 5 explains how it works.

**I8 — Trees really do not care.**
Fit a `DecisionTreeClassifier(random_state=42)` twice on the training rows —
once on the raw features, once on the standardised ones — and compare their
predictions on the *training* rows. Are they identical? Explain the result from
how a tree chooses a split. Then do the same with `KNeighborsClassifier` and
explain the difference.

---

## Challenge

**C1 — Reimplement `StandardScaler`.**
Without looking at scikit-learn's source, write a class with `fit`, `transform`
and `fit_transform` that standardises a dataframe's numeric columns, storing
`mean_` and `scale_`. Compare its output to `StandardScaler`'s on the training
rows to within 1e-12. Which denominator does scikit-learn use — the population
standard deviation (`ddof=0`) or the sample one (`ddof=1`) — and how did you find
out?

**C2 — Reimplement the stratified split.**
Using only `numpy` and `pandas`, write `my_stratified_split(frame, test_size,
random_state)` that shuffles within each class and takes the same fraction from
each. Verify that every class holds the same proportion on both sides. Then
explain what your implementation must do when a class's row count does not
divide evenly — and check what scikit-learn does in that case.

**C3 — Design the leak that a split cannot prevent.**
Suppose 200 of the 2,200 rows were duplicated measurements of the same 100
fields. A stratified random split would put some duplicates on both sides. Write
half a page on: how you would detect this, why the test score would be inflated,
what kind of splitter fixes it, and how the problem relates to a farm submitting
the same field twice through Week 10's API.

**C4 — Argue against saving scaled data.**
This week wrote both raw and scaled splits to `data/processed/`. Make the
strongest case that only the raw splits should be saved and that scaling should
always be re-done inside a pipeline. Then make the strongest case for keeping
the scaled copies. Which would you commit to a shared repository, and what would
change your mind?

**C5 — Preparation for serving, on paper.**
Week 10 will accept a single field as JSON and return a crop name. Write out —
in prose, no code — every transformation that request must pass through, in
order, and every artifact the server must have loaded to perform them. Where in
that list would a mistake produce training/serving skew, and how would you
detect it in production if it happened silently?

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| ◀ [§3.2 Learning notes](learning_notes.md) | [Chapter 3 — Data Preparation](README.md) · 🗺 [Roadmap](../README.md) | [§3.4 Validation](validation.md) ▶ |

<!-- nav:end -->
