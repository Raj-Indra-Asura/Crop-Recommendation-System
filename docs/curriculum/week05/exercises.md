# Week 5 — Exercises

Work through these in order. Beginner exercises check that you can reproduce
what the notes covered; intermediate exercises make you apply it somewhere new;
challenge exercises deliberately go slightly beyond the notes.

Keep scratch work in a new notebook or a scratch script. Do not edit
`notebooks/05_classification_models.ipynb`, `src/models/classical_models.py` or
`tests/test_classical_models.py` unless an exercise says so — and **never**
modify `data/raw/Crop_recommendation.csv`.

Everything this week happens on the **training rows only**.
`data/processed/test.csv` stays closed until Week 8; an exercise that needs
held-out data uses a cross-validation fold or an inner split of the training
set.

Most exercises start from the same lines:

```python
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluation import build_cv, cross_validated_accuracy, evaluate_model
from src.models import get_baseline_model, get_knn, get_logistic_regression, get_naive_bayes
from src.preprocessing import build_preprocessor

train = pd.read_csv("data/processed/train.csv")
FEATURES = list(FEATURE_COLUMNS)
X, y = train[FEATURES], train[TARGET_COLUMN]


def make_pipeline(model):
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])
```

`build_preprocessor()` with no arguments expects exactly the seven feature
columns. Any exercise that changes the columns — two features only (I1),
synthetic data (I2, C1), extra noise columns (I5) — must pass the new column
list, `build_preprocessor(list(frame.columns))`, exactly as the notebook's noise
demonstration does.

---

## Beginner

**B1 — Write the training loop from memory.**
Without looking at the notebook: fit `get_naive_bayes()` on `X, y` and predict
the first five rows. Which two method calls did you need, what does each of them
do, and which of them is allowed to see `y`?

**B2 — Count what each model stores.**
Fit all three models. Print `coef_.shape` and `intercept_.shape` for logistic
regression, `theta_.shape` and `var_.shape` for naive Bayes, and
`_fit_X.shape` for KNN (the leading underscore marks it as scikit-learn's own
private attribute — KNN has no public one, which is itself a hint about how
little it learns). How many numbers does each model keep? Which one keeps the
training data itself, and what does that imply for saving it to disk in Week 9
and serving it from an API in Week 10?

**B3 — Watch the probabilities.**
For the first field in `X`, print the top three classes and their probabilities
from each model's `predict_proba`. Which model is the most confident? Which
produces probabilities that can only be multiples of 0.2, and why?

**B4 — Reproduce the results table.**
Cross-validate the baseline and all three models with `cross_validated_accuracy`
through `make_pipeline`, and tabulate mean and std. Do your numbers match
[the learning notes §6](learning_notes.md)? If any differ, check the seed, the
fold count and whether you used the preprocessor.

**B5 — The gap that is not a gap.**
From your table, subtract logistic regression's mean from KNN's. Now compare
that difference with each model's standard deviation across folds. Write one
sentence explaining why this experiment does not rank those two models, and one
sentence explaining why it *does* rank naive Bayes above both.

**B6 — Sweep `k`.**
Cross-validate `get_knn(n_neighbors=k)` for `k` in `[1, 3, 5, 11, 25, 51, 101,
201, 401]`. Plot accuracy against `k`. Where is the overfitting end of the
curve, where is the underfitting end, and what value would the curve approach if
you kept going?

**B7 — Prove `k = 1` memorises.**
Fit `get_knn(n_neighbors=1)` on `X, y` and score it on `X, y` itself. What
accuracy do you get, and why is it guaranteed? Now cross-validate the same
model. Why is the second number the only informative one?

**B8 — Scaling, two models, two answers.**
Standardise `X` with `StandardScaler`, then re-fit KNN and naive Bayes on both
the raw and the standardised frames and compare their predictions row by row.
Which model's predictions changed and which did not? Explain both results in
terms of what the algorithm computes. Then multiply `K` by 1,000 instead and
re-check naive Bayes: seven rows move — re-read the `var_smoothing` note in
[§4 of the learning notes](learning_notes.md) and explain why an invariance that
is exact in theory is only nearly exact in floating point.

**B9 — Break naive Bayes' assumption on purpose.**
Add a column that is an exact copy of `K` and cross-validate naive Bayes again.
Did accuracy move? Now print `predict_proba` for a single row before and after.
What changed more, the prediction or the confidence — and why is that exactly
what §4 of the notes predicts?

**B10 — Run the guard rails.**
Run `pytest tests/test_classical_models.py` and `ruff check .`. How many tests
ran? Then execute the notebook end to end with
`jupyter nbconvert --to notebook --execute notebooks/05_classification_models.ipynb`
and confirm it exits 0.

---

## Intermediate

**I1 — Make the boundary visible.**
Pick two features (`humidity` and `rainfall` are a good pair) and three crops.
Fit logistic regression and KNN on those two columns only, then plot the
decision regions on a grid over the feature space. Which boundaries are straight
and which are not? Now pick a pair of features where the crops overlap and
repeat: which model degrades more gracefully?

**I2 — Find a problem logistic regression cannot solve.**
Generate a two-class dataset where one class forms a ring around the other
(`sklearn.datasets.make_circles`). Cross-validate logistic regression, KNN and
naive Bayes on it. Which fails, and why is its failure a statement about the
*shape* of the boundary rather than about the amount of data? Then add
`x1**2 + x2**2` as a third feature and re-run logistic regression. What happened,
and what does that say about "linear"?

**I3 — Where do the mistakes go?**
For each model, collect its cross-validated predictions with
`sklearn.model_selection.cross_val_predict` and list the rows it got wrong. It
is new here, and it is the only new function this section needs: give it the
same `cv=build_cv()` splitter from `src.evaluation` that
`cross_validated_accuracy` uses, and instead of a score it returns one
prediction per row, each made by a model that did not see that row.
Which crops does each model confuse? Do the three models make the *same*
mistakes or different ones? (You are not asked to interpret precision or recall
— that is Week 8; just count.)

**I4 — The cost of `C`.**
Cross-validate logistic regression for `C` in `[0.001, 0.01, 0.1, 1, 10, 100]`
and tabulate accuracy alongside `np.abs(model.coef_).mean()` from a fit on the
full training set. Describe in two sentences what small `C` does to the weights
and what it does to the score. Why is choosing the best `C` from this table
*not* a legitimate tuning procedure (yet)?

**I5 — Dilute the distance.**
Add 5, 20, 50 and 200 columns of standardised noise to `X` and cross-validate
KNN, logistic regression and naive Bayes on each version. Plot the three curves.
Which model degrades fastest, which slowest, and how does that match the
explanation of the curse of dimensionality in §3?

**I6 — Timing the three.**
Time `fit` and `predict` separately for each model, using the full training set
for both. Which model has the cheapest `fit`? The cheapest `predict`? Now repeat
with the training data duplicated ten times (17,600 rows) and describe how each
timing scaled.

**I7 — An unfair comparison, staged.**
Cross-validate logistic regression with `random_state=42` folds and KNN with
`random_state=7` folds. How big is the apparent difference? Repeat with several
seed pairs and record the largest gap you can manufacture between two models
that your fair table said were indistinguishable. Write one sentence for a
colleague explaining what you just demonstrated.

---

## Challenge

**C1 — Leak the scaler, and measure the damage.**
Fit `build_preprocessor()` on the *entire* training set, transform it, and then
cross-validate the three models on the pre-transformed matrix. Compare with the
correct pipeline results. Is the difference big here? Explain why the size of the
difference is not the point, and construct a small synthetic dataset (few rows,
extreme outliers) where the same mistake changes the reported score
substantially.

**C2 — Implement KNN yourself.**
Write a function that takes `X_train, y_train, X_query, k` and returns
predictions, using only NumPy. Check it agrees with `KNeighborsClassifier` on
100 query rows for `k` in `{1, 5, 11}`. Then explain, from your own code, why
prediction cost grows with the training set and why scaling changes the answer.

**C3 — Implement Gaussian naive Bayes yourself.**
Compute the per-class means and variances, then score a query row by summing
log-densities plus the log prior. Confirm your predictions match `GaussianNB` on
the training rows. Where in your code is the independence assumption located,
and what would you have to store instead to drop it?

**C4 — Is naive Bayes' lead real?**
Repeat the whole comparison with ten different cross-validation seeds
(`cross_validated_accuracy(..., random_state=seed)`), collect the ten means per
model, and plot the distributions. Does naive Bayes lead on all ten? Do logistic
regression and KNN ever swap? Write the two-sentence conclusion you would be
willing to defend — and note which parts of your procedure a later week will make
rigorous.

**C5 — A calibration check.**
Bucket every cross-validated prediction by its predicted probability (0.5-0.6,
0.6-0.7, ... 0.9-1.0) and compute the actual accuracy within each bucket. Get
the probabilities the same way I3 got the predictions —
`cross_val_predict(..., method="predict_proba")` — and take each row's largest
one as its confidence. Do this
for naive Bayes and for logistic regression. Which model's confidence matches
reality more closely, and how does the result support §4's claim that naive
Bayes' probabilities should be distrusted even when its predictions should not?

**C6 — Add a fourth model to the table.**
Pick an algorithm not covered this week — `LinearSVC`, `DecisionTreeClassifier`
or `SGDClassifier` — and add it to the comparison *without* changing the
protocol: same folds, same preprocessor, same metric. Where does it land? Then
write the factory function you would add to
`src/models/classical_models.py` for it, with the docstring it would need to
match the module's style. (Do not commit it; a later week decides which models the
project actually adopts.)
