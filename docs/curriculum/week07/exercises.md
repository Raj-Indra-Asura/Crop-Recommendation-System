# Week 7 — Exercises

Work through these in order. Beginner exercises check that you can reproduce what
the notes covered; intermediate exercises make you apply it somewhere new;
challenge exercises deliberately go slightly beyond the notes.

Keep scratch work in a new notebook or a scratch script. Do not edit
`notebooks/06_model_selection.ipynb`, `src/models/ensemble_models.py` or
`tests/test_ensemble_models.py` unless an exercise says so — and **never**
modify `data/raw/Crop_recommendation.csv`.

Everything this week happens on the **training rows only**.
`data/processed/test.csv` stays closed until Week 8; an exercise that needs
held-out data uses a cross-validation fold or an inner split of the training
set.

This week introduces ensembles, not hyperparameter search. The forest-size and
boosting-round sweeps below draw curves to explain mechanisms. They do not
license choosing a setting, because a real tuning protocol — `GridSearchCV`,
validation curves and friends — comes later. Week 8 also adds permutation
importance, SHAP, confusion matrices and the first test-set score. Do not use
any of those here.

Most exercises start from the same lines:

```python
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluation import build_cv, cross_validated_accuracy, evaluate_model
from src.models import (
    CLASSICAL_MODEL_FACTORIES,
    ENSEMBLE_MODEL_FACTORIES,
    GRADIENT_BOOSTING_BACKEND,
    get_baseline_model,
    get_decision_tree,
    get_gradient_boosting,
    get_knn,
    get_logistic_regression,
    get_naive_bayes,
    get_random_forest,
    get_svm,
)
from src.preprocessing import build_preprocessor

train = pd.read_csv("data/processed/train.csv")
FEATURES = list(FEATURE_COLUMNS)
X, y = train[FEATURES], train[TARGET_COLUMN]


def make_pipeline(model):
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])
```

`build_preprocessor()` with no arguments expects exactly the seven feature
columns. Any exercise that changes the columns — a duplicated feature (I4), a
synthetic frame (C4), or a hand-built two-feature frame — must pass the new
column list, `build_preprocessor(list(frame.columns))`, as in earlier weeks.

`get_gradient_boosting()` returns an XGBoost-backed estimator when the optional
`xgboost` package is importable and scikit-learn's
`GradientBoostingClassifier` otherwise. Always print or record
`GRADIENT_BOOSTING_BACKEND`; your code should work with either backend.
Expected scores below quote XGBoost first and the scikit-learn fallback where it
matters.

---

## Beginner

**B1 — Build the two ensemble estimators.**
Call `get_random_forest()` and `get_gradient_boosting()` with no arguments and
print the objects. What are the forest defaults for `n_estimators`, `max_depth`
and `max_features`? What are the boosting defaults for `n_estimators`,
`learning_rate` and `max_depth`? Print `GRADIENT_BOOSTING_BACKEND`. Confirm both
estimators are still unfitted by checking that `.predict(X)` on the bare object
raises.

**B2 — Reject the nonsense.**
Show that each of these raises `ValueError`, and read the message:
`get_random_forest(n_estimators=0)`, `get_random_forest(max_depth=0)`,
`get_random_forest(max_features="all")`, `get_random_forest(max_features=0)`,
`get_gradient_boosting(n_estimators=0)`, `get_gradient_boosting(max_depth=0)`
and `get_gradient_boosting(learning_rate=0)`. Which checks belong to forests,
which belong to boosting, and why is failing at factory time friendlier than
failing halfway through cross-validation?

**B3 — Extend the results table again.**
Cross-validate the baseline, the five Week 5/6 models and the two ensembles
through `make_pipeline`. Use the same five stratified folds and seed 42. You
should see about: baseline **0.0455**, KNN **0.9653**, logistic regression
**0.9682**, RBF SVM **0.9790**, unlimited tree **0.9852**, random forest
**0.9926 ±0.0058**, gradient boosting with XGBoost **0.9909 ±0.0033**, and
naive Bayes **0.9949 ±0.0042**. If your backend is `"sklearn"`, expect gradient
boosting nearer **0.9869 ±0.0034** instead. Did you use the preprocessor for
every row of the table?

**B4 — Say bagging and boosting in your own words.**
Write two short paragraphs, without copying the notes. The first must explain
bagging: bootstrap samples, independent overfit trees, averaging or voting, and
variance reduction. The second must explain boosting: weak trees fitted in
sequence, each correcting what remains wrong, `learning_rate` shrinking each
step, and the risk of overfitting if the chain is too long. End with one
sentence contrasting which family mainly lowers variance and which mainly lowers
bias.

**B5 — Count the bootstrap rows.**
Fit `get_random_forest()` on `X, y` without the pipeline, pull
`estimators_samples_[0]`, and count how many distinct row indices the first tree
saw. The draw has length **1,760** because it samples with replacement from the
1,760 training rows. You should find about **63%** distinct rows and about
**37%** out-of-bag rows for that tree (the notebook's seed gives 1,114 distinct
and 646 out-of-bag). In one sentence, explain why a row can be both absent from
one tree and present in another.

**B6 — Check feature randomness at the root.**
Fit a default forest and count the first split feature of each tree. You should
reproduce six root features across the 100 trees: humidity 25, K 20, P 19,
rainfall 17, N 16 and temperature 3. Fit one unlimited decision tree and confirm
its first split is rainfall. Why does `max_features="sqrt"` mean only **2 of the
7** features are offered at each split, and why does that make the trees less
correlated?

**B7 — Average overfit trees, honestly.**
Cross-validate `get_decision_tree()` and `get_random_forest()` through the same
pipeline. Expected: the single tree at **0.9852 ±0.0068**, the forest at
**0.9926 ±0.0058**. Now fit the forest on all `X, y` and score it on `X, y` with
`evaluate_model`; it should still score **1.0** on its own training rows. Explain
why this is not a contradiction: what overfits, and what the averaging improves.

**B8 — Read `feature_importances_`.**
Fit a default forest on all 1,760 training rows and make a horizontal bar plot
of `feature_importances_`, indexed by `FEATURES`. Confirm the values are
non-negative and sum to 1. Expected ranking: rainfall 0.2302, humidity 0.2242,
K 0.1754, P 0.1508, N 0.0964, temperature 0.0724, ph 0.0506. Which two
features dominate? Which feature is lowest? Write one cautious sentence that
starts "This tells me the fitted forest used ...", not "This proves ...".

---

## Intermediate

**I1 — Sweep forest size, but do not tune it.**
Cross-validate `get_random_forest(n_estimators=size)` for sizes
`[1, 3, 10, 30, 100, 300]`. Plot size against mean CV accuracy. Expected means:
0.9483, 0.9756, 0.9909, 0.9932, 0.9926 and 0.9932. Where does the curve flatten?
Why is `n_estimators` mostly a compute budget for a forest rather than a dial
that creates a bad overfitting end? Finally, state why picking 30 or 300 from
this plot is still not a formal tuning procedure.

**I2 — Watch boosting correct itself.**
Use depth-1 boosted trees, fit on the full training set, and record **training**
accuracy for rounds `[1, 2, 5, 20, 60]`. Expected: 0.6193, 0.6710, 0.6733,
0.9307 and 0.9881. Why is a single stump chain much weaker than one tree in the
forest-size sweep? Why are these training accuracies allowed for explaining the
mechanism, but not allowed as a model-selection result?

**I3 — Learning rate and rounds are paired.**
Pick three pairs such as `(learning_rate=0.2, n_estimators=50)`, `(0.1, 100)`
and `(0.05, 200)`, keeping `max_depth=3`. Cross-validate them and time their
fits. Do the slower, smaller-step models always win on this clean dataset? Write
two sentences explaining the tradeoff: a lower `learning_rate` makes each round
matter less, so more rounds are needed; more rounds cost time and can eventually
fit noise. Do not adopt a winner.

**I4 — Split the credit with a copied column.**
Create `X_duplicated = X.assign(humidity_copy=X["humidity"])`, fit a default
forest on it, and compare importances with the original forest. Expected:
humidity drops from **0.2242** to about **0.1485**, while the copy takes about
**0.1370**, with identical training accuracy. Explain why the model has not lost
information. Then connect this to Week 2's **0.74** correlation between `P` and
`K`: what can happen to importance when two columns carry similar information?

**I5 — Name the three importance limitations.**
Using your B8 and I4 plots, write three bullet points explaining why mean
decrease in impurity should not be quoted as the final scientific answer. Your
bullets must mention: it is computed on the **training data**; it is biased
towards high-cardinality features with many possible thresholds; and correlated
features can split credit. End by naming the two Week 8 tools that address these
problems: permutation importance and SHAP.

**I6 — Answer the Week 7 ranking question.**
From B3's table, answer this precisely: do the ensembles beat the Week 5 and
Week 6 models on this dataset? Your answer must say that the forest and booster
beat KNN, logistic regression, the RBF SVM and the unlimited decision tree. It
must also say that Gaussian naive Bayes still leads at **0.9949 ±0.0042**, while
the forest is **0.9926 ±0.0058** and the gap is inside the fold spread. The
honest headline is "level with the leader", not "ensembles win outright".

---

## Challenge

**C1 — Repeat the honest comparison across seeds.**
For seeds 0 through 9, run `cross_validated_accuracy(..., random_state=seed)`
for naive Bayes, the random forest and gradient boosting. Plot the ten means per
model. Does naive Bayes lead on every seed, or do the forest and booster
sometimes swap with it? Write the two-sentence conclusion you would defend, and
name what a later repeated-CV or paired-test protocol would make more rigorous.

**C2 — Bootstrap sampling by simulation.**
Without fitting a model, simulate 10,000 bootstrap samples of 1,760 row indices
with `np.random.default_rng(42)`. For each sample, count the number of distinct
rows and out-of-bag rows. Plot the distributions or report their 5th, 50th and
95th percentiles. How close is the median distinct share to 63%? Why is there a
distribution rather than one fixed number, and why does the forest not need each
tree to see every row?

**C3 — Plain bagging versus random forest.**
Cross-validate `get_random_forest(max_features=None)` and the default
`get_random_forest(max_features="sqrt")`, and for each count root-split features
across the fitted trees. `max_features=None` means every feature is available at
every split, so it is closer to plain bagging. Which version has more diverse
root questions? Which version scores higher or lower on these folds? Explain the
result using the rule that averaging cancels only errors the members do not
share.

**C4 — Make a noisier tabular problem.**
Construct a small synthetic classification problem with overlapping classes and
at least one useless noise column. Compare logistic regression, one unlimited
tree, a random forest and gradient boosting with the same cross-validation
protocol. Does the ensemble advantage become clearer than it was on the crop
data? If it does, explain why messier data gives variance reduction and
sequential error-correction more room to help. If it does not, inspect your data
generation choices before changing model settings.
