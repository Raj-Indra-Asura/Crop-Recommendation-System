# Week 6 — Exercises

> 🗺 [Roadmap](../README.md) › [Part II — Modelling (Weeks 4-7)](../README.md#part-ii--modelling-weeks-4-7) › [Chapter 6 — Margin-based and Tree-based Models](README.md) › **§6.3 Exercises**

Work through these in order. Beginner exercises check that you can reproduce what
the notes covered; intermediate exercises make you apply it somewhere new;
challenge exercises deliberately go slightly beyond the notes.

Keep scratch work in a new notebook or a scratch script. Do not edit
`notebooks/05_classification_models.ipynb`, `src/models/classical_models.py`,
`src/utils/visualization.py` or `tests/test_classical_models.py` unless an
exercise says so — and **never** modify `data/raw/Crop_recommendation.csv`.

Everything this week happens on the **training rows only**.
`data/processed/test.csv` stays closed until Week 8; an exercise that needs
held-out data uses a cross-validation fold or an inner split of the training set.

This week has no ensembles and no hyperparameter search. Both the `C` sweep and
the depth sweep below draw curves to explain a mechanism; nothing chosen off them
is "tuned", because a real search needs a protocol this course builds in a later
week. Where an exercise asks you to pick the best value from a sweep, it also
asks you to say why that is not yet a legitimate choice.

Most exercises start from the same lines:

```python
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluation import build_cv, cross_validated_accuracy, evaluate_model
from src.models import (
    CLASSICAL_MODEL_FACTORIES,
    get_decision_tree,
    get_logistic_regression,
    get_naive_bayes,
    get_svm,
)
from src.preprocessing import build_preprocessor
from src.utils.visualization import plot_decision_boundary

train = pd.read_csv("data/processed/train.csv")
FEATURES = list(FEATURE_COLUMNS)
X, y = train[FEATURES], train[TARGET_COLUMN]


def make_pipeline(model):
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])
```

`build_preprocessor()` with no arguments expects exactly the seven feature
columns. Any exercise that changes the columns — two features only (I1, I3),
synthetic data (C2) — must pass the new column list,
`build_preprocessor(list(frame.columns))`, exactly as Part 1's noise
demonstration did.

`cross_validated_accuracy(pipeline, X, y)` returns a dict with keys `scores`,
`mean`, `std` and `n_splits`; quote `mean` and `std`. `evaluate_model` needs a
**fitted** model. `CLASSICAL_MODEL_FACTORIES` is the five-entry registry from
[the learning notes §1](learning_notes.md); loop over it whenever an exercise
says "all five models" (B4).

---

## Beginner

**B1 — Build the two new estimators.**
Call `get_svm()` and `get_decision_tree()` with no arguments and print the
objects. What kernel and what `C` does the SVM default to? What `max_depth` and
`criterion` does the tree default to? Confirm neither is fitted yet by checking
that calling `.predict(X)` on the bare estimator raises.

**B2 — Reject the nonsense.**
Show that each of these raises `ValueError`, and read the message:
`get_svm(C=0)`, `get_svm(kernel="quadratic")`, `get_svm(gamma=-1)`,
`get_decision_tree(max_depth=0)`, `get_decision_tree(criterion="mse")`. Why is
raising here, rather than inside a later `fit`, the friendlier design?

**B3 — Extend the results table.**
Cross-validate `get_svm()` and `get_decision_tree()` through `make_pipeline`, and
add them to your Week 5 table. Do your numbers match [the learning notes
§7](learning_notes.md)? You should see the tree at about **0.9852 ±0.0068** and
the RBF SVM at about **0.9790 ±0.0103**. If they differ, check the seed, the fold
count and whether you used the preprocessor.

**B4 — Predict, then check: who leads?**
Before running anything, write down which of the five models you expect to have
the highest cross-validated accuracy. Now cross-validate all five and rank them.
Which model leads, and by how much over the second? (Expected: Gaussian naive
Bayes, 99.49%, ahead of the unlimited tree's 98.52%.) Was your guess right?

**B5 — The unlimited tree memorises.**
Fit `get_decision_tree()` on `X, y`, then score it on `X, y` itself with
`evaluate_model`. What accuracy do you get, and why is it guaranteed? Now read
`fitted.get_depth()` and `fitted.get_n_leaves()` off the fitted tree (reach it
through the pipeline's `named_steps["model"]`). Do they match the notes' depth 17
and 38 leaves? Which of these three numbers is a result, and which two are just
the algorithm's stopping rule?

**B6 — Print the first questions.**
Fit a tree pipeline, pull out the fitted tree, and print its rules with
`sklearn.tree.export_text(fitted, feature_names=FEATURES)`. What are the first
two questions it asks? (Expected: `rainfall <= 30.18`, then `humidity <=
27.98`.) Why does that make `humidity` and `rainfall` the natural pair for a
boundary plot?

**B7 — Gini versus entropy.**
Cross-validate `get_decision_tree(criterion="gini")` and
`get_decision_tree(criterion="entropy")`. How far apart are the two scores
relative to their fold spreads? Fit both on the full data and compare the first
few splits from `export_text`. Do the criteria disagree about *which* questions
to ask on this dataset?

**B8 — Trees ignore scaling.**
Fit a tree **without** the preprocessor (just `get_decision_tree()` on `X, y`),
and separately fit one on a hand-standardised copy of `X`
(`(X - X.mean()) / X.std()`). Compare their predictions on `X` row by row. Are
they identical? Explain the result in one sentence in terms of what a threshold
`feature <= t` actually selects. Which model from Week 5 would *not* survive this
test, and why?

**B9 — Count the support vectors.**
Fit `get_svm()` through `make_pipeline`, reach the fitted `SVC` via
`named_steps["model"]`, and print `len(model.support_)`. How many of the 1,760
rows are support vectors? (Expected: 943, about 54%.) Now do the same for
`get_svm(kernel="linear")` — is the count higher or lower, and what does that say
about which boundary is simpler? (Expected: 615.)

**B10 — Linear versus RBF.**
Cross-validate `get_svm(kernel="rbf")` and `get_svm(kernel="linear")`. Which
scores higher? (Expected: linear 0.9818 ±0.0077, rbf 0.9790 ±0.0103.) Is the
difference larger or smaller than the fold spread of either? Write one sentence
explaining why the flatter model is not beaten here, referring to the shape of
the crop blobs from [Week 2](../week02/learning_notes.md).

---

## Intermediate

**I1 — Draw the three boundaries.**
Choose `humidity` and `rainfall`. Fit three models — logistic regression, an RBF
SVM, and an unlimited tree — on **those two columns only** (remember
`build_preprocessor(["humidity", "rainfall"])` for the ones that need it), and
draw each with `plot_decision_boundary`. Which model gives straight seams, which
gives curves, and which gives axis-aligned boxes? Why is the printed accuracy of
each panel lower than its row in the §7 table, and why is quoting it forbidden?

**I2 — Break the boundary helper on purpose.**
Trigger each of `plot_decision_boundary`'s four `ValueError`s in turn: pass three
feature columns, pass a `y` of the wrong length, pass `resolution=1`, and pass
`padding=-0.1`. Read each message. Then pass a model that was fitted on the
*seven* features and a two-column `X_2d`: what goes wrong, and why is the
"fitted on exactly the two plotted columns" rule an honesty rule rather than a
technicality?

**I3 — Sweep `C` and watch the margin.**
Cross-validate `get_svm(C=c)` for `c` in `[0.01, 0.1, 1, 10, 100]`, and for each
also fit on the full data and record `len(model.support_)`. Tabulate `C`,
support-vector count and CV accuracy. Do you reproduce the notes' shape — 1,760
support vectors and 87.2% at `C = 0.01`, falling to about 610 and ~98% by
`C = 100`? Which end is underfitting? Why is picking the best `C` from this table
not yet a legitimate tuning procedure?

**I4 — Sweep the depth and plot two curves.**
For `max_depth` in `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, None]`, fit a tree on
the full training set and record its **training** accuracy (`evaluate_model` on
`X, y`) and its **validation** accuracy (`cross_validated_accuracy`). Plot both
against depth. Mark where the two curves sit on top of each other (high bias),
where they climb together, and where the training curve reaches 100% while
validation flattens near 98.5% (high variance). At which depth is validation
*above* training, and why is that a sign of noise rather than of a real gap?

**I5 — Name the tradeoff from your own plot.**
Using the plot from I4, write two short paragraphs a beginner could follow: one
defining **bias** using the shallow end, one defining **variance** using the deep
end. Then note the honest caveat from [§5](learning_notes.md): on this unusually
clean dataset the validation curve *flattens* rather than turning downwards.
Sketch (in words) what the same plot would look like on noisy data.

**I6 — A staircase for a diagonal.**
On the two-feature `humidity`/`rainfall` plane, fit trees at `max_depth` of 2, 5
and `None`, and draw each with `plot_decision_boundary`. Watch the boxes multiply
as depth grows. Explain, in terms of the `feature <= threshold` question shape,
why a tree can only approximate a diagonal boundary with a staircase — and why
that connects to overfitting.

**I7 — Where do the mistakes go?**
Using `sklearn.model_selection.cross_val_predict` with the same `cv=build_cv()`
splitter that `cross_validated_accuracy` uses, collect out-of-fold predictions
for the tree and the RBF SVM. List the rows each gets wrong. Do the two models
confuse the *same* pairs of crops, or different ones? (You are not asked for
precision or recall — that is Week 8; just count and compare.)

**I8 — The kernel in one sentence.**
Without using the word "space", write one sentence explaining what an SVM kernel
does. Then write a second sentence naming when you would reach for `rbf` over
`linear`, and a third saying why that reason does *not* apply to this dataset.
(This is a writing exercise; check yourself against [§3](learning_notes.md).)

---

## Challenge

**C1 — Is the tree-versus-SVM gap real?**
The tree (98.52%) leads the RBF SVM (97.90%) by 0.62 points, with fold spreads of
about 0.68 and 1.03. Repeat the comparison across ten cross-validation seeds
(`cross_validated_accuracy(..., random_state=seed)`), collect the ten means per
model, and plot the two distributions. Does the tree lead on all ten? Write the
two-sentence conclusion you would defend — and name the parts of your procedure
that a proper repeated-CV or paired-test protocol (a later week) would make
rigorous.

**C2 — A shape the flat boundary cannot cut.**
Generate a two-class problem where one class rings the other
(`sklearn.datasets.make_circles`). Cross-validate a linear SVM, an RBF SVM and a
decision tree on it (pass the new column list to the preprocessor). Which model
fails, and why is its failure a statement about the *shape* of the boundary
rather than the amount of data? Relate the winners to what you saw on the crop
blobs, where the ranking was the other way round.

**C3 — Turn `gamma` up until it overfits.**
Fix the RBF kernel and `C = 1`, and cross-validate `get_svm(gamma=g)` for a range
of `g` from very small to very large, alongside the support-vector count from a
full-data fit. Find the region where validation accuracy falls while the model
grows more support vectors. Explain, using the "bubble" picture from
[§3](learning_notes.md), why large `gamma` overfits — and why choosing `g` off
this curve is the same forbidden move as choosing `C` or `max_depth` off a sweep.

**C4 — Cost of the probability calibration.**
Time `fit` for `get_svm(probability=False)` and `get_svm(probability=True)` on
the full training set (wrap each in `make_pipeline`). How much slower is the
calibrated version, and what is scikit-learn doing in the extra time? Then check
whether `predict` and the arg-max of `predict_proba` ever disagree on the
training rows. Why is `probability=False` the right default for this project?

**C5 — Read the tree as prose.**
Fit a tree at `max_depth=4` and print it with `export_text`. By hand, translate
the path to one leaf into an English sentence a farmer could check ("if rainfall
is under … and humidity is under … then …"). Then fit the unlimited tree and try
the same for one of its 38 leaves. Which tree's rules are actually usable as
explanations, and how does that preview Week 7's use of trees for
explainability?

**C6 — Add a sixth model, honestly.**
Pick an algorithm not covered this week — for example `LinearSVC` or
`ExtraTreeClassifier` — and add it to the comparison **without** changing the
protocol: same folds, same preprocessor, same metric, appended to the same
table. Where does it land relative to the five? Then write the factory function
you would add to `src/models/classical_models.py` for it, with the docstring it
would need to match the module's style and the argument validation the other
factories perform. (Do not commit it; which models the project actually adopts is
a later week's decision.)

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| ◀ [§6.2 Learning notes](learning_notes.md) | [Chapter 6 — Margin-based and Tree-based Models](README.md) · 🗺 [Roadmap](../README.md) | [§6.4 Validation](validation.md) ▶ |

<!-- nav:end -->
