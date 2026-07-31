# Week 8 — Exercises

> 🗺 [Roadmap](../README.md) › [Part III — Evaluation and Explanation (Week 8)](../README.md#part-iii--evaluation-and-explanation-week-8) › [Chapter 8 — Model Evaluation & Explainability](README.md) › **§8.3 Exercises**

Work through these after reading [`learning_notes.md`](learning_notes.md) and
executing both notebooks. Nothing here needs a new dependency; everything runs
against the artefacts already in the repository.

Reminder before you start: **`data/processed/test.csv` was opened once, in
`notebooks/06_model_selection.ipynb`, after the model decisions were made.**
Several exercises below use it. That is fine for learning — but if you change a
model because of something you see there, the test measurement is spent and the
honest move is to say so in writing.

Most exercises start from the same lines, which rebuild the two finalists
`notebooks/06_model_selection.ipynb` §14 ended with:

```python
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data import DEFAULT_RANDOM_STATE, FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluation import (
    EXPLAINER_BACKEND,
    SHAP_AVAILABLE,
    confusion_frame,
    cross_validated_accuracy,
    evaluate_model,
    explain_prediction,
    permutation_feature_importance,
    tune_model,
)
from src.models import get_naive_bayes, get_random_forest
from src.preprocessing import build_preprocessor

FEATURES = list(FEATURE_COLUMNS)
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")
X_train, y_train = train[FEATURES], train[TARGET_COLUMN]
X_test, y_test = test[FEATURES], test[TARGET_COLUMN]


def make_pipeline(model):
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


# The chosen model, and the runner-up with the settings the grid search picked.
final_model = make_pipeline(get_naive_bayes()).fit(X_train, y_train)
tuned_forest = make_pipeline(
    get_random_forest(n_estimators=100, max_depth=10)
).fit(X_train, y_train)
background = X_train.sample(100, random_state=DEFAULT_RANDOM_STATE)
```

`evaluate_model(model, X, y)` returns `"accuracy"`, `"macro_f1"`,
`"weighted_f1"`, `"report"` and `"confusion_matrix"` — everything Exercises 1
and 2 need — and `print(SHAP_AVAILABLE, EXPLAINER_BACKEND)` says which explainer
your environment will use before you quote any contribution from it.

---

## Exercise 1 — Read the matrix out loud

Open `notebooks/06_model_selection.ipynb` (§12-§13) and find the tuned forest's
confusion matrix.

1. Locate every non-zero off-diagonal cell. There are two. For each one, write
   the sentence "`N` field(s) whose true crop was ___ were predicted as ___".
2. Sum the diagonal. Divide by 440. Does it match the reported accuracy?
3. For the `rice` row: how many rice fields were there, how many were found, and
   what is rice's **recall**?
4. For the `jute` column: how many predictions of jute were made, how many were
   really jute, and what is jute's **precision**?
5. Confirm both numbers against the `classification_report` printed in the
   notebook.

*The trap:* if you computed rice's recall by reading down the rice column, you
computed precision. Rows are recall.

---

## Exercise 2 — Macro and weighted by hand

The classification report prints 22 per-class F1 scores plus both averages.

1. Copy the 22 per-class F1 values into a list and take the plain mean. It should
   equal the reported macro F1 (0.9955 for the tuned forest).
2. Now take the mean weighted by support. Every support is 20, so predict the
   answer before computing it.
3. Delete 18 of the 20 `apple` rows from a *copy* of the test set, re-score, and
   print both averages again. Which moved, and in which direction?
4. Construct — on paper is fine — a 3-class example where weighted F1 exceeds
   0.95 while macro F1 is below 0.70. What does that model do?

---

## Exercise 3 — Cost of a search

Using `tune_model()` from `src/evaluation/tuning.py`:

1. Without running anything, predict `n_candidates` and `n_fits` for
   `{"model__n_estimators": [50, 100, 200], "model__max_depth": [None, 10, 20, 30]}`
   at the default 5 folds.
2. Run it on a small subsample of the training data and check both numbers in the
   returned dict.
3. Add a third key with 5 values. Predict the new cost, then confirm.
4. Now run the same space with `search="random", n_iter=10`. How many fits? What
   does `tune_model()` do if you ask for `n_iter=1000` on a 60-candidate grid?
   Read the source and explain why that behaviour is the right one.

---

## Exercise 4 — Does tuning actually help?

1. Cross-validate an untuned `get_random_forest()` pipeline on the training set
   with `cross_validated_accuracy`. Record the mean and the standard deviation.
2. Run the notebook's grid search. Record `best_score` and `best_std`.
3. Compute `best_score − untuned_mean`. Compare it to `best_std`.
4. Write one sentence, for a stakeholder, on whether the tuning was worth the 120
   model fits. Then write the sentence you would have written if you had only
   been shown the two mean scores.

---

## Exercise 5 — A hyperparameter that does nothing

The notebook searches `var_smoothing` over twelve values from 1e-11 to 1e-6 and
gets one distinct score.

1. Reproduce it, and print `cv_results["mean_test_score"].nunique()`.
2. Extend the grid upward: `1e-3`, `1e-1`, `1.0`, `10.0`. At what magnitude does
   the score finally move, and which direction does it move in?
3. Explain, in terms of what `var_smoothing` actually does, why the *small*
   values are indistinguishable but the large ones are not. What quantity does
   the smoothing floor have to exceed before it binds?
4. Does this experiment change the Week 8 decision to ship an untuned naive
   Bayes? Justify either answer.

---

## Exercise 6 — Anatomy of the two errors

For each of the two misclassified test rows (the `rice -> jute` field and the
`blackgram -> maize` field):

1. Print the row's seven measurements.
2. Print the training-set mean of each feature for the true crop and for the
   predicted crop.
3. Identify the features on which the two crops actually differ, and where this
   field sits between them.
4. Print `predict_proba` for the row and find the true crop's rank and
   probability.
5. Would a rule of "escalate to a human when the runner-up exceeds 15%" have
   caught it? How many *correct* predictions would the same rule have escalated
   unnecessarily across all 440 rows? Is that trade acceptable?

---

## Exercise 7 — MDI against permutation

1. Print the tuned forest's `feature_importances_` on the training data and its
   permutation importances on the test set, side by side, both sorted.
2. Two features change rank between the lists. Name them and explain the
   direction of the change.
3. `temperature` and `ph` get 0.074 and 0.051 from MDI, but breaking them costs
   under 0.01 accuracy. Which of Week 7's three stated limitations of MDI does
   that illustrate?
4. Now try `permutation_feature_importance()` on the **training** set instead.
   Do the numbers go up or down, and why is the held-out version the one to
   quote?

---

## Exercise 8 — Falling into the correlation trap deliberately

Week 2 measured a correlation of 0.74 between `P` and `K`.

1. Reproduce the joint-shuffle table from `notebooks/07_model_explainability.ipynb` (§2):
   baseline, `P` alone, `K` alone, `P` and `K` together.
2. Do the same for `temperature` and `ph`, which are not correlated. Compare the
   "together vs sum of parts" gap in the two cases.
3. A colleague proposes dropping `P` because its individual permutation
   importance is the second-lowest of the nutrients. Write the two-sentence
   reply, with the number that settles it.
4. Design one further experiment that would distinguish "P is redundant given K"
   from "P is genuinely unimportant". What result would you expect from each?

---

## Exercise 9 — Explain a prediction to a farmer

Pick any **correctly** classified test row.

1. Call `explain_prediction(final_model, X_test.iloc[[i]], background=...)`.
2. Write the explanation as a short paragraph aimed at someone with no ML
   background. It must name the recommended crop, the confidence, the runner-up
   and its probability, and the deciding measurement with the value that made it
   decisive.
3. Check your claim: is the deciding feature's value actually close to that
   crop's training mean and far from the runner-up's?
4. Repeat for a row where the top probability is below 0.9 — use
   `predict_proba(...).max(axis=1)` to find one. Which explanation is more
   useful, and why?

---

## Exercise 10 — Live without SHAP

`explain_prediction()` takes an explicit backend.

1. Explain the same row twice, once with `method="shap"` and once with
   `method="permutation"`. Print both contribution series.
2. Do they agree on the top feature? Do they agree on the ordering below it?
3. Explain why the two are not on the same scale, and why the SHAP numbers sum
   to something meaningful while the fallback's do not.
4. Now hide the library from the interpreter and confirm the code still works:

   ```bash
   mkdir -p /tmp/noshap && printf 'raise ImportError("hidden")\n' > /tmp/noshap/shap.py
   PYTHONPATH=/tmp/noshap pytest tests/test_explainability.py
   PYTHONPATH=/tmp/noshap python -c "from src.evaluation import SHAP_AVAILABLE, EXPLAINER_BACKEND; print(SHAP_AVAILABLE, EXPLAINER_BACKEND)"
   ```

   What does the test count do, and why is one test skipped in exactly one of the
   two configurations?

---

## Exercise 11 — Defend the decision

Week 8 ships Gaussian naive Bayes over a tuned random forest that scored
identically.

1. List the four criteria that broke the tie, in the order you would present them
   to a sceptical reviewer.
2. Write the strongest available argument for the **forest** instead. What would
   have to be true about future data for that argument to win?
3. Naive Bayes assumes the features are conditionally independent given the crop.
   Week 2 measured `P` and `K` at 0.74 correlation, so the assumption is
   violated. Why does the model work anyway? (Consider what the classifier needs
   to get right: the exact probabilities, or their ranking.)
4. State one measurable condition on future data that should trigger revisiting
   this decision.

---

## Exercise 12 — Extend the tooling

Choose one and implement it, with a test:

* Add a `scoring="f1_macro"` search to the notebook and compare its winner to the
  accuracy-scored one. Does optimising macro F1 pick a different model here?
* Extend `confusion_frame()` with a `normalize` option ("true", "pred", None) and
  say which normalisation answers "what fraction of rice fields did we find?".
* Add a `top_k` argument to `explain_prediction()` that trims the contributions
  series, and make sure the returned `top_feature` is unaffected.
* Write a `misclassified_rows(model, X, y)` helper returning a DataFrame of every
  wrong prediction with its true label, predicted label and top-2 probabilities —
  the thing Exercise 6 had to do by hand.

---

## Checkpoint

You are ready for Week 9 when you can, without looking anything up:

* read any cell of a confusion matrix in both directions;
* say why macro and weighted F1 are identical on this test set and when they
  would not be;
* explain why the cross-validation loop sits *inside* the hyperparameter search;
* say why the tuned forest's 99.43% is optimistic and its 99.55% is not;
* name the final model and give three reasons that are not accuracy;
* explain one prediction end to end, including the runner-up;
* state the correlation trap with the P/K numbers that demonstrate it;
* say which explainer produced any number you quote.

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| ◀ [§8.2 Learning notes](learning_notes.md) | [Chapter 8 — Model Evaluation & Explainability](README.md) · 🗺 [Roadmap](../README.md) | [§8.4 Validation](validation.md) ▶ |

<!-- nav:end -->
