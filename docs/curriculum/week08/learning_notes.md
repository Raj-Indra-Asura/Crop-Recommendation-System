# Week 8 — Learning Notes

Everything this week is about the same shift: **stop asking "how accurate is
it?" and start asking "where is it wrong, why did it say that, and which model
do we ship?"**

Accuracy has been the only number in the project since Week 4, and it has run out
of resolution. Week 7's table topped out at 99.49%; this week's held-out test
score is 99.55%, which is 438 correct out of 440. Two rows. Everything
interesting is in those two rows, in the search that did not improve anything,
and in the machinery for explaining a single recommendation to the person who has
to act on it.

> **SHAP availability, stated up front.** This week's optional dependency is
> `shap==0.46.0`. **It installed cleanly against the pinned
> numpy 2.2.1 / pandas 2.2.3 / scikit-learn 1.6.1, and SHAP is what
> `notebooks/07_model_explainability.ipynb` actually used** — every explanation
> in it is labelled `method: shap`. The documented fallback (§10) is implemented,
> tested and demonstrated anyway, because an optional dependency that has never
> been run without is not really optional.

---

## 1. Why accuracy stopped being enough

Accuracy is one number: correct predictions divided by all predictions. It is a
fair headline here — 22 balanced classes, no mistake obviously costlier than
another — and it is still incapable of answering any of these:

* Which crops does the model confuse, and with which?
* When it says "jute", how often is that right?
* Of the fields that really were rice, how many did it find?
* Is the error spread thinly over all 22 crops, or concentrated in one pair?

Two models can share an accuracy of 99.55% and fail completely differently. On
this test set they do: the tuned forest makes one `rice -> jute` error and one
`blackgram -> maize` error; Gaussian naive Bayes makes two `rice -> jute` errors.
Same headline, different failure mode, and the difference matters when choosing
which to deploy.

---

## 2. The confusion matrix

A confusion matrix is a table with one row per **true** class and one column per
**predicted** class. Cell `(i, j)` counts the examples whose true class is `i`
and which the model called `j`.

* The **diagonal** — where `i == j` — is the correct answers.
* Every **off-diagonal** cell is one specific, named mistake.

A tiny three-crop example, to fix the reading direction:

|  | pred rice | pred jute | pred maize |
| --- | --- | --- | --- |
| **true rice** | 18 | 2 | 0 |
| **true jute** | 0 | 20 | 0 |
| **true maize** | 0 | 0 | 20 |

Read the **row** to ask "what happened to the fields that really were rice?" —
18 correct, 2 sent to jute. Read the **column** to ask "when the model said jute,
what were those fields really?" — 20 real jute plus 2 rice. Rows are about
recall, columns are about precision, and confusing the two directions is the
single most common mistake when reading these tables.

With 22 classes the matrix is 22x22 = 484 cells, of which 462 are potential
mistakes. `confusion_frame()` in `src/evaluation/metrics.py` attaches the crop
names to both axes, because the raw NumPy array is unreadable:

```python
from src.evaluation import confusion_frame

matrix = confusion_frame(y_test, model.predict(X_test))
```

`evaluate_model()` now returns it under `"confusion_matrix"` alongside accuracy,
so no score in this project can be quoted without the breakdown being one key
away.

### What the real matrix looks like

Nearly all 462 off-diagonal cells are zero. The tuned forest fills exactly two of
them with a 1, naive Bayes exactly one with a 2. **That concentration is itself
the finding.** Errors scattered across many cells would suggest a model that has
not learned the structure; errors piled into one or two cells mean the model has
learned everything except the places where the classes genuinely overlap.

---

## 3. Precision, recall and F1

For one class — say jute — the four possibilities are:

|  | model says jute | model says something else |
| --- | --- | --- |
| **really jute** | true positive (TP) | false negative (FN) |
| **really not jute** | false positive (FP) | true negative (TN) |

* **Precision** = TP / (TP + FP) — *when the model says jute, how often is it
  right?* It is damaged by false alarms.
* **Recall** = TP / (TP + FN) — *of the fields that really are jute, how many
  does it find?* It is damaged by misses.
* **F1** = 2 · (precision · recall) / (precision + recall), the harmonic mean.
  The harmonic mean is used rather than the arithmetic one because it punishes
  imbalance: precision 1.0 with recall 0.0 gives F1 0.0, not 0.5.
* **Support** = how many rows of that class exist. 20 for every crop here.

The tension is real and it has a direction in this project. A model that
recommends jute for every ambiguous field has high jute *recall* and poor jute
*precision*, and the cost lands on farmers who planted jute on the advice.
The classification report on the tuned forest shows exactly this shape:

```
       jute       0.95      1.00      0.98        20
       rice       1.00      0.95      0.97        20
```

Jute has perfect recall (all 20 real jute fields found) and 0.95 precision (it
was also handed one rice field). Rice is the mirror image: perfect precision, one
field lost. Two numbers, one error, read from both ends.

---

## 4. Macro vs weighted averaging

Twenty-two per-class F1 scores are not a headline. Two ways to average them:

* **Macro F1** — the plain mean. Every class counts the same, however rare.
* **Weighted F1** — the mean weighted by support. Common classes dominate.

They are equal when every class has the same number of rows, and they diverge
sharply when it does not. The standard cautionary example: 990 rows of class A,
10 of class B, a model that always predicts A. Weighted F1 ≈ 0.99, macro F1 ≈
0.50. Quoting the weighted number alone would hide the total failure on B.

On this project's test set every crop has exactly 20 rows, because Week 3's
split was stratified. So:

| | tuned random forest | Gaussian naive Bayes |
| --- | --- | --- |
| accuracy | 0.9955 | 0.9955 |
| macro F1 | 0.9955 | 0.9954 |
| weighted F1 | 0.9955 | 0.9954 |

The two averages agree to four decimals, and the residual difference comes only
from *which* classes the handful of errors landed in. **Which to report?** Macro,
by default, because it is the one that would have caught a problem if there had
been one — and both, when they differ, because the gap between them *is* the
class-imbalance story.

Rule of thumb: report macro when every class matters equally (all 22 crops matter
to the farmer growing them); report weighted when you care about aggregate
performance over the population you will actually see.

---

## 5. Hyperparameters, and searching for them honestly

A **parameter** is learned from the data during `fit` — a tree's split
thresholds, naive Bayes' per-class means and variances. A **hyperparameter** is
chosen *before* fitting and controls how the fitting happens: `max_depth`,
`n_estimators`, `max_features`, `var_smoothing`.

Nothing in the training loop chooses a hyperparameter, so something else must.
The tempting options are both wrong:

* **"Whatever scores best on the test set."** A test set consulted while choosing
  is a training set with a misleading name, and every number derived from it
  afterwards is optimistic.
* **"Whatever looks best on the sweep I plotted."** Weeks 6 and 7 both plotted
  sweeps and both refused to adopt anything from them, for this reason: staring
  at validation scores and picking the peak is fitting to the validation set by
  hand.

The honest procedure puts cross-validation **inside** the search:

```
for each candidate setting:
    for each of the 5 stratified folds:
        fit on 4 folds, score on the 1 held out
    candidate score = mean of the 5 held-out scores
winner = the candidate with the best mean
refit the winner on all the training rows
```

Every candidate is scored only on rows it did not see, the training set is the
only data involved, and the test set stays sealed. `tune_model()` in
`src/evaluation/tuning.py` is that loop with the project's splitter and seed
already wired in:

```python
from src.evaluation import tune_model

result = tune_model(pipeline, {"model__max_depth": [None, 10, 20]}, X_train, y_train)
result["best_params"]     # the winning settings
result["best_score"]      # its mean over the 5 held-out folds
result["cv_results"]      # every candidate, sorted best first
result["n_fits"]          # candidates x folds - the real cost
```

### Grid vs randomised search

| | `GridSearchCV` (`search="grid"`) | `RandomizedSearchCV` (`search="random"`) |
| --- | --- | --- |
| Candidates | every combination | `n_iter` sampled at random |
| Cost | product of the list lengths | whatever you choose |
| Good for | small spaces, exhaustive claims | large spaces, continuous values |

The argument for randomised search is not that random beats exhaustive. It is
that in most spaces only two or three hyperparameters matter, and a grid spends
its budget re-testing the irrelevant ones at every level of the relevant ones.
Twenty random draws visit twenty *distinct* values of each important setting; a
budget-matched grid visits two or three.

Measured in the notebook: 20 draws from a 300-candidate space (100 fits instead
of 1,500) reached **exactly the same 99.43%** as the exhaustive search over the
smaller 24-candidate grid, with completely different settings.

### The winner's score is optimistic

Take the maximum of two dozen noisy numbers and part of that maximum is noise.
The more candidates tried, the larger the effect — this is the same selection
bias that makes a heavily-searched model look better than it is.

So the tuned forest's 99.43% is "the best of 24 candidates on these folds", not
"the accuracy of this model". The number worth quoting is the one measured
afterwards on data the search never saw. The gap in this project:

| Measurement | Score |
| --- | --- |
| Untuned forest, 5-fold CV on train (Week 7) | 99.26% (±0.58) |
| Tuned forest, best of 24 candidates | 99.43% (±0.60) |
| Tuned forest, held-out test set | 99.55% |

The tuning "gained" 0.17 points against a fold-to-fold standard deviation of 0.60
— **three times larger than the gain**. The correct reading is not "tuning
improved the forest" but "tuning found a candidate that is indistinguishable
from the default, and the search told us so".

### A search that changes nothing is a result

Gaussian naive Bayes has one hyperparameter, `var_smoothing`: a floor added to
every feature's variance so an almost-constant feature cannot produce an
infinitely confident prediction. Twelve values from 1e-11 to 1e-6, cross-
validated, produce **one distinct score**: 0.994886 every time.

That is not a failed experiment. It says the smoothing floor is never the binding
constraint on this data — the smallest real feature variance is orders of
magnitude larger — so the model is insensitive to the setting. Consequences:
nobody has to defend the choice, nobody may later claim a "tuned" naive Bayes,
and the leader of the table cannot be improved by tuning at all.

---

## 6. Choosing a final model

Two finalists, tied at 99.55% on the held-out set. When accuracy stops
discriminating, the decision is made on the criteria that were always there:

| Criterion | Why it matters |
| --- | --- |
| **Accuracy / macro F1** | The floor. Both clear it identically here. |
| **Error pattern** | *Which* mistakes, not how many. One kind of confusion is easier to document, monitor and warn about than two. |
| **Interpretability** | Can a prediction be explained to the person acting on it, and can the model be inspected without extra tooling? |
| **Training cost** | Refitting on new data, and re-running any search that came with it. |
| **Serving cost** | What the deployed system pays per prediction. |
| **Tuning risk** | Every setting chosen by search is a decision that may not transfer to fresh data. A model with nothing to tune has none. |

Applied here:

| | Tuned random forest | Gaussian naive Bayes |
| --- | --- | --- |
| Test accuracy | 99.55% | 99.55% |
| Macro F1 | 0.9955 | 0.9954 |
| Stored state | 100 trees, thousands of nodes | 308 numbers (22 crops x 7 features x mean+variance) |
| Fit time | ~0.3 s | ~0.008 s |
| Interpretability | needs SHAP or permutation importance | a mean and a variance per crop per feature, readable directly |
| Errors | `rice -> jute`, `blackgram -> maize` | `rice -> jute` x 2 |
| Tuning risk | 24 candidates tried, winner inside the noise | nothing to tune |

> **Decision: Gaussian naive Bayes is the final model.** It ties on every
> accuracy measure, costs ~40x less to fit and far less to serve, stores 308
> numbers instead of a hundred trees, has no hyperparameter that changes its
> behaviour on this data, and its errors are one kind rather than two. The tuned
> random forest is the recorded runner-up, to be revisited if the data ever grows
> features with strong interactions or non-Gaussian shapes — the conditions under
> which naive Bayes' independence assumption fails.

This is a claim about *this dataset*, not a general ranking. Week 2 showed the 22
crops sitting in compact, well-separated, roughly bell-shaped blobs, which is
precisely the shape naive Bayes assumes. On a messier problem the same reasoning
would land somewhere else.

---

## 7. Error analysis: the interesting content

Both finalists get 438 of 440 right, so this section is 0.45% of the test set and
most of the week's value.

### rice -> jute (both models)

Training-set averages:

| | N | P | K | temperature | humidity | ph | rainfall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **jute** | 78.1 | 47.1 | 39.9 | 24.9 | 79.1 | 6.7 | **175.8** |
| **rice** | 79.8 | 47.0 | 39.8 | 23.9 | 82.2 | 6.4 | **237.2** |
| *the misclassified field* | 67 | 43 | 39 | 26.0 | 85.0 | 6.0 | **186.8** |

Six of the seven features are effectively identical between the two crops. The
only separator is **rainfall**, and this field recorded 186.8 mm — an entirely
ordinary jute reading and an unusually dry one for rice. The model answered with
the more likely crop for that reading. A human agronomist given only these seven
numbers would have said the same thing.

### blackgram -> maize (the forest only)

| | N | P | K | temperature | humidity | ph | rainfall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **blackgram** | 39.6 | 67.7 | 19.2 | 29.8 | 65.0 | 7.1 | 67.7 |
| **maize** | 77.0 | 48.5 | 19.9 | 22.5 | 64.5 | 6.2 | 85.6 |
| *the misclassified field* | 60 | 59 | 22 | 31.9 | 66.7 | 7.2 | 74.2 |

Same story, different features: the two crops share potassium and humidity and
are separated by nitrogen and phosphorus, and this field sits between the two
profiles on both. The forest gave maize 0.57 with blackgram second at 0.25 — it
was visibly unsure rather than confidently wrong.

### Three conclusions

1. **Errors are concentrated, not scattered.** 462 possible confusion cells, all
   the mistakes in one or two of them. That is genuine class overlap, not a
   broken model.
2. **The model reports its own uncertainty.** Both errors left the true crop as a
   strong runner-up (0.32 and 0.25 for the forest, 0.16 and 0.26 for naive
   Bayes). A deployment rule of "route anything whose runner-up exceeds 15% to a
   human" would have caught every error on this test set.
3. **Error pattern is a selection criterion.** It is the tie-breaker used in §6,
   and it is only available because the confusion matrix was read.

---

## 8. Permutation importance

Week 7's `feature_importances_` (mean decrease in impurity) had three problems:
it is measured on the training data, it describes split bookkeeping rather than
reliance, and most models do not have it at all — including `GaussianNB`, the
model this project chose.

Permutation importance asks a blunter question: **break one column and see what
it costs.**

```
score the fitted model on held-out rows              -> baseline
for each feature:
    shuffle that column (values kept, alignment destroyed)
    score again
    importance = baseline - shuffled score
repeat n_repeats times, report mean and spread
```

Nothing is refitted. The model is unchanged; only its input is corrupted. Three
things follow, and they are the three reasons it is more trustworthy:

1. **It is computed on whatever data you hand it.** Give it the test set and the
   number describes generalisation, not memorisation.
2. **Its units are meaningful.** "0.317 for `humidity`" means this model loses
   31.7 percentage points of accuracy — 99.5% down to about 68% — when that
   column is scrambled. MDI's "0.219" is a share of impurity removed, which is
   not a quantity anyone can act on.
3. **It works on any fitted estimator**, because it only needs `predict`. This is
   what lets the chosen naive Bayes model be explained at all.

It also comes with a spread. Ten shuffles give a standard deviation per feature,
and two features whose error bars overlap are not distinguishable.

```python
from src.evaluation import permutation_feature_importance

permutation_feature_importance(model, X_test, y_test, n_repeats=10)
```

### The two methods disagree, and that is the point

For the tuned forest:

| Feature | Built-in MDI (train) | Permutation (test) |
| --- | --- | --- |
| humidity | 0.219 | **0.317** |
| N | 0.104 | **0.202** |
| rainfall | 0.216 | 0.169 |
| K | 0.190 | 0.148 |
| P | 0.146 | 0.111 |
| temperature | 0.074 | 0.007 |
| ph | 0.051 | 0.005 |

MDI ranks `rainfall` second and `N` fifth; held-out permutation reverses them.
And MDI credits `temperature` and `ph` with 0.074 and 0.051 where breaking them
costs almost nothing — splits that happened without helping, which is limitation
1 and 2 of MDI showing up in one table.

### The correlation trap

The one failure mode permutation importance keeps, and this dataset walks into
it. Week 2 measured a **0.74 correlation between `P` and `K`**. Shuffle `P` alone
and the model still has `K`, which carries much of the same information, so `P`
looks unimportant. Measured on the final model:

| Shuffled | Accuracy after | Cost |
| --- | --- | --- |
| nothing (baseline) | 0.9955 | — |
| `P` | 0.8164 | 0.179 |
| `K` | 0.5630 | 0.433 |
| **`P` and `K` together** | **0.4302** | **0.565** |
| `temperature` | 0.9105 | 0.085 |
| `ph` | 0.9475 | 0.048 |
| `temperature` and `ph` | 0.8502 | 0.145 |

The nutrients cost far more together than either alone; the uncorrelated control
pair costs about the sum of its parts. The general form: **a low permutation
importance means "the model can manage without this column *alone*", never "this
measurement does not matter".**

---

## 9. SHAP: explaining one prediction

Permutation importance is still a statement about a *set* of rows. SHAP
(SHapley Additive exPlanations) works one row at a time.

The idea is borrowed from cooperative game theory. Treat the prediction as a
payout and the features as players who produced it together. A feature's
**Shapley value** is its average marginal contribution across every order in
which the features could have been revealed — the unique attribution satisfying a
short list of fairness axioms. The practical consequence is **additivity**:

```
base value (the model's average output) + sum of the feature contributions
    = the model's output for this row
```

That is what permutation importance cannot offer. A SHAP explanation is a
complete account of one prediction, not a ranking of features.

The cost is compute — exact Shapley values need every subset of features — which
is why SHAP ships several explainers:

| Explainer | Works on | Speed |
| --- | --- | --- |
| `TreeExplainer` | tree models and ensembles | fast: exploits tree structure |
| `KernelExplainer` | anything with `predict_proba` | slow: approximates by sampling |

`explain_prediction()` tries the tree explainer first and falls back to the
kernel one, which is what the chosen naive Bayes pipeline needs. One detail worth
knowing: `TreeExplainer` explains the *model*, not the pipeline around it, so it
must be fed the scaled columns the forest was actually fitted on — the notebook
transforms the sample explicitly before handing it over.

### Reading the two plot types

* **Summary bar plot** — mean absolute contribution per feature, stacked over the
  22 crops. A ranking, like permutation importance, but built from per-row
  attributions.
* **Beeswarm** — one dot per row, horizontal position = that row's contribution
  to one class's score, colour = the feature's value. This is where SHAP earns
  its cost, because it shows **direction**: on the `rainfall` row of the rice
  beeswarm, high values (red) sit to the right and low values (blue) to the left,
  which is the model saying *rice wants water* in a form you can check against
  agronomy.

---

## 10. `explain_prediction()`, and the documented fallback

```python
from src.evaluation import explain_prediction

result = explain_prediction(final_model, X_test.iloc[[378]], background=X_train.sample(100))
result["prediction"]      # 'jute'
result["probability"]     # 0.8354
result["probabilities"]   # every class, largest first: jute 0.84, rice 0.16, ...
result["contributions"]   # per-feature, largest absolute value first
result["method"]          # 'shap' or 'permutation' - always recorded
```

Three pieces come back **together**, because none of them is an explanation on
its own: what was predicted, how confident the model was and what the runners-up
were, and which measurements drove it.

### The fallback, exactly as specified

If `shap` does not import, `explain_prediction()` does **not** improvise. It uses
these two things and nothing else:

1. **Per-sample permutation.** Take the row; replace one feature at a time with
   values drawn from a background sample of training rows; measure how far the
   predicted class's probability moves. Reported as
   `p(real value) − mean p(perturbed)`, so a positive number means the measured
   value supports the prediction.
2. **The raw `predict_proba` breakdown** across all 22 crops for that row.

The two backends are not the same quantity. SHAP values are additive on the
model's output scale; the fallback's numbers are drops in predicted probability,
which rank features but do not sum to anything. On the worked example they agree
on the decisive feature and disagree below it:

| Feature | SHAP (used) | Permutation fallback |
| --- | --- | --- |
| rainfall | **0.332** | **0.605** |
| temperature | 0.098 | 0.275 |
| P | 0.097 | 0.249 |
| N | 0.120 | 0.239 |
| K | 0.122 | 0.197 |
| ph | -0.023 | 0.039 |
| humidity | 0.024 | -0.005 |

Neither is a bug. Replacing this field's temperature with a random training value
often lands it in a season no crop in this band grows in, which costs a lot of
probability — the fallback measures that. SHAP instead asks what temperature
contributed *on average over every revelation order*, and much of that credit is
assigned to the features temperature was standing in for.

**Which was used here: SHAP.** `shap==0.46.0` installed cleanly against the
pinned dependencies, `EXPLAINER_BACKEND` reports `"shap"`, and every explanation
in `notebooks/07_model_explainability.ipynb` carries `method: shap`. The fallback
is exercised in the same notebook with `method="permutation"`, and the whole test
suite passes with `shap` hidden from the interpreter.

---

## 11. One prediction, explained in plain language

The point of all of this is a sentence a non-specialist can act on. For test row
378:

> The model recommended **jute** with **84%** confidence, and put **rice** second
> at **16%**. The measurement that decided it was **rainfall = 186.75 mm**.
> Fields of rice in the training data average **237 mm**; fields of jute average
> **176 mm**. On every other measurement the two crops are nearly identical —
> nitrogen 80 vs 78, phosphorus 47 vs 47, potassium 40 vs 40, humidity 82 vs 79 —
> so rainfall carried almost the whole decision, and this field's rainfall reads
> like jute.
>
> The field really was rice. The model was not malfunctioning: it reported the
> most likely crop given a genuinely ambiguous reading, and it flagged its own
> doubt by leaving 16% on rice, far above the ~0% it gave the other twenty crops.

Three properties make that usable:

1. **It names the deciding measurement**, not the model. "Because the random
   forest said so" is not an explanation; "because 186.75 mm of rainfall is a
   jute reading" is.
2. **It quantifies the alternative.** "The model was unsure" becomes a number.
3. **It suggests an action.** Any recommendation whose runner-up exceeds ~15%
   could be reviewed by a human — a rule that would have caught both errors on
   this test set at the cost of reviewing a handful of predictions.

---

## 12. What this week does not license

* **No causal claims.** Every number describes the fitted model. "The model
  relies on rainfall" is not "rainfall causes rice". SHAP attributes a
  *prediction*, never an outcome in the world.
* **No second look at the test set.** It was opened once, after the decisions.
  Any further tuning has to go back to cross-validating the training rows, and a
  new test measurement would need data nobody has looked at.
* **No production pipeline.** `models/` is empty, nothing is serialised, and
  there is no `predict()` entry point outside a notebook. The chosen model exists
  as a fitted object inside a kernel that will be shut down. Week 9 fixes that.
* **No nested cross-validation.** The searches here have one inner loop; an
  unbiased estimate of the *tuning procedure* would need an outer one too.
* **No calibration.** "0.84 confidence" is the model's number, and this week
  never checked whether things it calls 84% likely happen 84% of the time.

---

## Recap

| Question | Answer |
| --- | --- |
| What does cell `(i, j)` of a confusion matrix count? | Rows whose true class is `i` and predicted class is `j`. |
| Rows or columns for recall? | Rows. Columns are precision. |
| Macro vs weighted F1? | Equal weight per class vs weight by support; equal here because every crop has 20 test rows. |
| Why cross-validate inside a search? | So candidates are ranked on held-out folds and the test set is never consulted. |
| Grid or randomised? | Grid for small spaces, randomised when the space is big or continuous — cost is chosen, not inherited. |
| Did tuning help? | +0.17 points against a ±0.60 fold spread: no. And naive Bayes has nothing to tune. |
| Final model? | Gaussian naive Bayes — tied at 99.55%, far cheaper, interpretable, nothing to tune, one kind of error. |
| Which crops get confused? | `rice -> jute` (rainfall 186.8 in the overlap band) and `blackgram -> maize` (N and P between the two profiles). |
| Why is permutation importance more trustworthy? | Held-out data, meaningful units, works on any model — including one with no `feature_importances_`. |
| Its trap? | Correlated columns: `P` 0.179, `K` 0.433, the pair 0.565. |
| What does SHAP add? | Per-row, signed, additive attributions — direction as well as magnitude. |
| SHAP or the fallback, here? | **SHAP** (`shap==0.46.0`, installed cleanly); the fallback is implemented, tested and demonstrated. |
