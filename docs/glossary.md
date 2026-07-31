# Glossary

Alphabetical quick reference for terms used across the project. For the
teaching order and fuller treatment, see
[`docs/ml_concepts.md`](ml_concepts.md) and the weekly learning notes.

Terms are added as they are introduced. The marker on each entry gives the week
it was introduced — *(W1)* Week 1, *(W2)* Week 2, and so on.

---

**422 vs 500** *(W10)* — *You sent something invalid* vs *we failed*. A 422 names
the offending field and is fixed by sending a different request; a 500 means the
request was fine, the traceback belongs in the server log and not in the
response body, and retrying identically will not help.

**Accuracy** *(W4)* — The share of predictions that were correct. A fair
headline metric when the classes are balanced and every mistake costs the same,
as here — but never a complete one, because it cannot show *which* classes fail.

**API** *(W10)* — Application Programming Interface: the set of things one piece
of software promises another it can do, plus the exact way to ask. `predict()`
is an API; a *web* API is the same promise made reachable over a network, so the
caller need not share a language, a library or a machine.

**Artificial intelligence** *(W1)* — The broad field of building software that
performs tasks we would call intelligent. Machine learning is one family of
techniques within it; deep learning is one family within machine learning.

**ASGI** *(W10)* — The asynchronous interface between a Python web framework and
the server that runs it. FastAPI builds an ASGI application object; `uvicorn` is
the process that listens on a port and calls it — hence `uvicorn api.main:app`.

**Axis-aligned split** *(W6)* — A decision tree's only kind of cut: a threshold
on one feature, so its regions are rectangles with horizontal and vertical
edges. A diagonal boundary can only be approximated by a staircase.

**Bagging (bootstrap aggregating)** *(W7)* — Fitting many copies of the same
kind of model in parallel, each on its own bootstrap sample of the rows, and
combining them by voting or averaging. It attacks *variance*: the members stay
individually overfit, but their independent mistakes largely cancel. A random
forest is bagging plus feature randomness.

**Base value** *(W8)* — In a SHAP explanation, the model's average output over
the background data: the number the contributions are added to. Base value plus
the seven feature contributions reproduces this row's prediction exactly, which
is what makes a SHAP explanation complete rather than merely a ranking.

**Baseline model** *(W4)* — A deliberately unintelligent model, fitted and
scored exactly like a real candidate but ignoring the features entirely. Its
score is the floor: a model that fails to beat it has learned nothing from the
features. This project's baseline is 4.55% (`1/22`).

**Batch learning** *(W1)* — Training a model once on a complete dataset to
produce a fixed artifact, which is then deployed and periodically retrained.
Contrast with online learning, where the model updates continuously as data
arrives.

**Bias (model bias)** *(W6)* — Error caused by a model being too simple to
represent the truth. A depth-1 decision tree has high bias: on this dataset it
scores 9.09% on the training rows and 9.09% on held-out rows alike. High bias
shows up as two curves that are low *and together*.

**Bias-variance tradeoff** *(W6)* — The observation that total error is roughly
bias plus variance, and that every dial controlling how flexible a model is —
`max_depth`, `min_samples_leaf`, `k`, `C` — trades one for the other. Week 6
plots it with tree depth.

**Bin** *(W2)* — One slice of a histogram's range. Too few bins hide structure
(separate peaks merge); too many turn every bar into noise.

**Boosting** *(W7)* — Fitting weak models one after another, each trained on
what the chain so far still gets wrong, and adding them up. It attacks *bias*
rather than variance, is sequential by construction, and can eventually overfit
if the chain runs too long.

**Bootstrap sample** *(W7)* — A draw of `n` rows *with replacement* from a
training set of `n` rows, so some rows appear several times and others not at
all. About 63% of the rows are distinct; here the first tree of a default forest
saw 1,114 of the 1,760 rows.

**Boxplot** *(W2)* — A plot summarising a distribution as a box spanning the
interquartile range, a line at the median, whiskers reaching 1.5 IQR beyond the
box, and every point past the whiskers drawn individually.

**Class balance** *(W2)* — How many rows each class has. This dataset is
perfectly balanced: 22 crops with exactly 100 rows each. Balance decides whether
accuracy is a fair summary metric.

**Class imbalance** *(W2)* — The opposite: classes with substantially different
row counts, where a model can score well by ignoring the rare classes and
accuracy stops being informative.

**Class separation (eta-squared)** *(W2)* — The share of a feature's total
variance that lies *between* classes rather than within them. 0 means the
classes are indistinguishable on that feature alone; 1 means it separates them
perfectly. Computed by `separation_scores()` in `src/utils/eda.py`.

**Classification** *(W1)* — A supervised learning task whose output is a
category rather than a number. Predicting which of 22 crops suits a field is
classification.

**Classification report** *(W4)* — scikit-learn's per-class table of precision,
recall, F1 and support, returned as a string by `evaluate_model()`. Shown from
Week 4 so that accuracy is never read alone; the metrics in it are taught in
Week 8.

**`ColumnTransformer`** *(W3)* — A scikit-learn estimator that maps sets of
columns to the transformers applied to them, as `(name, transformer, columns)`
triples, with `remainder` deciding whether unnamed columns are dropped or passed
through. Built here by `build_preprocessor()` in
`src/preprocessing/preprocessor.py`.

**Conditional independence assumption** *(W5)* — Naive Bayes' "naive" step:
that within a class, each feature is independent of the others, so their joint
probability is the product of seven one-dimensional ones. False on this dataset
(`P` and `K` correlate at 0.74) and yet harmless to which class wins.

**Confusion matrix** *(W8)* — A table with one row per true class and one
column per predicted class; cell `(i, j)` counts the examples that really were
`i` and were called `j`. The diagonal is the correct answers, every off-diagonal
cell is a named mistake. Read rows for recall, columns for precision. Available
from `confusion_frame()` and from `evaluate_model()`'s `"confusion_matrix"` key.

**Correlation** *(W2)* — How strongly two numeric features move together, on a
scale from -1 to +1.

**Correlation heatmap** *(W2)* — The full feature-by-feature correlation matrix
drawn as a colour grid, so strong pairs are visible at a glance.

**Cross-validation (k-fold)** *(W4)* — Splitting the training data into `k`
folds and fitting the model `k` times, each time on `k - 1` folds and scoring on
the one left out, so every row is validated exactly once and the result is a
distribution of scores rather than a single number.

**`cross_val_score`** *(W4)* — scikit-learn's function running that loop. It
clones the estimator for every fold and returns a NumPy array with one score per
fold, in fold order — report its mean *and* its standard deviation.

**Curse of dimensionality** *(W5)* — As the number of features grows, data
becomes sparse and distances between points concentrate, so "nearest" stops
implying "similar". The reason KNN degrades on wide datasets; with seven
features this project is unaffected.

**Data leakage** *(W2)* — Training a model with information it would not have
at prediction time, so its measured performance flatters it and production
disappoints. The commonest cause is fitting preprocessing on the full dataset
instead of on the training set alone.

**Data preparation** *(W3)* — Everything that happens between the data as
recorded and the array a model is fitted on: encoding, splitting, scaling and,
where justified, feature engineering. Also called preprocessing.

**Dataframe** *(W1)* — pandas' table structure: rows and named, typed columns,
with built-in operations for filtering, grouping and summarising.

**Dataset** *(W1)* — The complete collection of instances available for
training and evaluation.

**Dataset contract** *(W1)* — The properties input data is required to satisfy
— here: exact column names and order, exactly 2,200 rows, numeric features, no
missing values, and exactly the 22 recorded crop names. Enforced by
`validate_dataset()` in `src/data/validate_schema.py`.

**`DatasetValidationError`** *(W1)* — This project's custom exception, raised
when loaded data violates the dataset contract. Subclasses `ValueError`.

**Decision boundary** *(W5)* — The surface in feature space where a model
switches from predicting one class to another. Logistic regression's is always
flat (linear); KNN's takes whatever shape the training rows imply; an RBF SVM's
curves; a decision tree's is made of rectangles. Week 6 draws all three by
classifying a fine grid of points over two features
(`src/utils/visualization.py`).

**Decision tree** *(W6)* — A classifier made of `feature <= threshold`
questions chosen greedily, each split picked to leave the two sides as pure as
possible. Readable, scale-invariant, and the model in this project most eager to
overfit: grown without limit it reaches depth 17, 38 leaves and a perfect,
worthless 100% training accuracy.

**Decorrelation** *(W7)* — Deliberately making ensemble members disagree —
through bootstrap samples and `max_features` — because averaging cancels only
the part of the error the members do not share. Identical members average to
exactly one member.

**Deep learning** *(W1)* — Machine learning using many-layered neural networks.
Not used in this project: with 2,200 rows and seven numeric features, classical
algorithms are both stronger and easier to explain.

**Descriptive statistic** *(W2)* — A single number summarising a property of a
whole column — its centre (mean, median), spread (standard deviation, IQR) or
shape (skewness).

**Discriminative model** *(W5)* — A model of `P(class | features)` — the
boundary between classes — without describing what each class's data looks
like. Logistic regression and KNN are discriminative.

**Distribution** *(W2)* — The pattern of which values a feature takes, and how
often.

**`DummyClassifier`** *(W4)* — scikit-learn's baseline estimator. `fit` looks
only at `y`; `predict` answers from the recorded label distribution using a
chosen strategy (`most_frequent`, `prior`, `stratified`, `uniform`, `constant`).

**Ensemble** *(W7)* — Several models combined into one prediction. It helps only
when the members are individually better than chance *and* wrong on different
rows; a hundred identical copies of a model are still that model.

**Entropy (split criterion)** *(W6)* — A measure of node impurity: the number
of bits needed to encode the node's labels. Zero for a single class, largest
when classes are evenly mixed. Interchangeable with Gini impurity in practice.

**Error analysis** *(W8)* — Studying the individual wrong predictions rather
than the aggregate score: which classes are confused with which, what those rows
measure, and how confident the model was. When accuracy saturates — 438 of 440
here — it is the only remaining source of information about the model.

**Evaluation protocol** *(W4)* — The metric, the fitting/scoring procedure and
the reference point, all fixed *before* any model is trained, so the judgement
cannot be shaped by the results. Here: accuracy with a per-class report, 5-fold
stratified cross-validation on the training rows with seed 42, against a 4.55%
baseline.

**Expected label set** *(W1)* — The exact set of 22 crop names recorded in Week
1 (`EXPECTED_LABELS`, and written out in
`docs/curriculum/week01/validation.md`). Every later week that touches `label`
must match against it and fail loudly if it differs.

**Exploratory data analysis (EDA)** *(W2)* — The stage of the lifecycle that
examines a dataset's statistics and shapes in order to understand it, before any
preparation or modelling. The last stage in which it is safe to look at every
row.

**F1 score** *(W8)* — The harmonic mean of precision and recall,
`2 · P · R / (P + R)`. The harmonic mean is used because it punishes imbalance:
precision 1.0 with recall 0.0 scores 0.0, not 0.5.

**Fail-fast validation** *(W1)* — Design principle: detect invalid input at the
moment it is read and stop, rather than continuing and producing a
plausible-looking but wrong result later.

**False negative / false positive** *(W8)* — A member of the class the model
missed, and a non-member it wrongly claimed. Misses damage recall, false alarms
damage precision.

**FastAPI** *(W10)* — The web framework used to serve the model. Its central
trick is that a type hint *is* the validation: from one Pydantic model it
generates request parsing, per-field error messages and the interactive `/docs`
page, so the documentation cannot drift from the behaviour.

**Feature** *(W1)* — One input variable used to make a prediction. This project
has seven: `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`.

**Feature importance (mean decrease in impurity)** *(W7)* — The
`feature_importances_` attribute of a fitted forest or booster: how much
impurity each column removed across all splits, normalised to sum to 1. Here
`rainfall` 0.2302 leads and `ph` 0.0506 trails. It describes *this fitted model
on its training data*, is biased towards columns with many candidate thresholds,
and splits credit between correlated columns — which is why Week 8 replaces it
with permutation importance and SHAP.

**Feature randomness** *(W7)* — A random forest's second source of diversity:
each split may only consider a random subset of columns, set by `max_features`.
The default `"sqrt"` offers 2 of these 7 features per split, so the 100 trees
open on six different features where a single tree always opens on `rainfall`.

**Feature scale** *(W2)* — The range of values a feature takes. In this dataset
`K` spans 200 units while `ph` spans about 6, which is why Week 3 must rescale
before using any distance-based model.

**Feature scaling** *(W3)* — Re-expressing columns so their numeric ranges are
comparable, removing the arbitrary weighting that different units would
otherwise impose on distances, gradients and penalties.

**`fit` / `transform` / `fit_transform`** *(W3)* — The three calls of a
scikit-learn transformer. `fit` learns parameters from data and stores them on
the object; `transform` applies stored parameters to any data; `fit_transform`
does both and is for training data only.

**Fitted state** *(W3)* — The attributes an estimator gains by being fitted —
`mean_`, `scale_`, `classes_`, later `coef_`. By convention they end in an
underscore, so an object with none has not been fitted.

**Fold** *(W4)* — One of the `k` equal parts a dataset is cut into for
cross-validation, used once as the validation part and `k - 1` times as part of
the training data.

**Gaussian naive Bayes** *(W5)* — Naive Bayes with every feature modelled as a
normal curve per class, so fitting reduces to one mean and one variance per
feature per class (308 numbers here). The best of Week 5's models at 99.49%.

**Generalisation** *(W1)* — Performing well on instances never seen during
training. The actual goal of machine learning, as opposed to reproducing the
training data.

**Generalisation gap** *(W6)* — Training accuracy minus validation accuracy.
For the unlimited decision tree here it is 100% - 98.52% = 1.5 points, and that
gap is precisely the part of the training score that does not survive contact
with unseen rows.

**Generative model** *(W5)* — A model of `P(features | class)`: it describes how
each class's data arises and derives the classification from that. Naive Bayes
is generative.

**Gini impurity** *(W6)* — The default split criterion for a decision tree: the
chance of mislabelling a random row in a node if you guessed at the node's own
class frequencies. Cheaper than entropy because it needs no logarithm, and it
almost always chooses the same splits.

**Gradient boosting** *(W7)* — Boosting in which each round fits a small tree to
the errors the running sum still makes, scaled by a `learning_rate`. Strong on
tabular data; supplied here by XGBoost when it is installed and by scikit-learn's
`GradientBoostingClassifier` otherwise.

**`GridSearchCV`** *(W8)* — scikit-learn's exhaustive hyperparameter search:
every combination of the grid, each scored by cross-validation on the training
rows. Cost is the product of the list lengths times the number of folds — 24
candidates and 120 fits for this project's forest grid. Wrapped by
`tune_model(..., search="grid")`.

**Health endpoint** *(W10)* — A cheap `GET` that reports whether the service can
actually serve — here, whether a model is loaded — rather than merely whether a
process is alive. It takes no dependencies, because it is asked precisely when
something is broken.

**Histogram** *(W2)* — A plot of a distribution: the feature's range is split
into equal-width bins and each bar counts the rows falling inside one.

**HTTP status code** *(W10)* — The three-digit verdict on a request. Read the
first digit: 2xx worked, 3xx look elsewhere, 4xx the client was wrong (422), 5xx
the server was wrong (500, 503). The class says who has to act.

**Hyperparameter** *(W5, W8)* — A setting chosen *before* fitting that controls
how the fitting happens (`max_depth`, `n_estimators`, `var_smoothing`), as
opposed to a parameter, which is learned from the data during `fit`. Nothing in
the training loop chooses one, so a search must.

**Hyperparameter search** *(W8)* — Trying candidate settings and keeping the
best, with cross-validation *inside* the loop so candidates are ranked on
held-out folds and the test set is never consulted. The winner's score is
optimistic — the maximum of many noisy numbers is partly noise — so it is not an
estimate of the model's accuracy.

**Inference** *(W1)* — Using a trained model to predict the label of a new,
unlabelled instance. Happens per request, in milliseconds. Contrast with
training.

**Instance** *(W1)* — One complete example: a set of feature values together
with its target. Also called a sample, observation or row.

**Interquartile range (IQR)** *(W2)* — `Q3 - Q1`: the width of the middle half
of the data. The basis of the boxplot box and of the 1.5 IQR outlier rule.

**JSON** *(W10)* — The text format requests and responses are written in. Every
language can read it, which is the point; it has no tuples, no `NaN` and no NumPy
types, so values must be converted before they are serialised.

**k-nearest neighbours (KNN)** *(W5)* — A classifier that stores the training
rows and predicts by majority vote among the `k` closest of them. Small `k`
overfits, large `k` underfits towards the baseline, and every distance depends
on the features' units — so it must be preceded by scaling.

**Kernel** *(W6)* — A function measuring the similarity between two rows, equal
to the inner product those rows would have in some higher-dimensional space. It
lets an SVM draw a *curved* boundary by measuring similarity instead of
positions.

**Kernel trick** *(W6)* — The reason a kernel is cheap: because an SVM only ever
needs inner products, it can fit a flat boundary in that higher-dimensional
space without computing a single coordinate there.

**Label** *(W1)* — The target value attached to an instance. Here, the crop
name; also the literal name of the target column.

**Label encoding** *(W3)* — Mapping class names onto the integers `0..k-1`,
alphabetically and losslessly, so an estimator can work with them.
`LabelEncoder` stores the mapping in `classes_`. The integers are identifiers,
not quantities.

**Lazy learning** *(W5)* — Learning that does almost nothing at `fit` time and
defers the work to `predict`. KNN is the standard example: its "model" is the
training set itself.

**`learning_rate` (shrinkage)** *(W7)* — The fraction of each boosting round's
correction that is actually applied. It is not independent of `n_estimators`:
smaller steps need more rounds to reach the same fit, so the two are one dial
with two handles.

**Linting** *(W1)* — Automatic inspection of source code for style violations
and likely errors. Performed here by `ruff`.

**Local explanation** *(W8)* — An explanation of one prediction rather than of
the model as a whole: what was predicted, with what probability, what came
second, and which measurement decided it. SHAP and the per-sample permutation
fallback both produce one; permutation importance does not.

**Logistic regression** *(W5)* — A linear classifier, despite the name: one
weighted sum of the features per class, converted to probabilities by softmax.
176 numbers on this dataset, readable, and restricted to flat decision
boundaries.

**Machine learning** *(W1)* — Building software by supplying examples of the
desired behaviour and letting an algorithm infer rules that reproduce them and
generalise to unseen cases.

**Macro average** *(W8)* — The mean of the per-class scores with equal weight
per class, whatever their support. The average that notices failure on a rare
class, and the default to report when every class matters equally.

**Margin** *(W6)* — The width of the empty corridor either side of an SVM's
boundary. Maximising it is the SVM's entire objective, and a wide margin is a
form of caution: a boundary in the middle of a wide corridor is not moved by one
new point.

**`max_depth`** *(W6)* — The maximum number of questions on any path from a
tree's root to a leaf. The most direct control over overfitting in this project:
shallow means high bias, unlimited means high variance.

**Mean** *(W2)* — The arithmetic average of a column. Pulled around by extreme
values, so it can differ sharply from the median in a skewed column.

**Median** *(W2)* — The middle value of a sorted column; half the rows lie below
it. Unaffected by extreme values.

**ML lifecycle** *(W1)* — The loop a project travels: frame the problem, get
the data, explore, prepare, model, evaluate and improve, productionize, deploy,
monitor — and back to framing.

**Multiclass classification** *(W1)* — Classification with more than two
possible categories. Contrast with binary classification, which has exactly
two.

**Multicollinearity** *(W2)* — Strongly correlated inputs to a linear model,
which make its fitted coefficients unstable and its per-feature explanation
unreliable. Tree-based models are largely untroubled by it.

**Multimodality** *(W2)* — More than one peak in a distribution, usually meaning
two different populations have been mixed into one column.

**Naive Bayes** *(W5)* — A probabilistic classifier applying Bayes' rule under
the assumption that features are conditionally independent given the class.
Cheap, assumption-driven, and a strong baseline even where the assumption fails.

**Nested cross-validation** *(W8, not used)* — Wrapping a whole hyperparameter
search inside an outer cross-validation loop, to estimate the *tuning procedure*
without bias. This project uses one inner loop plus a single held-out test set:
honest, but not the stronger protocol.

**Normalisation (min-max)** *(W3)* — Rescaling a column with
`(x - min) / (max - min)` so it lands in [0, 1]. Bounded, but highly sensitive
to a single extreme value. The word is used loosely in the wild — say which
operation you mean.

**One-hot encoding** *(W3)* — Expanding a categorical column into one 0/1 column
per category, so no ordering can be inferred from the codes. Needed for
categorical *inputs*; not needed for a classifier's target, and not needed by
this dataset, whose seven features are all numeric.

**One-vs-rest (OvR)** *(W5)* — Handling `k` classes by training `k` binary "this
class or not" models and taking the most confident. The alternative to a single
multinomial/softmax fit, which is what scikit-learn uses here.

**Out-of-bag (OOB) rows** *(W7)* — The roughly 37% of the training rows a given
bootstrap sample misses (646 of 1,760 for the first tree here). Because each
member never saw them, they act as free held-out data for that member, which is
what an OOB score averages over.

**Outlier** *(W2)* — A value beyond a boxplot's whiskers under the 1.5 IQR rule.
The output of an arithmetic rule, **not** a verdict that the value is wrong: it
may be a data error, a legitimate rare case, or — as in this dataset — ordinary
class structure showing through.

**Out-of-distribution input** *(W10)* — A request whose every field is inside its
allowed range but whose *combination* appears nowhere in the training data. A
classifier over 22 crops has no way to say "none of these", so it answers anyway
— often with near-perfect confidence.

**Overfitting** *(W1)* — When a model learns the training examples and their
noise instead of the pattern behind them: strong on training data, weak on
unseen data. Its opposite, underfitting, is being too simple to capture the
pattern at all. Week 6 makes both visible by plotting training and validation
accuracy against decision-tree depth.

**Pearson correlation** *(W2)* — The default correlation measure. It captures
*linear* association only, so a coefficient of 0 means "no straight-line
relationship", not "unrelated".

**Percentile** *(W2)* — The value below which a given share of the rows falls.
The 50th percentile is the median.

**Permutation importance** *(W7 named, W8 used)* — Shuffle one column of
held-out data, re-score the already-fitted model, and report the score lost.
Nothing is refitted. More trustworthy than `feature_importances_` because it is
measured off the training data, its units are accuracy lost, and it works on any
model with a `predict` method — including `GaussianNB`, which has no built-in
importances. Its one trap is correlated columns: shuffling `P` alone costs 0.179
and `K` alone 0.433, but the pair together costs 0.565, because each stands in
for the other.

**Pinned dependency** *(W1)* — A dependency specified at an exact version
(`pandas==2.2.3`) rather than loosely (`pandas`), so that every install
produces an identical environment.

**`Pipeline`** *(W3)* — A scikit-learn estimator chaining named steps into one
object with a single `fit`/`transform`/`predict`. Putting preprocessing inside
it makes train-only fitting structural, makes cross-validation correct by
construction, and makes the deployed artifact a single object.

**Precision** *(W8)* — `TP / (TP + FP)`: when the model names a class, how often
it is right. Damaged by false alarms, and the number that matters when acting on
a prediction is expensive.

**`predict_proba`** *(W5)* — The classifier method returning one probability per
class per row, in `classes_` order and summing to 1 across each row, instead of
the single label `predict` returns. `predict` is the class with the largest of
them. Useful, but only as trustworthy as the model's calibration — Gaussian
naive Bayes' are systematically overconfident.

**Processed data** *(W1)* — Data derived from the raw input by cleaning,
splitting or transformation. Written to `data/processed/`; never written back
over the raw data.

**Pydantic** *(W10)* — The validation library behind FastAPI. A Pydantic model
declares each field's type and constraints (`ge`, `le`, required,
`extra="forbid"`), and rejects anything that does not fit before a single line of
handler code runs.

**Quartile** *(W2)* — The 25th, 50th and 75th percentiles, which cut a column
into four equal-sized parts.

**Random forest** *(W7)* — Bagging plus feature randomness over unpruned
decision trees, combined by majority vote. On this dataset it scores 0.9926
against the single tree's 0.9852, while each member still memorises its own
training rows perfectly.

**`RandomizedSearchCV`** *(W8)* — A hyperparameter search over `n_iter` random
draws from the space instead of every combination. The cost is chosen rather
than inherited, and in spaces where only two or three settings matter it visits
more distinct values of each than a budget-matched grid. Here 20 draws from a
300-candidate space matched the exhaustive result. Wrapped by
`tune_model(..., search="random")`.

**`random_state` (seed)** *(W3)* — The fixed number that makes a shuffle
deterministic, so a split — and everything computed from it — is reproducible.
Recorded once as `DEFAULT_RANDOM_STATE = 42` in `src/data/split.py`. Its value is
arbitrary; tuning it to improve a score is overfitting the test set by hand.

**Raw data** *(W1)* — The original, unmodified dataset in `data/raw/`. Treated
as strictly read-only so it remains the recoverable source of truth.

**Recall** *(W8)* — `TP / (TP + FN)`: of the examples that really belong to a
class, how many the model finds. Damaged by misses.

**Regression** *(W1)* — A supervised learning task whose output is a number on
a continuous scale, such as predicting yield in kilograms.

**Regularisation strength (`C`)** *(W5)* — Logistic regression's *inverse*
penalty on large weights. Small `C` shrinks the weights and simplifies the
model; large `C` lets it fit the training data closely. Left at 1.0 throughout;
an SVM's `C` (W6) plays the same role for the soft margin.

**Reproducibility** *(W1)* — The property that the same code and data yield the
same results for any person at any time. The reason this project commits its
dataset and pins its dependency versions.

**REST** *(W10)* — A style for web APIs rather than a standard: resources have
paths, HTTP methods are used for what they mean, and every request carries
everything needed to answer it, so any replica can serve any request.

**Sanity check (smoke test)** *(W4)* — A cheap end-to-end run whose only job is
to prove the plumbing works. Fitting a baseline is the cheapest one available:
it exercises load, split, fit, predict and score in milliseconds and catches
shuffled labels, misaligned indexes and metrics wired to the wrong vector.

**Score variance (fold spread)** *(W4)* — How much a score moves between splits
of the same data. Measured as the standard deviation of the per-fold scores; a
difference between two models smaller than this spread is not yet a difference.

**Scale-invariant model** *(W3)* — A model whose output is unchanged by any
order-preserving rescaling of a feature, because it only compares values against
a learned threshold: decision trees and their ensembles. Scale-*sensitive*
models — KNN, SVM, logistic regression, neural networks, PCA — combine feature
values across columns and therefore need scaling.

**SHAP (SHapley Additive exPlanations)** *(W8)* — A library and method that
attributes one prediction across the features, with signed, additive values.
`TreeExplainer` is fast and exact for tree ensembles; `KernelExplainer` is slow
and sampled but works on anything with `predict_proba`. Optional here
(`shap==0.46.0`); when it is absent, `explain_prediction()` falls back to
per-sample permutation plus the raw `predict_proba` breakdown, and records which
ran under `"method"`.

**Shapley value** *(W8)* — From cooperative game theory: a player's average
marginal contribution to the payout across every order in which the players
could join. Applied to a prediction, it is the unique attribution satisfying a
short list of fairness axioms, and it is what makes SHAP values add up to the
model's output.

**Skewness** *(W2)* — How lopsided a distribution is. Positive means a long
right tail, negative a long left tail, zero symmetric.

**Soft margin** *(W6)* — The practical version of margin maximisation, in which
rows are allowed inside the corridor or across the boundary at a price set by
`C`. Small `C` buys a wider, smoother boundary by tolerating violations; large
`C` narrows it around the training rows.

**Softmax** *(W5)* — The function turning a vector of class scores into positive
probabilities that sum to 1, by exponentiating each and dividing by the total.
How logistic regression reaches 22 probabilities.

**Standard deviation** *(W2)* — Roughly the typical distance of a value from the
mean, expressed in the column's own units. Its square is the variance.

**Standardisation (z-score)** *(W3)* — Rescaling a column with
`(x - mean) / std`, so the data it was fitted on ends with mean 0 and standard
deviation 1. A linear transformation: it does not remove skew, outliers or
ordering. Implemented by `StandardScaler`.

**Statelessness** *(W10)* — The property that a server keeps no memory of a caller
between requests. It is what makes horizontal scaling possible, and it is why a
model held in memory (server state) is not a violation while a remembered user
(client state) would be.

**Stratified split** *(W3)* — A train/test split drawn within each class, so
class proportions are preserved on both sides. On this dataset it gives every
crop exactly 80 training and 20 test rows; an unstratified split of the same data
ranges from 11 to 27 test rows per crop.

**`StratifiedKFold`** *(W4)* — The cross-validation splitter that draws folds
within each class, so every class appears in every fold in the same proportion.
Built here by `build_cv()` in `src/evaluation/metrics.py` with
`n_splits=5, shuffle=True, random_state=42`.

**Streamlit** *(W10)* — A library that renders a Python script as a web page, used
here for the demo UI. Explicitly *not* a production frontend: it re-runs the
whole script on every interaction, keeps session state in server memory, and
ships with no authentication.

**Supervised learning** *(W1)* — Learning from examples in which the correct
answer is provided alongside each input.

**Support (classification report)** *(W8)* — The number of true examples of a
class in the evaluated set. 20 for every crop in this project's test set, which
is why macro and weighted averages agree.

**Support vector** *(W6)* — A training row sitting on the edge of an SVM's
margin. Only these rows decide where the boundary goes; on this dataset 943 of
the 1,760 training rows are support vectors at `C = 1`.

**Support vector machine (SVM)** *(W6)* — A classifier that chooses, among all
boundaries separating the classes, the one with the widest margin. Works in
distances, so it needs scaled features; reaches 97.90% here with an RBF kernel
and 98.18% with a linear one.

**Support-vector count as a diagnostic** *(W6)* — The number of support vectors
falls as `C` rises (1,760 at `C = 0.01`, 612 at `C = 100`), which makes the
margin narrowing visible without plotting anything.

**Target** *(W1)* — The value a model is trained to predict. In this project,
the `label` column.

**Test set** *(W1)* — The portion of the dataset held back from training and
used, ideally once, to estimate performance on unseen data. Implemented in
Week 3.

**`TestClient`** *(W10)* — FastAPI's in-process test driver. It exercises the full
request/response cycle — routing, validation, status codes — without opening a
port or starting a server, which is why `tests/test_api.py` runs in under two
seconds.

**Train-only fitting** *(W2)* — The rule that any preprocessing step is fitted
on the training set alone and then applied to the test set. Stated in Week 2,
enforced from Week 3 onward; breaking it causes data leakage.

**Training** *(W1)* — The offline process of fitting a model to labelled
examples. Contrast with inference.

**Training loop (`fit`/`predict`)** *(W5)* — The two calls every supervised
scikit-learn model shares: `fit(X_train, y_train)` learns parameters from
labelled rows, `predict(X)` applies them to rows the model has not seen. Named
in Week 5 because nothing after it changes.

**Training set** *(W1)* — The portion of the dataset a model is fitted on.

**Training/serving skew** *(W3)* — Preparing data at serving time in a way that
differs, even slightly, from how it was prepared at training time. A common cause
of production failures; the standard defence is shipping the fitted `Pipeline`
rather than reimplementing preparation on the server.

**Unsupervised learning** *(W1)* — Learning patterns or structure from data
where no correct answers are supplied.

**Uvicorn** *(W10)* — The ASGI server that listens on a TCP port and speaks HTTP,
calling the FastAPI application for each request. The framework and the server
are different things; only the server has a port.

**Validation set** *(W4)* — Data held out from training and used to *choose*
between models or hyperparameters, as often as needed. Distinct from the test
set, which answers "how good is the final choice?" once, at the end.
Cross-validation manufactures validation sets from the training data.

**Variance (model variance)** *(W6)* — Error caused by a model being flexible
enough to change a great deal when the training sample changes. An unlimited
decision tree has high variance: fit it on a different 1,760 rows and its
thresholds and leaves differ visibly.

**Variance** *(W2)* — The square of the standard deviation: the average squared
distance from the mean. Used by the mathematics; standard deviation is what
humans read, because it is in the original units.

**Virtual environment** *(W1)* — An isolated, per-project Python package
directory created with `venv`, preventing dependency conflicts between
projects. Stored in `venv/` (or `.venv/`) and never committed.

**Weak learner** *(W7)* — A model only slightly better than chance, used
deliberately as a boosting member. A depth-1 stump gets 61.93% of the training
rows right on its own; sixty of them in a chain reach 98.81%. Strength comes
from the sequence, not the link.

**Weighted average** *(W8)* — The mean of the per-class scores weighted by
support, so common classes dominate. Equal to the macro average only when the
classes are balanced; when they are not, quoting it alone can hide total failure
on a rare class.

**XGBoost** *(W7)* — An optional third-party gradient-boosting library, faster
than scikit-learn's implementation on this data. It is *not* a required
dependency of this project: `get_gradient_boosting()` falls back to
`GradientBoostingClassifier` when the import fails, and
`GRADIENT_BOOSTING_BACKEND` records which one is in use. Its own classifier
needs integer labels, so this project wraps it in a small label-encoding
adapter.
