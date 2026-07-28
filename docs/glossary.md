# Glossary

Alphabetical quick reference for terms used across the project. For the
teaching order and fuller treatment, see
[`docs/ml_concepts.md`](ml_concepts.md) and the weekly learning notes.

Terms are added as they are introduced. The marker on each entry gives the week
it was introduced — *(W1)* Week 1, *(W2)* Week 2, and so on.

---

**Accuracy** *(W4)* — The share of predictions that were correct. A fair
headline metric when the classes are balanced and every mistake costs the same,
as here — but never a complete one, because it cannot show *which* classes fail.

**Artificial intelligence** *(W1)* — The broad field of building software that
performs tasks we would call intelligent. Machine learning is one family of
techniques within it; deep learning is one family within machine learning.

**Baseline model** *(W4)* — A deliberately unintelligent model, fitted and
scored exactly like a real candidate but ignoring the features entirely. Its
score is the floor: a model that fails to beat it has learned nothing from the
features. This project's baseline is 4.55% (`1/22`).

**Batch learning** *(W1)* — Training a model once on a complete dataset to
produce a fixed artifact, which is then deployed and periodically retrained.
Contrast with online learning, where the model updates continuously as data
arrives.

**Bin** *(W2)* — One slice of a histogram's range. Too few bins hide structure
(separate peaks merge); too many turn every bar into noise.

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

**Deep learning** *(W1)* — Machine learning using many-layered neural networks.
Not used in this project: with 2,200 rows and seven numeric features, classical
algorithms are both stronger and easier to explain.

**Descriptive statistic** *(W2)* — A single number summarising a property of a
whole column — its centre (mean, median), spread (standard deviation, IQR) or
shape (skewness).

**Distribution** *(W2)* — The pattern of which values a feature takes, and how
often.

**`DummyClassifier`** *(W4)* — scikit-learn's baseline estimator. `fit` looks
only at `y`; `predict` answers from the recorded label distribution using a
chosen strategy (`most_frequent`, `prior`, `stratified`, `uniform`, `constant`).

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

**Fail-fast validation** *(W1)* — Design principle: detect invalid input at the
moment it is read and stop, rather than continuing and producing a
plausible-looking but wrong result later.

**Feature** *(W1)* — One input variable used to make a prediction. This project
has seven: `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`.

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

**Generalisation** *(W1)* — Performing well on instances never seen during
training. The actual goal of machine learning, as opposed to reproducing the
training data.

**Histogram** *(W2)* — A plot of a distribution: the feature's range is split
into equal-width bins and each bar counts the rows falling inside one.

**Inference** *(W1)* — Using a trained model to predict the label of a new,
unlabelled instance. Happens per request, in milliseconds. Contrast with
training.

**Instance** *(W1)* — One complete example: a set of feature values together
with its target. Also called a sample, observation or row.

**Interquartile range (IQR)** *(W2)* — `Q3 - Q1`: the width of the middle half
of the data. The basis of the boxplot box and of the 1.5 IQR outlier rule.

**Label** *(W1)* — The target value attached to an instance. Here, the crop
name; also the literal name of the target column.

**Label encoding** *(W3)* — Mapping class names onto the integers `0..k-1`,
alphabetically and losslessly, so an estimator can work with them.
`LabelEncoder` stores the mapping in `classes_`. The integers are identifiers,
not quantities.

**Linting** *(W1)* — Automatic inspection of source code for style violations
and likely errors. Performed here by `ruff`.

**Machine learning** *(W1)* — Building software by supplying examples of the
desired behaviour and letting an algorithm infer rules that reproduce them and
generalise to unseen cases.

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

**Normalisation (min-max)** *(W3)* — Rescaling a column with
`(x - min) / (max - min)` so it lands in [0, 1]. Bounded, but highly sensitive
to a single extreme value. The word is used loosely in the wild — say which
operation you mean.

**One-hot encoding** *(W3)* — Expanding a categorical column into one 0/1 column
per category, so no ordering can be inferred from the codes. Needed for
categorical *inputs*; not needed for a classifier's target, and not needed by
this dataset, whose seven features are all numeric.

**Outlier** *(W2)* — A value beyond a boxplot's whiskers under the 1.5 IQR rule.
The output of an arithmetic rule, **not** a verdict that the value is wrong: it
may be a data error, a legitimate rare case, or — as in this dataset — ordinary
class structure showing through.

**Overfitting** *(W1)* — When a model learns the training examples and their
noise instead of the pattern behind them: strong on training data, weak on
unseen data. Its opposite, underfitting, is being too simple to capture the
pattern at all.

**Pearson correlation** *(W2)* — The default correlation measure. It captures
*linear* association only, so a coefficient of 0 means "no straight-line
relationship", not "unrelated".

**Percentile** *(W2)* — The value below which a given share of the rows falls.
The 50th percentile is the median.

**Pinned dependency** *(W1)* — A dependency specified at an exact version
(`pandas==2.2.3`) rather than loosely (`pandas`), so that every install
produces an identical environment.

**`Pipeline`** *(W3)* — A scikit-learn estimator chaining named steps into one
object with a single `fit`/`transform`/`predict`. Putting preprocessing inside
it makes train-only fitting structural, makes cross-validation correct by
construction, and makes the deployed artifact a single object.

**Processed data** *(W1)* — Data derived from the raw input by cleaning,
splitting or transformation. Written to `data/processed/`; never written back
over the raw data.

**Quartile** *(W2)* — The 25th, 50th and 75th percentiles, which cut a column
into four equal-sized parts.

**`random_state` (seed)** *(W3)* — The fixed number that makes a shuffle
deterministic, so a split — and everything computed from it — is reproducible.
Recorded once as `DEFAULT_RANDOM_STATE = 42` in `src/data/split.py`. Its value is
arbitrary; tuning it to improve a score is overfitting the test set by hand.

**Raw data** *(W1)* — The original, unmodified dataset in `data/raw/`. Treated
as strictly read-only so it remains the recoverable source of truth.

**Regression** *(W1)* — A supervised learning task whose output is a number on
a continuous scale, such as predicting yield in kilograms.

**Reproducibility** *(W1)* — The property that the same code and data yield the
same results for any person at any time. The reason this project commits its
dataset and pins its dependency versions.

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

**Skewness** *(W2)* — How lopsided a distribution is. Positive means a long
right tail, negative a long left tail, zero symmetric.

**Standard deviation** *(W2)* — Roughly the typical distance of a value from the
mean, expressed in the column's own units. Its square is the variance.

**Standardisation (z-score)** *(W3)* — Rescaling a column with
`(x - mean) / std`, so the data it was fitted on ends with mean 0 and standard
deviation 1. A linear transformation: it does not remove skew, outliers or
ordering. Implemented by `StandardScaler`.

**Stratified split** *(W3)* — A train/test split drawn within each class, so
class proportions are preserved on both sides. On this dataset it gives every
crop exactly 80 training and 20 test rows; an unstratified split of the same data
ranges from 11 to 27 test rows per crop.

**`StratifiedKFold`** *(W4)* — The cross-validation splitter that draws folds
within each class, so every class appears in every fold in the same proportion.
Built here by `build_cv()` in `src/evaluation/metrics.py` with
`n_splits=5, shuffle=True, random_state=42`.

**Supervised learning** *(W1)* — Learning from examples in which the correct
answer is provided alongside each input.

**Target** *(W1)* — The value a model is trained to predict. In this project,
the `label` column.

**Test set** *(W1)* — The portion of the dataset held back from training and
used, ideally once, to estimate performance on unseen data. Implemented in
Week 3.

**Train-only fitting** *(W2)* — The rule that any preprocessing step is fitted
on the training set alone and then applied to the test set. Stated in Week 2,
enforced from Week 3 onward; breaking it causes data leakage.

**Training** *(W1)* — The offline process of fitting a model to labelled
examples. Contrast with inference.

**Training set** *(W1)* — The portion of the dataset a model is fitted on.

**Training/serving skew** *(W3)* — Preparing data at serving time in a way that
differs, even slightly, from how it was prepared at training time. A common cause
of production failures; the standard defence is shipping the fitted `Pipeline`
rather than reimplementing preparation on the server.

**Unsupervised learning** *(W1)* — Learning patterns or structure from data
where no correct answers are supplied.

**Validation set** *(W4)* — Data held out from training and used to *choose*
between models or hyperparameters, as often as needed. Distinct from the test
set, which answers "how good is the final choice?" once, at the end.
Cross-validation manufactures validation sets from the training data.

**Variance** *(W2)* — The square of the standard deviation: the average squared
distance from the mean. Used by the mathematics; standard deviation is what
humans read, because it is in the original units.

**Virtual environment** *(W1)* — An isolated, per-project Python package
directory created with `venv`, preventing dependency conflicts between
projects. Stored in `venv/` (or `.venv/`) and never committed.
