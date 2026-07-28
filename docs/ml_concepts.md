# ML Concepts

A running index of every machine learning concept taught in this course, in the
order it is introduced. Each entry gives a one-line definition and points at
the week that teaches it properly.

This file is updated **every week**, as concepts are introduced — not
consolidated at the end.

---

## Week 1 — Framing the problem and meeting the data

Taught in [`docs/curriculum/week01/learning_notes.md`](curriculum/week01/learning_notes.md).

| Concept | One-line definition | Section |
| --- | --- | --- |
| **Machine learning** | Building software by supplying examples of correct behaviour and letting an algorithm infer the rules, instead of writing the rules by hand. | §1 |
| **When to use ML** | Use it when rules are too numerous, too interacting or too unknown to write down, *and* labelled examples exist; otherwise prefer ordinary code. | §1 |
| **Supervised learning** | Learning from examples where the correct answer is attached to each input. | §2 |
| **Unsupervised learning** | Learning structure from data with no correct answers supplied. | §2 |
| **Classification** | Supervised learning where the prediction is a category. | §2 |
| **Regression** | Supervised learning where the prediction is a number on a continuous scale. | §2 |
| **Multiclass classification** | Classification with more than two possible categories — 22 crops, in this project. | §2 |
| **Categorical / unordered target** | A target whose values have no meaningful ordering, so treating their codes as numbers would be wrong. | §2 |
| **Batch learning** | Training once on a full dataset to produce a fixed artifact, retrained on a schedule (as opposed to online learning, which updates continuously). | §2 |
| **Feature** | A single input variable used to make a prediction. | §3 |
| **Target / label** | The value the model is trained to predict. | §3 |
| **Instance** | One complete example — a row of feature values together with its target. | §3 |
| **Dataset** | The full collection of instances. | §3 |
| **Problem statement** | The written framing of a task: inputs, output, problem type, success measure and explicit non-goals. | §3 |
| **Raw vs. processed data** | Raw data is treated as read-only source of truth; anything derived is written elsewhere, so the input is always recoverable. | §4 |
| **Artificial intelligence** | The broad goal of software performing tasks we would call intelligent; machine learning is one family of techniques within it. | §1 |
| **Deep learning** | Machine learning with many-layered neural networks; unnecessary for a small tabular problem like this one. | §1 |
| **Training set** | The portion of the dataset a model is fitted on. | §5 |
| **Test set** | The portion held back and looked at once, to estimate performance on unseen data. | §5 |
| **Generalisation** | Performing well on instances never seen during training — the actual goal of learning. | §5 |
| **Overfitting** | Learning the training examples and their noise rather than the underlying pattern; good on train, poor on test. | §5 |
| **Underfitting** | Being too simple to capture the pattern; poor on both train and test. | §5 |
| **Training** | The offline process of fitting a model on labelled data. | §5 |
| **Inference** | The online process of predicting a label for one unlabelled instance. | §5 |
| **ML lifecycle** | The loop from framing, through data, modelling and evaluation, to deployment and monitoring — and back. | §6 |
| **Expected label set** | The exact set of 22 crop names recorded in Week 1, which every later week must match against. | §8 |
| **Reproducibility** | The property that the same inputs and code produce the same results for everyone — the reason data and dependency versions are pinned. | §4, §7 |
| **Virtual environment** | An isolated per-project package directory, so projects cannot break each other's dependencies. | §7 |
| **Pinned dependency** | A dependency fixed to an exact version, so environments are identical over time and across machines. | §7 |
| **Dataframe** | A table of rows and named, typed columns — pandas' central data structure. | §7, §8 |
| **Dataset contract** | The set of properties input data must satisfy (columns, dtypes, size, label set, no nulls) to be considered valid. | §8 |
| **Fail-fast validation** | Raising an error the moment invalid input is read, rather than allowing a plausible-looking wrong answer downstream. | §8 |
| **Linting** | Automated checking of code for style and correctness problems, making a written standard machine-enforceable. | §9 |
| **Automated testing** | Executable checks that a system still behaves as specified, run on every change. | §9 |

---

## Week 2 — Exploratory data analysis

Taught in [`docs/curriculum/week02/learning_notes.md`](curriculum/week02/learning_notes.md).

| Concept | One-line definition | Section |
| --- | --- | --- |
| **Exploratory data analysis (EDA)** | Looking at a dataset's statistics and shapes to understand it, before preparing or modelling it. | §0 |
| **Descriptive statistic** | A single number summarising a property of a whole column, such as its centre or its spread. | §1 |
| **Mean** | The arithmetic average; sensitive to extreme values. | §1 |
| **Median** | The middle value when sorted; unaffected by extreme values, so it differs from the mean when a column is skewed. | §1 |
| **Standard deviation** | Roughly the typical distance of a value from the mean, in the column's own units; its square is the variance. | §1 |
| **Percentile / quartile** | The value below which a given share of rows falls; the 25th, 50th and 75th percentiles are the quartiles. | §1 |
| **Interquartile range (IQR)** | `Q3 - Q1`, the width of the middle half of the data. | §1, §5 |
| **Skewness** | How lopsided a distribution is: positive means a long right tail, negative a long left tail, zero symmetric. | §1, §3 |
| **Feature scale** | The range of values a feature takes; features on very different scales must be rescaled before distance-based models are used. | §1 |
| **Class balance / imbalance** | How many rows each class has; balanced classes make accuracy a fair metric, imbalanced ones do not. | §2 |
| **Distribution** | The pattern of which values a feature takes and how often. | §3 |
| **Histogram** | A plot of a distribution: the range split into equal-width bins, each bar counting the rows inside it. | §3 |
| **Bin** | One slice of a histogram's range; too few hide structure, too many show noise. | §3 |
| **Multimodality** | More than one peak in a distribution, usually meaning two populations have been mixed together. | §3 |
| **Correlation** | How strongly two numeric features move together, from -1 to +1. | §4 |
| **Pearson correlation** | The default correlation measure; captures *linear* association only, so 0 means "no straight-line relationship", not "unrelated". | §4 |
| **Correlation heatmap** | The whole feature-by-feature correlation matrix drawn as a colour grid. | §4 |
| **Multicollinearity** | Strongly correlated inputs making a linear model's fitted coefficients unstable and its explanation unreliable. | §4 |
| **Boxplot** | A five-number summary drawn as a box (IQR) with median, whiskers at 1.5 IQR, and points beyond drawn individually. | §5 |
| **Outlier** | A value beyond the whiskers under the 1.5 IQR rule — a description produced by arithmetic, not a verdict that the value is wrong. | §5 |
| **Class separation (eta-squared)** | The share of a feature's variance lying between classes rather than within them; 0 = the classes are indistinguishable on it, 1 = perfectly separated. | §5 |
| **Data leakage** | Training a model with information it would not have at prediction time, so its test score flatters it and production disappoints. | §6 |
| **Train-only fitting** | The rule that any preprocessing is fitted on the training set alone and then applied to the test set — stated here, enforced from Week 3. | §6 |

---

## Week 3 onwards

Not yet written.
