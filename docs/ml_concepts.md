# ML Concepts

> 🗺 [Roadmap](curriculum/README.md) › **Appendix A — Concepts by chapter**

A running index of every machine learning concept taught in this course, in the
order it is introduced. Each entry gives a one-line definition and points at
the week that teaches it properly.

This file is updated **every week**, as concepts are introduced — not
consolidated at the end. Week 12 completes it rather than rewriting it: the
twelve sections below *are* the consolidated glossary of the course, and the
fact that nothing needed backfilling at the end is the point.

**How to use it.** This file is ordered by *teaching order* — read down it to
follow the course, or search it to find the week that owns a concept.
[`docs/glossary.md`](glossary.md) holds the same terms alphabetically, with
fuller definitions, for when you know the word and want the meaning.

| Week | Section | Theme |
| --- | --- | --- |
| 1 | [Framing the problem and meeting the data](#week-1--framing-the-problem-and-meeting-the-data) | What ML is, what this problem is, and the dataset contract |
| 2 | [Exploratory data analysis](#week-2--exploratory-data-analysis) | Statistics, distributions, correlation, outliers, leakage |
| 3 | [Data preparation](#week-3--data-preparation) | Encoding, the stratified split, train-fitted preprocessing |
| 4 | [Baseline models and evaluation](#week-4--baseline-models-and-evaluation) | The floor to beat, and cross-validation |
| 5 | [Classification models](#week-5--classification-models) | Linear, distance-based and probabilistic families |
| 6 | [Margin-based and tree-based models](#week-6--margin-based-and-tree-based-models) | SVMs, trees, and overfitting made visible |
| 7 | [Ensembles](#week-7--ensembles) | Bagging, boosting, feature importance |
| 8 | [Evaluation & explainability](#week-8--evaluation--explainability) | Tuning, the test set, confusion matrices, SHAP |
| 9 | [Productionizing the model](#week-9--productionizing-the-model) | Config, pipeline, artifact, entry points |
| 10 | [Serving the model over HTTP](#week-10--serving-the-model-over-http) | FastAPI, schemas, status codes, a demo UI |
| 11 | [Containerisation and CI](#week-11--containerisation-and-continuous-integration) | Images, layers, GitHub Actions |
| 12 | [Final review and portfolio polish](#week-12--final-review-and-portfolio-polish) | Production-readiness, limitations, versioning, monitoring |

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

## Week 3 — Data preparation

Taught in [`docs/curriculum/week03/learning_notes.md`](curriculum/week03/learning_notes.md).

| Concept | One-line definition | Section |
| --- | --- | --- |
| **Data preparation / preprocessing** | Everything that happens between the data as recorded and the array a model is fitted on. | §0 |
| **Feature scaling** | Re-expressing columns so their numeric ranges are comparable, removing an arbitrary weighting caused purely by units. | §1, §2 |
| **Standardisation (z-score)** | `(x - mean) / std` per column; the data it was fitted on ends with mean 0 and standard deviation 1. | §2 |
| **Normalisation (min-max)** | `(x - min) / (max - min)`, mapping a column onto [0, 1]; bounded, but very sensitive to a single extreme value. | §2 |
| **Scale-sensitive model** | A model whose result depends on feature units — KNN, SVM, logistic regression, neural networks, PCA. | §1 |
| **Scale-invariant model** | A model unchanged by monotone rescaling, because it only compares a feature against a threshold — decision trees and their ensembles. | §1 |
| **Label encoding** | Mapping class names onto integers `0..k-1` (alphabetically, and losslessly) so an estimator can work with them. | §3 |
| **One-hot encoding** | Expanding a categorical column into one 0/1 column per category, so no false ordering is implied. Contrasted here; not yet needed. | §3 |
| **Stratified split** | Splitting within each class so that class proportions are preserved on both sides of the split. | §4 |
| **`random_state` / seed** | The fixed number that makes a shuffle deterministic, so a split — and every result depending on it — is reproducible. | §4 |
| **`fit`** | Learn parameters from data and store them on the estimator. Training data only. | §5 |
| **`transform`** | Apply the stored parameters to any data — train, test, or a single request in production. | §5 |
| **`fit_transform`** | Both at once; a convenience for training data, never for test data. | §5 |
| **Fitted state** | The attributes an estimator gains by being fitted (`mean_`, `scale_`, `classes_`); by convention they end in an underscore. | §5 |
| **Train-only fitting, enforced** | The Week 2 rule made procedural: exactly 0/1 statistics on train, near-but-not-equal on test, is the evidence it was obeyed. | §5 |
| **`ColumnTransformer`** | An estimator mapping sets of columns to the transformers applied to them, with `remainder` deciding the fate of the rest. | §6 |
| **`Pipeline`** | An estimator chaining named steps into one object with a single `fit`/`transform`/`predict`, so preparation travels with the model. | §6 |
| **Training/serving skew** | Preparing data differently at serving time than at training time; shipping the fitted pipeline is the standard defence. | §6 |

---

## Week 4 — Baseline models and evaluation

Taught in [`docs/curriculum/week04/learning_notes.md`](curriculum/week04/learning_notes.md).

| Concept | One-line definition | Section |
| --- | --- | --- |
| **Evaluation** | Measuring performance on data the model was not fitted on, with a metric, a protocol and a reference point all fixed in advance. | §0 |
| **Evaluation protocol** | The trio of metric, fitting/scoring procedure and baseline, chosen before any model is trained so the judgement cannot be shaped by the results. | §0 |
| **Accuracy** | The share of predictions that were correct; fair when classes are balanced and errors cost the same, never complete on its own. | §1 |
| **Baseline model** | A deliberately unintelligent model, fitted and scored like a real one but ignoring the features, whose score is the floor every real model must beat. | §2 |
| **`DummyClassifier`** | scikit-learn's baseline estimator: `fit` looks only at `y`, `predict` answers from the label distribution. | §3 |
| **Dummy strategies** | `most_frequent` and `prior` always predict one class; `stratified` and `uniform` guess randomly, with and without regard to class frequencies. | §3 |
| **The 1/k rule** | On a balanced k-class dataset a constant guess scores exactly 1/k — here 1/22 = 4.55%. | §4 |
| **Majority-class accuracy** | On imbalanced data the same naive model scores the majority class's share, which can look like an excellent result. | §5 |
| **Classification report** | The per-class table of precision, recall, F1 and support, printed beside accuracy so one number cannot hide 22 behaviours. | §5 |
| **Sanity check (smoke test)** | Fitting the baseline exercises the whole pipeline in milliseconds and catches shuffled labels, misaligned indexes and misdirected metrics. | §2 |
| **k-fold cross-validation** | Fitting `k` times on `k - 1` folds and scoring on the one left out, so every row is validated exactly once. | §7 |
| **Fold** | One of the `k` equal parts the data is cut into; validation part once, training part `k - 1` times. | §7 |
| **`StratifiedKFold`** | The splitter that draws folds within each class, so all 22 crops appear in every fold. | §7 |
| **`cross_val_score`** | The function running the loop; returns one score per fold, clones the estimator each time, and leaves the original unfitted. | §7 |
| **Score variance (fold spread)** | How much a score moves between splits; a gap between two models smaller than the spread is not yet a gap. | §6, §7 |
| **Validation set** | Data used to *choose* between models, as often as needed — distinct from the test set, which is read once at the end. | §8 |
| **Dataset difficulty / performance ceiling** | How much accuracy is achievable at all; on this unusually separable dataset it is ~99%, which is a property of the data, not of the modeller. | §9 |

---

## Week 5 — Classification models

Taught in [`docs/curriculum/week05/learning_notes.md`](curriculum/week05/learning_notes.md).

| Concept | One-line definition | Section |
| --- | --- | --- |
| **The training loop (`fit`/`predict`)** | `model.fit(X_train, y_train)` learns and stores parameters; `model.predict(X_test)` applies them to unseen rows. Every supervised model reuses it unchanged. | §1 |
| **Logistic regression** | A linear classifier: one weighted sum of the features per class, turned into probabilities and maximised against the training labels. | §2 |
| **Softmax** | The function converting a vector of class scores into positive probabilities that sum to 1, by exponentiating and normalising. | §2 |
| **One-vs-rest (OvR)** | Handling `k` classes with `k` binary "this class or not" models, taking the most confident; the alternative to a single multinomial fit. | §2 |
| **Linear decision boundary** | The flat surface where two classes' scores are equal — a line, plane or hyperplane. All logistic regression can draw. | §2 |
| **Regularisation strength `C`** | The inverse penalty on large weights: small `C` shrinks them towards zero, large `C` lets them fit the training data closely. | §2 |
| **k-nearest neighbours (KNN)** | Predicting by vote among the `k` closest stored training rows, with no boundary equation of its own. | §3 |
| **Lazy (instance-based) learning** | Learning that stores the training data and defers all work to prediction time. | §3 |
| **Effect of `k`** | `k = 1` memorises and overfits; very large `k` averages over everything and underfits towards the baseline. | §3 |
| **Curse of dimensionality** | As columns multiply, distances concentrate and data grows sparse, so "nearest" stops implying "similar". | §3 |
| **Naive Bayes** | Bayes' rule plus the assumption that features are independent given the class, reducing a joint distribution to a product. | §4 |
| **Gaussian naive Bayes** | Naive Bayes with each feature modelled as a normal curve per class: one mean and one variance per feature per class. | §4 |
| **Conditional independence assumption** | The "naive" part; false on this dataset (`P`/`K` correlate at 0.74) and yet harmless to the ranking of classes. | §4 |
| **Generative vs. discriminative** | Generative models describe how each class's data arises; discriminative models model the boundary between classes directly. | §4 |
| **Fair model comparison** | Same rows, same folds, same seed, same metric and same preparation for every candidate — otherwise the protocol is part of the result. | §5 |
| **Results table** | The running record of every model's cross-validated score beside the baseline, extended by later weeks rather than replaced. | §5, §6 |

---

## Week 6 — Margin-based and tree-based models

Taught in [`docs/curriculum/week06/learning_notes.md`](curriculum/week06/learning_notes.md).

| Concept | One-line definition | Section |
| --- | --- | --- |
| **Support vector machine (SVM)** | A classifier that places the boundary where the empty corridor between the classes is widest. | §9 |
| **Margin** | The width of that corridor; maximising it is the SVM's entire objective. | §9 |
| **Support vectors** | The training rows on the edge of the margin — the only ones that decide where the boundary sits (943 of 1,760 here). | §9 |
| **Soft margin / `C` for an SVM** | The price charged for a row inside or across the margin: small `C` buys a wider, smoother boundary, large `C` a tighter one. | §9 |
| **Kernel** | A similarity function between two rows, equal to their inner product in a higher-dimensional space. | §9 |
| **Kernel trick** | Fitting a flat boundary in that space without ever computing a coordinate there, so the boundary curves in the original features. | §9 |
| **RBF kernel and `gamma`** | Similarity decaying with distance; `gamma` sets how fast, and therefore how wiggly the boundary may be. | §9 |
| **Decision tree** | A chain of `feature <= threshold` questions, chosen greedily, ending in leaves that predict a class. | §10 |
| **Node purity (Gini, entropy)** | How mixed a node's labels are: 0 for a single class, largest when evenly mixed. The split chosen is the one that reduces it most. | §10 |
| **Information gain** | The drop in impurity a split buys, weighted by how many rows go each way. | §10 |
| **`max_depth`, `min_samples_leaf`** | The two dials that stop a tree splitting, and therefore stop it memorising. | §10, §11 |
| **Model capacity** | How much structure a model can express at all; the quantity every such dial adjusts. | §11 |
| **Generalisation gap** | Training accuracy minus validation accuracy — 100% against 98.52% for the unlimited tree here. | §11 |
| **Bias** | Error from a model too simple to represent the truth; a depth-1 tree scores 9.09% on training *and* validation rows. | §11 |
| **Variance** | Error from a model so flexible that it changes a great deal when the training sample does. | §11 |
| **Bias-variance tradeoff** | Total error as the sum of the two, with every complexity dial moving the model between them. | §11 |
| **Decision boundary, drawn** | Classify a fine grid of points and colour the plane by the answers; the seams are the boundary. | §12 |
| **Axis-aligned splits** | A tree can only cut parallel to the axes, so its regions are rectangles and diagonals become staircases. | §12 |
| **Scale invariance of trees** | A threshold split selects the same rows however the column is rescaled, so trees need no scaler. | §10 |

---

## Week 7 — Ensembles

Taught in [`docs/curriculum/week07/learning_notes.md`](curriculum/week07/learning_notes.md).

| Concept | One-line definition | Section |
| --- | --- | --- |
| **Ensemble** | Many models combined into one prediction, on the bet that their mistakes are less correlated than their correct answers. | §1 |
| **Wisdom-of-crowds condition** | Averaging only helps when the members are individually better than chance *and* err on different rows; identical members give identical predictions. | §1 |
| **Bagging** | Fit many deep models in parallel on bootstrap samples and vote; attacks **variance**, and cannot make the members individually better. | §1, §2 |
| **Boosting** | Fit weak models in sequence, each trained on what the chain so far still gets wrong; attacks **bias**, and can overfit if the chain runs too long. | §1, §3 |
| **Bootstrap sample** | A draw of `n` rows *with replacement* from `n` training rows; about 63% of the rows are distinct (1,114 of 1,760 here). | §2.1 |
| **Out-of-bag rows** | The ~37% a given tree never saw (646 here) — free held-out data for that member, and the basis of the OOB score. | §2.1 |
| **Feature randomness / `max_features`** | Offering each split only a random subset of columns; `"sqrt"` gives 2 of these 7, so the trees stop all asking the same first question. | §2.2 |
| **Decorrelation** | The point of both tricks: averaging cancels only the part of the error the members do not share. | §2.2 |
| **Random forest** | Bagging plus feature randomness over unpruned decision trees; 0.9926 here against 0.9852 for one tree. | §2 |
| **Majority voting** | The forest's combination rule for classification — the class most members name (or the highest averaged probability). | §2.3 |
| **`n_estimators` as a budget** | More trees never really hurt a forest; the curve flattens (0.9483 → 0.9932 from 1 to 30 trees) and buys compute, not risk. | §2.5 |
| **Weak learner** | A model barely better than chance, used deliberately: a depth-1 stump scores 0.6193 on the training rows before the chain improves it. | §3.2 |
| **Gradient boosting** | Each round fits a small tree to the current errors of the running sum, so the ensemble descends its loss stage by stage. | §3.1 |
| **`learning_rate` (shrinkage)** | The fraction of each round's correction actually applied; smaller steps need more rounds, so the two are one dial. | §3.3 |
| **XGBoost, and the fallback** | An optional faster boosting library; `get_gradient_boosting()` returns scikit-learn's `GradientBoostingClassifier` when it is absent, so nothing is blocked on it. | §3.4 |
| **`feature_importances_` (mean decrease in impurity)** | Total impurity each column removed across the fitted forest's splits, normalised to sum to 1: `rainfall` 0.2302 down to `ph` 0.0506. | §4 |
| **Importance is training-set, biased and splittable** | It is measured where the model fitted, favours columns with many candidate thresholds, and divides credit between correlated columns — duplicating `humidity` halves its score at identical accuracy. | §4 |
| **Permutation importance / SHAP (foreshadowed)** | The Week 8 replacements: shuffle a column and watch a held-out score fall, or attribute a single prediction across features. | §4 |
| **A tie is a result** | The ensembles beat every other single model but sit 0.0023 behind naive Bayes — inside the fold spread, so "level with the leader" is the defensible claim. | §5 |

---

## Week 8 — Evaluation & Explainability

Taught in [`docs/curriculum/week08/learning_notes.md`](curriculum/week08/learning_notes.md).

| Concept | One-line definition | Section |
| --- | --- | --- |
| **Confusion matrix** | One row per true class, one column per predicted class; cell `(i, j)` counts "really `i`, called `j`", the diagonal is correct. | §2 |
| **Reading direction** | Rows answer "what happened to the real rice fields?" (recall); columns answer "when it said jute, what were they?" (precision). | §2 |
| **Precision** | TP / (TP + FP) — how often a prediction of a class is right; damaged by false alarms. | §3 |
| **Recall** | TP / (TP + FN) — how much of a class the model finds; damaged by misses. | §3 |
| **F1** | Harmonic mean of precision and recall, so 1.0 with 0.0 scores 0.0 rather than 0.5. | §3 |
| **Macro average** | Mean of the per-class scores with equal weight per class — the one that notices failure on a rare class. | §4 |
| **Weighted average** | Mean weighted by support; identical to macro here only because every crop has exactly 20 test rows. | §4 |
| **Hyperparameter search** | Trying candidate settings and keeping the best, because nothing in `fit` chooses `max_depth` or `var_smoothing`. | §5 |
| **Inner cross-validation** | The 5 stratified folds *inside* the search, so candidates are ranked on held-out rows and the test set is never consulted. | §5 |
| **`GridSearchCV`** | Every combination; cost is the product of the list lengths (24 candidates, 120 fits here). | §5 |
| **`RandomizedSearchCV`** | `n_iter` random draws; cost is chosen, and 20 draws from a 300-candidate space matched the exhaustive result. | §5 |
| **Optimism of the winner** | The maximum of many noisy scores is partly noise: 0.9943 "best of 24" is not an estimate of the model's accuracy. | §5 |
| **A gain inside the noise** | Tuning bought +0.0017 against a ±0.0060 fold spread — the search's own answer is "the default was fine". | §5 |
| **A hyperparameter that does nothing** | Twelve `var_smoothing` values over five orders of magnitude give one identical CV score; the floor never binds on this data. | §5 |
| **Final model selection** | Accuracy, then error pattern, interpretability, training and serving cost, and tuning risk — Gaussian naive Bayes over a tuned forest at the same 0.9955. | §6 |
| **Error analysis** | The 2 wrong rows out of 440: `rice -> jute` (only rainfall separates them, 237 vs 176 mm) and `blackgram -> maize` (N and P). | §7 |
| **Concentrated vs scattered errors** | 462 possible confusion cells, all mistakes in one or two: genuine class overlap, not a broken model. | §7 |
| **Permutation importance** | Shuffle one column of held-out data and measure the score lost; no refitting, units of accuracy, works on any fitted model. | §8 |
| **Why it beats MDI** | Held-out rather than training data, actionable units, and available for models with no `feature_importances_` — including the chosen one. | §8 |
| **Correlation trap** | Correlated columns cover for each other: `P` alone costs 0.179, `K` alone 0.433, the pair together 0.565. | §8 |
| **SHAP / Shapley value** | A feature's average marginal contribution over every revelation order — the per-row, signed, additive attribution. | §9 |
| **Additivity** | Base value + contributions = this row's model output; a complete account of one prediction, not a ranking. | §9 |
| **`TreeExplainer` vs `KernelExplainer`** | Fast and exact for tree ensembles vs slow and sampled for anything with `predict_proba`. | §9 |
| **Local explanation** | An explanation of *one* prediction, naming the deciding measurement and the runner-up class with its probability. | §11 |
| **The documented fallback** | Without `shap`: per-sample permutation plus the raw `predict_proba` breakdown — fixed in advance, and the `"method"` key always records which ran. | §10 |

---

## Week 9 — Productionizing the model

Taught in [`docs/curriculum/week09/learning_notes.md`](curriculum/week09/learning_notes.md).

| Concept | One-line definition | Section |
| --- | --- | --- |
| **Hidden state** | A notebook's variables live in the kernel, not the file, so its displayed output is not evidence that its code produces that output. | §1.1 |
| **Execution order** | Cells run in whatever order you clicked them; a script has exactly one order, top to bottom, every time. | §1.2 |
| **No reuse** | A cell cannot be imported by a test, a script or a server — only copied, or re-run whole. | §1.3 |
| **Research vs production code** | Exploration optimises for looking at things; production optimises for running unattended, identically, elsewhere. | §1 |
| **`Pipeline` as one estimator** | Named steps chained into a single object: `fit` fits them in order, `predict` applies the fitted ones in order. | §2 |
| **Preprocessing inside the model object** | Puts the Week 3 `ColumnTransformer` where cross-validation re-fits it per fold and serving cannot forget it. | §2 |
| **Training/serving skew** | Training transformed the data and serving did not (or differently); the model then answers confidently and wrongly. | §2 |
| **`step__parameter` naming** | Naming pipeline steps is what makes `model__var_smoothing` addressable by a search — and `named_steps["model"]` readable. | §2 |
| **Serialization / pickling** | Writing a live Python object to bytes so a later process can rebuild it. | §3 |
| **`joblib.dump` / `joblib.load`** | Pickle with a fast path for NumPy arrays; the project's 6.3 KB `models/crop_model.joblib`. | §3 |
| **What an artifact does not carry** | Not the classes' code, not library versions, not the training data, not any metadata — only learned numbers and import paths. | §3 |
| **Unpickling executes code** | Loading an untrusted `.joblib` is equivalent to running an untrusted script. | §3 |
| **Reproducibility is four things** | Committed data + fixed seed + versioned code + pinned environment; `joblib` is none of them. | §4 |
| **Seeded end to end** | One `RANDOM_STATE`, used by every shuffling step; a seed on the split plus an unseeded shuffle later is no seed at all. | §4 |
| **"Works on my machine"** | The failure of the environment requirement: pinning does not make drift impossible, it makes it diagnosable. | §4 |
| **Config over hardcoding** | Paths, seed and chosen hyperparameters in one inert module, derived not duplicated, and still overridable per call. | §5 |
| **Source vs derived artifacts** | `data/raw/` is committed because nothing can regenerate it; `models/*.joblib` is git-ignored because the code can. | §6 |
| **Train on demand** | A missing artifact is the normal state of a clean clone, so `load_pipeline()` builds one rather than crashing — with an explicit opt-out. | §6 |
| **Entry point (`python -m`)** | A module runnable as a script with the repository root importable, so the same code serves the shell, the tests and the API. | §7 |
| **Validation at the boundary** | Missing, unexpected or non-numeric features are rejected before the model sees them, because Week 10 will pass in whatever a stranger sent. | §7.2 |
| **Runner-up probability at the point of use** | `predict_proba` shows the Week 8 `rice -> jute` ambiguity (0.7253 / 0.2747) where an operational routing rule can act on it. | §7.2 |

---

## Week 10 — Serving the model over HTTP

Taught in [`docs/curriculum/week10/learning_notes.md`](curriculum/week10/learning_notes.md).

| Concept | One-line definition | Section |
| --- | --- | --- |
| **API** | The set of things one program promises another it can do, plus the exact way to ask — `predict()` was already one. | §1.1 |
| **Web API** | The same promise reachable over a network, so the caller shares no language, library or machine — paid for in latency, parsing and untrusted input. | §1.2 |
| **Client / server** | The server runs and waits; the client speaks first. The server never initiates, and "up" is not "answering correctly". | §1.3 |
| **Request/response cycle** | Exactly one request (method, path, headers, body) and one response (status code, headers, body) per interaction. | §2.1 |
| **HTTP method** | The verb: `GET` reads and carries no body, `POST` sends one — which is why `/predict` is a `POST`. | §2.2 |
| **Status code classes** | 2xx worked, 3xx look elsewhere, 4xx the client was wrong, 5xx the server was wrong. | §2.3 |
| **JSON as the wire format** | Text every language can parse; it has no tuples, no `NaN` and no NumPy types, so values must be converted on the way out. | §2.4 |
| **REST** | A style, not a standard: resources have paths, methods are used uniformly, and every request is self-contained. | §3 |
| **Statelessness** | The server remembers nothing about a caller between requests, which is what allows any replica to answer any request. | §3 |
| **FastAPI** | The framework that reads a type hint and generates the validation, parsing and documentation from it. | §4.1 |
| **Pydantic model** | A class whose fields declare types and constraints; the contract *is* the enforcement, so the two cannot drift. | §4.2 |
| **Field constraints (`ge`/`le`, `...`)** | Inclusive bounds and "required with no default", checked before any handler code runs. | §4.2 |
| **`extra="forbid"`** | Unknown keys are an error, so a client's typo is reported rather than silently dropped. | §4.2 |
| **OpenAPI / `/docs`** | A machine-readable description of the API, generated from the same models, rendered as an interactive page that cannot go stale. | §4.3 |
| **Framework vs server (ASGI)** | FastAPI builds the application object; uvicorn is the process that listens on a port and calls it. | §4.4 |
| **Streamlit** | A Python script rendered as a web page — a demo tool, deliberately not a production frontend. | §5.1 |
| **Demo UI vs production frontend** | Re-runs the whole script per interaction, keeps session state in server memory, has no auth and little layout control. | §5.2 |
| **In-process call vs network call** | The UI calls `predict()` directly, so it runs without the API and Week 12's image contains only the API. | §5.3 |
| **Validation at the HTTP boundary** | The library check protects programmers with exceptions; the schema check answers strangers with status codes. Keep both. | §6.1 |
| **422 Unprocessable Entity** | Valid JSON that does not satisfy the contract — the client's problem, named per field, before any handler runs. | §6.2 |
| **500 Internal Server Error** | The request was fine and the server failed; log the traceback, return only that it happened. | §6.3 |
| **Not leaking internals** | A traceback in a response body hands a stranger your paths, versions and structure. | §6.3 |
| **503 Service Unavailable** | Up but not ready — a correct request that is worth retrying, unlike a 500. | §6.4 |
| **Out-of-distribution input** | In-range values in a combination the training data never contained; the model answers anyway, at 99.99997% confidence. | §6.5 |
| **Separation of concerns** | `api/` and `app/` both depend on `src/pipelines/`, on nothing of each other's, and `src/` on neither. | §7 |
| **Fat controller** | Business logic living inside a request handler, where nothing but an HTTP request can reach it. | §7 |
| **Lifespan / load once at start-up** | The model is loaded before the first request, so the cost is paid once and two requests cannot race on the file. | §8.1 |
| **Dependency injection (`Depends`)** | The endpoint declares what it needs; the tests override it with a model trained into `tmp_path`. | §8.2 |
| **Health endpoint** | A cheap, dependency-free answer to "can this process actually serve?", polled by containers and load balancers. | §8.3 |
| **`TestClient`** | Drives the app in-process — full request/response semantics, no port, milliseconds. | §9 |
| **Testing the contract, not the prediction** | Assert the label is *a* known crop and the response has the right shape; `== "jute"` breaks on every retrain. | §9 |

---

## Week 11 — Containerisation and continuous integration

Taught in [`docs/curriculum/week11/learning_notes.md`](curriculum/week11/learning_notes.md).

| Concept | One-line definition | Section |
| --- | --- | --- |
| **Environment consistency** | Pinned Python packages fix only half the environment; the interpreter, the libc and the OS userland are the half that breaks on someone else's machine. | §1.1 |
| **Container** | The program shipped together with its userland, libraries and start command, sharing the host kernel. | §1.2 |
| **Container vs virtualenv vs VM** | A virtualenv isolates packages, a container isolates the filesystem and process space, a VM isolates the kernel too — MBs, hundreds of MBs, GBs. | §1.2 |
| **Image vs container** | An image is a built, immutable, hashed artifact; a container is a process started from one, with a writable layer that dies with it. | §1.3 |
| **Base image** | The image yours starts `FROM`; `-slim` keeps glibc and drops the build tools, `alpine` swaps in musl and breaks scientific wheels. | §2.1 |
| **`WORKDIR` / `ENV`** | The working directory for every later instruction, and the interpreter settings (`PYTHONUNBUFFERED`, `PYTHONPATH`) a containerised process needs. | §2.2-2.3 |
| **`COPY` / `RUN`** | `COPY` brings files in from the build context; `RUN` executes a command at build time and keeps the resulting filesystem. | §2.4 |
| **Baking the artifact in** | Training during the build makes start-up a file read; the price is that retraining means rebuilding the image. | §2.5 |
| **Non-root container user** | The default user is root; a network-facing process should not be, and `/app` stays read-only to it. | §2.6 |
| **`EXPOSE` vs `-p`** | `EXPOSE` documents the port the process listens on; only `docker run -p` makes it reachable from the host. | §2.7 |
| **`HEALTHCHECK`** | Docker polling Week 10's `/health` from inside the container, so `docker ps` can say `healthy` rather than merely `up`. | §2.8 |
| **Exec form vs shell form** | The JSON-array `CMD` makes the server PID 1 and lets `docker stop`'s SIGTERM reach it; the string form hides it behind a shell. | §2.9 |
| **Binding to `0.0.0.0`** | Inside a container, `127.0.0.1` is reachable only from inside it, so a published port would forward to nothing. | §2.9 |
| **`CMD` vs `ENTRYPOINT`** | Arguments after the image name replace a `CMD` and are appended to an `ENTRYPOINT`. | §2.10 |
| **Layers and the build cache** | Each instruction is a cached filesystem diff; the first instruction whose inputs changed invalidates it and everything below. | §3.1 |
| **Layer ordering** | Least-frequently-changed first — dependencies before source — or every commit reinstalls the whole stack. | §3.2 |
| **Build context** | The directory uploaded to the daemon before any instruction runs; `COPY` can read nothing outside it. | §3.3 |
| **`.dockerignore`** | Keeps paths out of that upload — a speed question, a cache-stability question and, for secrets, a safety question. | §3.3 |
| **Trimmed deployment requirements** | The serving file lists what the server's imports actually reach; smaller image, smaller attack surface, and an honest statement of need. | §4 |
| **Version parity across requirements files** | Deployment pins must be identical, not merely compatible, to the development pins — an artifact loaded by a different scikit-learn is a silent bug. | §4.3 |
| **Continuous integration (CI)** | Every change automatically built and tested on a clean machine, so "does a fresh clone work?" is answered by a robot rather than by hope. | §5.1-5.2 |
| **Continuous delivery / deployment** | Automatically packaging every passing change into a release artifact, and (for deployment) shipping it — named this week, not implemented. | §5.1 |
| **Workflow, trigger, job, runner, step** | A YAML file; the event that starts it; a unit of work on its own fresh VM; the machine type; one command or reusable action, failing at the first non-zero exit. | §5.3 |
| **Status check** | The workflow's result attached to a commit or pull request, which a branch rule can require to be green before merging. | §5.5 |

---

## Week 12 — Final review and portfolio polish

Taught in [`docs/curriculum/week12/learning_notes.md`](curriculum/week12/learning_notes.md).

| Concept | One-line definition | Section |
| --- | --- | --- |
| **Production-ready** | Reproducible, installable, documented, bounded and independently verifiable — a green test suite is one of five conditions, not the whole of it. | §1.1-1.2 |
| **Documentation kinds (task / explanation / reference)** | Three different documents for three different readers; a task document that explains, or an explanation that lists flags, is broken. | §1.3 |
| **Reproducibility, strong form** | Not "it runs again for me" but "it produces the same numbers on a machine that has never seen it" — committed data, pinned versions, one seed, a derived artifact. | §1.4 |
| **Simulated fresh install** | A brand-new virtual environment inside the working copy: the closest an author can get to a stranger's clone, and the check CI performs on every push. | §1.4, validation §2 |
| **Repository audit / student review** | Reading your own project as a fixed persona with a fixed starting point, and recording every stumble as *"stopped at X because Y"*. | §2.1 |
| **Defect classes in documentation** | Broken links, stale code references, contradictory numbers, undefined jargon, unanswered forward references, placeholders — each mechanically checkable. | §2.2 |
| **README structure** | Problem, approach, results, how to run, what not to trust — in that order, because a reader who fails at question one never reaches question four. | §3.1 |
| **Showing command output** | Pasting what a command prints makes the reader's run self-checking and makes the claim auditable. | §3.2 |
| **Results table as evidence** | Listing the models that lost, with the protocol and the fold spread, is what shows the winner was chosen rather than found first. | §3.3 |
| **Known limitations** | What the data does not contain — region, season, soil, provenance, cost of error — stated where the reader passes it, not in a footnote. | §4.1-4.2 |
| **What an accuracy number is not** | A score on held-out rows of the same dataset is not evidence about a different sensor, country or year. | §4.3 |
| **Not-advice disclaimer** | A demonstration model must say plainly that it is not agronomic advice, and name what it does not know. | §4.4 |
| **Confidence is not certainty** | A normalised score across 22 classes, pushed toward the extremes by independent likelihoods, and least trustworthy exactly where it is highest — off-distribution. | §4.5 |
| **Presentation as the ethical failure mode** | The model is harmless; the caption claiming authority for it is not. | §4.6 |
| **Model versioning** | An artifact needs an identity: version string, data hash, code commit, library versions, hyperparameters, seed, metrics at training time. | §5.1 |
| **Model registry** | A store of artifacts against that metadata, with one marked `production`, the previous kept, and rollback as a pointer change. | §5.1 |
| **Semantic versioning for models** | Patch = retrained on more of the same; minor = new features or hyperparameters; major = the interface or label set changed. | §5.1 |
| **Model card** | The human-readable half of versioning: intended use, training data, performance, ethical considerations, caveats. | §5.1 |
| **Service monitoring vs model monitoring** | *Is the process healthy?* (rate, errors, latency) versus *are the answers still good?* — a silently wrong model returns 200 OK just as fast. | §5.2 |
| **Data drift (covariate shift)** | The inputs move, the relationship does not; detectable from requests alone, which is why it is monitored first. | §5.3 |
| **Concept drift** | The relationship itself moves, so past labels are now partly wrong; invisible to every input statistic, and needs outcomes. | §5.3 |
| **Ground-truth lag** | The label arrives a growing season later, partial (only the crop that was planted), biased (only those who took the advice) and noisy. | §5.4 |
| **Retraining trigger** | Scheduled, performance-based or drift-based — and always with the new model evaluated against the current one before it replaces it. | §5.5 |
| **Shadow deployment** | A candidate model answers every request in parallel with the live one; its answers are recorded, never returned. | §5.6 |
| **Canary release** | The candidate serves a small share of real traffic, so a bad model has a small blast radius. | §5.6 |
| **Rollback** | Returning to the previous known-good model without retraining — which requires that it still exists and can be named. | §5.1, §5.6 |
| **Naming an absence** | Saying precisely what you did not build, and what building it would take, is the honest form of a capability claim. | §5, §6 |
| **Withdrawing a promise in writing** | A forward reference that later weeks cannot honour is corrected out loud, not quietly dropped. | §6 |

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| ◀ [§12.5 Capstone reflection](curriculum/week12/capstone_reflection.md) | 🗺 [Roadmap](curriculum/README.md) | [Appendix B — Glossary](glossary.md) ▶ |

<!-- nav:end -->
