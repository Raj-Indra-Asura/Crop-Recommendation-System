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
