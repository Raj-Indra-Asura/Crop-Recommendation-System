# Capstone Reflection

Twelve weeks, one dataset, one running example. This is what was built, in
order, and what a person who worked through it can now do without help.

---

## What was built, week by week

| Week | What was built | The one idea it exists to teach |
| --- | --- | --- |
| **01** | `src/data/data_loader.py`, `src/data/validate_schema.py`, a pinned `requirements.txt`, a written problem statement, `notebooks/01_problem_definition.ipynb` | Frame the problem and refuse to trust the data: a dataset contract that fails loudly beats a plausible wrong answer downstream. |
| **02** | `src/utils/eda.py`, `src/utils/visualization.py`, `notebooks/02_EDA.ipynb` | Look before you model. Scale, class balance, correlation, outliers and leakage are all decided here, silently, whether or not you look. |
| **03** | `src/data/split.py`, `src/preprocessing/preprocessor.py`, `data/processed/{train,test}.csv` | Split first, fit the preprocessing on the training half only. The most common way to fool yourself is a scaler that has seen the test set. |
| **04** | `src/models/baseline.py`, `src/evaluation/metrics.py` | A number means nothing without something to beat. `DummyClassifier` sets the floor at 4.55% (1/22) and cross-validation replaces the single lucky split. |
| **05** | `src/models/classical_models.py` — logistic regression, KNN, Gaussian naive Bayes | Different model families make different assumptions; compare them on *identical* folds or you are comparing splits, not models. |
| **06** | SVM and decision trees in the same module; the depth sweep; decision-boundary plots | Overfitting made visible: an unlimited tree scores 100% on the rows it was fitted on and 98.52% on the ones it was not. |
| **07** | `src/models/ensemble_models.py` — random forest, gradient boosting (XGBoost with a scikit-learn fallback) | Many weak models beat one strong one — bagging attacks variance, boosting attacks bias — and a 0.23-point lead inside a 0.58-point fold spread is a tie. |
| **08** | `src/evaluation/tuning.py`, `src/evaluation/explainability.py`, the test set opened once | Tuning that gains 0.17 points against a 0.60-point spread gained nothing; the interesting content is the two wrong rows and why the model got them wrong. |
| **09** | `src/config.py`, `src/pipelines/training_pipeline.py`, `src/pipelines/predict_pipeline.py`, `models/crop_model.joblib` | A notebook is not a product. Preprocessing and model become one saved `Pipeline`, the artifact is derived rather than committed, and `predict({...})` is the whole interface. |
| **10** | `api/main.py`, `api/schemas.py`, `app/streamlit_app.py`, `docs/architecture.md` | A model reachable over HTTP with a typed contract: 422 when the caller is wrong, 500 when we are, and a demo UI that proves a human can use it. |
| **11** | `deployment/Dockerfile`, `deployment/requirements.txt`, `.dockerignore`, `.github/workflows/ci.yml`, `docs/deployment_guide.md` | "Works on my machine" is a bug report. An image ships the environment; CI proves a clean machine agrees. |
| **12** | This week: the audit, the finished `README.md`, the consolidated `docs/`, and the limitations written where a reader will see them | Finished is a claim somebody else tests. Reproducibility, documentation and honestly-stated limits are the difference between a project and a portfolio piece. |

Four things ran through every week and are worth naming separately, because
they are the habits rather than the topics:

* **Tests from Week 1.** 20 tests after Week 1, 404 after Week 10, and never a
  week where the suite was allowed to go red.
* **A written record in the week that incurred it.** Every concept entered
  `docs/ml_concepts.md` and `docs/glossary.md` in the same commit that first
  used it, which is why Week 12's audit had nothing to fix.
* **One seed, one config.** `RANDOM_STATE` in `src/config.py`, threaded through
  every split, fold and model, so any number in any document can be reproduced.
* **Logic in `src/`, never in a notebook.** Notebooks import; tests import the
  same thing. There is no code that only a kernel has ever run.

---

## The system, end to end

```
data/raw/Crop_recommendation.csv      2,200 rows, 22 crops, committed
        |  src/data/data_loader.py    validates columns, dtypes, size, labels
        v
   src/data/split.py                  stratified 80/20 -> 1,760 / 440
        |
        v
src/preprocessing/preprocessor.py     ColumnTransformer, fitted on train only
        |
        v
src/models/classical_models.py        GaussianNB(var_smoothing=1e-9)
        |
        v
src/pipelines/training_pipeline.py    one Pipeline -> models/crop_model.joblib
        |                             test accuracy 0.9955, macro F1 0.9954
        v
src/pipelines/predict_pipeline.py     predict({...}) -> "jute"
        |                                    \
        v                                     v
   api/main.py  POST /predict            app/streamlit_app.py  (in-process)
   api/schemas.py validates 7 fields
        |
        v
   {"crop":"jute","confidence":0.7253,...}     also inside deployment/Dockerfile
```

Every arrow is a file, every file is tested, and every decision behind an arrow
has a week that argues for it.

---

## What a graduate can do end to end

Given a new tabular dataset and a vague business question, a graduate of this
course can, unaided:

1. **Frame it** — write the problem statement, name the target, decide
   classification or regression, choose a metric, and write the non-goals down
   before any code.
2. **Ingest it defensively** — a loader with a schema contract that fails on the
   first invalid row rather than on a confusing result three weeks later.
3. **Explore it** — statistics, distributions, class balance, correlations,
   outliers, and a deliberate hunt for leakage.
4. **Prepare it** — a stratified split *first*, then a preprocessing pipeline
   fitted on the training half only, encoded labels, and the processed data on
   disk.
5. **Baseline it** — a dummy model and cross-validation, so that every later
   number has something to be better than.
6. **Model it** — fit and compare linear, distance-based, probabilistic,
   margin-based, tree-based and ensemble families on identical folds.
7. **Read the comparison honestly** — fold spread before mean, overfitting from
   the train/validation gap, and the discipline not to open the test set.
8. **Tune and evaluate** — grid and randomised search, and the judgement to say
   "this gained nothing"; then the test set once, a confusion matrix, and an
   error analysis in the language of the domain.
9. **Explain it** — permutation importance and SHAP on a single prediction, with
   the correlation trap named.
10. **Productionize it** — config, a fitted `Pipeline`, a saved artifact that is
    rebuilt rather than committed, and runnable entry points.
11. **Serve it** — a typed HTTP API with validation, honest status codes, a
    health endpoint and generated docs, plus a demo UI.
12. **Ship and check it** — a container that runs anywhere Docker runs, and CI
    that runs lint and the whole suite on a machine that is not theirs.
13. **Finish it** — audit the repository, write the README a stranger reads,
    state the limitations plainly, and describe what versioning and monitoring
    would add if it had real users.

The list is the course. The evidence that it was learned is that every item on
it exists in this repository, with the reasoning in
[`docs/ml_concepts.md`](../../ml_concepts.md) and the week that taught it
attached to every entry.

---

## The honest closing statement

The model in this repository is a Gaussian naive Bayes classifier that scores
99.55% on 440 held-out rows of one public dataset. It is a **demonstration**,
not agronomic advice, and the README says so in the reader's path rather than in
a footnote. There is no public deployment, no authentication, no model registry
and no monitoring; [Week 12's notes](learning_notes.md) §5 name each of them and
say what building them would involve.

Knowing what you did not build, and being able to say what you would build next
and why, is the last thing this course teaches — and the first thing anybody
reading a portfolio actually asks.
