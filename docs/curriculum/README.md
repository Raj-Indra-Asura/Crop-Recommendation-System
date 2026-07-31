# The Crop Recommendation Book — Roadmap

A twelve-chapter path from *what is machine learning* to a containerised,
CI-checked, documented service, built on one running example: predicting a crop
from seven soil and climate measurements.

Read it like a book. Chapters are in order, sections inside a chapter are in
order, and the code and notebooks named inside a chapter are read at the point
the chapter names them. Nothing later is needed to understand anything earlier.

---

## How to read this book

Every chapter has the same five sections, always in this order:

| Order | Section | What it is |
| --- | --- | --- |
| 0 | `README.md` (chapter cover) | where you are, what you need first, the ordered reading list, and the checklist that lets you leave |
| 1 | `syllabus.md` | objectives, prerequisites, concepts, deliverables — read first, it is the map of the chapter |
| 2 | `learning_notes.md` | the teaching text: what, why, where, how, and the common mistakes — read top to bottom |
| 3 | `exercises.md` | beginner, then intermediate, then challenge — do them before moving on |
| 4 | `validation.md` | run the commands, compare against the real recorded output |

Rules that make the order work:

1. **Do not skip the syllabus.** It states the prerequisites; if one is missing,
   the chapter it came from is named and you go back.
2. **Read the code the chapter builds before its exercises**, in the order the
   chapter cover lists it — helpers, then tests, then notebook.
3. **Run the validation commands yourself.** Each one has real recorded output
   to compare against; a mismatch is a problem to solve now, not later.
4. **Every page has a footer** with previous / up / next, so you can always walk
   the book without coming back here.

## Before Chapter 1

* Read the project [README](../../README.md) — problem, results, and what the
  finished system does.
* Have Python 3.11 (3.12 also works), git and a terminal. Everything else is
  installed in Chapter 1.
* You need basic Python — variables, functions, imports, running a script. No
  machine learning, statistics or pandas knowledge is assumed.

## The whole book at a glance

| Chapter | Title | Part | You leave able to |
| --- | --- | --- | --- |
| [1](week01/README.md) | Framing the Problem and Meeting the Data | Part I | You can state the problem as supervised multiclass classification, naming the seven inputs, the one output and the success measure |
| [2](week02/README.md) | Exploratory Data Analysis | Part I | You can describe the class balance, the strongest correlations and the outliers, and say what each means for modelling |
| [3](week03/README.md) | Data Preparation | Part I | You can explain why the scaler is fitted on the training set only, and what breaks if it is not |
| [4](week04/README.md) | Baseline Models | Part II | You can say what the baseline accuracy is (4.55%) and why it is 1/22 |
| [5](week05/README.md) | Classification Models | Part II | You can describe how each of the three algorithms decides, in one sentence each |
| [6](week06/README.md) | Margin-based and Tree-based Models | Part II | You can show overfitting with a tree-depth sweep and read the train/validation gap |
| [7](week07/README.md) | Ensembles | Part II | You can explain bagging and boosting mechanically, not just by name |
| [8](week08/README.md) | Model Evaluation & Explainability | Part III | You can read a confusion matrix and describe the two rows the model gets wrong |
| [9](week09/README.md) | Productionizing the Model | Part IV | `python -m src.pipelines.training_pipeline` writes `models/crop_model.joblib` and reports 99.55% |
| [10](week10/README.md) | Serving the Model Over HTTP | Part IV | You can start the API, call `/predict` with curl, and explain each status code it can return |
| [11](week11/README.md) | Containerization and Continuous Integration | Part IV | You can build the image, run it, and get a healthy `/health` response from the container |
| [12](week12/README.md) | Final Review and Portfolio Polish | Part V | `pytest` is green from a fresh virtual environment (404 passed, 1 skipped) |

---

## Part I — Foundations (Weeks 1-3)

The problem, the data, and the preparation everything else depends on.

### Chapter 1 — Framing the Problem and Meeting the Data — From a vague wish to a machine learning problem statement

*Start here:* [chapter cover](week01/README.md)

| Order | Read | Then |
| --- | --- | --- |
| 1 | [`week01/syllabus.md`](week01/syllabus.md) | §1.1 Syllabus |
| 2 | [`week01/learning_notes.md`](week01/learning_notes.md) | §1.2 Learning notes |
| 3 | [`src/data/data_loader.py`](../../src/data/data_loader.py) | `load_data()`, the single entry point to the data |
| 4 | [`src/data/validate_schema.py`](../../src/data/validate_schema.py) | `validate_dataset()`, the dataset contract |
| 5 | [`tests/test_data_loader.py`](../../tests/test_data_loader.py) | the contract enforced automatically |
| 6 | [`requirements.txt`](../../requirements.txt) | nine pinned packages |
| 7 | [`notebooks/01_problem_definition.ipynb`](../../notebooks/01_problem_definition.ipynb) | the written problem framing plus a first look at the dataframe |
| 8 | [`week01/exercises.md`](week01/exercises.md) | §1.3 Exercises |
| 9 | [`week01/validation.md`](week01/validation.md) | §1.4 Validation |

### Chapter 2 — Exploratory Data Analysis — Looking before leaping: understanding the data statistically and visually

*Start here:* [chapter cover](week02/README.md)

| Order | Read | Then |
| --- | --- | --- |
| 1 | [`week02/syllabus.md`](week02/syllabus.md) | §2.1 Syllabus |
| 2 | [`week02/learning_notes.md`](week02/learning_notes.md) | §2.2 Learning notes |
| 3 | [`src/utils/eda.py`](../../src/utils/eda.py) | nine documented exploration helpers |
| 4 | [`tests/test_eda.py`](../../tests/test_eda.py) | every helper run on a small synthetic frame |
| 5 | [`notebooks/02_EDA.ipynb`](../../notebooks/02_EDA.ipynb) | the full exploration, ending in four written findings |
| 6 | [`week02/exercises.md`](week02/exercises.md) | §2.3 Exercises |
| 7 | [`week02/validation.md`](week02/validation.md) | §2.4 Validation |

### Chapter 3 — Data Preparation — Turning raw data into model-ready data, correctly

*Start here:* [chapter cover](week03/README.md)

| Order | Read | Then |
| --- | --- | --- |
| 1 | [`week03/syllabus.md`](week03/syllabus.md) | §3.1 Syllabus |
| 2 | [`week03/learning_notes.md`](week03/learning_notes.md) | §3.2 Learning notes |
| 3 | [`src/data/split.py`](../../src/data/split.py) | `stratified_split()` and `class_proportions()` |
| 4 | [`src/preprocessing/preprocessor.py`](../../src/preprocessing/preprocessor.py) | `build_preprocessor()` and `build_preprocessing_pipeline()` |
| 5 | [`tests/test_preprocessing.py`](../../tests/test_preprocessing.py) | the split is stratified, reproducible, and the test rows are provably not part of the fit |
| 6 | [`notebooks/03_data_preparation.ipynb`](../../notebooks/03_data_preparation.ipynb) | the full preparation, writing five files to `data/processed/` |
| 7 | [`week03/exercises.md`](week03/exercises.md) | §3.3 Exercises |
| 8 | [`week03/validation.md`](week03/validation.md) | §3.4 Validation |

## Part II — Modelling (Weeks 4-7)

From a deliberately stupid baseline to five algorithms and two ensembles, all
compared on identical folds.

### Chapter 4 — Baseline Models — Establishing what "good" means, before building anything real

*Start here:* [chapter cover](week04/README.md)

| Order | Read | Then |
| --- | --- | --- |
| 1 | [`week04/syllabus.md`](week04/syllabus.md) | §4.1 Syllabus |
| 2 | [`week04/learning_notes.md`](week04/learning_notes.md) | §4.2 Learning notes |
| 3 | [`src/models/baseline.py`](../../src/models/baseline.py) | `get_baseline_model(strategy)` |
| 4 | [`src/evaluation/metrics.py`](../../src/evaluation/metrics.py) | `evaluate_model()`, `cross_validated_accuracy()` and `build_cv()`, extended in later chapters, never replaced |
| 5 | [`tests/test_baseline.py`](../../tests/test_baseline.py) | the factory, the folds and the 1/22 result on the real data |
| 6 | [`notebooks/04_baseline_models.ipynb`](../../notebooks/04_baseline_models.ipynb) | the number every future model must beat |
| 7 | [`week04/exercises.md`](week04/exercises.md) | §4.3 Exercises |
| 8 | [`week04/validation.md`](week04/validation.md) | §4.4 Validation |

### Chapter 5 — Classification Models — The first real algorithms: three ways to draw a boundary, compared fairly

*Start here:* [chapter cover](week05/README.md)

| Order | Read | Then |
| --- | --- | --- |
| 1 | [`week05/syllabus.md`](week05/syllabus.md) | §5.1 Syllabus |
| 2 | [`week05/learning_notes.md`](week05/learning_notes.md) | §5.2 Learning notes |
| 3 | [`src/models/classical_models.py`](../../src/models/classical_models.py) | `get_logistic_regression()`, `get_knn()`, `get_naive_bayes()` and the `CLASSICAL_MODEL_FACTORIES` registry |
| 4 | [`tests/test_classical_models.py`](../../tests/test_classical_models.py) | factories, the shared training loop and algorithm-specific behaviour |
| 5 | [`notebooks/05_classification_models.ipynb`](../../notebooks/05_classification_models.ipynb) | Part 1 — the four-row results table Chapters 6-8 extend |
| 6 | [`week05/exercises.md`](week05/exercises.md) | §5.3 Exercises |
| 7 | [`week05/validation.md`](week05/validation.md) | §5.4 Validation |

### Chapter 6 — Margin-based and Tree-based Models — Two more ways to draw a boundary, and the first honest look at overfitting

*Start here:* [chapter cover](week06/README.md)

| Order | Read | Then |
| --- | --- | --- |
| 1 | [`week06/syllabus.md`](week06/syllabus.md) | §6.1 Syllabus |
| 2 | [`week06/learning_notes.md`](week06/learning_notes.md) | §6.2 Learning notes |
| 3 | [`src/models/classical_models.py`](../../src/models/classical_models.py) | `get_svm()` and `get_decision_tree()` added beside the Chapter 5 factories |
| 4 | [`src/utils/visualization.py`](../../src/utils/visualization.py) | `plot_decision_boundary(model, X_2d, y)` |
| 5 | [`tests/test_classical_models.py`](../../tests/test_classical_models.py) | extended: support vectors, kernels, tree depth and the widening train/validation gap |
| 6 | [`notebooks/05_classification_models.ipynb`](../../notebooks/05_classification_models.ipynb) | Part 2 (§8-§15) — a six-row results table |
| 7 | [`week06/exercises.md`](week06/exercises.md) | §6.3 Exercises |
| 8 | [`week06/validation.md`](week06/validation.md) | §6.4 Validation |

### Chapter 7 — Ensembles — Many weak models beat one strong one — bagging, boosting, and feature importance

*Start here:* [chapter cover](week07/README.md)

| Order | Read | Then |
| --- | --- | --- |
| 1 | [`week07/syllabus.md`](week07/syllabus.md) | §7.1 Syllabus |
| 2 | [`week07/learning_notes.md`](week07/learning_notes.md) | §7.2 Learning notes |
| 3 | [`src/models/ensemble_models.py`](../../src/models/ensemble_models.py) | `get_random_forest()`, `get_gradient_boosting()`, the XGBoost fallback and the label adapter |
| 4 | [`tests/test_ensemble_models.py`](../../tests/test_ensemble_models.py) | variance reduction, decorrelation, error correction, and importance limits — with and without XGBoost installed |
| 5 | [`notebooks/06_model_selection.ipynb`](../../notebooks/06_model_selection.ipynb) | Part 1 (§0-§7) — an eight-row results table with feature importances |
| 6 | [`week07/exercises.md`](week07/exercises.md) | §7.3 Exercises |
| 7 | [`week07/validation.md`](week07/validation.md) | §7.4 Validation |

## Part III — Evaluation and Explanation (Week 8)

Tuning honestly, opening the test set once, and explaining what the model
learned.

### Chapter 8 — Model Evaluation & Explainability — Past accuracy: confusion matrices, honest hyperparameter search, and explaining a prediction

*Start here:* [chapter cover](week08/README.md)

| Order | Read | Then |
| --- | --- | --- |
| 1 | [`week08/syllabus.md`](week08/syllabus.md) | §8.1 Syllabus |
| 2 | [`week08/learning_notes.md`](week08/learning_notes.md) | §8.2 Learning notes |
| 3 | [`src/evaluation/tuning.py`](../../src/evaluation/tuning.py) | `tune_model()` over grid and randomised search |
| 4 | [`src/evaluation/explainability.py`](../../src/evaluation/explainability.py) | `permutation_feature_importance()` and `explain_prediction()` |
| 5 | [`src/evaluation/metrics.py`](../../src/evaluation/metrics.py) | `confusion_frame()`, macro/weighted F1 and the confusion matrix |
| 6 | [`tests/test_tuning.py`](../../tests/test_tuning.py) | the searches respect the project's folds and seed |
| 7 | [`tests/test_explainability.py`](../../tests/test_explainability.py) | permutation importance and the documented SHAP fallback |
| 8 | [`notebooks/06_model_selection.ipynb`](../../notebooks/06_model_selection.ipynb) | Part 2 (§8-§15) — the test set opened once and the final-model decision |
| 9 | [`notebooks/07_model_explainability.ipynb`](../../notebooks/07_model_explainability.ipynb) | the correlation trap, SHAP plots, and one prediction explained in plain language |
| 10 | [`week08/exercises.md`](week08/exercises.md) | §8.3 Exercises |
| 11 | [`week08/validation.md`](week08/validation.md) | §8.4 Validation |

## Part IV — Production (Weeks 9-11)

One artifact, one HTTP API, one image, one CI pipeline.

### Chapter 9 — Productionizing the Model — From notebook to pipeline: one fitted object, one saved artifact, one prediction function

*Start here:* [chapter cover](week09/README.md)

| Order | Read | Then |
| --- | --- | --- |
| 1 | [`week09/syllabus.md`](week09/syllabus.md) | §9.1 Syllabus |
| 2 | [`week09/learning_notes.md`](week09/learning_notes.md) | §9.2 Learning notes |
| 3 | [`src/config.py`](../../src/config.py) | paths, seed, chosen model name and hyperparameters — all inert |
| 4 | [`src/pipelines/training_pipeline.py`](../../src/pipelines/training_pipeline.py) | `build_model()`, `train_pipeline()`, `save_pipeline()` and `python -m src.pipelines.training_pipeline` |
| 5 | [`src/pipelines/predict_pipeline.py`](../../src/pipelines/predict_pipeline.py) | `load_pipeline()` (train-on-demand), `predict()` and `predict_proba()` |
| 6 | [`tests/test_training_pipeline.py`](../../tests/test_training_pipeline.py) | the pipeline trains, scores and saves |
| 7 | [`tests/test_predict_pipeline.py`](../../tests/test_predict_pipeline.py) | the artifact reloads and predicts from a plain dict |
| 8 | [`week09/exercises.md`](week09/exercises.md) | §9.3 Exercises |
| 9 | [`week09/validation.md`](week09/validation.md) | §9.4 Validation |

### Chapter 10 — Serving the Model Over HTTP — From `predict()` to `POST /predict`: an API, a demo UI, and the difference between them

*Start here:* [chapter cover](week10/README.md)

| Order | Read | Then |
| --- | --- | --- |
| 1 | [`week10/syllabus.md`](week10/syllabus.md) | §10.1 Syllabus |
| 2 | [`week10/learning_notes.md`](week10/learning_notes.md) | §10.2 Learning notes |
| 3 | [`api/schemas.py`](../../api/schemas.py) | `CropFeatures`, `PredictionResponse`, `HealthResponse` |
| 4 | [`api/main.py`](../../api/main.py) | `POST /predict`, `GET /health`, the lifespan loader and the test seam |
| 5 | [`app/streamlit_app.py`](../../app/streamlit_app.py) | the seven-field demo form |
| 6 | [`tests/test_api.py`](../../tests/test_api.py) | 200, 422, 500, 503 and `/health` |
| 7 | [`docs/architecture.md`](../architecture.md) | the request flow and the layering rule |
| 8 | [`week10/exercises.md`](week10/exercises.md) | §10.3 Exercises |
| 9 | [`week10/validation.md`](week10/validation.md) | §10.4 Validation |

### Chapter 11 — Containerization and Continuous Integration — "It works on my machine" is not a deployment: an image, trimmed requirements, and a CI pipeline

*Start here:* [chapter cover](week11/README.md)

| Order | Read | Then |
| --- | --- | --- |
| 1 | [`week11/syllabus.md`](week11/syllabus.md) | §11.1 Syllabus |
| 2 | [`week11/learning_notes.md`](week11/learning_notes.md) | §11.2 Learning notes |
| 3 | [`deployment/Dockerfile`](../../deployment/Dockerfile) | slim base, cached dependency layer, model trained at build, non-root user, healthcheck |
| 4 | [`deployment/requirements.txt`](../../deployment/requirements.txt) | seven pins, each justified |
| 5 | [`.dockerignore`](../../.dockerignore) | what never enters the build context |
| 6 | [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) | `ruff check .` and `pytest` on every push and PR |
| 7 | [`docs/deployment_guide.md`](../deployment_guide.md) | the exact build and run commands, and troubleshooting |
| 8 | [`week11/exercises.md`](week11/exercises.md) | §11.3 Exercises |
| 9 | [`week11/validation.md`](week11/validation.md) | §11.4 Validation |

## Part V — Review (Week 12)

Auditing the whole repository and defending the claim that it is finished.

### Chapter 12 — Final Review and Portfolio Polish — Finished is a claim you have to defend: auditing a repository and writing the reflection

*Start here:* [chapter cover](week12/README.md)

| Order | Read | Then |
| --- | --- | --- |
| 1 | [`week12/syllabus.md`](week12/syllabus.md) | §12.1 Syllabus |
| 2 | [`week12/learning_notes.md`](week12/learning_notes.md) | §12.2 Learning notes |
| 3 | [`README.md`](../../README.md) | problem, approach, results, limits — the front door |
| 4 | [`docs/architecture.md`](../architecture.md) | the full raw-CSV-to-response path |
| 5 | [`docs/ml_concepts.md`](../ml_concepts.md) | the complete twelve-chapter concept index |
| 6 | [`docs/glossary.md`](../glossary.md) | the alphabetical index |
| 7 | [`week12/exercises.md`](week12/exercises.md) | §12.3 Exercises |
| 8 | [`week12/validation.md`](week12/validation.md) | §12.4 Validation |
| 9 | [`week12/capstone_reflection.md`](week12/capstone_reflection.md) | §12.5 Capstone reflection |

---

## Appendices — reference, not chapters

| Appendix | File | Use it when |
| --- | --- | --- |
| Appendix A | [`docs/ml_concepts.md`](../ml_concepts.md) | every concept in the order the book introduces it |
| Appendix B | [`docs/glossary.md`](../glossary.md) | the same terms alphabetically, each naming the chapter that teaches it |
| Appendix C | [`docs/architecture.md`](../architecture.md) | the full path from raw CSV to HTTP response |
| Appendix D | [`docs/deployment_guide.md`](../deployment_guide.md) | build, run, check, troubleshoot |

## The exact file order, start to finish

This is the book's spine: the documents in the one order they are meant to be
read. Tick them off as you go.

* [x] **Roadmap — how to read this book** — you are here
* [ ] [Chapter 1 — Framing the Problem and Meeting the Data](week01/README.md)
* [ ] [§1.1 Syllabus](week01/syllabus.md)
* [ ] [§1.2 Learning notes](week01/learning_notes.md)
* [ ] [§1.3 Exercises](week01/exercises.md)
* [ ] [§1.4 Validation](week01/validation.md)
* [ ] [Chapter 2 — Exploratory Data Analysis](week02/README.md)
* [ ] [§2.1 Syllabus](week02/syllabus.md)
* [ ] [§2.2 Learning notes](week02/learning_notes.md)
* [ ] [§2.3 Exercises](week02/exercises.md)
* [ ] [§2.4 Validation](week02/validation.md)
* [ ] [Chapter 3 — Data Preparation](week03/README.md)
* [ ] [§3.1 Syllabus](week03/syllabus.md)
* [ ] [§3.2 Learning notes](week03/learning_notes.md)
* [ ] [§3.3 Exercises](week03/exercises.md)
* [ ] [§3.4 Validation](week03/validation.md)
* [ ] [Chapter 4 — Baseline Models](week04/README.md)
* [ ] [§4.1 Syllabus](week04/syllabus.md)
* [ ] [§4.2 Learning notes](week04/learning_notes.md)
* [ ] [§4.3 Exercises](week04/exercises.md)
* [ ] [§4.4 Validation](week04/validation.md)
* [ ] [Chapter 5 — Classification Models](week05/README.md)
* [ ] [§5.1 Syllabus](week05/syllabus.md)
* [ ] [§5.2 Learning notes](week05/learning_notes.md)
* [ ] [§5.3 Exercises](week05/exercises.md)
* [ ] [§5.4 Validation](week05/validation.md)
* [ ] [Chapter 6 — Margin-based and Tree-based Models](week06/README.md)
* [ ] [§6.1 Syllabus](week06/syllabus.md)
* [ ] [§6.2 Learning notes](week06/learning_notes.md)
* [ ] [§6.3 Exercises](week06/exercises.md)
* [ ] [§6.4 Validation](week06/validation.md)
* [ ] [Chapter 7 — Ensembles](week07/README.md)
* [ ] [§7.1 Syllabus](week07/syllabus.md)
* [ ] [§7.2 Learning notes](week07/learning_notes.md)
* [ ] [§7.3 Exercises](week07/exercises.md)
* [ ] [§7.4 Validation](week07/validation.md)
* [ ] [Chapter 8 — Model Evaluation & Explainability](week08/README.md)
* [ ] [§8.1 Syllabus](week08/syllabus.md)
* [ ] [§8.2 Learning notes](week08/learning_notes.md)
* [ ] [§8.3 Exercises](week08/exercises.md)
* [ ] [§8.4 Validation](week08/validation.md)
* [ ] [Chapter 9 — Productionizing the Model](week09/README.md)
* [ ] [§9.1 Syllabus](week09/syllabus.md)
* [ ] [§9.2 Learning notes](week09/learning_notes.md)
* [ ] [§9.3 Exercises](week09/exercises.md)
* [ ] [§9.4 Validation](week09/validation.md)
* [ ] [Chapter 10 — Serving the Model Over HTTP](week10/README.md)
* [ ] [§10.1 Syllabus](week10/syllabus.md)
* [ ] [§10.2 Learning notes](week10/learning_notes.md)
* [ ] [§10.3 Exercises](week10/exercises.md)
* [ ] [§10.4 Validation](week10/validation.md)
* [ ] [Chapter 11 — Containerization and Continuous Integration](week11/README.md)
* [ ] [§11.1 Syllabus](week11/syllabus.md)
* [ ] [§11.2 Learning notes](week11/learning_notes.md)
* [ ] [§11.3 Exercises](week11/exercises.md)
* [ ] [§11.4 Validation](week11/validation.md)
* [ ] [Chapter 12 — Final Review and Portfolio Polish](week12/README.md)
* [ ] [§12.1 Syllabus](week12/syllabus.md)
* [ ] [§12.2 Learning notes](week12/learning_notes.md)
* [ ] [§12.3 Exercises](week12/exercises.md)
* [ ] [§12.4 Validation](week12/validation.md)
* [ ] [§12.5 Capstone reflection](week12/capstone_reflection.md)
* [ ] [Appendix A — Concepts by chapter](../ml_concepts.md)
* [ ] [Appendix B — Glossary](../glossary.md)
* [ ] [Appendix C — Architecture](../architecture.md)
* [ ] [Appendix D — Deployment guide](../deployment_guide.md)

The code and notebooks each chapter builds are listed on that chapter's cover,
at the point in the chapter where they should be read.

## Keeping this order honest

`tests/test_curriculum_links.py` runs with the rest of the suite, on every push
and pull request. It checks that every relative link and `#anchor` in every
Markdown file resolves, that the list above names each curriculum document
exactly once, that all twelve chapters appear in order, and that each page's
previous / next footer agrees with the position the list gives it. If a document
is added, renamed or reordered, that test fails until this roadmap and the
footers are brought back into agreement.

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| _Start of the book_ | 🗺 [Roadmap](README.md) | [Chapter 1 — Framing the Problem and Meeting the Data](week01/README.md) ▶ |

<!-- nav:end -->
