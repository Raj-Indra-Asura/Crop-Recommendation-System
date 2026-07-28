# 🌱 Crop Recommendation System

A machine learning project **and** a beginner-friendly ML course in one
repository. It takes a student from zero knowledge to a deployed ML system,
using a single running example: given the growing conditions of a plot of land,
recommend the crop most suited to it.

```
Input:  N=90, P=42, K=43, temperature=25, humidity=80, ph=6.5, rainfall=200
Output: rice
```

The problem is supervised, multiclass classification over 22 crops and seven
numeric features.

---

## Quickstart

Targets Python 3.11. Python 3.12 also works — the Week 1 checks were recorded
on 3.12.3 — but 3.11 is the reference version the project pins its tooling to.

```bash
# 1. Clone and enter the repository
cd Crop-Recommendation-System

# 2. Create an isolated environment
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Check everything works
ruff check .
pytest

# 5. Open this week's notebook
jupyter notebook notebooks/04_baseline_models.ipynb
```

Step 5 starts a local Jupyter server and opens the notebook in your browser;
stop it with `Ctrl-C` in the terminal when you are done. Earlier weeks are
`notebooks/01_problem_definition.ipynb`, `notebooks/02_EDA.ipynb` and
`notebooks/03_data_preparation.ipynb`.

Load the data from Python:

```python
from src.data import load_data

frame = load_data()   # validates columns, dtypes, row count and label set
print(frame.shape)    # -> (2200, 8)
```

From Week 3 the prepared splits are also on disk, so a later week can start
without re-running the notebook:

```python
import pandas as pd

train = pd.read_csv("data/processed/train.csv")   # 1,760 rows, stratified
test = pd.read_csv("data/processed/test.csv")     # 440 rows, 20 per crop
```

From Week 4 the number to beat is fixed: a `DummyClassifier` scores **4.55%**
(1/22) under 5-fold stratified cross-validation, so any real model that does not
clear that is broken or trivial rather than merely weak.

If `data/raw/Crop_recommendation.csv` is ever missing, `load_data()` fails with
a message naming the file and where to obtain it. Restore the committed file —
never substitute randomly generated data, or every later week's results are
silently invalidated.

---

## Repository layout

| Path | Contents |
| --- | --- |
| `docs/curriculum/weekXX/` | The course: `syllabus`, `learning_notes`, `exercises`, `validation` per week |
| `docs/ml_concepts.md` | Running index of every concept, by week |
| `docs/glossary.md` | Alphabetical term reference |
| `data/raw/` | The original dataset — read-only, and deliberately **not** gitignored |
| `data/processed/` | Anything derived from the raw data — from Week 3, the train/test splits |
| `src/` | Reusable, tested implementation code |
| `notebooks/` | Exploratory analysis, importing from `src/` |
| `models/` | Trained artifacts — generated on demand, never committed |
| `tests/` | Automated test suite |

Logic lives in `src/` and is imported by notebooks, so that notebooks and tests
exercise the same code.

---

## Course progress

Filled in one row per week as the course proceeds.

| Week | Status | Notes |
| --- | --- | --- |
| 01 — Framing the problem, environment, loading & validating data | ✅ Complete | Dataset contract enforced by `validate_dataset()`; 20 tests passing. [Docs](docs/curriculum/week01/) |
| 02 — Exploratory data analysis | ✅ Complete | Statistics, distributions, correlation, outliers and data leakage; helpers in `src/utils/eda.py`, 42 tests passing. [Docs](docs/curriculum/week02/) |
| 03 — Data preparation | ✅ Complete | Label encoding, stratified 80/20 split and a train-fitted `ColumnTransformer`; helpers in `src/data/split.py` and `src/preprocessing/preprocessor.py`, 74 tests passing. [Docs](docs/curriculum/week03/) |
| 04 — Baseline models | ✅ Complete | `DummyClassifier` baseline at **4.55%** (1/22) under 5-fold stratified CV; helpers in `src/models/baseline.py` and `src/evaluation/metrics.py`, 112 tests passing. [Docs](docs/curriculum/week04/) |
| 05 — Classification models | ⬜ Not started | |
| 06 — Model selection & tuning | ⬜ Not started | |
| 07 — Model explainability | ⬜ Not started | |
| 08 — Pipelines | ⬜ Not started | |
| 09 — Testing & packaging | ⬜ Not started | |
| 10 — Serving an API | ⬜ Not started | |
| 11 — Streamlit application | ⬜ Not started | |
| 12 — Containerisation, CI & deployment | ⬜ Not started | |

### Week 1 Definition of Done

| Requirement | Status |
| --- | --- |
| `docs/curriculum/week01/{syllabus,learning_notes,exercises,validation}.md` exist | ✅ |
| New code has docstrings and passes lint (`ruff check .`) | ✅ |
| New behaviour has tests and `pytest` passes | ✅ 20 passed |
| `requirements.txt` updated, every dependency pinned | ✅ |
| README progress table updated | ✅ |
| `docs/ml_concepts.md` has an entry per new concept | ✅ |
| `validation.md` commands actually run, real output pasted | ✅ |
| `notebooks/01_problem_definition.ipynb` committed with executed output | ✅ |

### Week 2 Definition of Done

| Requirement | Status |
| --- | --- |
| `docs/curriculum/week02/{syllabus,learning_notes,exercises,validation}.md` exist | ✅ |
| New code has docstrings and passes lint (`ruff check .`) | ✅ |
| New behaviour has tests and `pytest` passes | ✅ 42 passed (20 + 22) |
| `requirements.txt` updated, every dependency pinned | ✅ no change — `matplotlib`/`seaborn` pinned in Week 1 |
| README progress table updated | ✅ |
| `docs/ml_concepts.md` has an entry per new concept | ✅ |
| `validation.md` commands actually run, real output pasted | ✅ |
| `notebooks/02_EDA.ipynb` committed with executed output | ✅ |

### Week 3 Definition of Done

| Requirement | Status |
| --- | --- |
| `docs/curriculum/week03/{syllabus,learning_notes,exercises,validation}.md` exist | ✅ |
| New code has docstrings and passes lint (`ruff check .`) | ✅ |
| New behaviour has tests and `pytest` passes | ✅ 74 passed (20 + 22 + 32) |
| `requirements.txt` updated, every dependency pinned | ✅ no change — `scikit-learn` pinned in Week 1 |
| README progress table updated | ✅ |
| `docs/ml_concepts.md` has an entry per new concept | ✅ |
| `validation.md` commands actually run, real output pasted | ✅ |
| `notebooks/03_data_preparation.ipynb` committed with executed output | ✅ |

### Week 4 Definition of Done

| Requirement | Status |
| --- | --- |
| `docs/curriculum/week04/{syllabus,learning_notes,exercises,validation}.md` exist | ✅ |
| New code has docstrings and passes lint (`ruff check .`) | ✅ |
| New behaviour has tests and `pytest` passes | ✅ 112 passed (20 + 22 + 32 + 38) |
| `requirements.txt` updated, every dependency pinned | ✅ no change — `scikit-learn` pinned in Week 1 |
| README progress table updated | ✅ |
| `docs/ml_concepts.md` has an entry per new concept | ✅ |
| `validation.md` commands actually run, real output pasted | ✅ |
| `notebooks/04_baseline_models.ipynb` committed with executed output | ✅ |

---

## Dataset

[Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
— 2,200 rows, 22 crop labels, 8 columns (`N`, `P`, `K`, `temperature`,
`humidity`, `ph`, `rainfall`, `label`). At roughly 150 KB it is committed to
version control so that every student and every CI run uses byte-identical
data.

## Course reference

The curriculum's structure is loosely inspired by the conceptual progression in
*Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* by Aurélien
Géron. All explanatory text in this repository is original and written for this
project.
