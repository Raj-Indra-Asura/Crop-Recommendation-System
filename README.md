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

## ⚠️ Current status: dataset missing

`data/raw/Crop_recommendation.csv` is **not present** in this repository, but
the project is built on the assumption that it is committed here. Until it is
restored:

* the seven dataset contract tests **skip** instead of running;
* `notebooks/01_problem_definition.ipynb` has not been created, because
  notebooks in this project are only committed with genuinely executed output.

To fix: download `Crop_recommendation.csv` from the
[Kaggle dataset page](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset),
place it at `data/raw/Crop_recommendation.csv`, and commit it. See
[`docs/curriculum/week01/validation.md`](docs/curriculum/week01/validation.md).

Do **not** substitute randomly generated data — every later week's results
would be silently invalidated.

---

## Quickstart

Requires Python 3.11.

```bash
# 1. Clone and enter the repository
cd Crop-Recommendation-System

# 2. Create an isolated environment
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Check everything works
ruff check .
pytest
```

Load the data from Python:

```python
from src.data import load_raw_data

frame = load_raw_data()   # validates shape, columns and class count
print(frame.shape)        # -> (2200, 8)
```

---

## Repository layout

| Path | Contents |
| --- | --- |
| `docs/curriculum/weekXX/` | The course: `syllabus`, `learning_notes`, `exercises`, `validation` per week |
| `docs/ml_concepts.md` | Running index of every concept, by week |
| `docs/glossary.md` | Alphabetical term reference |
| `data/raw/` | The original dataset — read-only, and deliberately **not** gitignored |
| `data/processed/` | Anything derived from the raw data |
| `src/` | Reusable, tested implementation code |
| `notebooks/` | Exploratory analysis, importing from `src/` |
| `models/` | Trained artifacts — generated on demand, never committed |
| `tests/` | Automated test suite |

Logic lives in `src/` and is imported by notebooks, so that notebooks and tests
exercise the same code.

---

## Course progress

| Week | Topic | Status | Docs |
| --- | --- | --- | --- |
| 01 | Framing the problem, environment setup, loading & validating data | ⚠️ Partial — blocked on missing dataset | [week01](docs/curriculum/week01/) |
| 02 | Exploratory data analysis | ⬜ Not started | — |
| 03 | Data preparation | ⬜ Not started | — |
| 04 | Baseline models | ⬜ Not started | — |
| 05 | Classification models | ⬜ Not started | — |
| 06 | Model selection & tuning | ⬜ Not started | — |
| 07 | Model explainability | ⬜ Not started | — |
| 08 | Pipelines | ⬜ Not started | — |
| 09 | Testing & packaging | ⬜ Not started | — |
| 10 | Serving an API | ⬜ Not started | — |
| 11 | Streamlit application | ⬜ Not started | — |
| 12 | Containerisation, CI & deployment | ⬜ Not started | — |

### Week 1 Definition of Done

| Requirement | Status |
| --- | --- |
| `docs/curriculum/week01/{syllabus,learning_notes,exercises,validation}.md` exist | ✅ |
| New code has docstrings and passes lint (`ruff check .`) | ✅ |
| New behaviour has tests and `pytest` passes | ⚠️ 3 passed, 7 skipped — contract tests need the dataset |
| `requirements.txt` updated, every dependency pinned | ✅ |
| README progress table updated | ✅ |
| `docs/ml_concepts.md` has an entry per new concept | ✅ |
| `validation.md` commands actually run, real output pasted | ✅ |
| `notebooks/01_problem_definition.ipynb` committed with executed output | ❌ Blocked on missing dataset |

Week 2 does not begin until the two blocked items are resolved.

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
