# Week 12 — Final Review and Portfolio Polish

> 🗺 [Roadmap](../README.md) › [Part V — Review (Week 12)](../README.md#part-v--review-week-12) › [Chapter 12 — Final Review and Portfolio Polish](README.md) › **§12.1 Syllabus**

## Title

**Finished is a claim you have to defend: auditing a repository, writing the
README a stranger reads first, and saying plainly what the model is not**

## Learning objectives

By the end of this week a student should be able to:

1. Define **production-ready** in terms a reviewer can check — reproducibility,
   documentation completeness, a stated failure surface and named limitations —
   and explain why "the tests pass" is a necessary but not sufficient condition.
2. Perform a **repository audit**: read the project as somebody who has never
   seen it, and turn every stumble into a specific, fixable defect.
3. Distinguish the three kinds of documentation this repository carries —
   **reference** (glossary), **explanation** (learning notes), **task**
   (quickstart, deployment guide) — and say who each is for.
4. Write a **README** that answers, in order: what problem, what approach, what
   results, how do I run it, and what should I not trust it for.
5. State the **limitations and ethics** of this specific model: it reproduces
   the patterns of one 2,200-row dataset of unknown provenance, it is not
   agronomic advice, and it must not be presented as authoritative for a real
   farming decision.
6. Explain **model versioning** at a portfolio level: why an artifact needs an
   identity, what a model registry stores, and why a prediction response should
   be able to name the model that produced it.
7. Explain **monitoring** at a portfolio level: the difference between service
   monitoring and model monitoring, what **data drift** and **concept drift**
   are, why ground truth for this problem arrives a growing season late, and
   what a retraining trigger would look like.
8. Name what a real deployment would add — registry, host, TLS, authentication,
   rate limiting, structured request logging, alerting, rollback — and say
   honestly that none of it is built here.
9. Trace the whole system end to end, from `data/raw/Crop_recommendation.csv` to
   a JSON response, naming the file responsible for each hop.
10. Point at the week that teaches any concept in the repository, using
    [`docs/ml_concepts.md`](../../ml_concepts.md) and
    [`docs/glossary.md`](../../glossary.md) as the index.
11. Verify the project from a **simulated fresh install** — a brand new virtual
    environment, the pinned requirements, the whole test suite — and say why
    that is the check that catches "works on my machine" in a way a `pytest` run
    in the environment you have been developing in cannot.

## Prerequisites

Weeks 1-11, in full. This week adds no model, no endpoint and no dependency; it
audits and finishes what the previous eleven built. It assumes and does **not**
re-explain:

* the dataset contract and the pinned environment
  ([Week 1 notes](../week01/learning_notes.md));
* the train/test split and the fitted preprocessing
  ([Week 3 notes](../week03/learning_notes.md));
* every model and every number in the results table
  ([Week 4](../week04/learning_notes.md) through
  [Week 8](../week08/learning_notes.md));
* the saved pipeline and `predict()` ([Week 9 notes](../week09/learning_notes.md));
* the API, its schemas and the Streamlit demo
  ([Week 10 notes](../week10/learning_notes.md));
* the image and the CI workflow ([Week 11 notes](../week11/learning_notes.md)).

**No new Python dependency this week.** `requirements.txt` is unchanged, no
module is added to `src/`, and the test count does not move: **404 passed, 1
skipped**, exactly as in Weeks 10 and 11.

## Concepts covered

| Concept | First introduced |
| --- | --- |
| Production-ready (beyond a green test suite) | Week 12 |
| Repository audit / student review | Week 12 |
| Documentation completeness: reference, explanation, task | Week 12 |
| Reproducibility as a checkable property | Week 1 (pins), Week 12 (verified end to end) |
| Simulated fresh install | Week 11 (CI does it), Week 12 (you do it) |
| README as the project's entry point | Week 12 |
| Results table as evidence | Week 8 (the decision), Week 12 (the presentation) |
| Known limitations, written down | Week 12 |
| Model ethics: scope, provenance, misuse | Week 2 (provenance), Week 12 (stated plainly) |
| Not-advice disclaimer | Week 12 |
| Model versioning | Week 12 |
| Model registry | Week 12 |
| Model card | Week 12 |
| Service monitoring vs model monitoring | Week 12 |
| Data drift | Week 2 (named), Week 12 (defined) |
| Concept drift | Week 12 |
| Ground-truth lag / feedback loop | Week 2 (named), Week 12 (defined) |
| Retraining trigger | Week 12 |
| Rollback, shadow and canary deployment | Week 12 |
| Portfolio narrative: problem, approach, results, limits | Week 12 |

Each also appears in [`docs/ml_concepts.md`](../../ml_concepts.md), which this
week completes as the twelve-week index, and in
[`docs/glossary.md`](../../glossary.md).

## Connection to the previous week

Week 11 ended with a blunt sentence: *"There is still no authentication and no
public address — the container runs on your machine, not on the internet. The
final review is Week 12."*

Week 12 does not remove that limitation. It **writes it down where a reader will
see it**, which is a different and, for a portfolio, more valuable act. The work
this week is auditing, consolidating and stating: reading all eleven weeks of
notes as a stranger would, fixing what is missing, and rewriting the README so
that somebody who arrives from a link — a reviewer, a recruiter, a future you —
can understand the project in two minutes and run it in five.

The one *new* technical idea is what a real deployment would add and this one
does not have: versioning and monitoring. Both are named and explained, neither
is built. Naming an absence precisely is the honest version of building it.

## Connection to the ML lifecycle

```
Frame the problem -> Get the data -> Explore -> Prepare -> Model
    -> Evaluate -> Improve -> Productionize -> Deploy -> [Monitor]
```

Twelve weeks have walked the loop once, left to right. Week 12 reaches the last
box and does not step into it: monitoring needs real users, real requests and
real outcomes, and this project has none of the three. What it does instead is
close the loop *on paper* — describe what would flow back from the monitor into
the data and the model, so that a student knows the shape of the arrow they are
not drawing.

That is also the honest boundary of a portfolio project, and stating it is the
last thing this course teaches.

## Expected student outcome

### The student CAN, after this week

* **Describe the entire pipeline from raw CSV to a served prediction**, naming
  each file: `data/raw/Crop_recommendation.csv` -> `src/data/data_loader.py`
  (validated against the Week 1 contract) -> `src/data/split.py` (1,760 / 440,
  stratified) -> `src/preprocessing/preprocessor.py` (a train-fitted
  `ColumnTransformer`) -> `src/models/classical_models.py` (Gaussian naive
  Bayes) -> `src/pipelines/training_pipeline.py` (one `Pipeline`, saved as
  `models/crop_model.joblib`) -> `src/pipelines/predict_pipeline.py`
  (`predict()`) -> `api/main.py` (`POST /predict`, validated by
  `api/schemas.py`) -> JSON on the wire.
* **Point to the week that taught any given concept**, via
  [`docs/ml_concepts.md`](../../ml_concepts.md) (teaching order) or
  [`docs/glossary.md`](../../glossary.md) (alphabetical), both of which carry a
  week marker on every entry.
* **Run the full test suite from a simulated fresh install** — a new virtual
  environment, `pip install -r requirements.txt`, `pytest` — and read the
  result as evidence that a stranger's clone will work.
* **Explain what "production-ready" means** for a project of this size, and
  audit somebody else's repository against the same list.
* **State the model's limitations without being asked**: one dataset of unknown
  provenance, 2,200 rows, exactly 100 per crop, no region, no season, no soil
  type, no year, no cost of being wrong — and therefore not agronomic advice.
* **Explain model versioning and monitoring** well enough to say what they
  would add first if this project got real users, and in what order.
* **Present the project**: the README's problem-approach-results-limits
  structure is the same structure as a two-minute spoken answer to "what did
  you build?"

### The student CANNOT yet

This is the final week, so there is no list. Everything the course set out to
teach has been taught.

What remains unbuilt is deliberately outside the course, and is now written down
rather than implied — a public host, TLS, authentication, rate limiting, a
model registry, request logging, drift detection, alerting and automated
retraining. Week 12's job was to name them, not to build them, and the README
and [`docs/architecture.md`](../../architecture.md) name them where a reader
will actually look.

## Deliverables for the week

* This week's four curriculum documents, plus
  [`capstone_reflection.md`](capstone_reflection.md) — what was built each week
  and what a graduate can now do end to end.
* A **Student Review of all twelve weeks'** `learning_notes.md`, with every gap
  found either fixed or recorded in [`validation.md`](validation.md).
* A finalized `README.md`: overview, architecture link, quickstart, the Weeks
  4-8 results table with the shipped model highlighted, a Streamlit screenshot,
  an API usage example, a plainly-worded limitations and ethics section, and the
  completed twelve-week progress table.
* Finalized `docs/architecture.md` (the full raw-CSV-to-response path, and what
  production would add), `docs/ml_concepts.md` (the complete twelve-week index),
  `docs/glossary.md` and `docs/deployment_guide.md`.
* `docs/images/streamlit_app.png` — a real screenshot of the Week 10 demo
  answering the project's example row.
* `pytest` green across the **whole** `tests/` directory, from a fresh virtual
  environment: 404 passed, 1 skipped.

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| ◀ [Chapter 12 — Final Review and Portfolio Polish](README.md) | [Chapter 12 — Final Review and Portfolio Polish](README.md) · 🗺 [Roadmap](../README.md) | [§12.2 Learning notes](learning_notes.md) ▶ |

<!-- nav:end -->
