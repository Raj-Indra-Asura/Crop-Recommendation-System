# Week 12 — Learning Notes

> 🗺 [Roadmap](../README.md) › [Part V — Review (Week 12)](../README.md#part-v--review-week-12) › [Chapter 12 — Final Review and Portfolio Polish](README.md) › **§12.2 Learning notes**

Eleven weeks built something. This week decides whether it is *finished*, and
finished turns out to be a claim somebody else gets to test.

Nothing about the model changes. That is now the fourth week in a row: accuracy
is still 99.55%, the example row is still `jute` at 0.7253, and the suite is
still 404 passed, 1 skipped. This week's work is entirely in the two things a
reviewer actually reads first — the README and the honesty of its limitations
section — and in one new idea that is *named* rather than built: what you would
add if this had real users.

---

## 1. What "production-ready" actually means

### 1.1 Green tests are the floor, not the ceiling

`pytest` says: *the behaviours somebody thought to check still hold, in the
environment I am running in.* That is a real and valuable statement, and it is
narrower than it sounds. It says nothing about whether:

* a stranger can install the project at all;
* the numbers in the README match the numbers the code produces;
* the model is appropriate for the decision somebody will make with it;
* a failure is diagnosable at 3am from a log line;
* the person who wrote it can be replaced.

Every one of those is a documentation or process property, and none of them is
testable by the test suite. "Production-ready" is the union.

### 1.2 A checkable definition

For a project this size, production-ready decomposes into five properties, and
each has a check that either passes or does not:

| Property | The check | Where this repo answers it |
| --- | --- | --- |
| **Reproducible** | A fresh environment, the pinned requirements, the committed data, and the same numbers come out | [`validation.md`](validation.md) Steps 1-4; `.github/workflows/ci.yml` on every push |
| **Installable** | Somebody who has never seen it gets from clone to a running thing without asking you a question | README quickstart; `docs/deployment_guide.md` |
| **Documented** | Every concept, every entry point and every decision has a written home | `docs/ml_concepts.md`, `docs/glossary.md`, `docs/architecture.md`, twelve weeks of notes |
| **Bounded** | The failure surface and the known limitations are stated where a reader will see them, not buried | README "Limitations and ethics"; the *What is not here* sections of `architecture.md` and `deployment_guide.md` |
| **Verifiable** | An automated check that a *machine you do not control* agrees | CI (Week 11); the fresh-venv run this week |

Notice what is *not* on that list: cleverness, coverage percentage, model
accuracy. A 99.55% model in an unreproducible repository is worth less to a
reader than a 90% model they can run.

### 1.3 The three documentations, and who reads them

Documentation is not one thing. This repository carries three kinds, and
confusing them is why so many projects have a README that is simultaneously too
long and useless:

| Kind | Question it answers | Read | This repo |
| --- | --- | --- | --- |
| **Task** | *How do I do X?* | While doing X, impatiently | README quickstart, `docs/deployment_guide.md`, each week's `validation.md` |
| **Explanation** | *Why is it like this?* | Once, in a chair, on purpose | The twelve `learning_notes.md`, `docs/architecture.md` |
| **Reference** | *What does this term/field mean?* | In two seconds, mid-task | `docs/glossary.md`, `docs/ml_concepts.md`, `/docs` (OpenAPI) |

A task document that stops to explain naive Bayes is a broken task document. An
explanation that lists command flags is a broken explanation. The README is
mostly task, with just enough explanation to make the task make sense, and it
*links* to the rest rather than absorbing it.

### 1.4 Reproducibility has a strong and a weak form

The weak form is *it runs again on my machine.* The strong form is *it runs, and
produces the same numbers, on a machine that has never seen it.* The repository
has been aiming at the strong form since Week 1 without saying so out loud:

* `data/raw/Crop_recommendation.csv` is committed, so nobody downloads a
  different version of "the dataset" (Week 1);
* every dependency is pinned to an exact version (Week 1);
* `RANDOM_STATE` lives in `src/config.py` and is passed to every split, every
  fold and every model that takes one (Weeks 3-9);
* the model artifact is **not** committed — it is derived, so committing it
  would be committing a result rather than a cause (Week 9);
* CI installs from scratch on a clean Ubuntu runner every push (Week 11).

This week adds the last link: a virtual environment created from nothing, in
[`validation.md`](validation.md), running the whole suite. Everything else was a
promise about what would happen in a fresh install; this is the fresh install.

---

## 2. The audit: reading your own repository as a stranger

### 2.1 The trick

You cannot un-know your own project. The workable substitute is to fix a
*persona* and a *starting point* and refuse to use anything outside them:

> A competent Python developer with no ML background has been sent the GitHub
> URL. They have fifteen minutes. They can read anything in the repository and
> nothing outside it. Where do they stop?

Then walk it in order — README first, because that is where they land — and
write down every place they would stop, in the form *"stopped at X because Y"*.
Each entry is a defect with a fix attached. "The README is a bit thin" is not a
finding; "the README never says what the input columns are, so a reader cannot
form the JSON body" is.

### 2.2 The Student Review, applied to all twelve weeks

The same review that each week ran against its own notes was run this week
against **all** of them at once, looking for six classes of defect:

1. **Broken relative links.** Every `](...)` target in every markdown file
   resolved against the filesystem. Result: 0 broken, out of several hundred.
2. **References to code that does not exist** — a function renamed in week 9
   but still named in week 5's prose, a module moved, a notebook renumbered.
   Result: 0.
3. **Numbers that contradict each other across weeks.** The dangerous class,
   because prose copies numbers and code does not. 2,200 rows / 22 crops / 100
   each; 1,760 train and 440 test; 4.55% baseline; 99.49% naive Bayes on the
   training folds; 99.55% (438/440) on the test set; `jute` at 0.7253. All
   consistent.
4. **Jargon used before it is defined**, checked against `glossary.md` and
   `ml_concepts.md` rather than against memory. Result: 0 terms undefined —
   which is a consequence of the rule the course followed from Week 1, that a
   term enters the glossary in the same commit that first uses it.
5. **Forward references** — every sentence anywhere in the repository that
   promised something of Week 12. There were twenty-two. §6 below is where they
   are answered.
6. **Placeholders** — `TODO`, `TBD`, an empty section, a table row that says
   "Not started". Result: one, and it was this week's row in the README
   progress table.

The audit's real output was not the defect list; it was the confirmation that
the discipline held. The reason there is nothing to fix in the notes is that
every week paid the documentation cost in the week that incurred it. Deferring
it to a "documentation week" would have produced a documentation week that
could not be finished.

### 2.3 What the audit *did* change

Three things, all outside the notes:

* `.gitignore` gained `fresh-venv/`, because this week's validation tells the
  reader to create a virtual environment by that name inside the repository, and
  a document that tells you to dirty your working tree is a defect.
* The README was rewritten. The old one was a *changelog* — eleven paragraphs,
  each starting "Week N…", in chronological order. That is the right structure
  for somebody following the course and the wrong structure for somebody
  evaluating the result, and the second reader is the more common one.
* `docs/architecture.md` gained the training path. It drew the *request* flow
  beautifully and never said where `models/crop_model.joblib` comes from, which
  is the question "describe the pipeline from raw CSV to a served prediction"
  actually asks.

---

## 3. The README a stranger reads

### 3.1 Five questions, in order

A README is not a description of the repository. It is an answer to five
questions, and the order matters because a reader who does not get past question
one will never see question four:

1. **What problem does this solve?** One sentence, then a concrete input and its
   output. Not "an end-to-end ML system" — *given N, P, K, temperature,
   humidity, pH and rainfall, recommend one of 22 crops.*
2. **What approach?** Enough to place it: supervised multiclass classification,
   classical models on tabular data, a fitted scikit-learn `Pipeline` served by
   FastAPI in a container. A diagram, linked or embedded.
3. **What results?** Numbers, in a table, with the chosen one marked and the
   evaluation protocol stated. A single "99.55% accuracy!" with no protocol is
   indistinguishable from a leak.
4. **How do I run it?** Copy-pasteable, in dependency order, with the expected
   output shown so the reader knows whether it worked.
5. **What should I not trust it for?** §4.

Everything else — repository layout, the course progress table, the dataset
citation — comes after, because it serves readers who have already decided to
stay.

### 3.2 Show the output

Every command in the finished README is followed by what it prints. This is
cheap and it is the single highest-value habit in technical writing: a reader
who sees `{"crop":"jute","confidence":0.7253,...}` knows in one glance whether
their run succeeded, and a reader who does not is left comparing their terminal
against their imagination.

It also makes the README *auditable*. A pasted output is a claim that can be
checked, and this week's [`validation.md`](validation.md) checks every one.

### 3.3 The results table is the evidence, not the boast

Weeks 4-8 trained eleven distinct model configurations. The finished README
lists **all** of them, including the ones that lost, with:

* the week that trained it, so a reader can go read why;
* the protocol — 5-fold stratified cross-validation on the 1,760 training rows,
  the same folds for every model, `data/processed/test.csv` untouched until
  Week 8;
* the fold standard deviation next to the mean, because 99.49 ± 0.42 and
  99.26 ± 0.58 overlap and a table without the spread invites a reader to
  believe a difference that is not there;
* the single test-set column, filled in for the two finalists only, because
  that is the only honest thing to put there.

Listing the losers is not modesty. It is the evidence that the winner was
*chosen* rather than *found first*, which is exactly the thing a reviewer is
trying to determine.

---

## 4. Limitations and ethics of *this* model

This is the section a portfolio needs most and skips most often. It is not a
disclaimer bolted on to avoid blame; it is the part that demonstrates the author
understands what they built.

### 4.1 What the model actually is

A Gaussian naive Bayes classifier. It stores, for each of 22 crops and each of 7
features, a mean and a variance — 308 numbers plus 22 priors — and answers by
asking which crop's numbers make the observed row least surprising, assuming the
seven features are independent given the crop. That assumption is false in the
data (Week 2 measured the correlations) and the model works anyway, which Week 5
explained and which does not make the assumption true.

It reflects **one dataset**: 2,200 rows, exactly 100 per crop, seven columns, no
identifiers, no dates, no locations. That is the entire evidential basis for
every number in this repository.

### 4.2 What the dataset does not contain

The honest list is longer than the feature list:

| Absent | Why it matters |
| --- | --- |
| **Region / country / soil type** | Nitrogen at 90 kg/ha means different things on clay in Kerala and on sand in Punjab. The model has never heard of either. |
| **Season / date / year** | No sowing window, no multi-year variation, no climate trend. |
| **Provenance** | The dataset is published on Kaggle without a stated collection method. The per-class counts are exactly 100 and the classes separate almost perfectly, which is not what field measurements usually look like. It may be simulated, augmented, or aggregated from agronomic tables. Nobody in this repository knows. |
| **Yield, cost, price** | The model recommends a crop, not a *profitable* crop. It has no idea what the farmer can sell, afford to plant, or store. |
| **Irrigation, fertiliser plans, rotation history** | Every one of which a real recommendation would depend on. |
| **The cost of being wrong** | All 22 classes are treated as equally valuable and every error as equally bad. A real error costs a season. |

### 4.3 What 99.55% is and is not

It is: the share of 440 held-out rows, drawn from the same dataset by the same
stratified split, that this model labelled correctly, measured once.

It is **not** a prediction of accuracy on a field that was measured by somebody
else's sensor, in a country the dataset never names, in a year after the data
was collected. There is no evidence in this repository about that number,
because there is no second dataset. A near-perfect score on a small, clean,
perfectly balanced dataset is more often a statement about the dataset than
about the model — Week 2 noticed the suspiciously clean separation, and Week 8
saw the model make only two mistakes, both on the one genuinely overlapping pair
(`rice` / `jute`, separated by rainfall alone).

### 4.4 The plain statement

> **This is not agronomic advice.** It is a demonstration model trained on one
> public dataset of unknown provenance. It has no knowledge of your region,
> soil, season, water access, market or budget, and its confidence numbers are
> the model's internal arithmetic, not a probability that planting will succeed.
> Do not use it to make a real planting decision. Talk to an agronomist or an
> extension service.

That paragraph is in the README, near the top, in the reader's path — not in a
footnote. The placement is the point: a limitation a reader has to hunt for has
not been disclosed, it has been hidden.

### 4.5 Confidence is not certainty

The API returns `confidence: 0.7253`. Three things are true about that number
and all three belong in the documentation:

* It is a *relative* score across 22 classes, normalised to sum to 1. It is not
  a probability that the crop will grow.
* Naive Bayes multiplies seven independent-ish likelihoods, which pushes its
  outputs towards the extremes. Week 10 found an out-of-distribution input that
  returned 99.99997% confidence — for a row the training data never contained.
  High confidence is *especially* untrustworthy off-distribution, because
  nothing in the model knows it is off-distribution.
* Presenting it as a percentage in a UI invites a user to read it as a
  probability of success. The Streamlit demo does exactly that, which is one
  more reason it is labelled a demo.

### 4.6 The ethical failure mode is presentation, not code

The model is harmless. A screenshot of the model, captioned "AI-powered crop
recommendation for farmers", is not: it makes an authoritative-sounding claim on
behalf of a system whose entire knowledge is 2,200 rows of unattributed data,
and the people most likely to act on it are the people least able to absorb a
lost season. The technical work is finished; the responsibility that remains is
about what you claim it does.

---

## 5. What real users would require — named, not built

This is the section that ends the course, and none of it is implemented. Saying
what you would build, in what order, and why, is a stronger portfolio signal
than half-building it.

### 5.1 Model versioning

Right now the artifact is `models/crop_model.joblib`. One filename, overwritten
by the next training run, mentioned in no response. That is fine for a project
where training is deterministic and the data never changes, and it fails the
first time somebody asks *"which model produced this answer?"*

A versioned model has an **identity**, and identity means recording the things
that would change the artifact:

| Recorded | Because |
| --- | --- |
| Version string (`crop-model-1.3.0`) | Humans need to say the name out loud |
| Training data hash / snapshot id | The same code on different data is a different model |
| Code commit SHA | The same data with different preprocessing is a different model |
| Library versions | scikit-learn's pickle format is not stable across versions |
| Hyperparameters and seed | Reproducing the fit |
| Metrics at training time | So a regression is visible before the model ships |

The cheap first step here would be a `MODEL_VERSION` in `src/config.py`,
embedded in the artifact and returned by `/health` and `/predict`. The full
version is a **model registry** — a service that stores artifacts against that
metadata, marks one as `production`, keeps the previous one so a **rollback** is
a pointer change rather than a retrain, and refuses to overwrite.

**Semantic versioning** works for models with a small reinterpretation: patch =
retrained on more of the same data; minor = new features or hyperparameters,
same interface; major = the interface or the label set changed, and every caller
must be updated. The label set is the one that bites — adding a 23rd crop
changes what every previous response meant.

A **model card** is the human-readable half: what the model is for, what data it
was trained on, how it performs, on whom it performs worse, and what it must not
be used for. §4 of this document is, in substance, this project's model card.

### 5.2 Monitoring: two different things

**Service monitoring** asks *is the process healthy?* — request rate, error
rate, latency percentiles, memory, restarts. This is ordinary software
operations and applies whether or not there is a model inside. The repository
has the seed of it: `GET /health` (Week 10) and Docker's `HEALTHCHECK` (Week 11)
answer "is it up", and nothing answers "how is it doing".

**Model monitoring** asks *are the answers still any good?* — and it is harder,
because a model that has silently become wrong returns 200 OK at 99.9%
confidence just as fast as one that is right. Nothing in the HTTP layer can tell
the difference.

The three signals, in increasing order of usefulness and difficulty:

1. **Input distribution** — compare the mean, standard deviation and range of
   each incoming feature against the training set. Cheap, needs no ground truth,
   catches broken sensors and unit changes immediately. Week 2's descriptive
   statistics are literally the baseline you would compare against.
2. **Output distribution** — the share of requests answered `rice` this week
   versus last, and the distribution of confidence. If confidence collapses or
   one class suddenly dominates, something moved.
3. **Accuracy against outcomes** — the only signal that actually measures the
   thing you care about, and the one you get last. §5.4.

### 5.3 Drift, and its two kinds

**Data drift** (covariate shift): the inputs change, the relationship does not.
A new region with heavier rainfall starts using the service. `P(X)` moved,
`P(y|X)` did not. The model is not wrong about the world; it is being asked
questions it has little evidence about, and its confidence will not admit it.

**Concept drift**: the relationship itself changes. A new drought-tolerant rice
cultivar is released and the rainfall a rice field needs is now different.
`P(y|X)` moved. Every stored label from before the change is now partly wrong,
and no amount of input monitoring will show it — only outcomes will.

Data drift is detectable from requests alone, which is why it is monitored
first. Concept drift needs ground truth, which is why it is the one that catches
teams out.

### 5.4 Ground truth arrives a season late

This is the specific, unavoidable property of *this* problem. The label is
"which crop was right for this plot", and the only way to observe it is to plant
something and wait for the harvest. That means:

* the **feedback loop** is months long, per prediction;
* it is **partial** — you learn something about the crop that was planted and
  nothing about the 21 that were not (the classic bandit problem: recommendation
  suppresses its own counterfactual);
* it is **biased** — you only observe outcomes for people who took the advice;
* and it is **noisy** — a failed harvest may be weather, pests or the farmer's
  irrigation rather than the recommendation.

So the honest monitoring design here is: watch inputs weekly (fast, cheap,
proximate), watch outputs monthly, and treat accuracy as a slow annual audit
against whatever outcomes can be collected, not as a dashboard number.

### 5.5 Retraining triggers

Three strategies, in ascending order of sophistication:

| Trigger | Retrain when | Cost |
| --- | --- | --- |
| **Scheduled** | Every N months, unconditionally | Simple; retrains when nothing changed, and waits when everything did |
| **Performance-based** | Measured accuracy drops below a threshold | Correct, and needs ground truth you may not have for a season |
| **Drift-based** | An input-distribution test crosses a threshold | Fast; may fire on a shift that does not actually hurt accuracy |

For this project the realistic answer is scheduled retraining with a drift-based
early alarm — and a hard rule that a retrained model is *evaluated against the
current production model on a held-out set before it replaces it*. Automated
retraining that ships whatever comes out is a way to automate getting worse.

### 5.6 The ordered list, if this got real users tomorrow

1. **Structured request logging** — one JSON line per prediction: timestamp,
   inputs, prediction, confidence, model version. Everything else needs this
   data to exist, and it costs an afternoon. Note that logging inputs is a
   privacy decision, not only a technical one: field measurements plus a
   timestamp can identify a farm.
2. **Model version in the response**, and in `/health`. Ten lines, and it is the
   difference between "the model was wrong" and "which model was wrong".
3. **Authentication and rate limiting** — an API key, a request budget. Anyone
   who can reach the port can currently spend the CPU.
4. **TLS and a real host** — a registry, a platform that runs the container, a
   domain, a certificate. This is the step that makes it public.
5. **Input-distribution monitoring** with a weekly report against the Week 2
   training statistics.
6. **A model registry and a rollback path**, once there is more than one model
   worth naming.
7. **Shadow deployment** — a new model answers every request in parallel with
   the live one and its answers are only recorded, never returned — followed by
   a **canary release** to a small share of real traffic. Both are ways to
   discover a bad model with a small blast radius.
8. **Alerting**, last, because an alert on a number nobody has yet watched for a
   month is an alert that will be muted in a week.

None of the eight is built. All eight are named because a reviewer's next
question after "you deployed it?" is "what would you do next?", and "I don't
know" is the only wrong answer.

---

## 6. The forward references, answered

Eleven weeks made twenty-two promises about Week 12. They fall into three
groups, and the third group is the interesting one.

**Delivered this week, as promised:**

* the final review, the README, and an honest account of what the repository can
  and cannot do (Weeks 8, 9, 11 notes; `deployment_guide.md`; the old README);
* monitoring and drift explained, promised as far back as Week 2's data-leakage
  section and Week 8's syllabus — §5.2-5.5 above;
* a capstone account of what was built and what a graduate can do —
  [`capstone_reflection.md`](capstone_reflection.md).

**Delivered early, in Week 11:** the container, the Dockerfile and the
`uvicorn api.main:app` start command were all promised to Week 12 by Week 10's
notes, and Week 11 built them. The promises were kept; the week number in the
prose was optimistic. Those references now read correctly.

**Not delivered, and now explicitly withdrawn:** Week 10's notes promised Week
12 would supply "a public address, a domain and a TLS certificate". It does not.
Publishing a service on the internet — a registry, a host, DNS, certificates,
authentication, and the ongoing responsibility of operating it — is a course of
its own, and the fourteen paragraphs it would take to fake it here would be the
one thing in this repository that a reviewer could catch as untrue. §5.6 names
what it would take instead.

Withdrawing a promise in writing is not a failure of the plan. Quietly dropping
it would have been.

### 6.1 Added after Week 12: the chapter covers and the test that guards them

Two things in this repository post-date the twelve weeks, so no week's narrative
introduced them as it was written. They are named here, because a curriculum
that leaves parts of its own repository unexplained has the defect §2 tells you
to look for.

* **Per-chapter `README.md` covers, and the roadmap
  ([`docs/curriculum/README.md`](../README.md)).** Forty-eight documents in
  twelve directories is a pile, not a book. Each chapter now opens with a cover
  that says what to read in what order — notes, then the code and notebooks it
  describes, then exercises, then validation — and the roadmap lists every
  document once, in reading order, with previous/next footers on every page.
  Nothing was rewritten to add them; it is navigation over existing content.
* **[`tests/test_curriculum_links.py`](../../../tests/test_curriculum_links.py).**
  Navigation that drifts is worse than none, and a stale link is exactly the
  kind of defect nobody notices while writing. Four tests check it
  mechanically: every relative link and `#anchor` in every Markdown file
  resolves; the roadmap's reading order names each curriculum document exactly
  once; and each page's previous/next footer agrees with the position the
  roadmap gives it. They are structural checks — they say nothing about the
  prose — and they run in CI with everything else.

That is why the suite now reports **408 passed, 1 skipped** where this week's
recorded output says 404: four tests, none of them about the model. The
recorded outputs are left as they were run, with the drift noted in
[§12.4 Step 3](validation.md#step-3--lint-and-the-whole-suite-from-the-fresh-environment) —
re-recording history to match today is the habit this course spent twelve weeks
avoiding.

---

## 7. What you can now do

Not *what the repository contains* — what a person who worked through it can do,
which is the only useful measure of a course:

* Take a CSV and a vague goal and turn it into a written problem statement with
  a success metric and explicit non-goals.
* Validate data against a contract that fails loudly, instead of trusting it.
* Explore a dataset well enough to know what will hurt you later — scale,
  balance, correlation, outliers, leakage.
* Split and preprocess without leaking, and explain to somebody else why fitting
  the scaler before the split is the single most common way to fool yourself.
* Establish a baseline before believing any result.
* Fit, cross-validate and compare eight families of classical model on identical
  folds, and read the fold spread before believing a difference.
* Recognise overfitting from a train/validation gap and demonstrate it
  deliberately.
* Tune hyperparameters, and — harder — recognise when tuning bought nothing and
  say so.
* Open a held-out test set once, read a confusion matrix, and explain the actual
  errors in the language of the domain.
* Explain a single prediction with permutation importance and SHAP, and name the
  traps in both.
* Turn a notebook into a package: config, pipeline, artifact, entry points,
  tests.
* Serve it over HTTP with validated schemas and honest status codes, and put a
  demo UI in front of it.
* Containerise it and let CI check it on a machine that is not yours.
* Audit the result, document it, and say clearly what it must not be used for.

That is the whole loop, once, with the evidence committed. The next project
starts at Week 1 again, faster.

---

## Further reading

* Sculley et al., *Hidden Technical Debt in Machine Learning Systems* (2015) —
  the paper behind §5; the model is the small box in the middle of the diagram.
* Mitchell et al., *Model Cards for Model Reporting* (2019) — the structure §4
  is an informal instance of.
* Gebru et al., *Datasheets for Datasets* (2018) — the questions §4.2 asks about
  provenance, asked systematically.
* Diátaxis (diataxis.fr) — the task/explanation/reference/tutorial split used in
  §1.3.
* Aurélien Géron, *Hands-On Machine Learning*, Chapter 2's "Launch, Monitor and
  Maintain Your System" — the same lifecycle, from the practitioner's side.

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| ◀ [§12.1 Syllabus](syllabus.md) | [Chapter 12 — Final Review and Portfolio Polish](README.md) · 🗺 [Roadmap](../README.md) | [§12.3 Exercises](exercises.md) ▶ |

<!-- nav:end -->
