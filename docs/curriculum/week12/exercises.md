# Week 12 — Exercises

> 🗺 [Roadmap](../README.md) › [Part V — Review (Week 12)](../README.md#part-v--review-week-12) › [Chapter 12 — Final Review and Portfolio Polish](README.md) › **§12.3 Exercises**

Work through these after reading [`learning_notes.md`](learning_notes.md) and
running the commands in [`validation.md`](validation.md).

This week's exercises are unlike every previous week's: almost none of them ask
you to write Python. They ask you to *read*, to *audit* and to *write plainly* —
which are the skills that turn eleven weeks of code into something somebody else
can use. Several ask you to write a paragraph. Write it; the exercise is the
writing, not the thinking about writing.

Rules for the week:

* Work on a **copy** or a scratch branch when an exercise asks you to change a
  document you might not want to keep. Revert with `git checkout -- <file>`.
* Anything you create as an experiment goes in `/tmp`, not in the repository.
* Several exercises have no single right answer. They have *defensible* answers,
  and the defence is the deliverable.

Do them in order. **Beginner** exercises (1-4) read the finished repository the
way a stranger would; **intermediate** exercises (5-9) write the documentation
the project still needs and trace it against the code; **challenge** exercises
(10-14) go beyond the notes by designing what this project deliberately did not
build and taking the habits elsewhere. The numbering is continuous across the
three tiers, so "Exercise 6" means the same exercise everywhere in the
curriculum.

---

## Beginner

*Read the finished repository the way a stranger would.*

### Exercise 1 — The fifteen-minute stranger

**Goal:** the audit from §2.1 of the notes, run on your own repository.

Set a timer for fifteen minutes. Adopt the persona: a competent Python developer
with no ML background, sent the GitHub URL, allowed to read anything inside the
repository and nothing outside it.

Start at `README.md` and try to reach a prediction. Every time you have to
guess, backtrack, or open a file the README did not send you to, stop the clock
and write one line:

```
stopped at <file/section> because <the specific thing that was missing>
```

**Expected:** between zero and four entries. Zero is a suspicious result — it
usually means you used knowledge the persona does not have. If you find
yourself thinking "well, obviously they'd know to…", that is a finding.

Now fix every entry, or write down why it is acceptable.

---

### Exercise 2 — Break the fresh install on purpose

**Goal:** see what the fresh-install check catches that `pytest` does not.

1. Delete one line from `requirements.txt` — `joblib`, say — and commit nothing.
2. In a *new* virtual environment (`/tmp/broken-venv`), install from that file
   and run `pytest`.
3. Does it fail? If not, why not — and what does that tell you about the
   difference between "declared dependencies" and "dependencies that happen to
   be installed"?
4. Restore the file with `git checkout -- requirements.txt`.

**Expected:** removing `joblib` alone does *not* break anything, because
scikit-learn depends on it and pip installs it anyway. That is precisely the bug
this check exists to find in the other direction: a package your code imports
that is present only as somebody else's transitive dependency will vanish the
day that somebody drops it. Repeat the experiment with `pandas` for a failure
you can see.

---

### Exercise 3 — Rewrite the first paragraph for a different reader

**Goal:** §3.1 of the notes, applied.

Write the README's opening — problem, input, output, in at most sixty words —
three times, for three readers:

1. a hiring manager skimming twelve portfolios in an hour;
2. a Python developer who wants to call your API this afternoon;
3. an agronomist who wants to know whether to believe it.

**Expected:** three genuinely different paragraphs. The third one is the hardest
and is the one whose content belongs in the limitations section. Then decide
which of the three the committed README should open with, and defend the choice
in one sentence.

---

### Exercise 4 — Audit the results table

**Goal:** check the evidence rather than trusting it.

For any **three** rows of the README's Weeks 4-8 results table:

1. Find the week's `validation.md` that produced the number.
2. Re-run the command that produced it.
3. Confirm the mean and the standard deviation match to four decimal places.

Then answer: which two rows of that table are closest together, and is the gap
between them larger or smaller than either one's fold spread? What is the
correct sentence to write about those two models?

**Expected:** naive Bayes 0.9949 ± 0.0042 and the random forest 0.9926 ± 0.0058
— a 0.0023 gap inside both spreads. The correct sentence says they are tied, and
gives a non-accuracy reason for choosing between them.

---

## Intermediate

*Write the documentation the project still needs, and trace it against the code.*

### Exercise 5 — Write the model card

**Goal:** turn §4 of the notes into the standard artifact it is an instance of.

Using the Model Cards structure (intended use; factors; metrics; training data;
evaluation data; ethical considerations; caveats and recommendations), write a
one-page model card for `models/crop_model.joblib` in `/tmp/model_card.md`.

Rules: every claim must be traceable to a file in this repository, and the
"intended use" section must contain an explicit *out of scope* list.

**Expected:** you will run out of things to say in "training data" almost
immediately, because the dataset's provenance is unknown. Write that down — "the
collection methodology is not published" is a finding, not a gap in your
writing.

---

### Exercise 6 — The disclaimer, three ways

**Goal:** find the line between honest and useless.

Write the not-agronomic-advice statement three times:

1. **Too weak** — technically present, easily missed, the sort of thing a
   marketing page would tolerate.
2. **Too strong** — so hedged that a reader concludes the whole project is
   worthless and stops reading.
3. **Right** — plain, specific about *what* it does not know (region, season,
   soil, market), and placed where the reader passes it anyway.

**Expected:** version 3 names the missing information rather than saying "may be
inaccurate". A limitation that names its cause is credible; one that gestures at
uncertainty is noise.

---

### Exercise 7 — Trace one request through every file

**Goal:** the end-to-end description, out loud, without notes.

Take the example row (`N=90, P=42, K=43, temperature=25, humidity=80, ph=6.5,
rainfall=200`). Write the sequence of files it passes through, from
`data/raw/Crop_recommendation.csv` to the JSON on the wire, and for each one:

* what it does to the data;
* which week built it;
* what happens if it fails.

Check yourself against [`docs/architecture.md`](../../architecture.md) **after**
you have written it, not during.

**Expected:** nine to twelve hops. The two people usually miss are
`src/preprocessing/preprocessor.py` — which is inside the saved artifact, not
called separately at predict time — and `src/config.py`, which is not in the
data path at all but decides where everything is.

---

### Exercise 8 — Find the week

**Goal:** prove the index works.

Without using search, and using only [`docs/ml_concepts.md`](../../ml_concepts.md)
and [`docs/glossary.md`](../../glossary.md), name the week that taught each of:

1. stratification, and why the split uses it;
2. why `fit_transform` on the test set is a bug;
3. what a support vector is;
4. why macro and weighted F1 are equal here;
5. what `EXPOSE` does not do;
6. why the Streamlit app does not call the API.

Time yourself. Anything that takes more than thirty seconds is a defect in the
index, not in you — fix the index.

---

### Exercise 9 — Add the model version (design only)

**Goal:** §5.1 of the notes, made concrete without building it.

Design — in prose, on one page, no code — the smallest change that would let a
`/predict` response say which model answered it. Cover:

* where the version string lives, and what makes it change;
* how it gets into the artifact (and why "read it from the filename" is fragile);
* which existing tests would need to change, and which would not;
* what `/health` should say;
* how a caller who received `crop-model-1.2.0` last week and `1.3.0` today
  should interpret a different answer to the same input.

**Expected:** the last bullet is the interesting one. A version number is only
useful if somebody can act on it.

---

## Challenge

*Design what this project deliberately did not build, and take it elsewhere.*

### Exercise 10 — Design the drift check

**Goal:** §5.2-5.3, made concrete.

You may log every incoming request. Design an input-drift check that runs
weekly:

1. Which statistics of which features do you compare, and against what
   baseline? (The baseline already exists — say where.)
2. What threshold fires an alert, and how did you choose it rather than guessing?
3. What is the false-alarm cost, and what is the missed-drift cost?
4. Give one drift the check would catch and one it would miss.

**Expected:** for (4), a broken humidity sensor reporting a constant 45% is
caught by a variance collapse; a *concept* drift — the same rainfall now suiting
a different rice cultivar — is invisible to every input statistic, and needs
outcomes.

---

### Exercise 11 — Cost the eight-item list

**Goal:** turn §5.6 from a list into a plan.

For each of the eight items, estimate: hours of work, whether it needs anything
you cannot buy or build alone, and what breaks first if you skip it. Then
reorder the list for a project where the *users are internal colleagues* rather
than the public, and say what moved and why.

**Expected:** authentication and TLS drop sharply for an internal tool; logging
and model versioning do not move, because they are what makes any later question
answerable.

---

### Exercise 12 — Explain the project in two minutes

**Goal:** the actual portfolio deliverable.

Out loud, timed, no notes: problem, approach, results, limitations, what you
would do next. Record it if you can bear to.

Rules: no more than one sentence of methodology jargon; the accuracy number must
be accompanied by its protocol; the limitations must come before anybody asks.

**Expected:** the first attempt runs long and spends ninety seconds on models.
The second attempt spends thirty seconds on the whole modelling section, because
"we compared eight families on identical folds and the simplest tied the best"
is the entire interesting content of Weeks 4-8.

---

### Exercise 13 — Audit somebody else's repository

**Goal:** the skill, transferred off this project.

Find a public ML portfolio repository. Score it against the five properties in
§1.2 — reproducible, installable, documented, bounded, verifiable — with one
line of evidence per property, and try, honestly, for ten minutes, to run it.

**Expected:** most public ML repositories fail *installable* and *bounded*
outright. Now go back and check that yours does not, using exactly the standard
you just applied to a stranger.

---

### Exercise 14 — Start the next project

**Goal:** the course was never about crops.

Pick a different tabular dataset. Do **Week 1 only**: the problem statement, a
loader, a schema contract, a pinned environment, and a test that fails when the
data is wrong.

**Expected:** an afternoon, not a week. That difference is what twelve weeks
bought.

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| ◀ [§12.2 Learning notes](learning_notes.md) | [Chapter 12 — Final Review and Portfolio Polish](README.md) · 🗺 [Roadmap](../README.md) | [§12.4 Validation](validation.md) ▶ |

<!-- nav:end -->
