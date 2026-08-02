# Week 11 — Exercises

> 🗺 [Roadmap](../README.md) › [Part IV — Production (Weeks 9-11)](../README.md#part-iv--production-weeks-9-11) › [Chapter 11 — Containerization and Continuous Integration](README.md) › **§11.3 Exercises**

Work through these after reading [`learning_notes.md`](learning_notes.md) and
running the commands in [`validation.md`](validation.md). Nothing here adds a
Python dependency; most of it needs Docker, and the last three need only a text
editor.

Rules for the week:

* Build from the **repository root**, always:
  `docker build -t crop-api -f deployment/Dockerfile .`
* Several exercises tell you to break something on purpose. Work on a **copy**
  (`cp deployment/Dockerfile /tmp/Dockerfile.experiment`) or revert with
  `git checkout -- <file>` when you are done. Never commit an experiment.
* Clean up containers you start: `docker ps` to find them,
  `docker stop <name>`. Reclaim disk with `docker image prune` when your
  experiments have left a pile of untagged images.
* If Docker is not available to you, Exercises 4, 6, 8, 9, 10, 11 and 12 are
  still fully doable — they are about reading, not running.

Do them in order. **Beginner** exercises (1-4) read the build log and the
Dockerfile you already have; **intermediate** exercises (5-8) break the image
and CI bindings on purpose and make you derive the project's rules yourself;
**challenge** exercises (9-12) go beyond the notes by tracing the whole
workflow, designing a new step and saying the week back in your own words. The
numbering is continuous across the three tiers, so "Exercise 6" means the same
exercise everywhere in the curriculum.

---

## Beginner

*Read the build log and the Dockerfile you already have.*

### Exercise 1 — Read the build log as a list of layers

**Goal:** connect each line of the Dockerfile to a line of the build output.

Rebuild from scratch, ignoring the cache:

```bash
docker build --no-cache -t crop-api-nocache -f deployment/Dockerfile .
```

1. Write down each `#N [step/total]` line and the Dockerfile instruction it
   came from.
2. Which single step takes the most time? Which produces the most bytes
   (`docker history crop-api-nocache`)?
3. The training step prints an accuracy. Which week produced that number, and
   why is it identical here?

**Expected:** the `pip install` dominates both time and size; the accuracy is
Week 8's 0.9955, unchanged because the seed and the data are committed.

---

### Exercise 2 — Break the cache on purpose

**Goal:** see the "first changed layer invalidates everything below" rule.

1. Rebuild with no changes and note the time — near-instant, everything
   `CACHED`.
2. Add a trailing comment line to `api/main.py`, rebuild, and record which
   steps say `CACHED` and which re-run.
3. Revert that, then add a comment to `deployment/requirements.txt` and rebuild
   again. Record the same thing.
4. Revert with `git checkout -- api/main.py deployment/requirements.txt`.

**Expected:** a source edit re-runs only from `COPY src/`… down (seconds); a
requirements edit re-runs the `pip install` too (tens of seconds), because the
`COPY` of that file changed and everything below a changed layer is rebuilt.

---

### Exercise 3 — Put the layers in the wrong order

**Goal:** measure the cost of the mistake §3.2 warns about.

Copy the Dockerfile to `/tmp/Dockerfile.badorder` and edit the copy so the
source is copied *before* the dependencies are installed:

```dockerfile
COPY src/ /app/src/
COPY api/ /app/api/
COPY data/raw/ /app/data/raw/
COPY deployment/requirements.txt /app/deployment/requirements.txt
RUN pip install --no-cache-dir -r /app/deployment/requirements.txt
```

Build it once, then edit one comment in `api/main.py` and build it again.
Time both. Compare against the same experiment on the real Dockerfile.

**Expected:** with the bad order, a one-character source change reinstalls
every dependency. Multiply that by the number of times you rebuild in an hour —
that is what layer order is worth.

---

### Exercise 4 — Explain the file to somebody else

**Goal:** the stated Week 11 outcome — *read the Dockerfile and explain what
each line does*.

Without looking at the learning notes, write one sentence per instruction in
`deployment/Dockerfile` (there are about a dozen). Then check yourself against
§2. Pay particular attention to the four that are easy to hand-wave:

* Why `-slim` and not `alpine`?
* Why is `USER appuser` after the `RUN` steps rather than before them?
* Why `--host 0.0.0.0` when Week 10 insisted on `127.0.0.1`?
* What does `EXPOSE 8000` actually do?

---

## Intermediate

*Break the image/CI bindings on purpose and derive the rules yourself.*

### Exercise 5 — Break the binding

**Goal:** make the `0.0.0.0` rule fail so you never have to memorise it.

On a copy of the Dockerfile, change the last line to bind to `127.0.0.1`.
Build it as `crop-api-loopback`, run it with `-p 8000:8000`, and `curl`
`/health`.

1. What does `curl` say?
2. What do the container's logs say — did the server start?
3. Now `docker exec <name> python -c "import urllib.request;
   print(urllib.request.urlopen('http://127.0.0.1:8000/health').read())"`.
   Why does that work when `curl` from your machine does not?

**Expected:** `curl: (52)` or a connection error from outside; a perfectly
healthy log inside; and the exec succeeds because it runs *inside* the
container, where 127.0.0.1 is the container's own loopback.

---

### Exercise 6 — Derive the deployment requirements yourself

**Goal:** the other stated outcome — *explain why the deployment requirements
file is smaller than the root one*.

Do not read `deployment/requirements.txt` first.

1. Start at `api/main.py` and follow every import, and every import of those
   modules, until nothing new appears. Write down the third-party packages you
   reach.
2. Compare your list with `deployment/requirements.txt`. Did you miss
   `uvicorn`? Why is it there even though no file imports it?
3. For each package in the root `requirements.txt` that is *not* in your list,
   write the reason in five words or fewer.
4. Which of the omissions do you think matters most: image size, attack
   surface, or honesty about what the service needs? Defend your answer.

---

### Exercise 7 — Ship the wrong requirements file

**Goal:** feel the difference rather than believe the table.

Build a second image from a copy of the Dockerfile that installs the **root**
`requirements.txt` instead of the trimmed one (you will need to copy that file
into the context too). Then compare:

```bash
docker images | grep crop-api
```

1. How much larger is it, and how much longer did it take?
2. Does the API behave any differently?
3. Name one thing an attacker who reaches a shell in the fat image can do that
   they cannot do in the slim one.

**Expected:** substantially larger and slower for zero behavioural gain; the
fat image contains `pytest`, `jupyter` and compilers' worth of tooling — more
ways to run code that is not the service.

---

### Exercise 8 — Read `.dockerignore` as a security control

**Goal:** understand the third reason from §3.3.

1. List the entries in `.dockerignore` and label each *speed*, *cache* or
   *safety* — some are more than one.
2. `data/processed/` is ignored but `data/raw/` is not. Why is that the right
   way round for this image?
3. Suppose a teammate adds `COPY . /app` to the Dockerfile "to be safe".
   Which `.dockerignore` entries would then be the only thing keeping the git
   history out of a published image?
4. If this project had a `.env` with an API key, would `.dockerignore` be
   enough to keep it out of the image? What else would be needed?

---

## Challenge

*Trace the whole workflow, design a new step, and say the week back.*

### Exercise 9 — Point CI at the wrong requirements file

**Goal:** the mistake §5.4 predicts.

On a scratch branch, edit `.github/workflows/ci.yml` so the install step reads
`pip install -r deployment/requirements.txt`. Do **not** push it — instead,
predict the failure in writing:

1. Which step fails, and with what message?
2. Would `ruff check .` fail too, or only `pytest`?
3. State in one sentence why CI installs the development file and the image
   installs the serving one.

Then revert the file.

---

### Exercise 10 — Trace the workflow

**Goal:** the third stated outcome — *explain what the CI workflow will do once
it runs*.

Answer from `.github/workflows/ci.yml` alone, without running anything:

1. A commit is pushed to a branch called `feature/x`, with no pull request open.
   Does CI run? Why?
2. A pull request is opened from `feature/x` to `main`. Does CI run now? On
   which commit?
3. Two commits are pushed to `main` thirty seconds apart. How many runs finish,
   and why?
4. `ruff` reports one error. Does `pytest` run at all?
5. The runner has no `models/crop_model.joblib`. Why does the suite pass anyway?
6. Write out, in order, the five things the job does, and the one thing each
   proves.

---

### Exercise 11 — Add a step (on paper)

**Goal:** understand jobs and steps well enough to extend the workflow.

Write the YAML — do not commit it — for each of these, and say what it would
cost in CI minutes and what it would catch:

1. Also run the tests on Python 3.12 (hint: the matrix is already there).
2. Build the Docker image in CI and fail the run if the build fails.
3. Start the container in CI, poll `/health`, and fail if it is not `ok` within
   thirty seconds.

Which of the three would you actually add to this repository, and which would
you leave out until there is a real deployment? Justify it in three sentences.

---

### Exercise 12 — Read the week back

**Goal:** state the boundary honestly, which is the whole point of the week.

Answer in your own words, in no more than a paragraph each:

1. Somebody says "the project is dockerized, so it is production-ready". Give
   three specific reasons that does not follow, using this repository.
2. The model is retrained tomorrow with better data. What exactly has to happen
   before a container serves the new model, and what would have to change for
   that not to require a rebuild?
3. CI is green on every commit. Name two kinds of bug it still cannot catch.
4. What remains before this repository is finished — and which week does that?

**Expected for (4):** the final review — README, documentation, an honest
account of capabilities and limits — and it is **Week 12**. Nothing this week
entitles anyone to call the repository done.

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| ◀ [§11.2 Learning notes](learning_notes.md) | [Chapter 11 — Containerization and Continuous Integration](README.md) · 🗺 [Roadmap](../README.md) | [§11.4 Validation](validation.md) ▶ |

<!-- nav:end -->
