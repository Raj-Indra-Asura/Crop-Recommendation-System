# Week 11 — Validation

> 🗺 [Roadmap](../README.md) › [Part IV — Production (Weeks 9-11)](../README.md#part-iv--production-weeks-9-11) › [Chapter 11 — Containerization and Continuous Integration](README.md) › **§11.4 Validation**

Run these in order from the repository root. Each step lists the exact command
and the output captured from a real run on 2026-07-31 (Python 3.12.3 on the
host, `python:3.11-slim` inside the image, Docker 28.0.4, the pinned
`requirements.txt`).

Image IDs, sizes, timings, PIDs and timestamps will differ on your machine.
Probabilities will not — the artifact is seeded.

**This week's validation is split in two**, and the split is deliberate:

* **Steps 0-7** were run by the agent that implemented the week, and the output
  below is real, not illustrative. A Docker daemon happened to be available in
  its sandbox, so the image really was built and really did answer.
* **Steps 8-9** are **yours**. Step 8 repeats the container run on your own
  machine — the point of a portable image is that it runs on a machine that is
  not the one that built it. Step 9 is the GitHub Actions tab, which nobody can
  check for their own in-progress push: the workflow only runs once the commit
  is on GitHub, which happens after the work is finished.

---

## Step 0 — Environment

```bash
python --version
docker --version
```

```
Python 3.12.3
Docker version 28.0.4, build b8034c0
```

If `docker --version` prints a version but later commands report
`Cannot connect to the Docker daemon`, the CLI is installed and the daemon is
not running. Start Docker Desktop, or `sudo systemctl start docker`.

**No new Python dependency this week.** The root `requirements.txt` is
unchanged; the new file is `deployment/requirements.txt`, which is installed
inside the image and never in your virtual environment.

---

## Step 1 — Lint and the whole suite, locally

Exactly what CI will run, run by hand first.

```bash
ruff check .
```

```
All checks passed!
```

```bash
pytest -q
```

```
404 passed, 1 skipped, 1018 warnings in 31.62s
```

Unchanged from Week 10: this week adds no Python code, so it adds no tests. The
Dockerfile and the workflow are validated by running them, not by `pytest`.

---

## Step 2 — Build the image

```bash
docker build -t crop-api -f deployment/Dockerfile .
```

The tail of a first build:

```
#12 17.10 Successfully installed annotated-types-0.8.0 anyio-4.14.2 click-8.4.2
 fastapi-0.115.6 h11-0.16.0 idna-3.18 joblib-1.5.3 numpy-2.2.1 pandas-2.2.3
 pydantic-2.10.4 pydantic-core-2.27.2 python-dateutil-2.9.0.post0 pytz-2026.3.post1
 scikit-learn-1.6.1 scipy-1.17.1 six-1.17.0 starlette-0.41.3 threadpoolctl-3.6.0
 typing-extensions-4.16.0 tzdata-2026.3 uvicorn-0.34.0
#12 DONE 17.8s

#13 [5/9] COPY src/ /app/src/
#13 DONE 0.0s

#14 [6/9] COPY api/ /app/api/
#14 DONE 0.0s

#15 [7/9] COPY data/raw/ /app/data/raw/
#15 DONE 0.0s

#16 [8/9] RUN python -m src.pipelines.training_pipeline
#16 1.684 Model:        naive_bayes {'var_smoothing': 1e-09}
#16 1.684 Train rows:   1760
#16 1.684 Test rows:    440
#16 1.684 Accuracy:     0.9955
#16 1.684 Macro F1:     0.9954
#16 1.684 Weighted F1:  0.9954
#16 1.684 Saved to:     /app/models/crop_model.joblib
#16 DONE 1.8s

#17 [9/9] RUN useradd --create-home --uid 10001 appuser
#17 DONE 0.2s

#18 exporting to image
#18 writing image sha256:c14848bfeb5385ca37230d9130fbc8b63569e303d0120f1765e731b559f9f8e5 done
#18 naming to docker.io/library/crop-api done
```

Two things to notice in that log:

1. **`Accuracy: 0.9955`** — the same number as Weeks 8, 9 and 10. The model was
   trained *inside the build*, from the committed CSV, with the seed from
   `src/config.py`. A different number here would mean the image is not
   reproducible.
2. **Only twenty packages installed**, all of them the seven pins plus their
   transitive dependencies. No Jupyter, no matplotlib, no pytest.

```bash
docker images crop-api
```

```
REPOSITORY   TAG       IMAGE ID       CREATED         SIZE
crop-api     latest    c14848bfeb53   2 minutes ago   489MB
```

---

## Step 3 — Prove the layer cache

Rebuild with nothing changed:

```bash
docker build -t crop-api -f deployment/Dockerfile .
```

```
#14 [8/9] RUN python -m src.pipelines.training_pipeline
#14 CACHED

#15 [9/9] RUN useradd --create-home --uid 10001 appuser
#15 CACHED

#16 exporting to image
#16 writing image sha256:c14848bfeb5385ca37230d9130fbc8b63569e303d0120f1765e731b559f9f8e5 done
#16 DONE 0.0s

0.80 s
```

**0.80 seconds** instead of tens, every layer `CACHED`, and the same image
hash. Exercise 2 makes you break the cache on purpose and watch which layers
survive.

Where the size actually is:

```bash
docker history crop-api --format "{{.CreatedBy}}\t{{.Size}}" | head -12
```

```
CMD ["uvicorn" "api.main:app" "--host" "0.0.…  0B
HEALTHCHECK {Test:[CMD-SHELL python -c "impo…  0B
EXPOSE [8000/tcp]                              0B
USER appuser                                   0B
RUN /bin/sh -c useradd --create-home --uid 1…  8.49kB
RUN /bin/sh -c python -m src.pipelines.train…  6.33kB
COPY data/raw/ /app/data/raw/ # buildkit       150kB
COPY api/ /app/api/ # buildkit                 14kB
COPY src/ /app/src/ # buildkit                 156kB
RUN /bin/sh -c pip install --no-cache-dir -r…  364MB
COPY deployment/requirements.txt /app/deploy…  1.63kB
WORKDIR /app                                   0B
```

364 MB of dependencies, 6 kB of model, 170 kB of source and data. That ratio is
the entire argument for §3's layer order: the expensive layer must not depend
on the file that changes every commit.

---

## Step 4 — Inspect the image without running it

```bash
docker image inspect crop-api \
  --format '{{.Os}}/{{.Architecture}} {{.Config.User}} {{.Config.Cmd}} {{.Config.ExposedPorts}}'
```

```
linux/amd64 appuser [uvicorn api.main:app --host 0.0.0.0 --port 8000] map[8000/tcp:{}]
```

Four claims from the Dockerfile, confirmed in the built artifact: the platform,
the **non-root** user, the exec-form command binding to `0.0.0.0`, and the
exposed port.

```bash
docker run --rm crop-api pip list
```

```
Package           Version
----------------- ------------
annotated-types   0.8.0
anyio             4.14.2
click             8.4.2
fastapi           0.115.6
h11               0.16.0
idna              3.18
joblib            1.5.3
numpy             2.2.1
packaging         26.2
pandas            2.2.3
pip               24.0
pydantic          2.10.4
pydantic_core     2.27.2
python-dateutil   2.9.0.post0
pytz              2026.3.post1
scikit-learn      1.6.1
scipy             1.17.1
setuptools        79.0.1
six               1.17.0
starlette         0.41.3
threadpoolctl     3.6.0
typing_extensions 4.16.0
tzdata            2026.3
uvicorn           0.34.0
wheel             0.46.3
```

That command also demonstrates §2.10: an argument after the image name
**replaced** the `CMD`, so no server started.

Read the list against the root `requirements.txt`. `jupyter`, `matplotlib`,
`seaborn`, `pytest`, `ruff`, `streamlit`, `httpx`, `xgboost` and `shap` are all
absent — the answer to "why is the deployment requirements file smaller?" in
one screenful.

---

## Step 5 — Run the container

```bash
docker run -d --rm --name crop-api-test -p 8000:8000 crop-api
sleep 12
docker logs crop-api-test
```

```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

`Started server process [1]` is the exec form working: uvicorn is PID 1, so
`docker stop` reaches it directly.

---

## Step 6 — Check health, three ways

### 6a — The endpoint

```bash
curl -s http://127.0.0.1:8000/health
```

```json
{"status":"ok","model_loaded":true,"n_classes":22}
```

`model_loaded: true` before the first prediction, because the model was loaded
at start-up from a file baked in at build time — not trained on demand, not
loaded lazily.

### 6b — Docker's own probe

```bash
docker ps --filter name=crop-api-test --format "{{.Status}} {{.Ports}}"
```

```
Up 12 seconds (healthy) 0.0.0.0:8000->8000/tcp
```

`(healthy)` is the `HEALTHCHECK` from the Dockerfile calling the same endpoint
from inside the container, and `0.0.0.0:8000->8000/tcp` is the port publishing
from `-p`.

### 6c — Who the process is

```bash
docker exec crop-api-test id
```

```
uid=10001(appuser) gid=10001(appuser) groups=10001(appuser)
```

Not root.

---

## Step 7 — Predict through the container

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"N":90,"P":42,"K":43,"temperature":25,"humidity":80,"ph":6.5,"rainfall":200}'
```

```json
{"crop":"jute","confidence":0.725342776978384,"probabilities":{"jute":0.725342776978384,"rice":0.2746571705151149,"coffee":5.250650112483512e-08}}
```

Digit for digit what Week 10 returned from a local `uvicorn`, and what Week 9
returned from `predict()` — same seed, same data, same pins.

The contract survives the container too:

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" -d '{"N":90}'
```

```
422
```

Then stop it:

```bash
docker stop crop-api-test
```

```
crop-api-test
```

`--rm` removes the container as it stops. The image stays; `docker images
crop-api` still lists it.

---

## Step 8 — The human-verified part: Docker on *your* machine

Everything above ran in the implementing agent's sandbox. An image that works
where it was built has proved something weaker than an image that works where
it was not — so run it yourself, on your own machine:

```bash
docker build -t crop-api -f deployment/Dockerfile .
docker run -p 8000:8000 crop-api
```

and in a second terminal:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"N":90,"P":42,"K":43,"temperature":25,"humidity":80,"ph":6.5,"rainfall":200}'
```

Confirm, in order:

1. The build ends with `naming to docker.io/library/crop-api`.
2. The training step inside the build prints `Accuracy: 0.9955`.
3. `/health` returns `{"status":"ok","model_loaded":true,"n_classes":22}`.
4. `/predict` returns `jute` at `0.7253` — the same number as your local Week 10
   run, from an image that carries its own Python.
5. <http://127.0.0.1:8000/docs> opens the interactive documentation.
6. `Ctrl-C` stops the container within a second or two, not after a ten-second
   kill timeout. That is the exec-form `CMD` forwarding SIGTERM.

If your machine is an arm64 Mac, the build is arm64 and everything above still
holds; the image is simply not the one an amd64 server would run
(learning notes §7).

If Docker is not installed at all, the week is **not** blocked: the commands and
their expected output are in
[`docs/deployment_guide.md`](../../deployment_guide.md) and above. Install
Docker when you can and come back — but do not claim the container works until
you have seen it.

---

## Step 9 — The human-verified part: the Actions tab

Nobody can watch CI check their own unpushed work. The workflow file has to
reach GitHub before GitHub can run it, and the run for the push that *adds*
`ci.yml` only exists after that push finished. So this step happens after the
pull request is open, and it is yours.

1. Open the repository on GitHub and click the **Actions** tab.
2. Find the run named **CI** for the latest commit on this branch.
3. Open it and expand the job **Lint and tests (Python 3.11)**.
4. Check each step in turn:
   * *Check out the repository* — succeeds.
   * *Set up Python 3.11* — prints the interpreter version.
   * *Install dependencies* — ends with `Successfully installed …`; this is the
     step that proves the pins resolve on a machine that is not yours.
   * *Lint* — `All checks passed!`.
   * *Run tests* — `404 passed, 1 skipped`, matching Step 1.
5. The run is green, and the commit shows a ✅ next to it in the commit list.
6. Open the pull request and confirm the same result appears as a **status
   check** at the bottom of the conversation.

If it is red, open the failing step: the log names the file and the line. The
usual first-run causes are a dependency that is installed on your laptop but
missing from `requirements.txt`, and a test that passes locally only because a
previous run left a file behind.

Optional, once it is green: add the badge to the README, replacing `OWNER` and
`REPO`:

```markdown
![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)
```

---

## Definition of Done — Week 11

| Requirement | Status |
| --- | --- |
| `docs/curriculum/week11/{syllabus,learning_notes,exercises,validation}.md` exist | ✅ |
| `deployment/Dockerfile` builds an image that serves the API via uvicorn | ✅ Step 2, agent-verified |
| `deployment/requirements.txt` is trimmed and fully pinned | ✅ 7 pins, versions identical to the root file; Step 4 lists what is absent |
| `.dockerignore` keeps `.git/`, tests, notebooks, docs and artifacts out of the context | ✅ |
| `.github/workflows/ci.yml` runs `pytest` on push and PR to `main` | ✅ file present; the run itself is Step 9, human-verified |
| `docs/deployment_guide.md` gives exact build/run/health commands and the ports | ✅ |
| Root `requirements.txt` unchanged, every dependency still pinned | ✅ no new Python dependency this week |
| Lint and tests still pass locally | ✅ Step 1 — `All checks passed!`, 404 passed, 1 skipped |
| README progress table updated | ✅ |
| `docs/ml_concepts.md` has an entry per new concept | ✅ |
| `validation.md` commands actually run, real output pasted | ✅ Steps 1-7 |
| `docker build` verified | ✅ **agent-verified** — a daemon was available; Step 2 |
| `docker run` + `curl /health` verified | ✅ **agent-verified** (Steps 5-7) and **to be human-verified on your own machine** (Step 8) |
| `ci.yml` observed green in the Actions tab | ⬜ **human-verified only** — Step 9, after this branch is pushed and the PR is open |
| Repository declared finished | ❌ **not this week.** Final review is Week 12 |

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| ◀ [§11.3 Exercises](exercises.md) | [Chapter 11 — Containerization and Continuous Integration](README.md) · 🗺 [Roadmap](../README.md) | [Chapter 12 — Final Review and Portfolio Polish](../week12/README.md) ▶ |

<!-- nav:end -->
