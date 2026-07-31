# Week 11 — Learning Notes

Ten weeks produced a service that runs *here*. This week it runs *anywhere*,
and the repository starts checking itself.

Nothing about the model changes. Again. That is the third week in a row where
the accuracy is 99.55% and the answer to the example row is `jute` at 0.7253 —
and it is the point. Deployment is a software-engineering problem that happens
to have a model inside it.

---

## 1. The problem: "it works on my machine"

### 1.1 What actually differs between two machines

Week 10 ends with an instruction:

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

That works on the machine it was written on. Handing it to somebody else means
handing over a list of assumptions that were never written down:

| Assumption | How it breaks |
| --- | --- |
| Python 3.11 or 3.12 is installed | The machine has 3.9, where this project's `X \| None` annotations are a syntax error |
| `python` means that interpreter | It means 2.7, or nothing, or a Homebrew build |
| A virtual environment is active | `pip install` fails on a system-managed Python, or pollutes it |
| The wheels exist for this platform | An M-series Mac, an ARM server, Alpine's musl libc |
| The C libraries are present | scikit-learn needs a BLAS; a slim base image may not have one |
| The working directory is the repository root | `src` is not importable from anywhere else |
| A compatible `libgomp`, `zlib`, `libstdc++` | Nothing in `requirements.txt` mentions them |

The pins in `requirements.txt` fix the *Python* half of that list and nothing
below it. A pinned `scikit-learn==1.6.1` still needs an operating system that
can run its wheel.

"It works on my machine" is not a joke about carelessness. It is an accurate
description of software whose environment was never specified.

### 1.2 What a container does about it

A **container** ships the environment with the program: the operating system
userland, the interpreter, the libraries, the code, the data it needs and the
command to run. What is *not* shipped is the kernel — a container uses the
host's — which is why an image is hundreds of megabytes rather than several
gigabytes, and why it starts in milliseconds rather than in a minute.

Three things are often confused:

| | Isolates | Contains | Starts in | Size |
| --- | --- | --- | --- | --- |
| **Virtual environment** | Python packages | Site-packages, a `python` symlink | instant | MBs |
| **Container** | Filesystem, processes, network, users | A whole userland + your app | ~1 s | 100s of MBs |
| **Virtual machine** | Everything, including the kernel | A whole operating system | ~1 min | GBs |

A virtual environment cannot help you if the other machine has the wrong
Python, the wrong libc or the wrong CPU architecture. A virtual machine can,
and charges a full operating system for it. A container is the middle: enough
isolation to be reproducible, cheap enough to start per request-handling
process.

### 1.3 Image vs container

* An **image** is a built artifact: layered, immutable, identified by a
  content hash, tagged with a name like `crop-api:latest`. It is a noun, and it
  is not running.
* A **container** is a process (or a small tree of them) started from an image,
  with a writable layer on top. It is a verb that has been conjugated.

One image, many containers — the same way one class makes many instances. Stop
a container and the writable layer goes with it; the image is untouched. That
is why "just SSH in and fix it" is not a container workflow: whatever you fix
is gone at the next `docker run`. Fixes go in the Dockerfile and produce a new
image.

### 1.4 What containers are *not*

* **Not a security boundary you should lean on.** Better than nothing, weaker
  than a VM; a container shares the host kernel.
* **Not a performance win.** The process runs at native speed, and the image is
  extra disk.
* **Not a substitute for tests.** A container reproducibly runs whatever you
  put in it, including a bug.
* **Not deployment.** An image on your laptop is portable, not public. §7.

---

## 2. The Dockerfile, line by line

The file is [`deployment/Dockerfile`](../../../deployment/Dockerfile). Read it
with this section beside it.

### 2.1 `FROM` — the base image

```dockerfile
FROM python:3.11-slim
```

Every image starts from another image. `python:3.11-slim` is the official
Python image on a trimmed Debian: a real glibc userland — which the numpy,
pandas and scikit-learn wheels are built against — without compilers,
documentation or the dozens of tools the full `python:3.11` carries.

Three families are worth knowing:

| Tag | Size | Trade-off |
| --- | --- | --- |
| `python:3.11` | ~1 GB | Everything; needlessly large for a server |
| `python:3.11-slim` | ~150 MB | glibc, no build tools — what we use |
| `python:3.11-alpine` | ~50 MB | musl libc: many scientific wheels do not exist, so pip compiles from source and the build becomes slow and fragile |

`alpine` is the trap. It looks like the obvious choice until scikit-learn takes
twenty minutes to compile.

The tag is pinned to a minor version for the same reason `requirements.txt`
pins packages: `FROM python:latest` means a rebuild next month may get a
different interpreter, and a build that is not reproducible cannot be debugged.

### 2.2 `ENV` — interpreter behaviour

```dockerfile
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app
```

* `PYTHONDONTWRITEBYTECODE` — no `.pyc` files. In a container they are written
  once into a layer and never reused across runs.
* `PYTHONUNBUFFERED` — Python's stdout buffering hides logs when output is a
  pipe (which it is, under Docker). Without this, a process that crashes takes
  its last and most interesting lines with it.
* `PIP_NO_CACHE_DIR` — pip's download cache would be baked into the image for
  no benefit.
* `PYTHONPATH=/app` — makes `import src` and `import api` work, exactly as
  `pythonpath = ["."]` does for pytest in `pyproject.toml`.

### 2.3 `WORKDIR` — the working directory

```dockerfile
WORKDIR /app
```

Creates the directory if needed and `cd`s into it for every later instruction
*and* for the running container. Without it, relative paths resolve against
`/`, which is how `data/raw/Crop_recommendation.csv` becomes "file not found"
in an image that clearly contains it.

### 2.4 `COPY` and `RUN` — dependencies before source

```dockerfile
COPY deployment/requirements.txt /app/deployment/requirements.txt
RUN pip install --no-cache-dir -r /app/deployment/requirements.txt

COPY src/ /app/src/
COPY api/ /app/api/
COPY data/raw/ /app/data/raw/
```

`COPY` moves files from the build context into the image. `RUN` executes a
command *during the build* and keeps the resulting filesystem.

The order is the interesting part, and §3 is about it.

Note what is *not* copied: `app/` (Streamlit), `tests/`, `notebooks/`,
`docs/`. The server imports none of them, and everything in an image is
something to download, store, scan and defend.

### 2.5 `RUN python -m src.pipelines.training_pipeline` — baking the model in

`models/crop_model.joblib` is git-ignored (Week 9), so it is not in the
repository and cannot be copied. There are three options:

| Option | Cost |
| --- | --- |
| Commit the artifact and `COPY` it | A binary in git; drifts from the code that made it |
| Train at container **start-up** | Every start pays seconds of training, and `/health` reports `model_loaded: false` while a load balancer watches |
| Train at **build** time | The build is slower once; every container starts by reading a file |

We train at build time. It works because the training is deterministic — the
seed lives in `src/config.py`, the data is committed — so the artifact inside
the image is the same artifact you get locally, with the same 99.55%.

The cost is real and worth stating: **retraining means rebuilding the image**.
A model that must update independently of code belongs in a model registry or
an object store, fetched at start-up. At this project's scale, that machinery
would cost more than it buys.

### 2.6 `USER` — not root

```dockerfile
RUN useradd --create-home --uid 10001 appuser
USER appuser
```

By default everything in a container runs as root. If a bug in a
network-facing process can be exploited, root inside the container is a much
better place to start from than an unprivileged user. Creating the user *after*
the build steps also leaves `/app` owned by root and read-only to the server —
the process cannot rewrite its own code or its own model.

### 2.7 `EXPOSE` — documentation, not a door

```dockerfile
EXPOSE 8000
```

`EXPOSE` records which port the process listens on. It opens nothing. The port
becomes reachable from your machine only because of `-p 8000:8000` on
`docker run`. Its value is that a reader — human or orchestrator — can see the
contract without reading the `CMD`.

### 2.8 `HEALTHCHECK` — the container's own probe

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "...urlopen('http://127.0.0.1:8000/health')..."
```

Docker runs that command inside the container on a schedule and labels the
container `healthy` or `unhealthy` in `docker ps`. This is where Week 10's
`/health` endpoint pays off: something automatic is now asking, and
"the process exists" was never the same claim as "the process can serve".

`python -c` rather than `curl` because the slim image ships no `curl`, and
adding one to make a health check work would be a package added for the check
rather than for the service. `127.0.0.1` here is correct — the probe runs
*inside* the container.

### 2.9 `CMD` — what the container runs

```dockerfile
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Two decisions in one line.

**Exec form vs shell form.** The JSON-array (exec) form runs uvicorn directly
as PID 1, so `docker stop`'s SIGTERM reaches it and the server shuts down
cleanly. The string (shell) form — `CMD uvicorn api.main:app` — runs
`/bin/sh -c "..."`, and the shell becomes PID 1 and does not forward the
signal; the container is killed ten seconds later instead of stopping.

**`0.0.0.0`, not `127.0.0.1`.** Inside a container, `127.0.0.1` is the
container's own loopback. A server bound there is reachable only from inside,
so `-p 8000:8000` forwards to nothing and `curl` reports connection refused.
`0.0.0.0` means "all interfaces of this container", which includes the one
Docker attached. Week 10 bound to `127.0.0.1` deliberately, to keep the
development server off the network; the container's isolation is what makes
`0.0.0.0` acceptable here.

### 2.10 `CMD` vs `ENTRYPOINT`

Both say what runs. They differ in what `docker run` arguments do:

* With `CMD`, anything you append **replaces** it:
  `docker run crop-api python -c "print(1)"` prints `1` and no server starts —
  handy for poking around inside the image.
* With `ENTRYPOINT`, anything you append is **passed as arguments** to it. The
  image becomes one fixed executable, and `docker run crop-api --port 9000`
  would add a flag to uvicorn.

`ENTRYPOINT` suits a container that *is* a command-line tool. This image is a
service that occasionally needs to be inspected, so `CMD` is the better fit.
(`ENTRYPOINT` plus `CMD` is a third pattern: the entrypoint is a shell script
that does set-up and then `exec "$@"`.)

---

## 3. Layers, the cache, and why order matters

### 3.1 Each instruction is a layer

`FROM`, `COPY` and `RUN` each produce a filesystem layer — a diff on top of the
one below. An image is that stack; a container is the stack plus one writable
layer.

Docker caches layers. Before running an instruction it checks whether it has
already built that exact instruction on that exact parent layer, and for a
`COPY` it also hashes the files being copied. If everything matches, it reuses
the result — the line prints `CACHED` and takes no time.

The rule that follows: **the first instruction whose inputs changed invalidates
that layer and every layer after it.**

### 3.2 The consequence

Our order is dependencies, then source:

```dockerfile
COPY deployment/requirements.txt ...
RUN pip install ...          # ~150 MB, tens of seconds
COPY src/ ...                # kilobytes, instant
COPY api/ ...
```

Edit `api/main.py` and rebuild: the requirements file is byte-identical, so the
install layer is reused and only the last few layers are redone. A rebuild
after a code change is a second or two.

Now imagine the other order:

```dockerfile
COPY . /app                  # changes on EVERY commit
RUN pip install -r deployment/requirements.txt
```

Every commit changes the context, so the `COPY` layer is new, so the install
below it can never be cached, so every rebuild re-downloads and reinstalls
scikit-learn. Same image, minutes instead of seconds, on every single build —
including every CI run, if the image is ever built there.

The general principle: **order instructions from least to most frequently
changed.** Base image, then system packages, then dependencies, then source.

### 3.3 The build context and `.dockerignore`

```bash
docker build -t crop-api -f deployment/Dockerfile .
```

The `.` is the **build context**: before executing a single instruction, the
CLI packages that directory and sends it to the Docker daemon. `COPY` can only
read from the context — that is why the build runs from the repository root
(the Dockerfile needs `src/`, `api/`, `data/raw/`) while the Dockerfile itself
lives in `deployment/`, named by `-f`.

`.dockerignore` excludes paths from that upload. It matters for three reasons,
in increasing order of importance:

1. **Speed.** `.git/` alone can be larger than the source; uploading it is
   wasted every build.
2. **Cache stability.** A file in the context can invalidate a `COPY` even when
   the code did not change — an edited notebook, a new `.pytest_cache`.
3. **Safety.** Anything in the context can be copied into an image, deliberately
   or by a careless `COPY . .`, and anything in an image can be extracted by
   anyone who pulls it. A `.env`, a key, a credentials file — this is the most
   common way a secret ends up in a published image.

Ours excludes `.git/`, `.github/`, virtual environments, Python caches,
`notebooks/`, `tests/`, `app/`, `docs/`, `models/` and `data/processed/`.

---

## 4. Two requirements files, and why the small one is small

### 4.1 The distinction

| | Root `requirements.txt` | `deployment/requirements.txt` |
| --- | --- | --- |
| Audience | A developer, and CI | One serving process |
| Contents | Everything the course uses | What `api/main.py` reaches by import |
| Size | 15 pins + transitive deps | 7 pins + transitive deps |
| Read by | `pip install -r requirements.txt` locally and in CI | the `RUN pip install` in the Dockerfile |

The serving list is derived mechanically, not by taste. Follow the imports:

```
api/main.py
  -> fastapi, sklearn (a type annotation), api/schemas.py -> pydantic
  -> src/config.py -> src/data/* -> pandas
  -> src/pipelines/predict_pipeline.py -> joblib, pandas, sklearn
       (and, on a cold start, training_pipeline -> numpy)
run by: uvicorn
```

That walk yields exactly seven names: `fastapi`, `pydantic`, `uvicorn`,
`numpy`, `pandas`, `scikit-learn`, `joblib`. Everything else in the root file
is reached by nothing the server executes.

### 4.2 What is missing, and why

| Absent | Why the server does not need it |
| --- | --- |
| `jupyter` | The image opens no notebook. A notebook server in a production image is a remote shell nobody asked for. |
| `matplotlib`, `seaborn` | The API returns JSON. It draws nothing. |
| `pytest`, `ruff` | Tests and lint run in CI, on the source, before the image is built. Shipping them ships a second way to execute arbitrary code. |
| `streamlit` | The demo UI is a separate process that calls `predict()` in-process (Week 10 §5.3). It is not part of the API image. |
| `httpx` | Only `fastapi.testclient` uses it, and the tests are not in the image. |
| `xgboost`, `shap` | Optional exploration packages; the shipped model is Gaussian naive Bayes. `xgboost` alone is >100 MB. |

Four reasons this is worth the extra file:

1. **Size.** Smaller downloads, faster pulls, faster deploys.
2. **Attack surface.** Every package is code that could have a vulnerability
   and tooling that could be abused. The safest dependency is the one that is
   not there.
3. **Honesty.** The file is a statement of what the service actually needs. If
   `api/` ever quietly grew a matplotlib import, this file would have to change
   to say so.
4. **Build speed.** §3 — a smaller install is a faster cached layer to rebuild
   when it is finally invalidated.

### 4.3 The rule that keeps them in step

Every version in the deployment file is **identical** to the root file's pin.
Not "compatible" — identical. A pipeline pickled by scikit-learn 1.6.1 loaded
by 1.7 may warn, may misbehave, or may fail on an attribute that moved; a
container that serves different numbers from your laptop is worse than one that
refuses to start.

The duplication is the price of the split. Whoever bumps a version in one file
must bump it in the other, and Exercise 6 makes that failure visible.

---

## 5. Continuous integration

### 5.1 What CI and CD mean

**Continuous integration** — every change is merged into the shared branch
often, and every change is *automatically* built and tested on a clean machine.
The word "continuous" is about frequency; the value is in "automatically" and
"clean machine".

**Continuous delivery** — every change that passes is automatically packaged
into a release artifact, ready to deploy at the press of a button.
**Continuous deployment** goes one step further and presses the button itself.

This week implements CI only. The workflow tests; it does not build the image,
push it to a registry, or release anything. Naming the other two matters
because "CI/CD" is said as one word and they are three different commitments.

### 5.2 Why bother on a solo project

The objection is reasonable: *I can run `pytest` myself.*

1. **You will forget.** Not out of laziness — because the change was one line
   in a docstring at 11pm.
2. **Your machine lies.** It has an artifact in `models/`, a stale
   `__pycache__`, a package you installed and never pinned, an environment
   variable you set in March. A CI runner has none of that. It is the only
   honest test of "does a clean clone work?", which is exactly what a reader of
   your repository will attempt.
3. **It documents the commands.** `ci.yml` is an executable answer to "how do I
   run this project's checks?" that cannot go stale, because it runs.
4. **It is a visible signal.** A green tick on every commit, and a badge in the
   README, tell a reviewer more about your engineering than a paragraph
   claiming the same thing.
5. **It catches the interaction you did not think of.** Python 3.11 on Ubuntu
   versus 3.12 on your laptop; a test that passes only because a previous test
   left a file behind; a missing pin that works locally because the package is
   already installed.
6. **It is the habit that scales.** On a team, CI is what makes "merge to main"
   safe. Learning it on a project with one contributor costs an afternoon;
   learning it on a project with ten costs an outage.

### 5.3 GitHub Actions vocabulary

The file is [`.github/workflows/ci.yml`](../../../.github/workflows/ci.yml).
The path is fixed: GitHub reads `.github/workflows/*.yml` and nowhere else.

* **Workflow** — one YAML file: when to run, and what to run.
* **Trigger / event** (`on:`) — what causes a run: a push, a pull request, a
  schedule, a manual click.
* **Job** — a unit of work with its own fresh virtual machine. Jobs run in
  parallel unless one `needs:` another.
* **Runner** (`runs-on:`) — the machine type. `ubuntu-latest` is GitHub-hosted
  and free for public repositories.
* **Step** — one command (`run:`) or one reusable action (`uses:`). Steps run
  in order in the same directory, and the job fails at the first non-zero exit
  code.
* **Action** — a packaged step someone else wrote, pinned by version:
  `actions/checkout@v4` clones your repository onto the runner (nothing is
  there by default — not even your code); `actions/setup-python@v5` installs an
  interpreter.

### 5.4 Reading our workflow

```yaml
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:
```

Three triggers: a push to `main` (the branch that must stay green), a pull
request targeting `main` (checked *before* it merges — this is the one that
prevents breakage rather than reporting it), and a manual button.

```yaml
permissions:
  contents: read
concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true
```

The automatic token is read-only, because this workflow reads code and writes
nothing; and a newer push to the same branch cancels the older run, because
results for a commit nobody will look at are just queue time.

The steps, in order, and what each proves:

| Step | Proves |
| --- | --- |
| `actions/checkout@v4` | The commit exists and can be fetched |
| `actions/setup-python@v5` (3.11, pip cache) | The project's target interpreter is available |
| `pip install -r requirements.txt` | **The pins actually resolve on a clean machine** — this is where an unpinned or mistyped dependency dies |
| `ruff check .` | Style and import order hold across the repository |
| `pytest` | All 404 tests pass with no local artifact, no cache, no model file |

Note which requirements file that is: the **root** one. CI runs the *tests*, and
the tests need `pytest`, `httpx` and the rest of the development environment.
The trimmed file is for the image, and only the image. A student who "tidies
up" by pointing CI at `deployment/requirements.txt` gets
`pytest: command not found` — which is Exercise 9.

The model file is git-ignored, so the runner has none. Nothing special is
needed: Week 9's `load_pipeline()` trains on demand, and the tests train into a
temporary directory.

### 5.5 What a run looks like

After the push, the repository's **Actions** tab lists a run per commit:

* a yellow dot while it runs;
* a green tick if every step exited 0;
* a red X otherwise, with the failing step expandable to its log.

On a pull request the same result appears as a **status check** at the bottom
of the conversation. A branch protection rule can require it to be green before
merging — the mechanism by which "main is never broken" stops being a promise
and starts being a rule.

**This is the part the agent that wrote this week cannot verify.** A workflow
file only runs once GitHub has the commit, and a run of the workflow *for the
push that adds it* happens after that push completes. Somebody with a browser
has to open the Actions tab and look. That is you, and
[`validation.md`](validation.md) says so explicitly.

---

## 6. What the container does *not* change

Worth stating plainly, because "dockerized" is often heard as "production
ready":

* **Same model, same numbers.** `jute` at 0.7253 inside the container, because
  it is the same seeded artifact from the same committed data.
* **Same security posture.** No authentication, no rate limiting, no TLS, no
  CORS policy, no request size limit. A container is a packaging format, not a
  firewall.
* **Same capacity.** One uvicorn worker in one process. `docker run` twice on
  different ports gives you two, and nothing in front of them to share traffic.
* **Same blindness.** No request logging beyond uvicorn's access lines, no
  metrics, no model version in the response, no drift detection.

What it *does* change: the environment is now specified, versioned and
executable, and it goes wherever the image goes.

---

## 7. What this week does not give you

* **A public URL.** The image is local. Publishing it means a registry
  (`docker push`) and a host that pulls and runs it, with a domain and a
  certificate. None of that is in this repository.
* **Continuous delivery.** CI tests. Nothing builds or ships the image
  automatically.
* **Multi-architecture images.** The image is built for the machine that built
  it. An arm64 laptop and an amd64 server need `docker buildx` and a manifest.
* **Smaller images by build stage.** A multi-stage build (compile in a fat
  stage, copy the result into a slim one) is the standard next step; ours is
  already small enough that it would add complexity for little gain.
* **Secrets management.** There are no secrets in this project, which is the
  only reason the subject can be skipped. The moment there is one, it belongs
  in the runtime environment (`docker run -e`, GitHub Actions secrets), never
  in a Dockerfile layer or a committed file.
* **A finished repository.** That is Week 12: the final review, the README, and
  an honest account of what was built, what it can do, and what it cannot.

---

## Summary

* A container ships the **environment** with the program, which is the half
  that `requirements.txt` never covered. Image = artifact; container = a
  running process from it.
* A Dockerfile is read top to bottom, and each `FROM`/`COPY`/`RUN` is a cached
  **layer**. Put what changes rarely first, so a code edit rebuilds seconds
  rather than minutes.
* The **build context** is uploaded before anything runs; `.dockerignore`
  controls what is in it, which is a speed, cache *and* safety question.
* `deployment/requirements.txt` lists what one serving process imports — seven
  packages, versions identical to the root file's. No Jupyter, no plots, no
  tests, no Streamlit, no XGBoost, no SHAP.
* The model is **trained during the build**, so containers start by reading a
  file; the price is that retraining means rebuilding.
* Inside a container, bind to `0.0.0.0`; `EXPOSE` documents, `-p` publishes;
  `CMD` in exec form makes uvicorn PID 1 so `docker stop` works.
* **CI** runs the checks automatically on a clean machine on every push and
  pull request. Even alone, it is the only honest answer to "does a clean clone
  of this repository work?".
* **CD** is named, not implemented. The container is portable, not public, and
  the repository is not finished — Week 12 is.
