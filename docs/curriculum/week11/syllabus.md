# Week 11 — Containerization and Continuous Integration

## Title

**"It works on my machine" is not a deployment: an image, a trimmed
requirements file, and a robot that runs the tests**

## Learning objectives

By the end of this week a student should be able to:

1. State the problem **containerization** solves — environment consistency —
   and say what a container includes that a virtual environment does not.
2. Distinguish an **image** (a built, immutable artifact) from a **container**
   (a running process started from one), and a container from a virtual machine.
3. Read `deployment/Dockerfile` line by line and say what each instruction
   contributes to the image.
4. Explain the **layer cache** and why dependencies are copied and installed
   *before* the source code.
5. Explain what the **build context** is, why the build runs from the repository
   root with `-f deployment/Dockerfile`, and what `.dockerignore` keeps out of it.
6. Justify why `deployment/requirements.txt` is **smaller** than the root
   `requirements.txt`, and name three packages that are deliberately missing.
7. Say why `--host 0.0.0.0` is required inside a container and what
   `EXPOSE` does and does not do.
8. Publish a port with `docker run -p`, and check the container is healthy via
   `GET /health` and via Docker's own `HEALTHCHECK`.
9. Define **CI** and **CD**, and argue why CI is worth having on a solo
   portfolio project.
10. Read `.github/workflows/ci.yml` and name its **trigger**, its **job**, its
    **runner** and its **steps**, and predict what happens on the next push.
11. Explain why CI installs the *root* requirements file while the image
    installs the trimmed one.

## Prerequisites

Weeks 1-10, in full. This week assumes and does **not** re-explain:

* the pinned `requirements.txt` and why every version is fixed
  ([Week 1 notes](../week01/learning_notes.md));
* the saved artifact, train-on-demand and `src/config.py`
  ([Week 9 notes](../week09/learning_notes.md));
* the FastAPI app, `uvicorn`, `/health` and `/predict`
  ([Week 10 notes](../week10/learning_notes.md));
* `pytest`, `ruff` and the 404-test suite (every week since Week 1).

**No new Python dependency this week.** The root `requirements.txt` is
unchanged; the new dependency is a tool outside Python — Docker — and a
service outside the repository — GitHub Actions.

## Concepts covered

| Concept | First introduced |
| --- | --- |
| Environment consistency ("works on my machine") | Week 11 |
| Container vs virtual environment vs virtual machine | Week 11 |
| Image vs container | Week 11 |
| Base image, and why `-slim` | Week 11 |
| Dockerfile instructions: `FROM`, `WORKDIR`, `ENV`, `COPY`, `RUN`, `EXPOSE`, `CMD` | Week 11 |
| `CMD` vs `ENTRYPOINT`; exec form vs shell form | Week 11 |
| Layers and the build cache | Week 11 |
| Build context and `.dockerignore` | Week 11 |
| Trimmed deployment requirements vs development requirements | Week 11 |
| Reproducible builds through pinning | Week 1 (pins), Week 11 (in an image) |
| Port publishing (`-p`) vs `EXPOSE` | Week 11 |
| Binding to `0.0.0.0` vs `127.0.0.1` | Week 11 |
| Running as a non-root user | Week 11 |
| `HEALTHCHECK` and container liveness | Week 10 (endpoint), Week 11 (probe) |
| Baking the model artifact into the image | Week 11 |
| Continuous integration | Week 11 |
| Continuous delivery / deployment | Week 11 |
| GitHub Actions: workflow, trigger, job, runner, step | Week 11 |
| Clean-runner reproducibility | Week 11 |
| Status checks on a pull request | Week 11 |

Each also appears in [`docs/ml_concepts.md`](../../ml_concepts.md).

## Connection to the previous week

Week 10 ended with a running service and a blunt limitation: *"Both commands
need a checkout, Python 3.11/3.12 and the pinned environment. There is no image,
no `Dockerfile`, no host, no domain, no HTTPS."*

Week 11 removes the first half of that. The API is now one `docker run` away on
any machine with Docker, with no checkout, no virtual environment and no `pip
install`. It also adds a second reader for the repository: a robot that runs
`ruff` and `pytest` on every push, on a machine that has never seen your laptop.

What is *not* removed: the host, the domain and the HTTPS. The container runs
locally. Publishing it to the internet is beyond this course's scope, and
pretending otherwise would be the one thing a portfolio reviewer would catch.

## Connection to the ML lifecycle

```
Frame the problem -> Get the data -> Explore -> Prepare -> Model
    -> Evaluate -> Improve -> Productionize -> [Deploy] -> Monitor
```

Week 10 made the model reachable over a network on one machine. Week 11 makes
it **portable** — the same artifact runs anywhere Docker runs — and makes the
repository **self-checking**. Both are deployment concerns, and neither touches
the model: accuracy is still 99.55%, and the same seeded pipeline answers
`jute` at 0.7253 inside the container as it did outside it.

## Expected student outcome

### The student CAN, after this week

* **Build the image**:
  `docker build -t crop-api -f deployment/Dockerfile .`
* **Run it and get a prediction over HTTP**, with no Python installed on the
  host: `docker run --rm -p 8000:8000 crop-api`, then `curl` `/health` and
  `/predict`.
* **Explain why `deployment/requirements.txt` is smaller than the root one** —
  it lists what one serving process imports, not what a developer needs; no
  Jupyter, no matplotlib, no pytest, no Streamlit, no XGBoost, no SHAP.
* **Read the Dockerfile and explain every line**: base image, environment
  variables, the dependency layer before the source layer, the training step,
  the non-root user, `EXPOSE`, `HEALTHCHECK`, `CMD`.
* **Explain why layer order affects build speed**, and predict which layers a
  one-line change to `api/main.py` invalidates.
* **Explain what the CI workflow will do once it runs**: on a push or pull
  request to `main`, GitHub starts a fresh Ubuntu runner, checks out the
  commit, installs Python 3.11 and the root `requirements.txt`, runs
  `ruff check .` and then `pytest`, and reports a green tick or a red X against
  the commit.
* **Find a failing run** in the Actions tab and read the step that failed.

### The student CANNOT yet

* **Call the repository finished.** The final review — README, documentation,
  a portfolio-ready account of what was built and what was learned — is
  **Week 12**.
* **Reach the container from anywhere but this machine.** No registry push, no
  host, no domain, no TLS certificate. `docker run` on your laptop is not a
  deployment to the public internet.
* **Deploy automatically.** The workflow is CI only: it tests, it does not
  build the image, push it anywhere, or release anything. The *CD* half is
  named and explained, not implemented.
* **Serve real traffic safely.** Everything Week 10 could not do — no auth, no
  rate limiting, no TLS, no CORS policy, one worker — is still true. A
  container does not add security; it adds portability.
* **Update the model without rebuilding.** The artifact is baked into the
  image at build time, so a retrained model means a new image.
* **Monitor anything.** No metrics, no request logging beyond uvicorn's access
  lines, no drift detection. Unchanged since Week 9.

## Deliverables for the week

* `deployment/Dockerfile` — `python:3.11-slim`, the trimmed requirements
  installed before the source, the model trained during the build, a non-root
  user, `EXPOSE 8000`, a `HEALTHCHECK` on `/health`, and
  `CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]`.
* `deployment/requirements.txt` — seven pinned packages, each version identical
  to the root file's, with the omissions listed and justified in comments.
* `.dockerignore` — `.git/`, `tests/`, `notebooks/`, `app/`, `docs/`, caches,
  virtual environments and local artifacts kept out of the build context.
* `.github/workflows/ci.yml` — triggered on push and pull request to `main`
  (plus manual dispatch); one job on `ubuntu-latest` that installs the root
  `requirements.txt`, runs `ruff check .` and runs `pytest`.
* `docs/deployment_guide.md` — the exact build and run commands, what the ports
  mean, how to check health, and a troubleshooting table.
* This week's four curriculum documents.
* `requirements.txt` unchanged; `docs/ml_concepts.md` and the README progress
  table updated.
