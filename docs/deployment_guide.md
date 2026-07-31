# Deployment Guide

How to build, run and check the Crop Recommendation API as a container.
Written in Week 11; the API itself is Week 10's
([`docs/architecture.md`](architecture.md) draws the request flow).

Everything below runs from the **repository root**.

---

## What gets built

| File | Role |
| --- | --- |
| `deployment/Dockerfile` | The recipe: base image, dependencies, source, the trained model, the command |
| `deployment/requirements.txt` | The *serving* dependencies — pinned, and a subset of the root `requirements.txt` |
| `.dockerignore` | What never enters the build context (`.git/`, `tests/`, `notebooks/`, `app/`, `docs/`, …) |

The image contains `src/`, `api/`, `data/raw/`, a model trained during the
build, and seven Python packages. It does **not** contain the notebooks, the
tests, the Streamlit demo, Jupyter, matplotlib, seaborn, XGBoost or SHAP.

---

## Prerequisites

* Docker installed and its daemon running — `docker version` must print both a
  *Client* and a *Server* block. On Docker Desktop, "daemon running" means the
  Desktop application is open.
* Nothing else. Python, the virtual environment and the pinned development
  requirements are **not** needed to build or run the image; that is the point
  of the exercise.

---

## Build

```bash
docker build -t crop-api -f deployment/Dockerfile .
```

Read the command right to left:

* `.` — the **build context**: the directory sent to the daemon. It must be the
  repository root, because the Dockerfile copies `src/`, `api/` and `data/raw/`.
* `-f deployment/Dockerfile` — the recipe lives in a subdirectory, the context
  does not. Running `docker build .` from inside `deployment/` would send a
  context containing only two files and fail on the first `COPY`.
* `-t crop-api` — the **tag**, the name you will use to run it.

A first build downloads the base image and the wheels and takes a couple of
minutes. A rebuild after editing `api/main.py` takes seconds: the dependency
layer is cached and only the layers below the changed `COPY` are redone.

Verify the image exists:

```bash
docker images crop-api
```

```
REPOSITORY   TAG       IMAGE ID       CREATED         SIZE
crop-api     latest    c14848bfeb53   2 minutes ago   489MB
```

---

## Run

```bash
docker run --rm -p 8000:8000 crop-api
```

* `-p 8000:8000` — **publish** the container's port 8000 as port 8000 on your
  machine. The left half is the host port and can be changed
  (`-p 9000:8000` serves it on `http://127.0.0.1:9000`); the right half is
  fixed by the `CMD` in the Dockerfile.
* `--rm` — delete the stopped container afterwards. The image stays.

The logs end with:

```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

`0.0.0.0` inside the container, not `127.0.0.1`: a server bound to loopback
inside a container is reachable only from inside that container, and the
published port would connect to nothing.

Stop it with `Ctrl-C`. To run it in the background instead:

```bash
docker run -d --rm --name crop-api -p 8000:8000 crop-api
docker logs -f crop-api      # follow the logs
docker stop crop-api         # stop it (and, with --rm, remove it)
```

### Ports

| Port | Where | What listens |
| --- | --- | --- |
| 8000 | inside the container | uvicorn, via `CMD` and `EXPOSE 8000` |
| 8000 | on your machine | only because of `-p 8000:8000` |

`EXPOSE` is documentation. It opens nothing by itself — `-p` does.

---

## Check the container is healthy

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status":"ok","model_loaded":true,"n_classes":22}
```

Three claims, in order of strength:

1. **The container is running** — `docker ps` lists it.
2. **The port is published and the process is answering** — you got a response
   rather than `Connection refused`.
3. **It can actually serve** — `"model_loaded":true` and 22 classes. A reply of
   `{"status":"degraded","model_loaded":false,"n_classes":0}` means the process
   is up and useless; check `docker logs`.

The Dockerfile also declares a `HEALTHCHECK`, so Docker polls `/health` itself:

```bash
docker ps --filter name=crop-api --format "{{.Status}} {{.Ports}}"
```

```
Up 12 seconds (healthy) 0.0.0.0:8000->8000/tcp
```

`(healthy)` comes from that probe. `(unhealthy)` is a signal an orchestrator can
act on — restart the container, or stop routing traffic to it.

### A real prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"N":90,"P":42,"K":43,"temperature":25,"humidity":80,"ph":6.5,"rainfall":200}'
```

```json
{"crop":"jute","confidence":0.725342776978384,"probabilities":{"jute":0.725342776978384,"rice":0.2746571705151149,"coffee":5.250650112483512e-08}}
```

Identical to Week 10's local `uvicorn` answer, to the last digit, because it is
the same code, the same pinned versions and the same seeded artifact.

The interactive documentation is published too: open
<http://127.0.0.1:8000/docs>.

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Cannot connect to the Docker daemon` | The daemon is not running | Start Docker Desktop / `sudo systemctl start docker` |
| `failed to compute cache key: "/src": not found` | Built from inside `deployment/` | Build from the repository root with `-f deployment/Dockerfile` |
| `Bind for 0.0.0.0:8000 failed: port is already allocated` | Something else holds port 8000 — often Week 10's local `uvicorn` | Stop it, or publish elsewhere: `-p 9000:8000` |
| `curl: (7) Failed to connect` | `-p` omitted, or the container exited | `docker ps -a`, then `docker logs <name>` |
| `{"model_loaded":false}` | Start-up failed to load the model | `docker logs <name>` — the traceback is there; `/health` stays up on purpose |
| The image is huge / builds slowly | Root `requirements.txt` used, or `.dockerignore` ignored | Build with `deployment/requirements.txt`; check `.dockerignore` is at the repository root |

---

## What this deployment is not

* **Not public.** The image runs on your machine. There is no host, no domain,
  no HTTPS and no registry push.
* **Not authenticated.** Anyone who can reach the port can spend your CPU.
* **Not scaled.** One uvicorn worker in one container; no replicas, no load
  balancer, no queue.
* **Not observed.** Nothing records which model answered, or what it was asked.
  Monitoring and drift detection remain out of reach.

Those limits are the honest end of Week 11. Week 12 is the final review.

---

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request to `main`: it
checks out the repository on a clean Ubuntu runner, installs the **root**
`requirements.txt` (CI runs the tests, so it needs the development
environment), runs `ruff check .`, then runs `pytest`. A red X on a commit or a
pull request means the suite failed there; open the run in the repository's
**Actions** tab to read the log.

The workflow does not build the image and does not deploy anything. It answers
one question — *does this commit still pass on a machine that is not yours?* —
and that is the question a portfolio reviewer asks first.
