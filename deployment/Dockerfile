# syntax=docker/dockerfile:1
#
# Crop Recommendation API - serving image (Week 11)
#
# Build from the REPOSITORY ROOT, not from this directory, because the build
# context has to contain `api/`, `src/` and `data/`:
#
#     docker build -t crop-api -f deployment/Dockerfile .
#     docker run --rm -p 8000:8000 crop-api
#     curl http://127.0.0.1:8000/health
#
# Layer order below is deliberate: the things that change rarely (base image,
# dependencies) come first, and the things that change on every commit (the
# source) come last, so an ordinary code edit rebuilds only the final layers.

# --- Base image --------------------------------------------------------------
# `slim` is the small Debian variant: a real glibc userland (which the numpy,
# pandas and scikit-learn wheels need) without the compilers, docs and tooling
# of the full image. The tag is pinned to a minor version so that a rebuild
# next month gets the same interpreter as today.
FROM python:3.11-slim

# --- Interpreter behaviour ---------------------------------------------------
# PYTHONDONTWRITEBYTECODE: no .pyc files to bloat the layer; the container is
#   thrown away, so the cache never pays for itself.
# PYTHONUNBUFFERED: logs reach `docker logs` immediately instead of sitting in
#   a buffer, which matters when the process dies before flushing.
# PIP_NO_CACHE_DIR: pip's download cache would be baked into the image for
#   nothing.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# --- Dependencies ------------------------------------------------------------
# Copied and installed BEFORE the source. Docker caches each instruction against
# the files it read: as long as deployment/requirements.txt is byte-identical,
# this ~150 MB install is reused and a source edit costs seconds, not minutes.
# Copying the whole repository first would invalidate it on every commit.
COPY deployment/requirements.txt /app/deployment/requirements.txt
RUN pip install --no-cache-dir -r /app/deployment/requirements.txt

# --- Application source ------------------------------------------------------
# Only what the server actually reads at run time. `app/` (Streamlit), `tests/`,
# `notebooks/` and `docs/` are not copied - see .dockerignore for the rest.
# `data/raw/` is copied because the model is trained during the build below.
COPY src/ /app/src/
COPY api/ /app/api/
COPY data/raw/ /app/data/raw/

# --- The model artifact ------------------------------------------------------
# `models/crop_model.joblib` is git-ignored, so it is never in the build
# context. Training it here bakes a deterministic artifact into the image
# (seeded in src/config.py) and keeps start-up fast: the container loads a file
# instead of fitting a model while a load balancer waits for /health.
RUN python -m src.pipelines.training_pipeline

# --- Run as a non-root user --------------------------------------------------
# The default user in a container is root. A web server that is reachable from
# a network should not be: created after the build steps so that /app stays
# root-owned and read-only to the server process.
RUN useradd --create-home --uid 10001 appuser
USER appuser

# --- Network -----------------------------------------------------------------
# Documentation, not a firewall rule: EXPOSE records the port the process
# listens on. Publishing it to the host is `docker run -p 8000:8000`.
EXPOSE 8000

# The container's own liveness probe, using the endpoint Week 10 built. Python
# is used rather than curl because the slim image ships no curl.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status == 200 else 1)"

# --- What the container runs -------------------------------------------------
# Exec form (a JSON array), so uvicorn is PID 1 and receives `docker stop`'s
# SIGTERM directly instead of being wrapped in a shell that swallows it.
# `--host 0.0.0.0` is required: 127.0.0.1 inside a container is reachable only
# from inside that container, so a published port would connect to nothing.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
