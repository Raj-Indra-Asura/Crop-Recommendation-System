# Chapter 11 — Containerization and Continuous Integration

> 🗺 [Roadmap](../README.md) › [Part IV — Production (Weeks 9-11)](../README.md#part-iv--production-weeks-9-11) › **Chapter 11 — Containerization and Continuous Integration**

**"It works on my machine" is not a deployment: an image, trimmed requirements,
and a CI pipeline**

Part IV — Production (Weeks 9-11) · Chapter 11 of 12 · the curriculum calls this
**Week 11**, and the directory is `docs/curriculum/week11/`.

---

## Before you open this chapter

* **You should already have finished:** Chapter 10: the API runs locally and its
  tests pass.
* **Previous chapter:** [Chapter 10 — Serving the Model Over
  HTTP](../week10/README.md)
* **Environment:** the virtual environment from Chapter 1, with
  `requirements.txt` installed.

## Read this chapter in this order

Top to bottom. Every row assumes the rows above it are done.

| Step | What | Where | Why here |
| --- | --- | --- | --- |
| 1 | §11.1 Syllabus | [`syllabus.md`](syllabus.md) | objectives, prerequisites, concepts, deliverables — read first, it is the map of the chapter |
| 2 | §11.2 Learning notes | [`learning_notes.md`](learning_notes.md) | the teaching text: what, why, where, how, and the common mistakes — read top to bottom |
| 3 | Code | [`deployment/Dockerfile`](../../../deployment/Dockerfile) | slim base, cached dependency layer, model trained at build, non-root user, healthcheck |
| 4 | Code | [`deployment/requirements.txt`](../../../deployment/requirements.txt) | seven pins, each justified |
| 5 | Code | [`.dockerignore`](../../../.dockerignore) | what never enters the build context |
| 6 | Code | [`.github/workflows/ci.yml`](../../../.github/workflows/ci.yml) | `ruff check .` and `pytest` on every push and PR |
| 7 | Code | [`docs/deployment_guide.md`](../../deployment_guide.md) | the exact build and run commands, and troubleshooting |
| 8 | §11.3 Exercises | [`exercises.md`](exercises.md) | beginner, then intermediate, then challenge — do them before moving on |
| 9 | §11.4 Validation | [`validation.md`](validation.md) | run the commands, compare against the real recorded output |

## What this chapter adds to the repository

* [`deployment/Dockerfile`](../../../deployment/Dockerfile) — slim base, cached
  dependency layer, model trained at build, non-root user, healthcheck
* [`deployment/requirements.txt`](../../../deployment/requirements.txt) — seven
  pins, each justified
* [`.dockerignore`](../../../.dockerignore) — what never enters the build
  context
* [`.github/workflows/ci.yml`](../../../.github/workflows/ci.yml) — `ruff check
  .` and `pytest` on every push and PR
* [`docs/deployment_guide.md`](../../deployment_guide.md) — the exact build and
  run commands, and troubleshooting

The full list, including documentation updates, is in the
[syllabus](syllabus.md#deliverables-for-the-week).

## Do not move on until

* [ ] You can build the image, run it, and get a healthy `/health` response from
  the container.
* [ ] You can explain why dependencies are installed before the source is
  copied.
* [ ] The exercises in [`exercises.md`](exercises.md) are done, not skimmed.
* [ ] Every command in [`validation.md`](validation.md) produced the output
  recorded there.

Then go to [Chapter 12 — Final Review and Portfolio
Polish](../week12/README.md).

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| ◀ [§10.4 Validation](../week10/validation.md) | 🗺 [Roadmap](../README.md) | [§11.1 Syllabus](syllabus.md) ▶ |

<!-- nav:end -->
