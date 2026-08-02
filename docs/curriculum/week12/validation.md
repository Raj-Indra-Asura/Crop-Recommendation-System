# Week 12 — Validation

> 🗺 [Roadmap](../README.md) › [Part V — Review (Week 12)](../README.md#part-v--review-week-12) › [Chapter 12 — Final Review and Portfolio Polish](README.md) › **§12.4 Validation**

Run these in order from the repository root. Each step lists the exact command
and the output captured from a real run on 2026-07-31 (Ubuntu, Python 3.12.3,
Docker 28.0.4, the pinned `requirements.txt`).

Timings, pip resolver chatter, image IDs and warning counts will differ on your
machine. The **numbers** will not — the data is committed and the pipeline is
seeded.

**This week's validation is split in two**, and the split is deliberate:

* **Steps 0-9** were run by the agent that implemented the week, in place. The
  output below is real, not illustrative.
* **Step 10 is yours**, and it is the one check nobody can perform on
  themselves: a genuine `git clone` of the merged `main` into an empty
  directory. The agent works *inside* the repository it is changing, so the
  closest it can get is a brand-new virtual environment in place (Step 2).
  A real fresh clone additionally proves that everything needed is actually
  committed.

---

## Step 0 — Environment

```bash
python --version
git rev-parse --short HEAD
docker --version
```

```
Python 3.12.3
3c33cbd9
Docker version 28.0.4, build b8034c0
```

The project targets Python 3.11; 3.12 is what this run used, and both are
supported. **No new Python dependency this week** — `requirements.txt` is
byte-identical to Week 11's.

---

## Step 1 — The audit: every markdown link resolves

The Student Review's first mechanical check. Every `](target)` in every markdown
file, resolved against the filesystem:

```bash
python - <<'PY'
import re, os, glob
bad, files = [], glob.glob('docs/**/*.md', recursive=True) + ['README.md', 'data/raw/README.md']
for f in files:
    d, fenced = os.path.dirname(f), False
    for i, line in enumerate(open(f), 1):
        if line.lstrip().startswith('```'):
            fenced = not fenced
        if fenced:
            continue                      # skip code samples, which are not links
        line = re.sub(r'`[^`]*`', '', line)   # and skip inline code, same reason
        for m in re.finditer(r'\]\(([^)]+)\)', line):
            t = m.group(1).split()[0]
            if t.startswith(('http', '#', 'mailto')):
                continue
            p = os.path.normpath(os.path.join(d, t.split('#')[0]))
            if not os.path.exists(p):
                bad.append((f, i, t))
for b in bad:
    print(b)
print("markdown files:", len(files))
print("broken links :", len(bad))
PY
```

```
markdown files: 55
broken links : 0
```

Fifty-five markdown files, several hundred relative links, none broken. Run this
again after any documentation change; it takes a second and it catches the
single most common documentation defect.

The two `continue`/`re.sub` lines matter more than they look. Without them the
checker reports two "broken links" that are really this file and the learning
notes *talking about* the link syntax — a checker that cries wolf on its own
documentation gets switched off within a week.

---

## Step 2 — Simulated fresh install (agent-run)

This is the check the week exists for. A **new** virtual environment, built from
nothing, installing only what `requirements.txt` declares.

```bash
python -m venv fresh-venv
source fresh-venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

```
Successfully uninstalled pip-24.0
Successfully installed pip-26.2
...
Successfully installed MarkupSafe-3.0.3 altair-5.5.0 annotated-types-0.8.0
anyio-4.14.2 ... fastapi-0.115.6 joblib-1.5.3 matplotlib-3.10.0 numpy-2.2.1
pandas-2.2.3 pydantic-2.10.4 pytest-8.3.4 ruff-0.8.4 scikit-learn-1.6.1
scipy-1.18.0 seaborn-0.13.2 shap-0.46.0 streamlit-1.41.1 uvicorn-0.34.0
xgboost-2.1.3
```

```bash
pip list --format=freeze | wc -l
```

```
148
```

Fifteen direct pins in `requirements.txt` pull 148 packages in total. The
directly pinned versions all appear verbatim in that install line — which is the
property that makes the numbers below reproducible.

`fresh-venv/` is git-ignored (`.gitignore`, "Virtual environments"), so this
step cannot dirty your working tree. Delete it with `rm -rf fresh-venv` when you
are done.

---

## Step 3 — Lint and the whole suite, from the fresh environment

```bash
ruff check .
```

```
All checks passed!
```

```bash
pytest
```

```
================ 404 passed, 1 skipped, 1032 warnings in 29.62s ================
```

**404 passed, 1 skipped** — identical to Weeks 10 and 11, because Week 12 adds
no Python code. The skip is Week 8's SHAP test on the documented fallback path.

> **Recorded in Week 12; the count has since moved.** Running the suite today
> gives **408 passed, 1 skipped**. The four extra tests are
> [`tests/test_curriculum_links.py`](../../../tests/test_curriculum_links.py),
> added *after* Week 12 finished, when the per-chapter `README.md` covers and
> the roadmap's reading order were introduced: with a book-shaped curriculum,
> a broken link or a footer pointing at the wrong chapter is a real defect, so
> it is checked mechanically rather than by re-reading. The test is structural
> only — it resolves every relative link and `#anchor`, checks the roadmap names
> each document exactly once, and checks each page's previous/next footer
> against the roadmap's order; it says nothing about the prose. See
> [the roadmap's "Keeping this order honest"](../README.md#keeping-this-order-honest).
> The output above is left as recorded rather than rewritten — the same rule
> every other recorded output in this curriculum follows. Expect `404 + 4`
> wherever this week quotes 404, and twelve test files rather than eleven.

The whole `tests/` directory, not this week's additions:

```bash
pytest --collect-only -q | tail -2
pytest --collect-only -q | grep '::' | cut -d: -f1 | sort | uniq -c
```

```
405 tests collected in 1.75s

     27 tests/test_api.py
     38 tests/test_baseline.py
    119 tests/test_classical_models.py
     20 tests/test_data_loader.py
     22 tests/test_eda.py
     61 tests/test_ensemble_models.py
     32 tests/test_explainability.py
     17 tests/test_predict_pipeline.py
     32 tests/test_preprocessing.py
     15 tests/test_training_pipeline.py
     22 tests/test_tuning.py
```

Eleven test files, one per capability the course added, from Week 1's loader to
Week 10's API. Every week is still checked. (A twelfth file,
`tests/test_curriculum_links.py`, joined them after this was recorded; it checks
the documentation's navigation, not a capability of the model.)

---

## Step 4 — The artifact rebuilds from nothing

Week 9's rule: the model is *derived*, never committed. Deleting it must be a
non-event.

```bash
rm -f models/crop_model.joblib
python -m src.pipelines.training_pipeline
```

```
Model:        naive_bayes {'var_smoothing': 1e-09}
Train rows:   1760
Test rows:    440
Accuracy:     0.9955
Macro F1:     0.9954
Weighted F1:  0.9954
Saved to:     /home/runner/.../models/crop_model.joblib
```

The same 0.9955 that Week 8 chose the model on, Week 9 packaged, Week 11 baked
into the image and this week's README quotes.

```bash
python -m src.pipelines.predict_pipeline
```

```
Input:      {'N': 90, 'P': 42, 'K': 43, 'temperature': 25, 'humidity': 80, 'ph': 6.5, 'rainfall': 200}
Prediction: jute
Top 3:      jute 0.7253, rice 0.2747, coffee 0.0000
```

`jute` at 0.7253, unchanged since Week 9 — and, per Week 8, the `rice`/`jute`
pair is exactly the one the model finds hard.

---

## Step 5 — The container still serves it

Week 11's image, rebuilt on this week's commit:

```bash
docker build -t crop-api -f deployment/Dockerfile .
```

```
#16 [8/9] RUN python -m src.pipelines.training_pipeline
#16 1.729 Model:        naive_bayes {'var_smoothing': 1e-09}
#16 1.729 Accuracy:     0.9955
#16 1.729 Saved to:     /app/models/crop_model.joblib
#16 DONE 2.4s
#18 writing image sha256:abbb738f8a82558ad87947ef48321f34e92d318ea8451e9332176b24316908f5 done
#18 naming to docker.io/library/crop-api done

real    0m32.751s
```

```bash
docker run -d --rm --name crop-api-w12 -p 8000:8000 crop-api
sleep 15
curl -s http://127.0.0.1:8000/health
```

```
{"status":"ok","model_loaded":true,"n_classes":22}
```

```bash
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"N":90,"P":42,"K":43,"temperature":25,"humidity":80,"ph":6.5,"rainfall":200}'
```

```
{"crop":"jute","confidence":0.725342776978384,"probabilities":{"jute":0.725342776978384,"rice":0.2746571705151149,"coffee":5.250650112483512e-08}}
```

The container's answer is bit-for-bit the answer Step 4 got outside it. That is
the whole promise of a seeded pipeline in a pinned image.

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" -d '{"N":90}'
docker ps --format "table {{.Names}}\t{{.Status}}"
docker images crop-api --format "{{.Repository}}:{{.Tag}}  {{.Size}}"
docker stop crop-api-w12
```

```
422
NAMES          STATUS
crop-api-w12   Up 15 seconds (healthy)
crop-api:latest  490MB

crop-api-w12
```

Week 10's 422 on a malformed body, and Week 11's `HEALTHCHECK` reporting
`(healthy)` rather than merely `Up`.

---

## Step 6 — The Streamlit screenshot in the README is real

```bash
streamlit run app/streamlit_app.py
```

Opened at `http://localhost:8501`, the pre-filled example row submitted, and the
result captured to `docs/images/streamlit_app.png`:

```
Recommended crop: jute
Confidence 72.5%
Top 3 candidates  [bar chart]
```

72.5% is 0.7253 rounded — the same prediction as Steps 4 and 5, produced by the
third of the three entry points. The screenshot in the README is that page, not
a mock-up.

Stop the server with `Ctrl-C` when you are done.

---

## Step 7 — No placeholders left

```bash
grep -rn "TODO\|TBD\|FIXME\|Not started" --include=*.md . | grep -v "week12/"
```

```
(no output)
```

The only surviving occurrences are in Week 12's own notes and in this file,
where the words are being *discussed* rather than left behind — which is why
`week12/` is excluded. The README progress table's last row no longer says
"Not started".

---

## Step 8 — Every promise about Week 12 is answered

```bash
grep -rn "Week 12" --include=*.md docs README.md | wc -l
```

```
75
```

Twenty of those lines sit in weeks 1-11 and carry twenty-two distinct promises
about this week, all written before it existed; each is
either delivered, delivered early in Week 11, or explicitly withdrawn in
[`learning_notes.md`](learning_notes.md) §6. The rest are this week's own
documents.

---

## Step 9 — The twelve weeks are complete on disk

```bash
ls -d docs/curriculum/week*/ | wc -l
for w in docs/curriculum/week*/; do
  for f in syllabus learning_notes exercises validation; do
    [ -f "$w$f.md" ] || echo "MISSING $w$f.md"
  done
done
ls docs/curriculum/week12/
```

```
12
capstone_reflection.md  exercises.md  learning_notes.md  syllabus.md  validation.md
```

No `MISSING` line: twelve weeks, four documents each, plus this week's
`capstone_reflection.md`.

---

## Step 10 — The real fresh clone (human-run, after merge)

Everything above ran *inside* the repository being changed. This is the check
that cannot be simulated: a clone into an empty directory, from the merged
`main`, on your own machine.

```bash
git clone https://github.com/Raj-Indra-Asura/Crop-Recommendation-System.git fresh-clone
cd fresh-clone
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
pytest
docker build -t crop-api -f deployment/Dockerfile . && docker run -p 8000:8000 crop-api
curl http://127.0.0.1:8000/health
```

What you should see, in order:

| Command | Expected |
| --- | --- |
| `git clone` | ~1 MB; `data/raw/Crop_recommendation.csv` present, `models/` empty |
| `pip install` | 148 packages, the pinned versions of Step 2 |
| `pytest` | `404 passed, 1 skipped` (`408` since `tests/test_curriculum_links.py` was added — see Step 3) |
| `docker build` | The training line printing `Accuracy: 0.9955` |
| `docker run` | uvicorn on `0.0.0.0:8000`, then `Ctrl-C` to stop it |
| `curl .../health` | `{"status":"ok","model_loaded":true,"n_classes":22}` |

If any of those differs, the difference is a defect and belongs in an issue.
Two things are worth checking specifically, because they are what a fresh clone
proves and an in-place run cannot:

* **`models/` is empty after the clone** and everything still works — the
  artifact really is derived, not committed.
* **No file outside the clone is needed.** No dataset download, no environment
  variable, no secret, no "ask the author".

---

## What this week's validation demonstrates

| Claim | Evidence |
| --- | --- |
| Reproducible | Step 2's fresh environment; Steps 4-6 producing the identical prediction three ways |
| Installable | Step 2, from `requirements.txt` alone; Step 10 from a clone |
| Documented | Step 1 (no broken links), Step 7 (no placeholders), Step 9 (all twelve weeks complete) |
| Bounded | The README's limitations section, and [`learning_notes.md`](learning_notes.md) §4-5 |
| Verifiable | Step 3 from a clean environment, and `.github/workflows/ci.yml` on every push |

Those are the five properties of §1.2 of this week's notes, each with a command
next to it. That is what "finished" means here — not a feeling, a checklist
somebody else can re-run.

<!-- nav:start -->

---

| ◀ Previous | ▲ Up | Next ▶ |
| --- | --- | --- |
| ◀ [§12.3 Exercises](exercises.md) | [Chapter 12 — Final Review and Portfolio Polish](README.md) · 🗺 [Roadmap](../README.md) | [§12.5 Capstone reflection](capstone_reflection.md) ▶ |

<!-- nav:end -->
