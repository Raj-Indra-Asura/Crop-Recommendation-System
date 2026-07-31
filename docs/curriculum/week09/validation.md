# Week 9 — Validation

Run these in order from the repository root. Each step lists the exact command
and the output captured from a real run on 2026-07-31 (Python 3.12.3, the pinned
`requirements.txt`).

Numbers that come from a seeded computation should match yours exactly. Timings
and file timestamps will not.

---

## Step 0 — Environment

```bash
python --version
pip install -r requirements.txt
python -c "import joblib, sklearn, numpy; print(joblib.__version__, sklearn.__version__, numpy.__version__)"
```

```
Python 3.12.3
1.5.3 1.6.1 2.2.1
```

The only new pin this week:

```
joblib==1.5.3
```

It was already present as a scikit-learn dependency. It is named explicitly
because Week 9 calls `joblib.dump` and `joblib.load` directly, and an artifact
must be reloaded by a version that can read it.

---

## Step 1 — Start from a clean `models/`

The trained artifact is **not** committed, so a fresh clone looks like this. If
you have run the pipeline before, delete the file to reproduce the state.

```bash
rm -f models/crop_model.joblib
ls -la models/
```

```
total 8
drwxrwxr-x  2 runner runner 4096 Jul 31 14:05 .
drwxr-xr-x 11 runner runner 4096 Jul 31 14:04 ..
-rw-rw-r--  1 runner runner    0 Jul 31 13:58 .gitkeep
```

Only `.gitkeep` — the empty file that keeps the directory itself in version
control. Confirm git is ignoring the artifact rather than merely not having seen
it yet:

```bash
git check-ignore -v models/crop_model.joblib
```

```
.gitignore:20:models/*.joblib	models/crop_model.joblib
```

---

## Step 2 — Train from the command line

```bash
python -m src.pipelines.training_pipeline
```

```
Model:        naive_bayes {'var_smoothing': 1e-09}
Train rows:   1760
Test rows:    440
Accuracy:     0.9955
Macro F1:     0.9954
Weighted F1:  0.9954
Saved to:     /.../Crop-Recommendation-System/models/crop_model.joblib

real	0m2.335s
```

Three things to check against earlier weeks:

* **1,760 / 440** is the Week 3 stratified split, seed 42, unchanged.
* **0.9955 accuracy, 0.9954 macro F1** are the Week 8 test numbers for Gaussian
  naive Bayes, to four decimal places. Week 9 changed no modelling decision, and
  the identical numbers are the evidence.
* The whole run takes about **2.3 seconds**, which is why training on demand
  (Step 6) is a reasonable fallback rather than a disaster.

The file now exists:

```bash
ls -l models/
```

```
-rw-rw-r-- 1 runner runner 6329 Jul 31 14:05 crop_model.joblib
```

**6.3 KB.** The training data is 150 KB and scikit-learn is tens of megabytes;
the artifact is neither. It holds learned numbers and references to classes, and
nothing else.

The script takes two flags for the cases where the defaults are wrong —
`--model-path` to write elsewhere, and `--model-name` to try a different model
from the Week 5-7 registry:

```bash
python -m src.pipelines.training_pipeline --model-name random_forest --model-path /tmp/rf.joblib
```

```
Model:        random_forest {}
Train rows:   1760
Test rows:    440
Accuracy:     0.9955
Macro F1:     0.9955
Weighted F1:  0.9955
Saved to:     /tmp/rf.joblib
```

`--model-name` drops `FINAL_MODEL_PARAMS` (a `var_smoothing` means nothing to a
forest). The flag is for experiments; `src/config.py` is still where the
project's *decision* is recorded.

---

## Step 3 — The artifact is derived, not source

```bash
git status --short
```

```
```

Empty. A 6.3 KB binary was just created inside the repository and git has
nothing to say about it, because `.gitignore` line 20 excludes it. Only
`data/raw/` — the input nothing can regenerate — is committed.

---

## Step 4 — Look inside the reloaded pipeline

```bash
python -c "
import joblib
p = joblib.load('models/crop_model.joblib')
print(type(p))
print(p.named_steps)
print(p.named_steps['preprocess'].named_transformers_['numeric'].mean_)
print(p.named_steps['model'].class_prior_[:3], len(p.classes_))
"
```

```
<class 'sklearn.pipeline.Pipeline'>
{'preprocess': ColumnTransformer(transformers=[('numeric', StandardScaler(),
                                 ['N', 'P', 'K', 'temperature', 'humidity',
                                  'ph', 'rainfall'])],
                  verbose_feature_names_out=False), 'model': GaussianNB()}
[ 50.54772727  53.33977273  48.14318182  25.6094092   71.41676188
   6.47383024 103.4515899 ]
[0.04545455 0.04545455 0.04545455] 22
```

One object holding both halves: the Week 3 `ColumnTransformer` with its seven
learned means (of the **training** rows — note `rainfall` at 103.45, not the
full dataset's mean), and the Week 8 `GaussianNB` with 22 classes and a flat
`1/22 = 0.04545` prior, because every crop has exactly 100 rows.

Note what had to happen for that command to work: Python imported
`sklearn.pipeline.Pipeline`, `StandardScaler` and `GaussianNB` **from your
installed scikit-learn**. The file names those classes; it does not contain
them.

---

## Step 5 — Predict

The one-liner from the week's brief:

```bash
python -c "from src.pipelines.predict_pipeline import predict; print(predict({'N':90,'P':42,'K':43,'temperature':25,'humidity':80,'ph':6.5,'rainfall':200}))"
```

```
jute
```

And the module's own manual check, which prints the runner-ups too:

```bash
python -m src.pipelines.predict_pipeline
```

```
Input:      {'N': 90, 'P': 42, 'K': 43, 'temperature': 25, 'humidity': 80, 'ph': 6.5, 'rainfall': 200}
Prediction: jute
Top 3:      jute 0.7253, rice 0.2747, coffee 0.0000
```

> **Read this carefully — it is the most instructive output of the week.** The
> README uses these measurements to illustrate the problem with `rice` as the
> intended answer, and the model says **jute at 0.7253, rice second at 0.2747**.
> That is not a bug in Week 9; it is Week 8's documented `rice -> jute`
> confusion pair, now visible at the point of use. The two crops differ mainly
> in rainfall (training means ≈237 mm for rice, ≈176 mm for jute), and 200 mm
> sits between them, closer to jute. A runner-up at 27% is exactly the case
> Week 8 said should be routed to a human rather than answered flatly.

A clearly separated input behaves as you would expect:

```bash
python -c "
from src.pipelines.predict_pipeline import predict_proba
print(predict_proba({'N':20,'P':130,'K':200,'temperature':22,'humidity':92,'ph':5.9,'rainfall':110}, top_k=3))
"
```

```
{'apple': 1.0, 'grapes': 3.0217067108534656e-60, 'banana': 0.0}
```

---

## Step 6 — Training on demand from a clean state

Delete the artifact and predict anyway:

```bash
rm -f models/crop_model.joblib
time python -c "from src.pipelines.predict_pipeline import predict; print(predict({'N':90,'P':42,'K':43,'temperature':25,'humidity':80,'ph':6.5,'rainfall':200}))"
```

```
jute

real	0m2.534s
```

No crash: `load_pipeline()` found no file, ran the training pipeline, saved it,
and then predicted. The ~2.5 s is the training run; a second call loads the
saved file instead.

The behaviour is opt-out, and the refusal names its own fix:

```bash
python -c "
from src.pipelines.predict_pipeline import load_pipeline
load_pipeline('models/absent.joblib', train_if_missing=False)
"
```

```
FileNotFoundError: No trained model at models/absent.joblib. Model artifacts are not committed; run `python -m src.pipelines.training_pipeline` to build one, or call load_pipeline(train_if_missing=True).
```

---

## Step 7 — Reproducibility

Two consecutive training runs, compared byte for byte:

```bash
python -m src.pipelines.training_pipeline > /dev/null && md5sum models/crop_model.joblib
python -m src.pipelines.training_pipeline > /dev/null && md5sum models/crop_model.joblib
```

```
4f6e0b3a68748c20bec2d24f72bd3895  models/crop_model.joblib
4f6e0b3a68748c20bec2d24f72bd3895  models/crop_model.joblib
```

Identical. Two properties produce that: the split is seeded
(`RANDOM_STATE = 42`), and Gaussian naive Bayes has no random component at all —
given the same rows it computes the same means and variances.

**This does not make the result reproducible on its own.** The file was
identical because the *data*, the *seed*, the *code* and the *environment* were
all identical. Change any one of them and the guarantee is gone — which is why
`data/raw/` is committed, the seed lives in `src/config.py`, the code is in git,
and `requirements.txt` pins every version. `joblib` contributes none of those
four; it only writes down the result.

---

## Step 8 — Input validation

```bash
python -c "
from src.pipelines.predict_pipeline import predict
try:
    predict({'N':90,'P':42,'K':43,'temperature':25,'humidity':80,'ph':6.5})
except ValueError as e: print('ValueError:', e)
try:
    predict({'N':90,'P':42,'K':43,'temperature':25,'humidity':80,'ph':'6.5','rainfall':200})
except TypeError as e: print('TypeError:', e)
"
```

```
ValueError: Missing feature(s): ['rainfall']. Required: ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'].
TypeError: Feature 'ph' must be a number, got '6.5'.
```

Week 10 will hand this function whatever a stranger sent over HTTP, so the check
belongs here rather than in the API.

---

## Step 9 — The week's tests

```bash
pytest tests/test_training_pipeline.py tests/test_predict_pipeline.py
```

```
32 passed, 909 warnings in 1.37s
```

15 for the training pipeline and 17 for the prediction pipeline. Both write into
pytest's temporary directories, so the suite never touches
`models/crop_model.joblib` and passes on a clone with no artifact present. The
training tests fit on a 220-row stratified sample (10 rows per crop) to stay
fast; the prediction tests train once per module on the full 2,200 rows, because
"the answer is one of the 22 crops" is only a meaningful assertion for a model
that has seen every crop's real spread.

The warnings are the pre-existing NumPy deprecation notices from `shap`'s
import, unrelated to this week.

---

## Step 10 — The whole suite and the linter

```bash
ruff check .
pytest
```

```
All checks passed!
377 passed, 1 skipped, 1018 warnings in 27.55s
```

345 passed + 1 skipped at the end of Week 8, plus this week's 32. The single
skip is Week 8's SHAP-dependent test, which skips or runs depending on whether
the optional package is installed.

---

## Checklist

| Claim | Command | Evidence |
| --- | --- | --- |
| Training runs from the shell and saves a model | `python -m src.pipelines.training_pipeline` | Step 2 — 6,329-byte `models/crop_model.joblib` |
| Modelling decisions unchanged from Week 8 | same | accuracy 0.9955, macro F1 0.9954, 1,760/440 |
| The artifact is not version-controlled | `git status --short` | Step 3 — empty output, `.gitignore:20` |
| The pipeline holds preprocessing *and* model | Step 4 | `ColumnTransformer` + `GaussianNB` in one object |
| Predicting returns a valid crop label | Step 5 | `jute`, with `rice` second at 0.2747 |
| A missing artifact trains rather than crashes | Step 6 | prediction after `rm`, in 2.5 s |
| The seed makes the run deterministic | Step 7 | identical md5 across two runs |
| Bad input is rejected before the model | Step 8 | named `ValueError` / `TypeError` |
| The week is tested | Step 9 | 32 passed |
| Nothing earlier broke | Step 10 | 377 passed, 1 skipped; ruff clean |

## What this does **not** show

* Any of it working **over a network**. Every command above needs a local Python
  interpreter with this repository importable. There is no server, no port and
  no JSON contract yet — that is Week 10.
* A model **version history**. One filename, overwritten on every run, with no
  record beside it of which data or which library versions produced it.
* That the artifact loads under a **different** scikit-learn. It was written and
  read by 1.6.1 here; that is the pinned version, and the pin is the reason the
  question does not arise.
