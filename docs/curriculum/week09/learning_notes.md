# Week 9 — Learning Notes

**Productionizing the model: pipelines, artifacts and reproducibility**

Everything in this course so far has produced *findings*. This week produces
*software*. The model does not change — same estimator, same seed, same 0.9955
— but the way it is obtained changes completely: from "open a notebook and run
the cells in order" to "run one command, or import one function".

---

## §1 — Why notebooks are for exploration, not production

Notebooks are the right tool for Weeks 1-8. They interleave code, output and
prose; they keep a dataframe in memory while you look at it from six angles;
they show a plot next to the code that drew it. Nothing here is an argument
against them. The argument is against *shipping* them.

### 1.1 Hidden state

A notebook's variables live in a kernel, not in the file. The file shows

```python
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
```

but what `X_train` actually *is* depends on how many times you ran that cell.
Run it twice and you have standardised the standardised data. The notebook looks
identical either way, and the output cell underneath may have been written by a
third, even earlier run.

The general form: **a notebook's displayed output is not evidence that its code
produces that output.** Only re-running from a fresh kernel proves anything, and
nothing in the file forces you to do that.

### 1.2 Execution order

Cells run in whatever order you click them. The execution counters (`In [7]`,
`In [3]`, `In [12]`) record what happened, and a notebook whose counters are out
of order is a notebook whose result nobody can reproduce — including its author
tomorrow. A deleted cell that defined a still-referenced variable is invisible:
the variable is in the kernel, and the code that made it is gone.

Scripts have exactly one execution order: top to bottom, every time, in a fresh
interpreter.

### 1.3 No reuse

This is the one that actually blocks Week 10. A function in `src/` can be
imported by a test, a script, a web server and another notebook. A cell in
`06_model_selection.ipynb` can be imported by nothing. The only ways to reuse
notebook logic are to copy-paste it — creating two copies that drift — or to run
the whole notebook, plots and all, to get one number out of it.

There is a fourth problem worth naming: notebooks **diff badly**. The `.ipynb`
format is JSON with embedded outputs and base64 images, so a one-character
change can produce a thousand-line diff and two people editing the same notebook
produce conflicts no reviewer can read.

### 1.4 What this project does about it

Nothing new, as it happens — the discipline has been in place since Week 1. All
real logic lives in `src/`, and the notebooks import it:

```python
from src.data import load_data, stratified_split
from src.preprocessing import build_preprocessor
```

Week 9 simply completes the pattern. The last two things that only existed as
notebook cells — "fit the chosen model on the training split" and "predict for
one new row" — become `src/pipelines/training_pipeline.py` and
`src/pipelines/predict_pipeline.py`. After this week, deleting every notebook in
the repository would break no test and no command.

---

## §2 — One `Pipeline`, not two objects

Week 3 built a `ColumnTransformer` and made the argument for it. Week 9 acts on
the argument.

```python
Pipeline([
    ("preprocess", build_preprocessor(FEATURE_COLUMNS)),   # Week 3
    ("model",      get_naive_bayes(var_smoothing=1e-9)),   # Week 8
])
```

That is the shape of what `build_training_pipeline()` returns in
`src/pipelines/training_pipeline.py`. The real function looks the model up by
name (`FINAL_MODEL_NAME`) instead of calling `get_naive_bayes` directly, so
changing the model is a config edit rather than a code edit — §5 — but the
object it builds is exactly the two steps above.

A `Pipeline` is itself an estimator. Calling `fit(X, y)` fits the preprocessor
on `X`, transforms `X`, and fits the model on the result. Calling `predict(X)`
applies the **already-fitted** preprocessor and then the model. Two steps, one
object, one `fit`, one `predict`.

Why that matters, in three separate places:

**Cross-validation.** `cross_val_score(pipeline, X, y, cv=5)` re-fits the scaler
inside every fold, using only that fold's training rows. Scale first and
cross-validate afterwards and every fold's validation rows contributed to the
mean and standard deviation the model was trained under — the leakage of Week 2,
in the form that is easiest to commit by accident.

**Serialization.** One object is one file. A scaler saved beside its model is a
pair that can be reunited in the wrong order, or with one half rebuilt and the
other stale, and nothing will raise an error — the numbers will just be wrong.

**Serving.** Next week a request arrives as raw measurements: `ph = 6.5`,
`rainfall = 200`. The pipeline applies exactly the transformation the training
rows received, because it *is* the object that received them. The failure this
prevents has a name — **training/serving skew** — and it is the most common
serious bug in deployed ML systems: the training code standardised, the serving
code forgot, and the model quietly returns nonsense with high confidence.

Naming the steps (`"preprocess"`, `"model"`) rather than relying on their
position is what makes `pipeline.named_steps["model"]` readable, and what let
Week 8's searches address `model__var_smoothing` — the `step__parameter`
convention. The same trick works one level down: the preprocess step is a
`ColumnTransformer`, and the scaler inside it is
`pipeline.named_steps["preprocess"].named_transformers_["numeric"]`, because
Week 3 named that branch `"numeric"` too.

---

## §3 — Serialization: what `joblib` saves, and what it does not

**Serialization** is turning a live Python object into a stream of bytes that a
later process can turn back into an equivalent object. Python's built-in format
for this is called **pickle**, and the two directions have names: *pickling* is
writing, *unpickling* is reading. `joblib` is pickle with a faster path for
large NumPy arrays, which is why scikit-learn's own documentation recommends it
for fitted estimators. Nothing here is specific to machine learning — a fitted
pipeline is just a Python object with a lot of numbers in it.

```python
import joblib

joblib.dump(pipeline, "models/crop_model.joblib")   # write
pipeline = joblib.load("models/crop_model.joblib")  # read
```

Two calls, and that is the whole API this week needs: `dump` takes the object
and a destination, `load` takes a path and gives the object back. `dump` walks
the object graph — the pipeline, the two steps inside it, and every array they
hold — and writes out, for our pipeline:

* the **learned parameters** — `StandardScaler`'s seven means and seven scales
  (`mean_`, `scale_`), `GaussianNB`'s per-class means, variances and priors
  (`theta_`, `var_`, `class_prior_`: 22 classes x 7 features x 2, plus 22
  priors). The trailing underscore is scikit-learn's mark for "learned during
  `fit`", from Week 5;
* the **hyperparameters** — `var_smoothing=1e-9`, the column list, the step
  names;
* **references to the classes** by import path: "this is a
  `sklearn.pipeline.Pipeline` containing a
  `sklearn.preprocessing._data.StandardScaler`".

The whole thing is 6.3 KB. That size is itself informative: the artifact holds
*numbers*, not code and not data.

What it does **not** contain:

| Not in the file | Consequence |
| --- | --- |
| The code of `Pipeline`, `StandardScaler`, `GaussianNB` | Loading imports them from your installed scikit-learn. No scikit-learn, no load. |
| The **versions** of scikit-learn, NumPy, Python | A different version may warn (`InconsistentVersionWarning`), silently change behaviour, or fail to unpickle at all. |
| The training data | The file cannot tell you what it was fitted on, or when. |
| Any metadata | No accuracy, no git commit, no timestamp — unless you write them yourself. |

And one security note that matters the moment a file arrives from elsewhere:
**unpickling executes code**. `joblib.load` on an untrusted file is equivalent to
running an untrusted script. Only load artifacts you or your own pipeline
produced.

---

## §4 — Reproducibility is four things, and `joblib` is one of them

"Here is the model file" is not a reproducible result. Reproducing a result means
being able to *rebuild* it, and that needs four things fixed:

1. **The data.** `data/raw/Crop_recommendation.csv` is committed — 150 KB, 2,200
   rows, byte-identical for everyone. Every downstream number is defined against
   it.
2. **The seed.** `RANDOM_STATE = 42`, used by the split and by any model with a
   random component. One constant, in `src/config.py`, re-exported from Week 3's
   `DEFAULT_RANDOM_STATE` so the two cannot drift apart. Fixed *end to end*: a
   seed on the split and a fresh unseeded shuffle later is no better than no
   seed at all.
3. **The code.** Version-controlled, and the artifact deliberately is not — see
   §6.
4. **The environment.** `requirements.txt`, every package pinned to an exact
   version. An unpinned `scikit-learn>=1.0` is a different program every month.

"Works on my machine" is the failure of item 4, and it is not a joke about
sloppiness — it is a genuine problem. The same code, the same seed and the same
data can produce different numbers under a different NumPy version, a different
Python, or even a different **BLAS** build — BLAS being the low-level linear
algebra library NumPy delegates its arithmetic to, and which may sum a column of
floats in a different order on a different machine. Pinning does not make that
impossible; it
makes it *diagnosable*, because the environment is written down and can be
compared.

On this project the training run is fully deterministic — two runs write
**byte-identical** files (verified in
[`validation.md`](validation.md) with `md5sum`) — because Gaussian naive Bayes
has no random component and the split is seeded. That is a property of this
model, not a law: a random forest would still be deterministic given its own
`random_state`, but only because it is passed one.

---

## §5 — Config over hardcoding

Up to Week 8 the project's constants lived where they were used: the raw path in
`src/data/data_loader.py`, the seed in `src/data/split.py`, the chosen
`var_smoothing` in a notebook cell. That was fine while a human drove everything.
It stops being fine when a script, a test suite and (next week) a web server all
have to agree.

`src/config.py` is the agreement:

```python
MODEL_PATH = PROJECT_ROOT / "models" / "crop_model.joblib"
RANDOM_STATE = DEFAULT_RANDOM_STATE          # 42, re-exported, not re-typed
FINAL_MODEL_NAME = "naive_bayes"             # the Week 8 decision, as data
FINAL_MODEL_PARAMS = {"var_smoothing": 1e-9}
```

Three properties are deliberate:

* **Inert.** Importing config reads no file, fits nothing, has no side effect. A
  test or a server can import it for free, and it can never fail halfway.
* **Derived, not duplicated.** `RANDOM_STATE` is imported from `src.data.split`
  rather than re-typed. A constant written down twice is a constant that will
  eventually disagree with itself.
* **Overridable by argument.** Every function that uses a config value also takes
  it as a keyword argument, which is why the tests can train into `tmp_path`
  instead of the repository's `models/`. Config is the *default*, not a global
  that behaviour is locked to.

What "hardcoding" actually costs: `FINAL_MODEL_PARAMS` recorded in one place
means swapping the model is a one-line edit that the training script, both
tests and next week's API all pick up. The same value copy-pasted into four
files means a change is four edits, three of which someone will forget, and the
resulting disagreement produces no error — only a wrong answer.

---

## §6 — The artifact is not committed

`.gitignore` contains `models/*.joblib`. That is a decision worth defending,
because it looks at first like the opposite of reproducibility.

**Why the data is committed but the model is not.** The CSV is *source*: nothing
in the repository can regenerate it, it is small, and it never changes. The
`.joblib` is *derived*: it is a deterministic function of the data plus the code
plus the config, all three of which are version-controlled. Committing a
binary that changes on every retrain would bloat the history (git stores every
version forever), produce diffs nobody can read, and create a second source of
truth that can silently disagree with the code.

**The consequence, and the design it forces.** In a fresh clone there is no
model file. If `predict()` simply raised `FileNotFoundError`, then the tests,
next week's API and CI — *continuous integration*, the automated test run a
service like GitHub Actions performs on every push — would all fail on a clean
checkout until somebody remembered to run the trainer. So `load_pipeline()`
trains one on demand:

```python
if not path.is_file():
    train_pipeline(model_path=path)   # ~2.3 s, then saved
return joblib.load(path)
```

First call pays a couple of seconds; every later call loads the file. The escape
hatch is explicit — `load_pipeline(path, train_if_missing=False)` raises a
`FileNotFoundError` whose message says exactly what to run — because in a real
deployment "the model is missing, so train one on the web server" is precisely
the wrong behaviour, and the code should let you say so.

---

## §7 — The two entry points

### 7.1 Training

```bash
python -m src.pipelines.training_pipeline
```

`python -m` runs a module *as a script* with the repository root on `sys.path`,
which is why `import src.config` works without installing the project. The
script's shape is the whole course in six calls (a sketch of `main()`, not
literal code):

```
load_data()                  # Week 1 — read and validate
stratified_split()           # Week 3 — 1,760 / 440, seeded
build_training_pipeline()    # Week 3 preprocessing + Week 8 model
pipeline.fit(X_train, y_train)
evaluate_model(pipeline, X_test, y_test)   # Week 4/8 metrics
save_pipeline(pipeline, MODEL_PATH)        # Week 9
```

It prints what it did — the model and its hyperparameters, the split sizes,
accuracy, macro and weighted F1, and the path written — because a training run
that reports nothing cannot be trusted by whoever finds the file later.

Two flags exist for the cases where the defaults are wrong:

```bash
python -m src.pipelines.training_pipeline --model-path /tmp/scratch.joblib
python -m src.pipelines.training_pipeline --model-name random_forest
```

`--model-path` writes somewhere other than `MODEL_PATH`; `--model-name` picks a
different model from the registry the script merges out of Weeks 5-7
(`naive_bayes`, `decision_tree`, `random_forest`, …). Passing a name that is not
in the registry fails immediately, listing the ones that are. Note that
`--model-name` also drops `FINAL_MODEL_PARAMS`, because `var_smoothing` means
nothing to a random forest; the flag is for experiments, and `src.config` is
still where a *decision* is recorded.

`train_pipeline()` is the importable version of the same thing, and it takes the
data as an argument. That is what lets `tests/test_training_pipeline.py` train
on 220 sampled rows in about a second while the script trains on all 2,200. It
returns a dictionary rather than printing: `"pipeline"` (the fitted object),
`"metrics"` (the Week 4 evaluation dictionary, so
`result["metrics"]["accuracy"]` is the held-out score), `"model_path"`,
`"n_train"`, `"n_test"`, `"model_name"` and `"model_params"`. `main()` is only a
formatter for those seven keys, which is why a caller never has to parse
printed output.

### 7.2 Prediction

```python
from src.pipelines.predict_pipeline import predict

predict({"N": 90, "P": 42, "K": 43, "temperature": 25,
         "humidity": 80, "ph": 6.5, "rainfall": 200})   # -> 'jute'
```

Three things happen before the model sees anything:

1. **Validation.** Missing feature, unexpected feature, or a non-numeric value
   is rejected with a message that names it. A web API will hand this function
   whatever a stranger typed, and Week 10 inherits the check rather than
   reinventing it.
2. **Ordering.** The dictionary is rebuilt as a one-row dataframe in
   `FEATURE_COLUMNS` order. A `ColumnTransformer` fitted on named columns needs
   the names it was fitted with; a caller should not have to know the order.
3. **Loading.** The artifact is loaded, or trained first if absent (§6). A caller
   making many predictions passes `pipeline=` to load once.

That seven-key dictionary is exported from the module as `EXAMPLE_INPUT`, so
the demo, the docs and the tests all quote one copy of it rather than four:
`predict(EXAMPLE_INPUT)` is the same call as the one above.

`predict_proba()` returns all 22 probabilities, sorted, or the best `top_k` of
them if you ask for a number. For the input above:
`jute 0.7253`, `rice 0.2747`, everything else effectively zero. Week 8's
`rice -> jute` confusion pair, met in the confusion matrix, is now something you
can see at the point of use — and it is exactly the material for the
"route anything whose runner-up exceeds 15% to a human" rule that week proposed.

Note what the label answers: this is the crop the *model* considers most likely
given seven numbers, not agronomic advice. Those particular measurements are
genuinely ambiguous between rice and jute, and the honest output is the pair of
probabilities rather than the single word.

---

## §8 — What the tests are actually checking

Two files, 32 tests — 15 for training and 17 for prediction — and neither of
them tries to prove the model is good — that
was Week 8's job.

`tests/test_training_pipeline.py` checks that the *script* behaves like a
script: the pipeline has the two named steps and comes back unfitted, training
on a small sample writes a non-empty file, that file reloads into something that
predicts valid labels, the reported split sizes add up, two seeded runs agree
exactly, an unknown model name raises before anything is written, and `main()`
exits `0` after writing where it was told to.

`tests/test_predict_pipeline.py` is the round trip — train, save, reload,
predict, and assert the answer is one of the 22 known crops — plus the
train-on-demand behaviour in both directions, the input validation cases, the
fact that key order in the request does not matter, and the agreement between
`predict` and the top of `predict_proba`.

Every test writes into `tmp_path`. A test suite that overwrites
`models/crop_model.joblib` would be a test suite with a side effect on the
developer's working copy, and side effects are what this whole week is about
removing. The two files pay different prices for that: the training tests fit on
a 220-row stratified sample (10 rows per crop) passed in as `frame=`, while the
prediction tests need a *realistic* model to make "is the answer one of the 22
crops" meaningful, so their fixture trains once per module on the full 2,200
rows — into `tmp_path_factory`, still never into `models/`.

---

## §9 — What is still missing

The honest list, in the order the industry usually hits it:

* **No network.** Everything here needs a Python interpreter with this
  repository importable. Week 10.
* **No versioning.** One filename, overwritten in place. Nothing records which
  data, which commit or which library versions produced the file sitting in
  `models/` right now. A production system writes that metadata beside the
  artifact, or uses a **model registry** — a service that stores every trained
  model under a version number together with its metrics, its inputs and its
  stage (staging, production, archived), so a deployment can name a version
  rather than a file path, and roll back to the previous one.
* **No monitoring.** Nothing notices if next season's soil measurements drift
  away from the 2,200 rows this model was fitted on. The model will keep
  answering confidently regardless.
* **No retraining trigger.** A human runs the trainer.
* **Python only.** A pickle is a Python object. Serving it from another language
  means exporting to a neutral format (ONNX, PMML) — out of scope here.

---

## Summary

* Notebooks hide state, run in arbitrary order and cannot be imported; that is
  fine for exploration and disqualifying for production.
* A `Pipeline` bundles the Week 3 preprocessing with the Week 8 model into one
  estimator — one `fit`, one `predict`, one file, and no training/serving skew.
* `joblib.dump` saves learned numbers and class *references*; it does not save
  code, versions or data, and unpickling untrusted files runs code.
* Reproducibility = committed data + fixed seed + versioned code + pinned
  environment. The artifact is a convenience, not a record.
* `src/config.py` holds paths, the seed and the chosen hyperparameters in one
  inert place; every function still accepts them as arguments.
* The artifact is git-ignored because it is derived; `load_pipeline()` therefore
  trains on demand so a clean clone works.
* `python -m src.pipelines.training_pipeline` produces the model;
  `predict({...})` consumes it. Week 10 puts HTTP in front of the second one.
