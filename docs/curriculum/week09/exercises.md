# Week 9 — Exercises

Work through these after reading [`learning_notes.md`](learning_notes.md) and
running the two commands in [`validation.md`](validation.md). Nothing here needs
a new dependency, and nothing here needs a notebook — that is the point of the
week.

Most exercises start from the same imports:

```python
import joblib
import pandas as pd

from src.config import FINAL_MODEL_NAME, FINAL_MODEL_PARAMS, MODEL_PATH, RANDOM_STATE
from src.data import FEATURE_COLUMNS, TARGET_COLUMN, load_data
from src.pipelines.predict_pipeline import EXAMPLE_INPUT, load_pipeline, predict, predict_proba
from src.pipelines.training_pipeline import build_training_pipeline, train_pipeline
```

Write any throwaway files into a temporary directory, not into `models/`.

---

## Exercise 1 — Break a notebook on purpose

**Goal:** feel hidden state rather than read about it.

Open `notebooks/06_model_selection.ipynb` — the cell near the top that does
`train = pd.read_csv(...)`. Add a cell **below** it containing
`train = train.head(len(train) // 2)`, run that cell **twice**, then add one
more cell containing `print(train.shape)` and run it.

1. What does `train.shape` report, and what would it have reported if you had
   restarted the kernel and run every cell once, top to bottom? Why does
   *halving* show the problem in a way that `train.head(100)` would not?
2. Nothing about the file records that you ran the cell twice. Which piece of
   evidence *would* have recorded it, and why is it not saved in the `.ipynb`?
3. Write the equivalent script — load, take the head, print the shape — and say
   what would have to happen for it to produce a different answer on two runs.
4. Undo your edits: delete the two cells you added and restart the kernel.

---

## Exercise 2 — Take the pipeline apart

**Goal:** see that a fitted pipeline is just the two Week 3 / Week 8 objects.

```python
pipeline = load_pipeline()
print(pipeline.named_steps)
scaler = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
model = pipeline.named_steps["model"]
```

1. Print `scaler.mean_` and `scaler.scale_`. Which entry corresponds to
   `rainfall`, and how do you know without counting by eye?
2. Compare `scaler.mean_` to `load_data()[list(FEATURE_COLUMNS)].mean()`. They
   are close but **not** equal. Explain the difference in one sentence.
3. `model.theta_` has shape `(22, 7)`. What is `model.theta_[i, j]`, and what is
   `model.class_prior_`? Why is every prior exactly `1/22` here?
4. The scaler's means are in the *original* units and the model's `theta_` are
   not. Explain why, in terms of the order the two steps run.

---

## Exercise 3 — What is in the artifact, and what is not

**Goal:** state the limits of serialization precisely.

1. Report the file size of `models/crop_model.joblib`. The training data is
   ~150 KB; the artifact is a fraction of that. What does the difference tell
   you about what was saved?
2. Count the numbers the model actually stores: `theta_`, `var_` and
   `class_prior_` for 22 classes and 7 features, plus the scaler's `mean_` and
   `scale_`. Does your count explain the file size, at 8 bytes per float?
3. Without running it, predict what
   `python -c "import joblib; joblib.load('models/crop_model.joblib')"` does in
   an environment with **no scikit-learn installed**. Then explain the failure
   in terms of what `dump` wrote.
4. Someone emails you a `.joblib` file and asks you to load it. Give the
   security reason to refuse.

---

## Exercise 4 — Reproducibility, four ways to break it

**Goal:** separate the four requirements and test three of them.

1. Run the training script twice and compare `md5sum models/crop_model.joblib`.
   Explain why the two files are identical, naming the two properties of this
   model and split that make it so.
2. Call `train_pipeline(model_path=None, random_state=7)` and compare
   `result["metrics"]["accuracy"]` to the seeded run's. Which of the four
   reproducibility requirements did you just violate, and is the difference you
   see evidence that either model is better?
3. Train a random forest into a scratch path twice —
   `python -m src.pipelines.training_pipeline --model-name random_forest
   --model-path /tmp/rf.joblib` — and repeat step 1's `md5sum` on `/tmp/rf.joblib`.
   Is the artifact still byte-identical across two runs? What in
   `src/models/ensemble_models.py` decides the answer?
4. `requirements.txt` pins `scikit-learn==1.6.1`. Describe — do not attempt —
   what could happen if a colleague loads your artifact under 1.9, and name the
   warning scikit-learn issues for exactly this case.

---

## Exercise 5 — Config over hardcoding

**Goal:** prove the single-source-of-truth claim by changing one line.

1. Point at every place the number `42` is *defined* in `src/`. There should be
   one. Where is it re-exported, and why is re-exporting better than re-typing?
2. Change `FINAL_MODEL_NAME` in `src/config.py` to `"decision_tree"` and set
   `FINAL_MODEL_PARAMS` to `{}` (a tree has no `var_smoothing`). Re-run the
   training script. How many files did you have to edit for the trainer, both
   tests and `predict()` to agree?
3. Now run `pytest tests/test_training_pipeline.py tests/test_predict_pipeline.py`
   with that config in place. Exactly two tests fail. Read their names and their
   assertions: are they failing because they assert on `FINAL_MODEL_NAME` /
   `FINAL_MODEL_PARAMS`, or because they assert something the config does *not*
   control? Should a test that pins the project's *decision* fail when the
   decision changes? Argue both sides in three sentences, then put the config
   back.
4. `src/config.py` imports nothing that touches the filesystem. Name two things
   that would go wrong if it loaded the CSV at import time.

---

## Exercise 6 — Train on demand

**Goal:** understand the design forced by not committing the artifact.

1. Delete `models/crop_model.joblib` and call `predict(EXAMPLE_INPUT)`. Time it.
   Call it again and time it again. Explain the two numbers.
2. Now call `load_pipeline("models/crop_model.joblib", train_if_missing=False)`
   with the file deleted. Quote the error message. Which of the two behaviours
   belongs in a web server, and why?
3. `data/raw/` is committed and `models/` is not. Write the rule that decides
   this in one sentence, general enough to apply to a file you have not seen.
4. Suppose the artifact *were* committed. Describe a scenario where the file in
   git and the code in git disagree, and say who would notice.

---

## Exercise 7 — The ambiguous example

**Goal:** connect Week 8's error analysis to the thing users actually call.

```python
ranked = predict_proba(EXAMPLE_INPUT)
```

1. `EXAMPLE_INPUT` is the row the README uses to illustrate the problem, with
   `rice` as the intended answer. What does the model return, and with what
   runner-up probability?
2. Look up the two crops' mean `rainfall` in the training data. Which single
   feature is doing the work, and does 200 mm sit closer to rice or to jute?
3. Week 8 proposed routing any prediction whose runner-up exceeds 15% to a
   human. Would this input be routed? Write the three lines that implement that
   rule on top of `predict_proba`.
4. Raise `rainfall` until the label flips. Report the threshold to the nearest
   10 mm, and say what that number is *not* evidence of (hint: one row, one
   model).

---

## Exercise 8 — Extend the pipeline

**Goal:** add a step without touching anything else.

1. Insert a `SimpleImputer(strategy="median")` before the scaler, inside the
   `ColumnTransformer`'s numeric branch in `src/preprocessing/preprocessor.py`.
   (`SimpleImputer` is scikit-learn's transformer for missing values: it learns
   one fill value per column during `fit` — here the median — and substitutes it
   for every `NaN` during `transform`. It is new this week, and the branch has
   to become a small `Pipeline` of two steps to hold both it and the scaler.)
   Re-run the tests. Which tests had to change, and which did not?
2. The dataset has no missing values, so the imputer changes no number. Argue
   for and against shipping a step that currently does nothing.
3. Now imagine the API in Week 10 receives a request with `ph` absent. With the
   imputer in the pipeline, what happens? Compare with what `predict()`'s
   validation does today, and say which behaviour you want and why.
4. Revert your change.

---

## Exercise 9 — Write the metadata the artifact lacks

**Goal:** turn §9's "no versioning" from a complaint into code.

1. Extend `train_pipeline()` to write a `models/crop_model.json` beside the
   artifact, recording: the model name and hyperparameters, the seed, the test
   accuracy, the row count, the current `sklearn.__version__` and
   `platform.python_version()`.
2. Which of those fields would let a future reader detect the failure described
   in Exercise 4.4 *before* trusting a prediction?
3. Should this JSON be committed? Apply your rule from Exercise 6.3 and defend
   the answer.
4. What is still missing that a real model registry would give you? Name two
   things — §9 of the notes describes what a registry stores.

---

## Exercise 10 — Explain it back

**Goal:** the week's oral exam. No code.

Answer each in three sentences or fewer.

1. A colleague says "I've sent you the `.joblib`, so you can reproduce my
   results." What is wrong with that sentence?
2. Why is the preprocessing inside the pipeline rather than applied before it?
3. Why does `predict()` train a model when the file is missing, when a normal
   program would raise?
4. What can you do at the end of Week 9 that you could not at the end of Week 8,
   and what can you still not do?
