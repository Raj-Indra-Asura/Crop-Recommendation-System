# ML Concepts

A running index of every machine learning concept taught in this course, in the
order it is introduced. Each entry gives a one-line definition and points at
the week that teaches it properly.

This file is updated **every week**, as concepts are introduced — not
consolidated at the end.

---

## Week 1 — Framing the problem and meeting the data

Taught in [`docs/curriculum/week01/learning_notes.md`](curriculum/week01/learning_notes.md).

| Concept | One-line definition | Section |
| --- | --- | --- |
| **Machine learning** | Building software by supplying examples of correct behaviour and letting an algorithm infer the rules, instead of writing the rules by hand. | §1 |
| **When to use ML** | Use it when rules are too numerous, too interacting or too unknown to write down, *and* labelled examples exist; otherwise prefer ordinary code. | §1 |
| **Supervised learning** | Learning from examples where the correct answer is attached to each input. | §2 |
| **Unsupervised learning** | Learning structure from data with no correct answers supplied. | §2 |
| **Classification** | Supervised learning where the prediction is a category. | §2 |
| **Regression** | Supervised learning where the prediction is a number on a continuous scale. | §2 |
| **Multiclass classification** | Classification with more than two possible categories — 22 crops, in this project. | §2 |
| **Categorical / unordered target** | A target whose values have no meaningful ordering, so treating their codes as numbers would be wrong. | §2 |
| **Batch learning** | Training once on a full dataset to produce a fixed artifact, retrained on a schedule (as opposed to online learning, which updates continuously). | §2 |
| **Feature** | A single input variable used to make a prediction. | §3 |
| **Target / label** | The value the model is trained to predict. | §3 |
| **Instance** | One complete example — a row of feature values together with its target. | §3 |
| **Dataset** | The full collection of instances. | §3 |
| **Problem statement** | The written framing of a task: inputs, output, problem type, success measure and explicit non-goals. | §3 |
| **Raw vs. processed data** | Raw data is treated as read-only source of truth; anything derived is written elsewhere, so the input is always recoverable. | §4 |
| **Artificial intelligence** | The broad goal of software performing tasks we would call intelligent; machine learning is one family of techniques within it. | §1 |
| **Deep learning** | Machine learning with many-layered neural networks; unnecessary for a small tabular problem like this one. | §1 |
| **Training set** | The portion of the dataset a model is fitted on. | §5 |
| **Test set** | The portion held back and looked at once, to estimate performance on unseen data. | §5 |
| **Generalisation** | Performing well on instances never seen during training — the actual goal of learning. | §5 |
| **Overfitting** | Learning the training examples and their noise rather than the underlying pattern; good on train, poor on test. | §5 |
| **Underfitting** | Being too simple to capture the pattern; poor on both train and test. | §5 |
| **Training** | The offline process of fitting a model on labelled data. | §5 |
| **Inference** | The online process of predicting a label for one unlabelled instance. | §5 |
| **ML lifecycle** | The loop from framing, through data, modelling and evaluation, to deployment and monitoring — and back. | §6 |
| **Expected label set** | The exact set of 22 crop names recorded in Week 1, which every later week must match against. | §8 |
| **Reproducibility** | The property that the same inputs and code produce the same results for everyone — the reason data and dependency versions are pinned. | §4, §7 |
| **Virtual environment** | An isolated per-project package directory, so projects cannot break each other's dependencies. | §7 |
| **Pinned dependency** | A dependency fixed to an exact version, so environments are identical over time and across machines. | §7 |
| **Dataframe** | A table of rows and named, typed columns — pandas' central data structure. | §7, §8 |
| **Dataset contract** | The set of properties input data must satisfy (columns, dtypes, size, label set, no nulls) to be considered valid. | §8 |
| **Fail-fast validation** | Raising an error the moment invalid input is read, rather than allowing a plausible-looking wrong answer downstream. | §8 |
| **Linting** | Automated checking of code for style and correctness problems, making a written standard machine-enforceable. | §9 |
| **Automated testing** | Executable checks that a system still behaves as specified, run on every change. | §9 |

---

## Week 2 onwards

Not yet written.
