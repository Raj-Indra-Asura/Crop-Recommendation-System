"""The first three real classifiers this project trains (Week 5).

Week 4 fixed the floor: a :class:`~sklearn.dummy.DummyClassifier` scores 4.55%
(1/22) under 5-fold stratified cross-validation. This module supplies the first
estimators that are allowed to look at ``N``, ``P``, ``K``, temperature,
humidity, pH and rainfall, so the floor finally has something standing on top of
it.

Three algorithms, chosen because they fail in different ways
------------------------------------------------------------
**Logistic regression** — a *linear* model. It learns one weight per feature per
class, adds them up, and turns the 22 resulting scores into probabilities with
the softmax function; the class with the largest score wins. The boundary
between any two crops is therefore a flat surface in seven-dimensional space. It
is fast, its weights are readable (Week 7 cashes that in), and it is helpless
whenever the true boundary is curved in a way no straight line can follow.

**k-nearest neighbours** — a *distance-based* model. ``fit`` stores the training
rows and learns nothing; ``predict`` finds the ``k`` stored rows closest to the
query and takes a vote. The boundary it draws is whatever shape the data
happens to have, which is its strength and the reason it cannot explain itself.
Because it works entirely in distances, it is the model in this project that is
most sensitive to feature scaling — hence the Week 3 preprocessor in front of it,
without which ``K`` (spanning ~200 units) would drown out ``ph`` (spanning ~6).

**Gaussian naive Bayes** — a *probabilistic* model. It applies Bayes' rule with
one deliberately wrong assumption: that within a class the features are
independent of one another. It therefore stores just a mean and a variance per
feature per class — 22 x 7 x 2 numbers here — which makes it the cheapest of the
three to fit by a wide margin, and a strong baseline-above-the-baseline even
when the assumption is violated, as it is here (Week 2 measured a 0.74
correlation between ``P`` and ``K``).

The pattern all three share
---------------------------
Every factory returns an **unfitted** estimator with the same scikit-learn API:

.. code-block:: python

    model.fit(X_train, y_train)      # learn from the training rows
    predictions = model.predict(X_test)   # answer for rows it has not seen

That is *the* training loop, and it does not change again for the rest of the
course — Weeks 6, 7 and 8 reuse it for tuned models, ensembles and the final
evaluation. Returning the estimators unfitted is what lets them be dropped
into a :class:`~sklearn.pipeline.Pipeline` behind
:func:`src.preprocessing.preprocessor.build_preprocessor` and re-fitted inside
every cross-validation fold, which is the only way to scale features without
leaking a validation fold's statistics into training.

Hyperparameters are given sensible, explicit defaults here and are *not* tuned:
searching for better ones is Week 6's subject, and doing it by hand now would
mean choosing settings by peeking at scores.
"""

from __future__ import annotations

from collections.abc import Callable

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from src.data.split import DEFAULT_RANDOM_STATE

#: Default inverse regularisation strength for logistic regression. Smaller
#: values shrink the learned weights harder; 1.0 is scikit-learn's own default
#: and is kept until Week 6 tunes it.
DEFAULT_LOGISTIC_C: float = 1.0

#: Default iteration budget for logistic regression's solver. scikit-learn ships
#: 100, which is not enough for 22 classes here and produces a
#: ``ConvergenceWarning``; 1,000 converges comfortably on scaled features.
DEFAULT_LOGISTIC_MAX_ITER: int = 1_000

#: Default number of neighbours consulted by KNN. Odd values avoid ties in the
#: two-class case; with 22 classes ties are broken by class order regardless.
DEFAULT_K_NEIGHBORS: int = 5

#: Weighting scheme KNN uses by default: every one of the ``k`` neighbours gets
#: an equal vote. ``"distance"`` weights each vote by 1/distance instead.
DEFAULT_KNN_WEIGHTS: str = "uniform"

#: Voting schemes :func:`get_knn` accepts.
KNN_WEIGHT_OPTIONS: tuple[str, ...] = ("uniform", "distance")

#: Variance floor added to every feature variance by Gaussian naive Bayes, so a
#: feature that is constant within a class cannot produce a division by zero.
DEFAULT_VAR_SMOOTHING: float = 1e-9


def get_logistic_regression(
    C: float = DEFAULT_LOGISTIC_C,
    max_iter: int = DEFAULT_LOGISTIC_MAX_ITER,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> LogisticRegression:
    """Build an unfitted multiclass logistic regression classifier.

    The model scores each of the 22 crops with a weighted sum of the seven
    features, ``w . x + b``, and converts the 22 scores into probabilities with
    the **softmax** function, so they are positive and sum to 1. Training picks
    the weights that make the training labels most probable. Because each score
    is linear in the features, the surface separating any two crops is flat — a
    line in two dimensions, a plane in three, a hyperplane in seven.

    scikit-learn 1.6 fits this as a single softmax (multinomial) model rather
    than 22 separate one-vs-rest fits, because the default solver (``lbfgs``)
    supports it. One-vs-rest — train one "is it rice or not?" classifier per
    crop and take the most confident — is the alternative formulation; it is
    available via ``OneVsRestClassifier`` and is not needed here.

    ``C`` is the **inverse** regularisation strength: large ``C`` means weak
    regularisation and larger weights, small ``C`` means the weights are pulled
    towards zero. It is the hyperparameter Week 6 searches over.

    The features must be standardised before they reach this model — not for
    correctness but for the solver, which converges far faster when the columns
    share a scale. Put :func:`src.preprocessing.preprocessor.build_preprocessor`
    in front of it in a :class:`~sklearn.pipeline.Pipeline`.

    Args:
        C: Inverse regularisation strength; must be strictly positive. Defaults
            to :data:`DEFAULT_LOGISTIC_C`.
        max_iter: Maximum solver iterations. Defaults to
            :data:`DEFAULT_LOGISTIC_MAX_ITER` (1,000), which avoids the
            ``ConvergenceWarning`` scikit-learn's default of 100 raises here.
        random_state: Seed, so a fit is reproducible. Defaults to the
            project-wide :data:`src.data.split.DEFAULT_RANDOM_STATE`.

    Returns:
        An unfitted :class:`~sklearn.linear_model.LogisticRegression`.

    Raises:
        ValueError: If ``C`` is not positive, or ``max_iter`` is less than 1.
    """
    if C <= 0:
        raise ValueError(f"`C` must be strictly positive, got {C}.")
    if max_iter < 1:
        raise ValueError(f"`max_iter` must be at least 1, got {max_iter}.")
    return LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)


def get_knn(
    n_neighbors: int = DEFAULT_K_NEIGHBORS,
    weights: str = DEFAULT_KNN_WEIGHTS,
) -> KNeighborsClassifier:
    """Build an unfitted k-nearest-neighbours classifier.

    KNN does no work at fit time beyond memorising the training rows — it is the
    standard example of a *lazy* learner, and its cost is paid at prediction
    time instead, when every query row is compared against all of them. To
    classify a field it finds the ``n_neighbors`` training fields closest to it
    in feature space (Euclidean distance by default) and returns the label most
    common among them.

    ``n_neighbors`` controls how smooth the resulting decision boundary is:

    * ``k = 1`` follows the training data exactly, so training accuracy is a
      perfect and completely uninformative 100% while a single mislabelled row
      claims its whole neighbourhood — the overfitting end;
    * large ``k`` averages over an ever wider region until the model approaches
      "predict the most common class overall" — the underfitting end;
    * the useful values lie in between, and finding them is Week 6's job.

    Two properties matter more here than the exact value of ``k``. First,
    distances are meaningless across mismatched units, so the Week 3 scaler must
    precede this model. Second, in high-dimensional spaces distances concentrate
    — every point ends up roughly equidistant from every other, so "nearest"
    stops meaning "similar". That is the **curse of dimensionality**, and with
    only seven features this project is comfortably clear of it; it is the
    reason KNN is a poor choice on, say, text data with thousands of columns.

    Args:
        n_neighbors: How many neighbours vote; must be at least 1. Defaults to
            :data:`DEFAULT_K_NEIGHBORS` (5).
        weights: ``"uniform"`` (every neighbour votes equally) or ``"distance"``
            (closer neighbours count for more). Defaults to
            :data:`DEFAULT_KNN_WEIGHTS`.

    Returns:
        An unfitted :class:`~sklearn.neighbors.KNeighborsClassifier`.

    Raises:
        ValueError: If ``n_neighbors`` is less than 1, or ``weights`` is not one
            of :data:`KNN_WEIGHT_OPTIONS`.
    """
    if n_neighbors < 1:
        raise ValueError(f"`n_neighbors` must be at least 1, got {n_neighbors}.")
    if weights not in KNN_WEIGHT_OPTIONS:
        raise ValueError(
            f"Unsupported KNN weighting {weights!r}. "
            f"Choose one of: {', '.join(KNN_WEIGHT_OPTIONS)}."
        )
    return KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)


def get_naive_bayes(var_smoothing: float = DEFAULT_VAR_SMOOTHING) -> GaussianNB:
    """Build an unfitted Gaussian naive Bayes classifier.

    Bayes' rule turns "how likely is this data under each crop?" into "how
    likely is each crop given this data": ``P(crop | x)`` is proportional to
    ``P(x | crop) * P(crop)``. The hard part is ``P(x | crop)``, the joint
    distribution of seven features at once. **Naive** Bayes assumes the features
    are independent *given the class*, so that joint probability factorises into
    a product of seven one-dimensional ones, and **Gaussian** naive Bayes models
    each of those as a normal distribution. Fitting therefore reduces to
    computing a mean and a variance per feature per class, in a single pass over
    the data.

    The independence assumption is false here — Week 2 measured a correlation of
    0.74 between ``P`` and ``K`` — and the model still works well, which is the
    interesting part. Classification only needs the *ranking* of the 22 class
    scores to be right, not their calibration; correlated features make the
    probabilities badly overconfident while usually leaving the winner
    unchanged. Treat the probabilities it reports with suspicion, but not its
    predictions.

    Being cheap, assumption-driven and free of hyperparameters worth tuning,
    it is the natural "second baseline": a model that beats the dummy but that a
    well-chosen algorithm should beat in turn.

    Unlike the other two, this model is unaffected by standardising the features
    — a linear rescaling shifts each class mean and variance identically — so the
    Week 3 preprocessor is kept in front of it only to keep the comparison
    between models about the models.

    Args:
        var_smoothing: Fraction of the largest feature variance added to every
            variance for numerical stability; must not be negative. Defaults to
            :data:`DEFAULT_VAR_SMOOTHING` (1e-9).

    Returns:
        An unfitted :class:`~sklearn.naive_bayes.GaussianNB`.

    Raises:
        ValueError: If ``var_smoothing`` is negative.
    """
    if var_smoothing < 0:
        raise ValueError(f"`var_smoothing` must not be negative, got {var_smoothing}.")
    return GaussianNB(var_smoothing=var_smoothing)


#: The Week 5 models, in the order the notebook reports them, mapped to their
#: zero-argument factories. Later weeks add entries rather than rewriting the
#: comparison loop that consumes this mapping.
CLASSICAL_MODEL_FACTORIES: dict[str, Callable[[], BaseEstimator]] = {
    "logistic_regression": get_logistic_regression,
    "knn": get_knn,
    "naive_bayes": get_naive_bayes,
}
