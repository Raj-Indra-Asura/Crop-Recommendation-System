"""The naive baseline every later model has to beat (Week 4).

A **baseline** is a model that is deliberately not intelligent. It ignores the
features entirely and guesses from the distribution of the labels alone, which
makes its score the price of admission: any "real" model that fails to beat it
has learned nothing from ``N``, ``P``, ``K``, temperature, humidity, pH or
rainfall, whatever its accuracy happens to look like in isolation.

Without such a number, an accuracy is uninterpretable. "97%" sounds excellent
until you learn that 96% of the rows carry one label and a constant guess scores
96 too. On this project the arithmetic runs the other way — 22 classes with 100
rows each mean a constant guess scores about 1/22 ≈ 4.5% — but the reasoning is
identical, and it is the reasoning, not the number, that transfers to the next
dataset.

:class:`~sklearn.dummy.DummyClassifier` implements this idea with the ordinary
scikit-learn API, so a baseline can be dropped into exactly the same
``fit``/``predict``/``cross_val_score`` machinery as a random forest. The
strategies this project uses:

``"most_frequent"``
    Always predict the class that appeared most often in the training data. On
    a balanced dataset ties are broken by class order, so it collapses to "always
    predict one particular crop".

``"prior"``
    Predicts the same labels as ``"most_frequent"``; the difference is in
    ``predict_proba``, which returns the observed class distribution rather than
    a hard 1 for the winning class.

``"stratified"``
    Draw a random guess *in proportion to* the training class frequencies. It is
    the "monkey with a weighted die" baseline, and unlike ``"most_frequent"`` its
    score varies from run to run, which is why it takes a ``random_state``.

``"uniform"``
    Draw a random guess with every class equally likely — indistinguishable from
    ``"stratified"`` on a perfectly balanced dataset, and very different on an
    imbalanced one.

``"constant"`` is deliberately **not** supported here: it needs an extra
``constant=`` argument naming the class to predict, and a factory whose only
argument is a strategy name cannot supply it. Instantiate
:class:`~sklearn.dummy.DummyClassifier` directly if you ever need it.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier

from src.data.split import DEFAULT_RANDOM_STATE

#: The strategies :func:`get_baseline_model` accepts, in the order the notebook
#: reports them. Frozen as a tuple so a caller cannot append to it by accident.
BASELINE_STRATEGIES: tuple[str, ...] = ("most_frequent", "stratified", "uniform", "prior")

#: The strategy quoted whenever this project says "the baseline". It is the
#: strictest sensible choice on balanced data — a stratified guess can beat it
#: by luck, so requiring a real model to beat *this* number is the weaker,
#: fairer demand.
DEFAULT_BASELINE_STRATEGY: str = "most_frequent"


def get_baseline_model(
    strategy: str = DEFAULT_BASELINE_STRATEGY,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> DummyClassifier:
    """Build an unfitted baseline classifier.

    The returned estimator ignores its features completely: ``fit`` looks only at
    ``y``, and ``predict`` answers from the label distribution it recorded there.
    That is the point — its score measures what "knowing nothing about this
    field" is worth, so a real model's improvement over it is the part
    attributable to the features.

    The object comes back **unfitted**, matching
    :func:`src.preprocessing.preprocessor.build_preprocessor`, so that the
    caller decides what it is fitted on and cross-validation can re-fit it
    inside every fold.

    Args:
        strategy: One of :data:`BASELINE_STRATEGIES`. Defaults to
            :data:`DEFAULT_BASELINE_STRATEGY` (``"most_frequent"``).
        random_state: Seed for the strategies that guess randomly
            (``"stratified"`` and ``"uniform"``). Defaults to the project-wide
            :data:`src.data.split.DEFAULT_RANDOM_STATE` so that a reported
            baseline can be reproduced exactly. It is ignored by
            ``"most_frequent"`` and ``"prior"``, which are deterministic.

    Returns:
        An unfitted :class:`~sklearn.dummy.DummyClassifier`.

    Raises:
        ValueError: If ``strategy`` is not one of :data:`BASELINE_STRATEGIES`.
    """
    if strategy not in BASELINE_STRATEGIES:
        raise ValueError(
            f"Unsupported baseline strategy {strategy!r}. "
            f"Choose one of: {', '.join(BASELINE_STRATEGIES)}."
        )
    return DummyClassifier(strategy=strategy, random_state=random_state)
