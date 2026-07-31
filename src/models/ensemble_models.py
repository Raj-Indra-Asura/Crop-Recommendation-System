"""Ensembles: many weak models voted or corrected into one strong model (Week 7).

Week 6 ended with a decision tree that scored a perfect 100% on the rows it was
fitted on and 98.52% on rows it had not seen. That gap is **variance**: a single
unlimited tree memorises the accidents of one particular sample of 1,760 fields,
and a different sample would produce a visibly different tree. Week 7 does not
fix that by making the tree simpler. It fixes it by fitting *many* such trees
and combining their answers.

Two ways to combine, and they are opposites
-------------------------------------------
**Bagging** — *bootstrap aggregating* — fits its models **in parallel**, each on
a different random resample of the training rows, and averages them. Every
member sees a slightly different world, so each one memorises different noise;
averaging cancels the disagreement while keeping the signal they agree on. The
members are deliberately allowed to overfit, because the averaging is what
removes the variance. Bagging lowers **variance** and leaves **bias** roughly
where the individual model put it. :func:`get_random_forest` is bagging plus one
extra trick.

**Boosting** fits its models **sequentially**, each one trained to correct the
mistakes the accumulated ensemble is still making. The members are deliberately
*weak* — shallow trees, "stumps" in the extreme — and the ensemble becomes strong
by adding hundreds of small corrections. Boosting lowers **bias** primarily, and
because each round chases the current residual error it *can* overfit if left to
run too long, which is what ``learning_rate`` and ``n_estimators`` control
together. :func:`get_gradient_boosting` is boosting.

The one-line version: **bagging averages independent opinions to cancel their
noise; boosting builds a chain of specialists, each hired to fix what the
previous ones got wrong.**

Random forest = bagging + feature randomness
--------------------------------------------
Averaging only cancels error that the members do not share. Trees fitted on
bootstrap resamples of the same data are still highly correlated: if one feature
is strongly predictive, nearly every tree splits on it first and they all make
the same kind of mistake. A random forest therefore adds a second source of
randomness — at **every node**, each tree may only consider a random subset of
the features (``max_features``, ``"sqrt"`` by default: 2 or 3 of this project's
7). That forces the trees apart, decorrelates their errors, and is what makes
the average worth more than its members.

Gradient boosting = sequential error-correction
------------------------------------------------
Fit a small tree. Look at what the ensemble still gets wrong. Fit the next small
tree on *that* — the direction that would most reduce the remaining loss, which
is where the word *gradient* comes from — and add a shrunken version of it to
the running total. Repeat ``n_estimators`` times. ``learning_rate`` is the
shrinkage: small steps mean each tree matters less and more of them are needed,
which is slower but usually generalises better. The full derivation is not
needed to use it, and is not given here.

``feature_importances_``
------------------------
Both models expose a ``feature_importances_`` array, one non-negative number per
feature, summing to 1. For tree ensembles it is **mean decrease in impurity**:
every time a feature is used for a split, the drop in Gini impurity it achieved
is weighted by how many rows passed through that node, and the totals are summed
over all nodes of all trees and normalised. It is cheap — the numbers fall out of
fitting — and it is the first thing in this project that answers "which
measurements matter?".

It also has three limitations worth knowing before quoting it:

* it is computed **on the training data**, so a feature the model overfitted on
  scores highly regardless of whether it helps on new rows;
* it is **biased towards high-cardinality features** — continuous columns offer
  many candidate thresholds, so they win splits more often than they deserve;
* **correlated features split the credit** arbitrarily. Week 2 measured a 0.74
  correlation between ``P`` and ``K``; whichever of the two a tree happens to
  split on first absorbs the importance, and the other looks less useful than it
  is. Dropping the "unimportant" one can then cost accuracy.

It says which features the *model* used, never which features *cause* the
outcome, and never why one individual field was classified as it was. Week 8
adds permutation importance (shuffle a column, measure how much held-out
accuracy falls) and SHAP (attribute a single prediction across features), which
address the first and third limitations respectively.

XGBoost, and the fallback
-------------------------
:func:`get_gradient_boosting` prefers `XGBoost <https://xgboost.readthedocs.io>`_
when it is importable and falls back to scikit-learn's
:class:`~sklearn.ensemble.GradientBoostingClassifier` when it is not. Both
implement the same idea; XGBoost is faster and adds regularisation terms, while
the scikit-learn version needs no extra install. The fallback exists so that
this week's material can be worked through in any environment — see
:data:`GRADIENT_BOOSTING_BACKEND` for which one is active, and Week 7's
``learning_notes.md`` for the discussion.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from src.data.split import DEFAULT_RANDOM_STATE

try:  # pragma: no cover - which branch runs depends on the environment
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE: bool = True
except ImportError:  # pragma: no cover - see above
    XGBClassifier = None  # type: ignore[assignment,misc]
    XGBOOST_AVAILABLE = False

#: Default number of trees in a random forest. More trees never *hurt* accuracy
#: — averaging more members cannot increase variance — they only cost time, so
#: this is one of the few "hyperparameters" that is really a budget. 100 is
#: scikit-learn's own default and is ample for 1,760 rows.
DEFAULT_FOREST_N_ESTIMATORS: int = 100

#: Default depth limit for the forest's trees: ``None``, i.e. unlimited. This is
#: deliberate and is the opposite of the advice for a *single* tree. Bagging
#: removes variance by averaging, so its members are supposed to be low-bias and
#: high-variance; pruning them would trade away accuracy the averaging was going
#: to recover for free.
DEFAULT_FOREST_MAX_DEPTH: int | None = None

#: Default number of features considered at each split. ``"sqrt"`` gives
#: ``round(sqrt(7)) = 3`` of this project's seven columns — the setting that
#: decorrelates the trees and turns bagged trees into a random forest.
DEFAULT_FOREST_MAX_FEATURES: str | int | float | None = "sqrt"

#: Values :func:`get_random_forest` accepts for ``max_features`` as a string.
FOREST_MAX_FEATURES_OPTIONS: tuple[str, ...] = ("sqrt", "log2")

#: Default number of boosting rounds, i.e. how many small trees are chained
#: together. Unlike a forest's ``n_estimators`` this one can overfit: every
#: extra round fits the error that remains, including the part of it that is
#: noise.
DEFAULT_BOOSTING_N_ESTIMATORS: int = 100

#: Default shrinkage applied to each boosting round's contribution. Lower values
#: take smaller steps and generalise better but need more rounds; ``learning_rate``
#: and ``n_estimators`` trade off against one another and are tuned together.
DEFAULT_BOOSTING_LEARNING_RATE: float = 0.1

#: Default depth of each boosted tree. Boosting wants **weak** learners: a depth
#: of 3 can express interactions between at most three features, and the
#: strength comes from chaining a hundred of them rather than from any one.
DEFAULT_BOOSTING_MAX_DEPTH: int = 3


class XGBoostStringLabelClassifier(ClassifierMixin, BaseEstimator):
    """An :class:`xgboost.XGBClassifier` that accepts this project's string labels.

    XGBoost's scikit-learn wrapper is *almost* a drop-in estimator, with one
    gap that matters here: it requires the target to be the integers
    ``0..n_classes-1`` and raises ``ValueError`` on anything else. This
    project's target is the crop name (``"rice"``, ``"maize"``, ...), which
    every scikit-learn classifier accepts directly.

    This adapter closes that gap and nothing else: it label-encodes ``y`` on the
    way into ``fit`` and decodes the integers back to crop names on the way out
    of ``predict``, so the model can be dropped into the same
    :class:`~sklearn.pipeline.Pipeline`, the same
    :func:`~src.evaluation.metrics.cross_validated_accuracy` call and the same
    results table as every other model in the course.

    Args:
        n_estimators: Number of boosting rounds.
        learning_rate: Shrinkage applied to each round's contribution.
        max_depth: Depth of each boosted tree.
        random_state: Seed, so a fit is reproducible.
    """

    def __init__(
        self,
        n_estimators: int = DEFAULT_BOOSTING_N_ESTIMATORS,
        learning_rate: float = DEFAULT_BOOSTING_LEARNING_RATE,
        max_depth: int = DEFAULT_BOOSTING_MAX_DEPTH,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        """Record the hyperparameters; nothing is built until ``fit``."""
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> XGBoostStringLabelClassifier:
        """Encode the labels, fit the underlying XGBoost model, and return self.

        Args:
            X: Training features.
            y: Training labels, of any dtype scikit-learn accepts.

        Returns:
            This estimator, fitted.

        Raises:
            ImportError: If XGBoost is not installed.
        """
        if not XGBOOST_AVAILABLE:  # pragma: no cover - guarded by the factory
            raise ImportError(
                "XGBoost is not installed. Use `get_gradient_boosting()`, which falls "
                "back to scikit-learn's GradientBoostingClassifier automatically."
            )
        self.encoder_ = LabelEncoder().fit(y)
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self.model_.fit(X, self.encoder_.transform(y))
        self.classes_ = self.encoder_.classes_
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict crop names for ``X``.

        Args:
            X: Rows to classify.

        Returns:
            An array of labels in the original (string) label space.
        """
        return self.encoder_.inverse_transform(self.model_.predict(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities for ``X``.

        Args:
            X: Rows to classify.

        Returns:
            An ``(n_rows, n_classes)`` array whose columns follow ``classes_``.
        """
        return self.model_.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        """One importance per feature, taken from the fitted XGBoost model."""
        return self.model_.feature_importances_


def get_random_forest(
    n_estimators: int = DEFAULT_FOREST_N_ESTIMATORS,
    max_depth: int | None = DEFAULT_FOREST_MAX_DEPTH,
    max_features: str | int | float | None = DEFAULT_FOREST_MAX_FEATURES,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> RandomForestClassifier:
    """Build an unfitted random forest classifier.

    A random forest is a **bagged** ensemble of decision trees with one addition.

    1. **Bootstrap sampling.** Each tree is fitted on its own resample of the
       training set, drawn *with replacement* and of the same size — so about
       63% of the rows appear in any given tree's sample, some of them more than
       once, and the remaining ~37% are that tree's *out-of-bag* rows.
    2. **Feature randomness.** At every node, a tree may only choose among
       ``max_features`` randomly selected features rather than all seven.
    3. **Voting.** Predictions are the majority vote of the trees (scikit-learn
       averages their probability estimates, which is the smoother version of
       the same thing).

    Steps 1 and 2 exist for one reason: **averaging only cancels errors the
    members do not share**. Bootstrap sampling alone leaves the trees highly
    correlated, because a dominant feature is chosen first by nearly all of
    them; restricting the features available at each node forces different trees
    to build different structures, so their mistakes are made in different
    places and the average is better than any member.

    That is also why the members are grown to full depth by default. A single
    unlimited tree overfits (Week 6 plotted it: 100% training, 98.52%
    validation), and here that is the *point* — low bias, high variance, and the
    variance is removed by averaging 100 of them rather than by pruning any one.

    Args:
        n_estimators: Number of trees; must be at least 1. Defaults to
            :data:`DEFAULT_FOREST_N_ESTIMATORS` (100).
        max_depth: Depth limit per tree, or ``None`` for unlimited. Must be at
            least 1 if given. Defaults to :data:`DEFAULT_FOREST_MAX_DEPTH`.
        max_features: Features considered per split — ``"sqrt"``, ``"log2"``, an
            ``int`` count, a ``float`` share of the columns, or ``None`` for all
            of them (which turns the forest back into plain bagged trees).
            Defaults to :data:`DEFAULT_FOREST_MAX_FEATURES`.
        random_state: Seed for the bootstrap samples and the per-node feature
            draws. Defaults to the project-wide
            :data:`src.data.split.DEFAULT_RANDOM_STATE`.

    Returns:
        An unfitted :class:`~sklearn.ensemble.RandomForestClassifier`.

    Raises:
        ValueError: If ``n_estimators`` is less than 1, ``max_depth`` is less
            than 1, or ``max_features`` is an unknown string or a non-positive
            number.
    """
    if n_estimators < 1:
        raise ValueError(f"`n_estimators` must be at least 1, got {n_estimators}.")
    if max_depth is not None and max_depth < 1:
        raise ValueError(f"`max_depth` must be at least 1 or None, got {max_depth}.")
    if isinstance(max_features, str):
        if max_features not in FOREST_MAX_FEATURES_OPTIONS:
            raise ValueError(
                f"Unsupported `max_features` {max_features!r}. Choose one of: "
                f"{', '.join(FOREST_MAX_FEATURES_OPTIONS)}, an int, a float, or None."
            )
    elif max_features is not None and max_features <= 0:
        raise ValueError(f"`max_features` as a number must be positive, got {max_features}.")
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        random_state=random_state,
    )


def get_gradient_boosting(
    n_estimators: int = DEFAULT_BOOSTING_N_ESTIMATORS,
    learning_rate: float = DEFAULT_BOOSTING_LEARNING_RATE,
    max_depth: int = DEFAULT_BOOSTING_MAX_DEPTH,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> BaseEstimator:
    """Build an unfitted gradient boosting classifier.

    Where a forest fits its trees independently and averages them, boosting
    fits them **one after another**, each on what the ensemble so far still gets
    wrong:

    1. start with a constant prediction;
    2. measure the error that remains — formally, the gradient of the loss with
       respect to the current predictions, which is where "gradient" comes from;
    3. fit a small tree to *that*;
    4. add ``learning_rate`` times the new tree to the running prediction;
    5. go back to 2, ``n_estimators`` times.

    Each member is deliberately weak (``max_depth=3``: it can combine at most
    three features), and the ensemble is strong because a hundred small
    corrections accumulate. This reduces **bias** — the ensemble can express
    what no single member could — which is the mirror image of what bagging
    does, and it is why boosting *can* overfit: rounds 200, 300 and 400 keep
    fitting whatever error is left, including the part that is noise.
    ``learning_rate`` is the brake, and it trades directly against
    ``n_estimators``: halve the rate and you need roughly twice the rounds.

    **XGBoost or scikit-learn?** This factory returns an XGBoost model when
    ``xgboost`` is importable and scikit-learn's
    :class:`~sklearn.ensemble.GradientBoostingClassifier` when it is not, so the
    week works in any environment rather than depending on an optional install.
    The two implement the same algorithm and behave the same way through
    ``fit``/``predict``/``feature_importances_``; XGBoost is considerably faster
    (it grows all 22 classes' trees in parallel over compressed feature bins)
    and adds L1/L2 penalties on the leaf values. The XGBoost path is wrapped in
    :class:`XGBoostStringLabelClassifier` because ``XGBClassifier`` refuses
    non-integer labels and this project's target is the crop name.
    :data:`GRADIENT_BOOSTING_BACKEND` records which implementation is in use, so
    a notebook can report it rather than assume it.

    Args:
        n_estimators: Number of boosting rounds; must be at least 1. Defaults to
            :data:`DEFAULT_BOOSTING_N_ESTIMATORS` (100).
        learning_rate: Shrinkage applied to each round; must be strictly
            positive. Defaults to :data:`DEFAULT_BOOSTING_LEARNING_RATE` (0.1).
        max_depth: Depth of each boosted tree; must be at least 1. Defaults to
            :data:`DEFAULT_BOOSTING_MAX_DEPTH` (3).
        random_state: Seed, so a fit is reproducible. Defaults to the
            project-wide :data:`src.data.split.DEFAULT_RANDOM_STATE`.

    Returns:
        An unfitted classifier: :class:`XGBoostStringLabelClassifier` if XGBoost
        is installed, otherwise
        :class:`~sklearn.ensemble.GradientBoostingClassifier`.

    Raises:
        ValueError: If ``n_estimators`` or ``max_depth`` is less than 1, or
            ``learning_rate`` is not positive.
    """
    if n_estimators < 1:
        raise ValueError(f"`n_estimators` must be at least 1, got {n_estimators}.")
    if learning_rate <= 0:
        raise ValueError(f"`learning_rate` must be strictly positive, got {learning_rate}.")
    if max_depth < 1:
        raise ValueError(f"`max_depth` must be at least 1, got {max_depth}.")
    if XGBOOST_AVAILABLE:
        return XGBoostStringLabelClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )


#: Which gradient boosting implementation :func:`get_gradient_boosting` returns
#: in this environment: ``"xgboost"`` if the package imported, otherwise
#: ``"sklearn"``. Notebooks and tests read this instead of assuming.
GRADIENT_BOOSTING_BACKEND: str = "xgboost" if XGBOOST_AVAILABLE else "sklearn"

#: The Week 7 ensembles, mapped to their zero-argument factories, in the order
#: the notebook reports them. Kept separate from
#: :data:`src.models.classical_models.CLASSICAL_MODEL_FACTORIES` so that "the
#: single models" and "the ensembles" stay distinguishable in the results table.
ENSEMBLE_MODEL_FACTORIES: dict[str, Callable[[], BaseEstimator]] = {
    "random_forest": get_random_forest,
    "gradient_boosting": get_gradient_boosting,
}
