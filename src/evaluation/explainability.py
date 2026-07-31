"""Explainability: why *this* field was told to grow *that* crop.

Week 7's ``feature_importances_`` answers a question about the fitted model as a
whole ("which columns did the trees split on most profitably?"), on the training
data, and with the credit for correlated columns split arbitrarily between them.
Week 8 replaces it with two things it cannot do:

* :func:`permutation_feature_importance` — shuffle one column of **held-out**
  data, re-score, and report how much accuracy the model loses. It is measured
  on rows the model never trained on, it works for any estimator (naive Bayes
  has no ``feature_importances_`` at all), and it measures reliance on a feature
  rather than the mechanics of a split.
* :func:`explain_prediction` — attribute a **single** prediction across the
  seven measurements that produced it, so a recommendation can be defended to
  the person who has to act on it.

**Two backends, and the module says which one it used.** When ``shap`` imports,
:func:`explain_prediction` uses it: SHAP attributes a prediction across features
with a game-theoretic guarantee (the contributions plus a base value reconstruct
the model's output). When ``shap`` is missing — the same risk XGBoost carries in
Week 7 — the fallback is fixed and deliberately unimaginative:

1. **Per-sample permutation.** Take the row, replace one feature at a time with
   values drawn from a background sample, and measure how far the predicted
   class's probability moves. A feature whose real value the prediction depends
   on shows a large drop; an irrelevant one shows none.
2. **The raw ``predict_proba`` breakdown** across all classes for that row,
   which says how confident the model was and what its runners-up were.

:data:`EXPLAINER_BACKEND` records which one is active, and every result carries
a ``"method"`` key, so no explanation in this project can be quoted without
knowing how it was produced.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance

from src.data.split import DEFAULT_RANDOM_STATE

try:  # pragma: no cover - exercised by whichever environment runs the tests
    import shap

    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover - the documented fallback path
    shap = None
    SHAP_AVAILABLE = False

#: Which single-prediction explainer this environment will use: ``"shap"`` when
#: the package imported, ``"permutation"`` for the documented fallback.
EXPLAINER_BACKEND: str = "shap" if SHAP_AVAILABLE else "permutation"

#: Repeats used by :func:`permutation_feature_importance`. Each repeat is a
#: fresh shuffle of one column, so the spread over repeats says how much of the
#: reported drop is noise.
DEFAULT_N_REPEATS: int = 10

#: Background rows drawn per feature by the fallback explainer. Thirty draws is
#: enough for the ranking to be stable on this dataset and cheap enough to run
#: inside a notebook cell.
DEFAULT_BACKGROUND_SAMPLES: int = 30


def permutation_feature_importance(
    model: BaseEstimator,
    X: Any,
    y: Any,
    n_repeats: int = DEFAULT_N_REPEATS,
    scoring: str = "accuracy",
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Rank features by the score a **fitted** model loses when each is shuffled.

    The procedure is one idea repeated: score the model on ``X``; shuffle one
    column so its values no longer line up with their rows (the column keeps its
    distribution and loses its meaning); score again; the difference is that
    feature's importance. Repeat ``n_repeats`` times per feature, because a
    single shuffle is one random draw.

    This is more trustworthy than ``feature_importances_`` for three reasons.
    It is computed on whatever data you pass — give it the held-out set and the
    number describes generalisation rather than memorisation. It applies to
    *any* fitted estimator, including ones with no notion of a split. And it
    measures the model's dependence on the information in a column, not the
    bookkeeping of how a tree happened to spend its splits.

    Its one well-known trap survives here: when two columns are correlated
    (``P`` and ``K`` correlate at 0.74 in this dataset), shuffling either alone
    leaves the model able to recover most of the signal from the other, so both
    can look unimportant even though the pair matters a great deal.

    Args:
        model: A fitted estimator exposing ``predict``.
        X: Features to permute — ideally rows the model was **not** fitted on.
        y: True labels for ``X``.
        n_repeats: Shuffles per feature. Defaults to :data:`DEFAULT_N_REPEATS`.
        scoring: Any scikit-learn scorer name. Defaults to ``"accuracy"``.
        random_state: Seed for the shuffles, so the table is reproducible.

    Returns:
        A :class:`~pandas.DataFrame` indexed by feature name and sorted most
        important first, with columns ``"importance_mean"`` (the average score
        lost) and ``"importance_std"`` (its spread across repeats).

    Raises:
        ValueError: If ``X`` and ``y`` have different lengths, or ``n_repeats``
            is less than 1.
    """
    if len(X) != len(y):
        raise ValueError(f"`X` has {len(X)} rows but `y` has {len(y)} labels; they must match.")
    if n_repeats < 1:
        raise ValueError(f"`n_repeats` must be at least 1, got {n_repeats}.")

    outcome = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        scoring=scoring,
        random_state=random_state,
    )
    if hasattr(X, "columns"):
        names = list(X.columns)
    else:
        names = [f"feature {i}" for i in range(len(outcome.importances_mean))]
    return pd.DataFrame(
        {
            "importance_mean": outcome.importances_mean,
            "importance_std": outcome.importances_std,
        },
        index=pd.Index(names, name="feature"),
    ).sort_values("importance_mean", ascending=False)


def _as_single_row(X_row: Any) -> pd.DataFrame:
    """Coerce one example into a one-row dataframe, whatever shape it arrived in."""
    if isinstance(X_row, pd.DataFrame):
        if len(X_row) != 1:
            raise ValueError(f"`X_row` must hold exactly one row, got {len(X_row)}.")
        return X_row
    if isinstance(X_row, pd.Series):
        return X_row.to_frame().T
    array = np.asarray(X_row)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[0] != 1:
        raise ValueError(f"`X_row` must hold exactly one row, got {array.shape[0]}.")
    return pd.DataFrame(array)


def _probability_breakdown(model: BaseEstimator, row: pd.DataFrame) -> pd.Series:
    """Return the model's class probabilities for one row, largest first."""
    probabilities = np.asarray(model.predict_proba(row)).ravel()
    return pd.Series(probabilities, index=list(model.classes_)).sort_values(ascending=False)


def _shap_contributions(
    model: BaseEstimator,
    row: pd.DataFrame,
    predicted_class: Any,
    background: pd.DataFrame | None,
) -> tuple[pd.Series, float]:
    """Per-feature SHAP values for ``predicted_class``, plus the base value.

    Tries the fast tree explainer first and falls back to the model-agnostic
    kernel explainer, which needs a background sample to define "average".
    """
    class_index = int(np.where(np.asarray(model.classes_) == predicted_class)[0][0])
    try:
        explainer = shap.TreeExplainer(model)
    except Exception as error:  # noqa: BLE001 - any explainer failure means "try the other one"
        if background is None:
            raise TypeError(
                "SHAP could not build a tree explainer for this model and no "
                "`background` was supplied for the model-agnostic one."
            ) from error
        columns = list(row.columns)

        def predict_proba_frame(values: np.ndarray) -> np.ndarray:
            """Keep column names attached, which a fitted pipeline requires."""
            return model.predict_proba(pd.DataFrame(np.asarray(values), columns=columns))

        explainer = shap.KernelExplainer(
            predict_proba_frame,
            shap.kmeans(background, min(len(background), 25)),
        )

    values = explainer.shap_values(row)
    if isinstance(values, list):                      # one array per class
        contributions = np.asarray(values[class_index]).ravel()
    else:
        values = np.asarray(values)
        if values.ndim == 3:
            contributions = values[0, :, class_index]
        else:
            contributions = values.ravel()

    expected = explainer.expected_value
    expected = np.asarray(expected).ravel()
    base_value = float(expected[class_index]) if expected.size > 1 else float(expected[0])
    return pd.Series(contributions, index=list(row.columns)), base_value


def _permutation_contributions(
    model: BaseEstimator,
    row: pd.DataFrame,
    predicted_class: Any,
    background: pd.DataFrame,
    n_samples: int,
    random_state: int,
) -> pd.Series:
    """The documented fallback: perturb one feature of this row at a time.

    For each feature, the row's real value is replaced by ``n_samples`` values
    drawn from the background column and the predicted class's probability is
    re-read. The contribution reported is ``p(real) - mean(p(perturbed))``:
    positive when the measured value supports the prediction, negative when it
    argues against it.
    """
    rng = np.random.default_rng(random_state)
    class_index = int(np.where(np.asarray(model.classes_) == predicted_class)[0][0])
    actual_probability = float(np.asarray(model.predict_proba(row)).ravel()[class_index])

    draws = min(n_samples, len(background))
    contributions = {}
    for feature in row.columns:
        replacements = rng.choice(np.asarray(background[feature]), size=draws, replace=True)
        perturbed = pd.concat([row] * draws, ignore_index=True)
        perturbed[feature] = replacements
        mean_probability = float(
            np.asarray(model.predict_proba(perturbed))[:, class_index].mean()
        )
        contributions[feature] = actual_probability - mean_probability
    return pd.Series(contributions)


def explain_prediction(
    model: BaseEstimator,
    X_row: Any,
    background: Any = None,
    method: str = "auto",
    n_samples: int = DEFAULT_BACKGROUND_SAMPLES,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Explain one prediction: what was predicted, how sure, and on what evidence.

    Three pieces come back together, because none of them is an explanation on
    its own:

    * **the prediction** and the model's probability for it;
    * **the runners-up** — the full ``predict_proba`` breakdown, which turns
      "rice" into "rice at 0.98, with jute a distant second at 0.01" and shows
      whether the model was confident or merely breaking a tie;
    * **per-feature contributions** — how much each of the seven measurements
      pushed the model towards or away from the predicted crop.

    The contributions come from SHAP when it is installed, and from the
    fixed fallback (per-sample permutation, see
    :func:`_permutation_contributions`) when it is not. They are *not* the same
    quantity: SHAP values are additive on the model's output scale and sum with
    the base value to reproduce it, while the fallback's numbers are drops in
    predicted probability. Both rank features for one row; only SHAP adds up.
    The ``"method"`` key in the result always says which was used.

    Args:
        model: A **fitted** classifier exposing ``predict_proba`` and ``classes_``.
        X_row: The single example to explain — a one-row
            :class:`~pandas.DataFrame`, a :class:`~pandas.Series`, or a 1-D
            array of the model's features in order.
        background: Reference rows describing "a typical field", normally the
            training features. Required by the fallback (there is nothing to
            perturb towards without it) and by SHAP's model-agnostic explainer;
            unused by SHAP's tree explainer.
        method: ``"auto"`` (SHAP if available, else the fallback), ``"shap"`` to
            insist on SHAP, or ``"permutation"`` to insist on the fallback.
        n_samples: Background draws per feature in the fallback. Defaults to
            :data:`DEFAULT_BACKGROUND_SAMPLES`.
        random_state: Seed for those draws.

    Returns:
        A dictionary with ``"prediction"``, ``"probability"``,
        ``"probabilities"`` (a :class:`~pandas.Series` over all classes, largest
        first), ``"contributions"`` (a Series over features, largest absolute
        value first), ``"method"`` (``"shap"`` or ``"permutation"``),
        ``"base_value"`` (SHAP only, otherwise ``None``) and ``"top_feature"``.

    Raises:
        ValueError: If ``X_row`` does not hold exactly one row, if ``method`` is
            unknown, or if the fallback is requested without ``background``.
        AttributeError: If ``model`` has no ``predict_proba``.
    """
    if method not in ("auto", "shap", "permutation"):
        raise ValueError(f"`method` must be 'auto', 'shap' or 'permutation', got {method!r}.")
    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            f"{type(model).__name__} has no `predict_proba`; this helper explains "
            "probability outputs, so fit a classifier that provides them."
        )

    row = _as_single_row(X_row)
    if background is not None and not isinstance(background, pd.DataFrame):
        background = pd.DataFrame(np.asarray(background), columns=list(row.columns))
    if (
        isinstance(background, pd.DataFrame)
        and isinstance(row.columns, pd.RangeIndex)
        and background.shape[1] == row.shape[1]
    ):
        # ``X_row`` arrived as a bare array, so borrow the background's names.
        row = row.set_axis(list(background.columns), axis="columns")

    probabilities = _probability_breakdown(model, row)
    predicted_class = probabilities.index[0]

    chosen = EXPLAINER_BACKEND if method == "auto" else method
    if chosen == "shap" and not SHAP_AVAILABLE:
        raise ValueError("`method='shap'` was requested but the `shap` package is not installed.")

    base_value: float | None = None
    if chosen == "shap":
        try:
            contributions, base_value = _shap_contributions(model, row, predicted_class, background)
        except Exception:  # noqa: BLE001 - a failed explainer must not lose the explanation
            if background is None:
                raise
            chosen = "permutation"

    if chosen == "permutation":
        if background is None:
            raise ValueError(
                "The permutation fallback needs `background` rows (normally the training "
                "features) to perturb this row's values towards."
            )
        contributions = _permutation_contributions(
            model, row, predicted_class, background, n_samples, random_state
        )
        base_value = None

    contributions = contributions.reindex(
        contributions.abs().sort_values(ascending=False).index
    )

    return {
        "prediction": predicted_class,
        "probability": float(probabilities.iloc[0]),
        "probabilities": probabilities,
        "contributions": contributions,
        "top_feature": str(contributions.index[0]),
        "method": chosen,
        "base_value": base_value,
    }
