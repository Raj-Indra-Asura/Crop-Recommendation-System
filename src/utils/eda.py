"""Plotting and summarising helpers for exploratory data analysis (Week 2).

Every function here is deliberately small, pure and free of side effects that a
test cannot observe:

* nothing calls ``plt.show()`` — the caller decides when (or whether) to render;
* nothing writes a file;
* nothing mutates the dataframe it is given.

That is what lets ``tests/test_eda.py`` run all of them head-lessly on a tiny
synthetic frame, and what lets ``notebooks/02_EDA.ipynb`` import the same code
the test suite exercises instead of copy-pasting plotting code into cells.

The functions take the column names as arguments rather than hard-coding
``FEATURE_COLUMNS``, so they stay usable on any tabular dataset — including the
small synthetic frames used in tests.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

#: Default number of histogram bins. 30 is a compromise: fine enough to expose
#: a second peak in a 2,200-row column, coarse enough not to show noise.
DEFAULT_BINS: int = 30


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    """Check that every requested column exists, and return them as a list.

    Args:
        frame: The dataframe to check against.
        columns: Column names the caller intends to use.

    Returns:
        The requested column names as a list, in the order given.

    Raises:
        KeyError: If any requested column is absent from ``frame``.
    """
    requested = list(columns)
    missing = [name for name in requested if name not in frame.columns]
    if missing:
        raise KeyError(
            f"Column(s) not found in dataframe: {missing}. "
            f"Available: {list(frame.columns)}"
        )
    return requested


def describe_features(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Summarise numeric columns with count, mean, spread, quartiles and skew.

    This is ``DataFrame.describe()`` transposed — one row per feature, which
    reads far better than one column per feature when there are several — plus
    two extra columns:

    * ``median``, stated explicitly next to ``mean`` so the two can be compared
      at a glance. A large gap between them signals a skewed distribution.
    * ``skew``, Fisher-Pearson skewness. Roughly: 0 is symmetric, positive
      means a long right tail, negative a long left tail.

    Args:
        frame: The dataframe to summarise.
        columns: The numeric columns to include, in the order to report them.

    Returns:
        A new dataframe indexed by feature name, with the statistics as
        columns. The input frame is not modified.

    Raises:
        KeyError: If any requested column is absent from ``frame``.
    """
    requested = _require_columns(frame, columns)
    summary = frame[requested].describe().T
    summary["median"] = frame[requested].median()
    summary["skew"] = frame[requested].skew()
    return summary


def class_balance(frame: pd.DataFrame, target: str = "label") -> pd.DataFrame:
    """Count the rows belonging to each class, with proportions.

    Args:
        frame: The dataframe containing the target column.
        target: Name of the target column. Defaults to ``"label"``.

    Returns:
        A dataframe indexed by class value with two columns, ``count`` and
        ``proportion``, sorted from the most to the least frequent class.

    Raises:
        KeyError: If ``target`` is absent from ``frame``.
    """
    _require_columns(frame, [target])
    counts = frame[target].value_counts()
    return pd.DataFrame(
        {"count": counts, "proportion": frame[target].value_counts(normalize=True)}
    )


def plot_class_balance(
    frame: pd.DataFrame,
    target: str = "label",
    ax: Axes | None = None,
) -> Axes:
    """Draw a horizontal bar chart of how many rows each class has.

    A perfectly balanced dataset produces bars of identical length. Any raggedness
    is worth noticing now, because it changes which evaluation metric is
    trustworthy later (Week 8).

    Args:
        frame: The dataframe containing the target column.
        target: Name of the target column. Defaults to ``"label"``.
        ax: Existing axes to draw on. A new figure is created when omitted.

    Returns:
        The axes the chart was drawn on.

    Raises:
        KeyError: If ``target`` is absent from ``frame``.
    """
    counts = class_balance(frame, target)["count"]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(3.0, 0.3 * len(counts) + 1)))

    order = list(counts.index)
    sns.barplot(x=counts.to_numpy(), y=order, ax=ax, hue=order, legend=False)
    ax.set_xlabel("rows")
    ax.set_ylabel(target)
    ax.set_title(f"Class balance: rows per {target}")
    return ax


def plot_feature_histograms(
    frame: pd.DataFrame,
    columns: Sequence[str],
    bins: int = DEFAULT_BINS,
    n_cols: int = 3,
) -> Figure:
    """Draw one histogram per feature on a shared grid of subplots.

    A histogram answers "what values does this feature actually take, and how
    often?" — the single fastest way to spot a skew, a hard floor or ceiling, or
    a second hidden peak (which usually means two sub-populations are mixed
    together).

    Args:
        frame: The dataframe to plot from.
        columns: The numeric columns to plot, one subplot each.
        bins: Number of histogram bins. Defaults to :data:`DEFAULT_BINS`.
        n_cols: Number of subplots per row of the grid. Defaults to 3.

    Returns:
        The figure holding the grid. Any unused subplot slots are removed.

    Raises:
        KeyError: If any requested column is absent from ``frame``.
        ValueError: If ``columns`` is empty, or ``bins`` or ``n_cols`` is
            less than 1.
    """
    requested = _require_columns(frame, columns)
    if not requested:
        raise ValueError("`columns` must name at least one column to plot.")
    if bins < 1:
        raise ValueError(f"`bins` must be at least 1, got {bins}.")
    if n_cols < 1:
        raise ValueError(f"`n_cols` must be at least 1, got {n_cols}.")

    n_rows = -(-len(requested) // n_cols)  # ceiling division
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 3.2 * n_rows), squeeze=False
    )
    flat_axes = list(axes.ravel())

    for ax, name in zip(flat_axes, requested, strict=False):
        sns.histplot(data=frame, x=name, bins=bins, ax=ax, kde=False)
        ax.set_title(name)
        ax.set_xlabel("")

    for unused in flat_axes[len(requested) :]:
        fig.delaxes(unused)

    fig.suptitle("Distribution of each feature")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(
    frame: pd.DataFrame,
    columns: Sequence[str],
    method: str = "pearson",
    ax: Axes | None = None,
) -> Axes:
    """Draw an annotated heatmap of the pairwise correlations between features.

    Correlation here measures *linear* association only: +1 means the two
    features move together proportionally, -1 means they move in exact
    opposition, 0 means no linear relationship (which is not the same as "no
    relationship").

    Args:
        frame: The dataframe to compute correlations from.
        columns: The numeric columns to correlate.
        method: Correlation method passed to :meth:`pandas.DataFrame.corr` —
            ``"pearson"`` (linear, the default), ``"spearman"`` (rank-based) or
            ``"kendall"``.
        ax: Existing axes to draw on. A new figure is created when omitted.

    Returns:
        The axes the heatmap was drawn on.

    Raises:
        KeyError: If any requested column is absent from ``frame``.
    """
    requested = _require_columns(frame, columns)
    matrix = frame[requested].corr(method=method)

    if ax is None:
        _, ax = plt.subplots(figsize=(1.1 * len(requested) + 2, 1.0 * len(requested) + 1))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"{method.capitalize()} correlation between features")
    return ax


def plot_boxplot_by_label(
    frame: pd.DataFrame,
    feature: str,
    target: str = "label",
    ax: Axes | None = None,
) -> Axes:
    """Draw one box per class showing how ``feature`` is distributed within it.

    A box spans the interquartile range (25th to 75th percentile) with the
    median as the line inside it; the whiskers reach 1.5 IQR beyond the box, and
    anything past them is drawn as an individual point — a candidate outlier,
    not a proven error.

    Read side by side, the boxes answer the question that matters for
    classification: does this feature *separate* the classes, or do all the
    boxes sit on top of one another?

    Args:
        frame: The dataframe to plot from.
        feature: Name of the numeric column to summarise.
        target: Name of the grouping (class) column. Defaults to ``"label"``.
        ax: Existing axes to draw on. A new figure is created when omitted.

    Returns:
        The axes the boxplot was drawn on.

    Raises:
        KeyError: If ``feature`` or ``target`` is absent from ``frame``.
    """
    _require_columns(frame, [feature, target])
    n_classes = frame[target].nunique()

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6.0, 0.45 * n_classes + 2), 4.5))

    sns.boxplot(data=frame, x=target, y=feature, ax=ax, hue=target, legend=False)
    ax.set_title(f"{feature} by {target}")
    ax.tick_params(axis="x", rotation=90)
    return ax


def plot_boxplots_by_label(
    frame: pd.DataFrame,
    columns: Sequence[str],
    target: str = "label",
) -> Figure:
    """Stack one :func:`plot_boxplot_by_label` per feature into a single figure.

    Args:
        frame: The dataframe to plot from.
        columns: The numeric columns to plot, one row of the figure each.
        target: Name of the grouping (class) column. Defaults to ``"label"``.

    Returns:
        The figure holding one boxplot panel per feature.

    Raises:
        KeyError: If any requested column, or ``target``, is absent.
        ValueError: If ``columns`` is empty.
    """
    requested = _require_columns(frame, columns)
    _require_columns(frame, [target])
    if not requested:
        raise ValueError("`columns` must name at least one column to plot.")

    n_classes = frame[target].nunique()
    fig, axes = plt.subplots(
        len(requested),
        1,
        figsize=(max(8.0, 0.45 * n_classes + 2), 3.6 * len(requested)),
        squeeze=False,
    )

    for ax, name in zip(axes.ravel(), requested, strict=True):
        plot_boxplot_by_label(frame, feature=name, target=target, ax=ax)

    fig.tight_layout()
    return fig


def count_outliers_iqr(
    frame: pd.DataFrame,
    columns: Sequence[str],
    whisker: float = 1.5,
) -> pd.DataFrame:
    """Count values outside the boxplot whiskers, per feature.

    This puts a number on what a boxplot shows visually. A value is flagged when
    it falls below ``Q1 - whisker * IQR`` or above ``Q3 + whisker * IQR``, which
    is exactly the rule matplotlib uses to decide which points to draw
    individually.

    Flagged is *not* the same as wrong. On this dataset the flagged points are
    mostly the tails of legitimate per-crop sub-populations, which is why Week 2
    counts them and removes none of them.

    Args:
        frame: The dataframe to inspect.
        columns: The numeric columns to check.
        whisker: Multiplier applied to the interquartile range. Defaults to
            1.5, the boxplot convention.

    Returns:
        A dataframe indexed by feature name with columns ``lower_bound``,
        ``upper_bound``, ``n_outliers`` and ``pct_outliers``.

    Raises:
        KeyError: If any requested column is absent from ``frame``.
        ValueError: If ``whisker`` is negative.
    """
    requested = _require_columns(frame, columns)
    if whisker < 0:
        raise ValueError(f"`whisker` must be non-negative, got {whisker}.")

    rows = []
    for name in requested:
        values = frame[name]
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - whisker * iqr, q3 + whisker * iqr
        flagged = int(((values < lower) | (values > upper)).sum())
        rows.append(
            {
                "feature": name,
                "lower_bound": lower,
                "upper_bound": upper,
                "n_outliers": flagged,
                "pct_outliers": 100.0 * flagged / len(values) if len(values) else 0.0,
            }
        )

    return pd.DataFrame(rows).set_index("feature")


def separation_scores(
    frame: pd.DataFrame,
    columns: Sequence[str],
    target: str = "label",
) -> pd.Series:
    """Rank features by how much of their variance is explained by the class.

    The statistic is the *correlation ratio* eta-squared: the share of a
    feature's total variance that lies **between** classes rather than within
    them. It runs from 0 (every class has the same distribution — the feature
    tells you nothing) to 1 (classes are perfectly separated on this feature
    alone).

    This is a descriptive summary of the boxplots, not a model and not a feature
    selection decision: a low score here does not license dropping a feature,
    because a feature can be useless alone and valuable in combination.

    Args:
        frame: The dataframe to score.
        columns: The numeric columns to score.
        target: Name of the class column. Defaults to ``"label"``.

    Returns:
        A series indexed by feature name holding eta-squared, sorted from the
        most to the least separating feature. Constant features score 0.0.

    Raises:
        KeyError: If any requested column, or ``target``, is absent.
    """
    requested = _require_columns(frame, columns)
    _require_columns(frame, [target])

    scores = {}
    for name in requested:
        values = frame[name].astype(float)
        grand_mean = values.mean()
        group_means = values.groupby(frame[target]).mean()
        group_sizes = values.groupby(frame[target]).size()
        between = float((group_sizes * (group_means - grand_mean) ** 2).sum())
        total = float(((values - grand_mean) ** 2).sum())
        scores[name] = 0.0 if total == 0 else between / total

    return pd.Series(scores, name="eta_squared").sort_values(ascending=False)
