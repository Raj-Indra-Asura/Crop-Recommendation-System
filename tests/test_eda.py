"""Smoke tests for :mod:`src.utils.eda`.

These are deliberately *smoke* tests: they confirm that every helper runs on a
small synthetic frame, returns the documented type, and refuses obviously
invalid input. They do not assert pixel-level plot contents — that would test
matplotlib, not this project.

The synthetic frame is used rather than the real CSV so that the whole file runs
even if the dataset is missing, and so that the helpers stay honest about being
general-purpose rather than hard-wired to the crop data.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

# A non-interactive backend, chosen before pyplot is imported anywhere: these
# tests must run on machines with no display attached (CI included).
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from src.utils.eda import (  # noqa: E402
    class_balance,
    count_outliers_iqr,
    describe_features,
    plot_boxplot_by_label,
    plot_boxplots_by_label,
    plot_class_balance,
    plot_correlation_heatmap,
    plot_feature_histograms,
    separation_scores,
)

SYNTHETIC_FEATURES = ("alpha", "beta", "gamma")
SYNTHETIC_TARGET = "label"


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    """Build a tiny, deterministic frame with three features and three classes.

    ``alpha`` separates the classes cleanly, ``beta`` is pure noise shared by
    all of them, and ``gamma`` is an exact multiple of ``alpha`` so that the
    correlation helper has a perfect +1 pair to find.
    """
    rng = np.random.default_rng(seed=0)
    rows_per_class = 20
    frames = []
    for offset, name in enumerate(["one", "two", "three"]):
        alpha = rng.normal(loc=10.0 * offset, scale=1.0, size=rows_per_class)
        frames.append(
            pd.DataFrame(
                {
                    "alpha": alpha,
                    "beta": rng.normal(loc=5.0, scale=1.0, size=rows_per_class),
                    "gamma": 2.0 * alpha,
                    SYNTHETIC_TARGET: name,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(autouse=True)
def close_figures():
    """Close every figure a test opened, so the suite does not leak memory."""
    yield
    plt.close("all")


# --- summarising helpers ----------------------------------------------------


def test_describe_features_returns_a_row_per_feature(synthetic_frame):
    summary = describe_features(synthetic_frame, SYNTHETIC_FEATURES)
    assert isinstance(summary, pd.DataFrame)
    assert list(summary.index) == list(SYNTHETIC_FEATURES)
    assert {"mean", "median", "std", "skew", "min", "max"} <= set(summary.columns)


def test_describe_features_does_not_mutate_its_input(synthetic_frame):
    before = synthetic_frame.copy()
    describe_features(synthetic_frame, SYNTHETIC_FEATURES)
    pd.testing.assert_frame_equal(synthetic_frame, before)


def test_describe_features_rejects_an_unknown_column(synthetic_frame):
    with pytest.raises(KeyError):
        describe_features(synthetic_frame, ["alpha", "not_a_column"])


def test_class_balance_counts_every_class(synthetic_frame):
    balance = class_balance(synthetic_frame, SYNTHETIC_TARGET)
    assert list(balance.columns) == ["count", "proportion"]
    assert balance["count"].sum() == len(synthetic_frame)
    assert balance["proportion"].sum() == pytest.approx(1.0)
    assert set(balance.index) == {"one", "two", "three"}


def test_class_balance_rejects_a_missing_target(synthetic_frame):
    with pytest.raises(KeyError):
        class_balance(synthetic_frame, "crop")


def test_count_outliers_iqr_returns_a_row_per_feature(synthetic_frame):
    counts = count_outliers_iqr(synthetic_frame, SYNTHETIC_FEATURES)
    assert list(counts.index) == list(SYNTHETIC_FEATURES)
    assert (counts["n_outliers"] >= 0).all()
    assert (counts["lower_bound"] <= counts["upper_bound"]).all()


def test_count_outliers_iqr_flags_an_extreme_value(synthetic_frame):
    baseline = count_outliers_iqr(synthetic_frame, ["beta"]).loc["beta", "n_outliers"]
    spiked = synthetic_frame.copy()
    spiked.loc[0, "beta"] = 10_000.0
    after = count_outliers_iqr(spiked, ["beta"]).loc["beta", "n_outliers"]
    assert after == baseline + 1


def test_count_outliers_iqr_rejects_a_negative_whisker(synthetic_frame):
    with pytest.raises(ValueError):
        count_outliers_iqr(synthetic_frame, SYNTHETIC_FEATURES, whisker=-1.0)


def test_separation_scores_rank_the_separating_feature_first(synthetic_frame):
    scores = separation_scores(synthetic_frame, SYNTHETIC_FEATURES, SYNTHETIC_TARGET)
    assert isinstance(scores, pd.Series)
    assert set(scores.index) == set(SYNTHETIC_FEATURES)
    assert ((scores >= 0.0) & (scores <= 1.0)).all()
    # `alpha` is built to separate the classes; `beta` is built not to.
    assert scores.idxmax() in {"alpha", "gamma"}
    assert scores["beta"] < scores["alpha"]


def test_separation_scores_are_zero_for_a_constant_feature(synthetic_frame):
    flat = synthetic_frame.assign(alpha=1.0)
    assert separation_scores(flat, ["alpha"], SYNTHETIC_TARGET)["alpha"] == 0.0


# --- plotting helpers -------------------------------------------------------


def test_plot_class_balance_returns_axes(synthetic_frame):
    ax = plot_class_balance(synthetic_frame, SYNTHETIC_TARGET)
    assert isinstance(ax, Axes)


def test_plot_class_balance_draws_on_supplied_axes(synthetic_frame):
    _, ax = plt.subplots()
    assert plot_class_balance(synthetic_frame, SYNTHETIC_TARGET, ax=ax) is ax


def test_plot_feature_histograms_returns_one_subplot_per_feature(synthetic_frame):
    fig = plot_feature_histograms(synthetic_frame, SYNTHETIC_FEATURES, bins=5)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == len(SYNTHETIC_FEATURES)


def test_plot_feature_histograms_handles_a_single_feature(synthetic_frame):
    fig = plot_feature_histograms(synthetic_frame, ["alpha"], bins=5, n_cols=3)
    assert len(fig.axes) == 1


def test_plot_feature_histograms_rejects_empty_columns(synthetic_frame):
    with pytest.raises(ValueError):
        plot_feature_histograms(synthetic_frame, [])


def test_plot_feature_histograms_rejects_non_positive_bins(synthetic_frame):
    with pytest.raises(ValueError):
        plot_feature_histograms(synthetic_frame, SYNTHETIC_FEATURES, bins=0)


def test_plot_correlation_heatmap_returns_axes(synthetic_frame):
    ax = plot_correlation_heatmap(synthetic_frame, SYNTHETIC_FEATURES)
    assert isinstance(ax, Axes)


def test_plot_correlation_heatmap_accepts_spearman(synthetic_frame):
    ax = plot_correlation_heatmap(synthetic_frame, SYNTHETIC_FEATURES, method="spearman")
    assert isinstance(ax, Axes)


def test_plot_correlation_heatmap_rejects_an_unknown_column(synthetic_frame):
    with pytest.raises(KeyError):
        plot_correlation_heatmap(synthetic_frame, ["alpha", "not_a_column"])


def test_plot_boxplot_by_label_returns_axes(synthetic_frame):
    ax = plot_boxplot_by_label(synthetic_frame, "alpha", SYNTHETIC_TARGET)
    assert isinstance(ax, Axes)


def test_plot_boxplots_by_label_returns_one_panel_per_feature(synthetic_frame):
    fig = plot_boxplots_by_label(synthetic_frame, SYNTHETIC_FEATURES, SYNTHETIC_TARGET)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == len(SYNTHETIC_FEATURES)


def test_plot_boxplots_by_label_rejects_a_missing_target(synthetic_frame):
    with pytest.raises(KeyError):
        plot_boxplots_by_label(synthetic_frame, SYNTHETIC_FEATURES, target="crop")
