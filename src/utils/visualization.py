"""Model-facing plotting helpers (Week 6).

Where :mod:`src.utils.eda` draws the *data*, this module draws what a *model*
does with it. It holds one function so far,
:func:`plot_decision_boundary`, which turns the phrase "decision boundary" into
a picture: the regions of a two-dimensional feature space in which a fitted
classifier answers each class, with the training rows drawn on top.

The same conventions as :mod:`src.utils.eda` apply, and for the same reason —
they are what let ``tests/test_classical_models.py`` exercise this code
head-lessly:

* nothing calls ``plt.show()`` — the caller decides when (or whether) to render;
* nothing writes a file;
* nothing mutates its inputs, and nothing fits a model.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

#: Number of grid points per axis used to paint the background. 200 x 200 =
#: 40,000 predictions, which is fast for every model in this project and fine
#: enough that the boundary does not look like a staircase.
DEFAULT_GRID_RESOLUTION: int = 200

#: Fraction of each feature's range added as padding around the data, so points
#: at the extremes are not drawn on the edge of the plot.
DEFAULT_PADDING: float = 0.05


def plot_decision_boundary(
    model,
    X_2d: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    ax: Axes | None = None,
    resolution: int = DEFAULT_GRID_RESOLUTION,
    padding: float = DEFAULT_PADDING,
    title: str | None = None,
) -> Axes:
    """Draw the regions a fitted classifier assigns to each class, in 2-D.

    A **decision boundary** is the surface where the model switches its answer
    from one class to another. In seven dimensions it cannot be drawn; in two it
    can, and the method is brute force rather than mathematics: cover the plane
    with a fine grid, ask the model to classify every grid point, and colour
    each point by the answer. The colours are the model's regions; the seams
    between them are the boundary.

    That makes the difference between algorithms visible rather than asserted —
    a linear model produces straight seams, an RBF-kernel SVM curved ones, and a
    decision tree only horizontal and vertical ones, because every question it
    can ask is of the form ``feature <= threshold``.

    This is an **illustration, not a result**. The model must have been fitted on
    exactly the two columns being plotted, so the picture describes a model
    trained on two features, not the seven-feature model whose accuracy the
    results table reports. Never quote a score from a plot made this way.

    Args:
        model: A **fitted** classifier (or pipeline) whose ``predict`` accepts
            two columns in the same order as ``X_2d``.
        X_2d: The two feature columns to plot, as a dataframe with two columns
            or an array of shape ``(n_rows, 2)``. The first is drawn on the
            x-axis, the second on the y-axis.
        y: The label of each row in ``X_2d``; used only to colour the scatter.
        ax: Existing axes to draw on. A new figure is created when omitted.
        resolution: Grid points per axis. Defaults to
            :data:`DEFAULT_GRID_RESOLUTION`; lower it for a faster, coarser plot.
        padding: Fraction of each feature's range left as a margin around the
            points. Defaults to :data:`DEFAULT_PADDING`.
        title: Title for the axes. A generic one is used when omitted.

    Returns:
        The axes the plot was drawn on.

    Raises:
        ValueError: If ``X_2d`` does not have exactly two columns, if ``y`` has a
            different length, if ``resolution`` is less than 2, or if ``padding``
            is negative.
    """
    features = pd.DataFrame(X_2d).reset_index(drop=True)
    if features.shape[1] != 2:
        raise ValueError(
            f"`X_2d` must have exactly two columns to be plottable, got {features.shape[1]}."
        )
    labels = pd.Series(np.asarray(y)).reset_index(drop=True)
    if len(labels) != len(features):
        raise ValueError(f"`X_2d` has {len(features)} rows but `y` has {len(labels)}.")
    if resolution < 2:
        raise ValueError(f"`resolution` must be at least 2, got {resolution}.")
    if padding < 0:
        raise ValueError(f"`padding` must not be negative, got {padding}.")

    first, second = (str(name) for name in features.columns)
    x_values = features.iloc[:, 0].to_numpy(dtype=float)
    y_values = features.iloc[:, 1].to_numpy(dtype=float)
    x_grid, y_grid = _mesh(x_values, y_values, resolution, padding)

    # Ask the model for the grid in the same container it was fitted on, so a
    # model fitted on a plain array is not handed feature names it never saw.
    flattened = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    grid = (
        pd.DataFrame(flattened, columns=[first, second])
        if isinstance(X_2d, pd.DataFrame)
        else flattened
    )
    predictions = np.asarray(model.predict(grid))

    # contourf needs numbers, so map the predicted labels (often strings) onto
    # the class order the training labels define, keeping colours consistent
    # between the background and the scatter.
    classes = np.unique(np.concatenate([labels.to_numpy(), predictions]))
    codes = np.searchsorted(classes, predictions).reshape(x_grid.shape)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    ax.contourf(
        x_grid,
        y_grid,
        codes,
        levels=np.arange(len(classes) + 1) - 0.5,
        cmap="tab20",
        alpha=0.25,
    )
    for index, name in enumerate(classes):
        rows = labels.to_numpy() == name
        if not rows.any():
            continue
        ax.scatter(
            x_values[rows],
            y_values[rows],
            s=14,
            color=plt.get_cmap("tab20")(index % 20),
            edgecolor="black",
            linewidth=0.3,
            label=str(name),
        )
    ax.set_xlabel(first)
    ax.set_ylabel(second)
    ax.set_title(title or f"Decision boundary on {first} and {second}")
    return ax


def _mesh(
    x_values: np.ndarray,
    y_values: np.ndarray,
    resolution: int,
    padding: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the grid of points the model is asked to classify.

    Args:
        x_values: The first feature's values, used to set the horizontal range.
        y_values: The second feature's values, used to set the vertical range.
        resolution: Grid points per axis.
        padding: Fraction of each range left as a margin.

    Returns:
        The pair of ``(resolution, resolution)`` coordinate arrays produced by
        :func:`numpy.meshgrid`.
    """
    axes = []
    for values in (x_values, y_values):
        low, high = float(values.min()), float(values.max())
        # A constant column would give a zero-width axis, so fall back to 1.
        margin = padding * (high - low) or 1.0
        axes.append(np.linspace(low - margin, high + margin, resolution))
    return np.meshgrid(*axes)
