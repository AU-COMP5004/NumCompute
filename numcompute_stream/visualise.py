"""Reusable matplotlib plots for streaming experiments.

Every function:

- accepts an optional ``ax`` to draw into an existing figure,
- saves the figure when ``save_path`` is given,
- calls ``plt.show()`` only when ``show=True`` (notebooks display the
  returned figure inline anyway),
- returns the ``matplotlib.figure.Figure``.

NaN values (e.g. chunks scored before the first training) are drawn as gaps.
"""

import numpy as np

__all__ = ["plot_metric_over_time", "compare_models", "plot_predictions_vs_ground_truth",
           "plot_confusion_matrix"]


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _finish(fig, save_path, show):
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        _plt().show()
    return fig


def _axes(ax, figsize):
    if ax is not None:
        return ax.figure, ax
    fig, ax = _plt().subplots(figsize=figsize)
    return fig, ax


def _as_series(values, name):
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D (one value per chunk), got shape {arr.shape}")
    return arr


def plot_metric_over_time(metric_values, title="Metric over time", ylabel="Accuracy",
                          xlabel="Chunk", ax=None, save_path=None, show=False, marks=None):
    """Plot one metric across chunks.

    Parameters
    ----------
    metric_values : array_like, shape (n_chunks,)
    title, ylabel, xlabel : str
    ax : matplotlib Axes or None
    save_path : str or None
        File to save the figure to (format from the extension, e.g. ".png").
    show : bool, default False
    marks : array_like of int or None
        Chunk indices to mark with vertical lines (e.g. retrain events).

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``metric_values`` is not 1-D.

    Complexity
    ----------
    Time O(n_chunks).
    """
    y = _as_series(metric_values, "metric_values")
    fig, ax = _axes(ax, (8, 3.5))
    x = np.arange(y.size)
    ax.plot(x, y, marker="o", markersize=3, linewidth=1.8)
    for m in ([] if marks is None else np.asarray(marks)):
        ax.axvline(m, color="grey", linestyle=":", linewidth=1)
    if marks is not None and len(marks):
        ax.plot([], [], color="grey", linestyle=":", label="retrain")
        ax.legend(loc="lower right")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.3)
    return _finish(fig, save_path, show)


def compare_models(metric1, metric2, labels=("Model 1", "Model 2"), title="Model comparison",
                   ylabel="Accuracy", xlabel="Chunk", ax=None, save_path=None, show=False):
    """Plot two models' streaming metrics on the same axes.

    Parameters
    ----------
    metric1, metric2 : array_like, shape (n_chunks,)
        The two series may have different lengths.
    labels : tuple of two str
    title, ylabel, xlabel : str
    ax, save_path, show : see :func:`plot_metric_over_time`

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If a series is not 1-D or ``labels`` does not have two entries.

    Complexity
    ----------
    Time O(n_chunks).
    """
    a, b = _as_series(metric1, "metric1"), _as_series(metric2, "metric2")
    if len(labels) != 2:
        raise ValueError("labels must contain exactly two names")
    fig, ax = _axes(ax, (8, 3.5))
    ax.plot(np.arange(a.size), a, marker="o", markersize=3, label=labels[0])
    ax.plot(np.arange(b.size), b, marker="s", markersize=3, label=labels[1])
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    return _finish(fig, save_path, show)


def plot_predictions_vs_ground_truth(y_true, y_pred, title="Predictions vs ground truth (latest chunk)",
                                     ax=None, save_path=None, show=False):
    """Show true and predicted classes for each sample of a chunk.

    True labels are drawn as circles, predictions as crosses; wrong
    predictions are highlighted in red. The title reports the accuracy.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n_samples,)
    title : str
    ax, save_path, show : see :func:`plot_metric_over_time`

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If the arrays differ in length or are not 1-D.

    Complexity
    ----------
    Time O(n_samples log n_samples).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must be 1-D with the same length, "
                         f"got {y_true.shape} and {y_pred.shape}")
    labels = np.unique(np.concatenate([y_true, y_pred]))
    t, p = np.searchsorted(labels, y_true), np.searchsorted(labels, y_pred)
    wrong = t != p
    fig, ax = _axes(ax, (8, 3.5))
    x = np.arange(y_true.size)
    ax.scatter(x, t, s=40, facecolors="none", edgecolors="tab:blue", label="ground truth")
    ax.scatter(x[~wrong], p[~wrong], s=18, marker="x", color="tab:green", label="correct prediction")
    ax.scatter(x[wrong], p[wrong], s=30, marker="x", color="tab:red", label="wrong prediction")
    ax.set_yticks(np.arange(labels.size), [str(v) for v in labels])
    acc = 1.0 - wrong.mean() if y_true.size else float("nan")
    ax.set(title=f"{title} — accuracy {acc:.1%}", xlabel="Sample in chunk", ylabel="Class")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8, frameon=False)
    ax.grid(alpha=0.3, axis="y")
    return _finish(fig, save_path, show)


def plot_confusion_matrix(cm, labels=None, title="Confusion matrix", ax=None, save_path=None,
                          show=False):
    """Heatmap of a confusion matrix (rows = true, columns = predicted).

    Parameters
    ----------
    cm : array_like, shape (k, k)
    labels : array_like of length k or None
    title : str
    ax, save_path, show : see :func:`plot_metric_over_time`

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``cm`` is empty or not square.

    Complexity
    ----------
    Time O(k**2).
    """
    cm = np.asarray(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or cm.shape[0] == 0:
        raise ValueError(f"cm must be a non-empty square matrix, got shape {cm.shape}")
    labels = np.arange(cm.shape[0]) if labels is None else np.asarray(labels)
    fig, ax = _axes(ax, (4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ticks = np.arange(cm.shape[0])
    ax.set_xticks(ticks, [str(v) for v in labels])
    ax.set_yticks(ticks, [str(v) for v in labels])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(title=title, xlabel="Predicted", ylabel="True")
    return _finish(fig, save_path, show)
