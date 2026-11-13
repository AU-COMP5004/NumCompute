"""Streaming classification metrics.

Every metric object has the same API:

- ``update(y_true_chunk, y_pred_chunk) -> self``  add one chunk
- ``result()``                                    current value
- ``reset() -> self``                             forget everything

With ``window=None`` a metric is cumulative over the whole stream; with
``window=k`` it covers only the most recent ``k`` chunks (rolling window).
Rolling windows keep a running sum: the newest chunk is added and the chunk
that leaves the window is subtracted, so an update costs the same with or
without a window.

Accuracy, precision, recall and F1 are all computed from one accumulated
confusion matrix (:class:`StreamingConfusionMatrix`), so they always agree.
The batch functions (``accuracy_score`` etc.) give the same numbers for a
single array and are used in the tests to check the streaming versions.
"""

from collections import deque

import numpy as np

from .utils import check_positive_int, check_y

__all__ = ["accuracy_score", "confusion_matrix", "precision_recall_f1", "roc_auc_score",
           "StreamingConfusionMatrix", "StreamingAccuracy", "StreamingPrecision",
           "StreamingRecall", "StreamingF1", "StreamingAUC"]

_AVERAGES = ("binary", "macro", "micro", "weighted")


# ----------------------------------------------------------------- helpers
def _check_pair(y_true, y_pred):
    y_true = check_y(y_true, name="y_true")
    return y_true, check_y(y_pred, y_true.shape[0], name="y_pred")


def _pad(mat, k):
    """Zero-pad a square matrix to shape (k, k)."""
    if mat.shape[0] == k:
        return mat
    out = np.zeros((k, k), dtype=mat.dtype)
    out[:mat.shape[0], :mat.shape[1]] = mat
    return out


def _auc_from_counts(pos, neg):
    """AUC from positive / negative counts per score level (sorted ascending).

    Each positive scores 1 against every negative at a lower level and 1/2
    against every negative at the same level (ties).
    """
    pos, neg = pos.astype(np.float64), neg.astype(np.float64)
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    neg_below = np.cumsum(neg) - neg
    return float((pos @ neg_below + 0.5 * (pos @ neg)) / (n_pos * n_neg))


# ----------------------------------------------------------- batch functions
def accuracy_score(y_true, y_pred):
    """Fraction of correct predictions.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n_samples,)

    Returns
    -------
    accuracy : float
        NaN for empty input.

    Raises
    ------
    ValueError
        If the lengths differ or a label is missing.

    Complexity
    ----------
    Time O(n), space O(n).
    """
    y_true, y_pred = _check_pair(y_true, y_pred)
    return float(np.mean(y_true == y_pred)) if y_true.size else float("nan")


def confusion_matrix(y_true, y_pred, labels=None):
    """Confusion matrix, rows = true label, columns = predicted label.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n_samples,)
    labels : array_like or None
        Label order; defaults to the sorted union of both arrays. Samples
        with a label outside ``labels`` are ignored.

    Returns
    -------
    cm : ndarray of int64, shape (n_labels, n_labels)
    labels : ndarray, shape (n_labels,)

    Raises
    ------
    ValueError
        If the lengths differ.

    Complexity
    ----------
    Time O(n log n + k**2), space O(n + k**2) for k labels.
    """
    y_true, y_pred = _check_pair(y_true, y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred])) if labels is None else np.asarray(labels)
    k = labels.size
    order = np.argsort(labels, kind="stable")
    ti = np.searchsorted(labels, y_true, sorter=order)
    pi = np.searchsorted(labels, y_pred, sorter=order)
    ok = (ti < k) & (pi < k)
    ok[ok] &= (labels[order[ti[ok]]] == y_true[ok]) & (labels[order[pi[ok]]] == y_pred[ok])
    ti, pi = order[ti[ok]], order[pi[ok]]
    cm = np.bincount(ti * k + pi, minlength=k * k).reshape(k, k)
    return cm, labels


def _prf_from_cm(cm, labels, average, pos_label, zero_division):
    """Precision, recall and F1 from a confusion matrix (vectorised)."""
    if average not in _AVERAGES:
        raise ValueError(f"average must be one of {_AVERAGES}, got {average!r}")
    tp = np.diag(cm).astype(np.float64)
    pred = cm.sum(axis=0).astype(np.float64)
    true = cm.sum(axis=1).astype(np.float64)

    def ratio(a, b):
        return np.divide(a, b, out=np.full_like(a, zero_division, dtype=np.float64), where=b > 0)

    if average == "micro":
        p = ratio(np.array([tp.sum()]), np.array([pred.sum()]))[0]
        return float(p), float(p), float(p)
    if average == "binary" and labels.size > 2:
        raise ValueError(f"average='binary' needs at most 2 labels, but {labels.size} were seen "
                         f"({labels.tolist()}); use average='macro', 'micro' or 'weighted'")
    precision, recall = ratio(tp, pred), ratio(tp, true)
    f1 = ratio(2 * precision * recall, precision + recall)
    if average == "binary":
        hit = np.flatnonzero(labels == pos_label)
        if hit.size == 0:
            return zero_division, zero_division, zero_division
        i = hit[0]
        return float(precision[i]), float(recall[i]), float(f1[i])
    if labels.size == 0:
        return zero_division, zero_division, zero_division
    if average == "weighted" and true.sum() > 0:
        w = true / true.sum()
    else:
        w = np.full(labels.size, 1.0 / labels.size)
    return float(w @ precision), float(w @ recall), float(w @ f1)


def precision_recall_f1(y_true, y_pred, average="binary", pos_label=1, zero_division=0.0):
    """Precision, recall and F1 for a batch.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n_samples,)
    average : {"binary", "macro", "micro", "weighted"}, default "binary"
    pos_label : label, default 1
        Positive class for ``average="binary"``.
    zero_division : float, default 0.0
        Value used when a ratio has a zero denominator.

    Returns
    -------
    precision, recall, f1 : float

    Raises
    ------
    ValueError
        If ``average`` is unknown, or ``average="binary"`` is used with more
        than two labels.

    Complexity
    ----------
    Time O(n log n + k**2) for k labels.
    """
    cm, labels = confusion_matrix(y_true, y_pred)
    return _prf_from_cm(cm, labels, average, pos_label, zero_division)


def roc_auc_score(y_true, scores, pos_label=1):
    """Exact binary ROC AUC (Mann-Whitney U, ties count 1/2).

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
    scores : array_like, shape (n_samples,)
        Higher score = more likely ``pos_label``.
    pos_label : label, default 1

    Returns
    -------
    auc : float
        NaN if only one class is present.

    Raises
    ------
    ValueError
        If ``scores`` does not have the shape of ``y_true`` or contains NaN.

    Complexity
    ----------
    Time O(n log n), space O(n).
    """
    y_true = check_y(y_true, name="y_true")
    s = np.asarray(scores, dtype=np.float64)
    if s.shape != y_true.shape:
        raise ValueError(f"scores has shape {s.shape}, expected {y_true.shape}")
    if np.isnan(s).any():
        raise ValueError("scores contain NaN")
    levels, inv = np.unique(s, return_inverse=True)
    pos = y_true == pos_label
    k = levels.size
    return _auc_from_counts(np.bincount(inv.ravel()[pos], minlength=k),
                            np.bincount(inv.ravel()[~pos], minlength=k))


# --------------------------------------------------------- streaming metrics
class StreamingConfusionMatrix:
    """Confusion matrix accumulated over chunks, with a growing label set.

    Parameters
    ----------
    labels : array_like or None
        Fixed label set. If None, labels are added as they appear.
    window : int or None, default None
        If set, :meth:`result` covers only the last ``window`` chunks.

    Attributes
    ----------
    n_samples_seen_ : int
        Samples added since the last reset (all chunks, not just the window).

    Notes
    -----
    New labels are appended to an internal label list, so earlier counts never
    move; :meth:`result` returns the matrix in sorted label order. Labels
    outside a fixed ``labels`` set are ignored.
    """

    def __init__(self, labels=None, window=None):
        check_positive_int(window, "window", allow_none=True)
        self.labels = labels
        self.window = window
        self.reset()

    def reset(self):
        """Forget all chunks.

        Returns
        -------
        self

        Complexity
        ----------
        O(1).
        """
        self._labels = None if self.labels is None else np.asarray(self.labels)
        self._sum = np.zeros((0, 0), dtype=np.int64)
        self._chunks = deque()
        self.n_samples_seen_ = 0
        return self

    def _codes(self, values):
        """Map labels to internal indices (vectorised); unknown -> -1."""
        uniq, inv = np.unique(values, return_inverse=True)
        if self._labels is None:
            self._labels = uniq
        elif self.labels is None:
            new = uniq[~np.isin(uniq, self._labels)]
            if new.size:
                self._labels = np.concatenate([self._labels, new])
        order = np.argsort(self._labels, kind="stable")
        pos = np.searchsorted(self._labels, uniq, sorter=order)
        idx = order[np.minimum(pos, order.size - 1)] if order.size else np.zeros(uniq.size, dtype=np.int64)
        valid = (pos < order.size) & (self._labels[idx] == uniq) if order.size else np.zeros(uniq.size, bool)
        idx = np.where(valid, idx, -1)
        return idx[inv.ravel()]

    def update(self, y_true, y_pred):
        """Add one chunk of (true, predicted) labels.

        Parameters
        ----------
        y_true, y_pred : array_like, shape (n_samples,)

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the lengths differ or a label is missing.

        Complexity
        ----------
        Time O(m log m + k**2) per chunk of m samples and k labels, with or
        without a window; space O(k**2 * window).
        """
        y_true, y_pred = _check_pair(y_true, y_pred)
        ti, pi = self._codes(y_true), self._codes(y_pred)
        k = 0 if self._labels is None else self._labels.size
        ok = (ti >= 0) & (pi >= 0)
        cm = np.bincount(ti[ok] * k + pi[ok], minlength=k * k).reshape(k, k)
        self._sum = _pad(self._sum, k) + cm
        if self.window is not None:
            self._chunks.append(cm)
            if len(self._chunks) > self.window:
                old = self._chunks.popleft()
                self._sum[:old.shape[0], :old.shape[1]] -= old
        self.n_samples_seen_ += y_true.shape[0]
        return self

    def result(self):
        """Current matrix and its labels.

        Returns
        -------
        cm : ndarray of int64, shape (k, k)
            Rows = true label, columns = predicted label, sorted label order.
        labels : ndarray, shape (k,)

        Complexity
        ----------
        Time O(k log k + k**2).
        """
        if self._labels is None:
            return np.zeros((0, 0), dtype=np.int64), np.empty(0)
        k = self._labels.size
        order = np.argsort(self._labels, kind="stable")
        cm = _pad(self._sum, k)
        return cm[np.ix_(order, order)], self._labels[order]


class _CMMetric:
    """Base class for metrics computed from a streaming confusion matrix.

    Parameters
    ----------
    window : int or None
        Rolling window in chunks (None = cumulative).
    labels : array_like or None
        Fixed label set (None = learn labels from the stream).
    """

    def __init__(self, window=None, labels=None):
        self.window = window
        self.labels = labels
        self.cm_ = StreamingConfusionMatrix(labels=labels, window=window)

    def update(self, y_true, y_pred):
        """Add one chunk of (true, predicted) labels.

        Parameters
        ----------
        y_true, y_pred : array_like, shape (n_samples,)

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the lengths differ or a label is missing.

        Complexity
        ----------
        Time O(m log m + k**2) per chunk (see :class:`StreamingConfusionMatrix`).
        """
        self.cm_.update(y_true, y_pred)
        return self

    def reset(self):
        """Forget all chunks.

        Returns
        -------
        self

        Complexity
        ----------
        O(1).
        """
        self.cm_.reset()
        return self

    @property
    def n_samples_seen_(self):
        """Samples added since the last reset (int)."""
        return self.cm_.n_samples_seen_


class StreamingAccuracy(_CMMetric):
    """Cumulative or rolling-window accuracy.

    Parameters
    ----------
    window : int or None, default None
        Rolling window in chunks.
    labels : array_like or None
    """

    def result(self):
        """Fraction of correct predictions.

        Returns
        -------
        accuracy : float
            NaN before any sample has been seen.

        Complexity
        ----------
        Time O(k**2).
        """
        cm, _ = self.cm_.result()
        total = cm.sum()
        return float(np.trace(cm) / total) if total else float("nan")


class _PRFMetric(_CMMetric):
    """Base class for streaming precision / recall / F1.

    Parameters
    ----------
    average : {"binary", "macro", "micro", "weighted"}, default "binary"
    pos_label : label, default 1
    zero_division : float, default 0.0
    window : int or None, default None
    labels : array_like or None
    """

    _which = None

    def __init__(self, average="binary", pos_label=1, zero_division=0.0, window=None, labels=None):
        if average not in _AVERAGES:
            raise ValueError(f"average must be one of {_AVERAGES}, got {average!r}")
        super().__init__(window=window, labels=labels)
        self.average = average
        self.pos_label = pos_label
        self.zero_division = zero_division

    def result(self):
        """Current value of the metric.

        Returns
        -------
        value : float
            NaN before any sample has been seen.

        Raises
        ------
        ValueError
            If ``average="binary"`` but more than two labels have been seen.

        Complexity
        ----------
        Time O(k**2).
        """
        cm, labels = self.cm_.result()
        if cm.sum() == 0:
            return float("nan")
        return _prf_from_cm(cm, labels, self.average, self.pos_label, self.zero_division)[self._which]


class StreamingPrecision(_PRFMetric):
    """Streaming precision (parameters as :func:`precision_recall_f1`, plus ``window``)."""
    _which = 0


class StreamingRecall(_PRFMetric):
    """Streaming recall (parameters as :func:`precision_recall_f1`, plus ``window``)."""
    _which = 1


class StreamingF1(_PRFMetric):
    """Streaming F1 (parameters as :func:`precision_recall_f1`, plus ``window``)."""
    _which = 2


class StreamingAUC:
    """Exact binary ROC AUC accumulated over chunks.

    One positive and one negative counter is kept per distinct score, so the
    result always equals :func:`roc_auc_score` on all data seen. A decision
    tree produces few distinct scores (one probability per leaf), so the
    state stays small.

    Parameters
    ----------
    pos_label : label, default 1

    Attributes
    ----------
    n_samples_seen_ : int
    """

    def __init__(self, pos_label=1):
        self.pos_label = pos_label
        self.reset()

    def reset(self):
        """Forget all chunks.

        Returns
        -------
        self

        Complexity
        ----------
        O(1).
        """
        self._levels = np.empty(0)
        self._pos = np.empty(0)
        self._neg = np.empty(0)
        self.n_samples_seen_ = 0
        return self

    def update(self, y_true, scores):
        """Add one chunk.

        Parameters
        ----------
        y_true : array_like, shape (n_samples,)
        scores : array_like, shape (n_samples,), probabilities in [0, 1]

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If ``scores`` has the wrong shape, contains NaN, or is outside [0, 1].

        Complexity
        ----------
        Time O((m + L) log(m + L)) per chunk of m samples and L distinct
        scores seen so far; space O(L).
        """
        y_true = check_y(y_true, name="y_true")
        s = np.asarray(scores, dtype=np.float64)
        if s.shape != y_true.shape:
            raise ValueError(f"scores has shape {s.shape}, expected {y_true.shape}")
        if np.isnan(s).any() or np.any((s < 0) | (s > 1)):
            raise ValueError("scores must be probabilities in [0, 1] without NaN")
        is_pos = (y_true == self.pos_label).astype(np.float64)
        levels, inv = np.unique(np.concatenate([self._levels, s]), return_inverse=True)
        inv = inv.ravel()
        self._pos = np.bincount(inv, np.concatenate([self._pos, is_pos]), levels.size)
        self._neg = np.bincount(inv, np.concatenate([self._neg, 1 - is_pos]), levels.size)
        self._levels = levels
        self.n_samples_seen_ += y_true.shape[0]
        return self

    def result(self):
        """Current AUC.

        Returns
        -------
        auc : float
            NaN until both classes have been seen.

        Complexity
        ----------
        Time O(L) for L distinct scores.
        """
        return _auc_from_counts(self._pos, self._neg)
