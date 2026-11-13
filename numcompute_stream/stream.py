"""Streaming training loop with per-chunk logging.

:class:`StreamTrainer` wraps a model or :class:`~numcompute_stream.pipeline.Pipeline`
and runs the usual "test-then-train" (prequential) evaluation: each new
chunk is first used to score the current model, then to update it. Every
chunk produces one log record with the chunk accuracy, cumulative and
rolling-window metrics, fit time, whether the tree retrained, buffer fill
and memory footprint.
"""

import time

import numpy as np

from .metrics import StreamingAccuracy
from .utils import NotFittedError, check_X_y, memory_usage

__all__ = ["StreamTrainer"]


def _final_estimator(model):
    return getattr(model, "final_estimator", model)


class StreamTrainer:
    """Train and evaluate a model chunk by chunk, logging metrics.

    Parameters
    ----------
    model : estimator or Pipeline
        Must implement ``partial_fit(X, y)`` and ``predict(X)``.
    metrics : dict of name -> streaming metric, or None
        Extra cumulative metrics (objects with ``update(y_true, y_pred)``
        and ``result()``), e.g. ``{"f1": StreamingF1(average="macro")}``.
        A cumulative accuracy is always tracked.
    window : int, default 5
        Window (in chunks) of the rolling accuracy that is also logged.
    test_then_train : bool, default True
        If True, :meth:`step` scores a chunk before training on it, so every
        score is on data the model has not seen.
    verbose : bool, default False
        Print one log line per chunk.

    Attributes
    ----------
    log_ : list of dict
        One record per chunk (see :meth:`step`).
    """

    def __init__(self, model, metrics=None, window=5, test_then_train=True, verbose=False):
        if not hasattr(model, "partial_fit") or not hasattr(model, "predict"):
            raise TypeError("model must implement partial_fit() and predict()")
        self.model = model
        self.metrics = dict(metrics or {})
        self.window = window
        self.test_then_train = test_then_train
        self.verbose = verbose
        self.reset()

    def reset(self):
        """Clear logs and metrics (the model itself is not reset).

        Returns
        -------
        self

        Complexity
        ----------
        O(number of metrics).
        """
        self.log_ = []
        self.cumulative_accuracy_ = StreamingAccuracy()
        self.rolling_accuracy_ = StreamingAccuracy(window=self.window)
        for m in self.metrics.values():
            m.reset()
        self._pending = {}
        return self

    # ------------------------------------------------------------ chunk API
    def fit_chunk(self, X, y):
        """Update the model with one chunk.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
        y : array_like, shape (n_samples,)

        Returns
        -------
        info : dict
            ``fit_time`` (s), ``retrained`` (bool), ``n_retrains``,
            ``buffer_fill`` (samples waiting) and ``memory_bytes``.

        Raises
        ------
        ValueError
            Propagated from the model for invalid input.

        Complexity
        ----------
        The model's ``partial_fit`` cost plus O(model arrays) for the memory
        estimate.
        """
        final = _final_estimator(self.model)
        before = getattr(final, "n_retrains_", None)
        start = time.perf_counter()
        self.model.partial_fit(X, y)
        info = {
            "fit_time": time.perf_counter() - start,
            "retrained": before is not None and final.n_retrains_ > before,
            "n_retrains": getattr(final, "n_retrains_", None),
            "buffer_fill": getattr(final, "buffer_count_", None),
            "memory_bytes": memory_usage(self.model),
        }
        self._pending.update(info)
        return info

    def score_chunk(self, X, y):
        """Score the current model on one chunk and update all metrics.

        Before the model has been trained, nothing is updated and the chunk
        accuracy is NaN.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
        y : array_like, shape (n_samples,)

        Returns
        -------
        info : dict
            ``chunk_accuracy``, ``cumulative_accuracy``, ``rolling_accuracy``
            and one entry per extra metric.

        Raises
        ------
        ValueError
            If X and y have different lengths.

        Complexity
        ----------
        The model's ``predict`` cost plus O(m log m + k**2) per metric.
        """
        X, y = check_X_y(X, y)
        try:
            y_pred = self.model.predict(X) if X.shape[0] else y[:0]
        except NotFittedError:
            y_pred = None
        if y_pred is not None and y.size:
            chunk_acc = float(np.mean(y_pred == y))
            self.cumulative_accuracy_.update(y, y_pred)
            self.rolling_accuracy_.update(y, y_pred)
            for m in self.metrics.values():
                m.update(y, y_pred)
        else:
            chunk_acc = float("nan")
        info = {
            "chunk_accuracy": chunk_acc,
            "cumulative_accuracy": self.cumulative_accuracy_.result(),
            "rolling_accuracy": self.rolling_accuracy_.result(),
        }
        info.update({name: m.result() for name, m in self.metrics.items()})
        self.last_predictions_ = (y, y_pred)
        self._pending.update(info)
        return info

    def step(self, X, y):
        """Process one chunk (score + fit in the configured order) and log it.

        Returns
        -------
        record : dict
            The new entry of :attr:`log_`: ``chunk``, ``n_samples`` plus the
            fields of :meth:`score_chunk` and :meth:`fit_chunk`.

        Raises
        ------
        ValueError
            If X and y are invalid or have different lengths.

        Complexity
        ----------
        :meth:`score_chunk` plus :meth:`fit_chunk`.
        """
        X, y = check_X_y(X, y)
        self._pending = {"chunk": len(self.log_), "n_samples": int(X.shape[0])}
        if self.test_then_train:
            self.score_chunk(X, y)
            self.fit_chunk(X, y)
        else:
            self.fit_chunk(X, y)
            self.score_chunk(X, y)
        record = dict(self._pending)
        self.log_.append(record)
        if self.verbose:
            print(self.format_record(record))
        return record

    def run(self, chunks):
        """Process an iterable of ``(X_chunk, y_chunk)`` pairs.

        Parameters
        ----------
        chunks : iterable of (X_chunk, y_chunk)

        Returns
        -------
        self

        Complexity
        ----------
        One :meth:`step` per chunk.
        """
        for X, y in chunks:
            self.step(X, y)
        return self

    # --------------------------------------------------------------- logs
    def history(self, key):
        """Values of one log field across chunks.

        Parameters
        ----------
        key : str
            A log field, e.g. ``"chunk_accuracy"`` or ``"fit_time"``.

        Returns
        -------
        values : ndarray of float64, shape (n_chunks,)
            None (e.g. accuracy before the first training) becomes NaN.

        Complexity
        ----------
        Time O(n_chunks).
        """
        return np.array([np.nan if r.get(key) is None else r.get(key) for r in self.log_],
                        dtype=np.float64)

    @staticmethod
    def format_record(r):
        """One-line text version of a log record.

        Parameters
        ----------
        r : dict
            A record from :attr:`log_`.

        Returns
        -------
        line : str

        Complexity
        ----------
        O(1).
        """
        def pct(v):
            return "  n/a " if v is None or np.isnan(v) else f"{v:6.1%}"
        retrain = " retrained" if r.get("retrained") else ""
        return (f"chunk {r['chunk']:3d} | n={r['n_samples']:4d} | acc {pct(r['chunk_accuracy'])}"
                f" | cum {pct(r['cumulative_accuracy'])} | roll {pct(r['rolling_accuracy'])}"
                f" | buffer {r.get('buffer_fill')} | mem {r.get('memory_bytes', 0) / 1024:7.1f} KB"
                f" | fit {r.get('fit_time', 0) * 1000:6.2f} ms{retrain}")

    def summary(self):
        """Final cumulative metrics and totals.

        Returns
        -------
        summary : dict
            ``chunks``, ``samples``, ``cumulative_accuracy``,
            ``total_fit_time``, ``n_retrains`` and one entry per extra metric.

        Complexity
        ----------
        Time O(n_chunks).
        """
        out = {"chunks": len(self.log_),
               "samples": int(sum(r["n_samples"] for r in self.log_)),
               "cumulative_accuracy": self.cumulative_accuracy_.result(),
               "total_fit_time": float(sum(r.get("fit_time", 0.0) for r in self.log_)),
               "n_retrains": getattr(_final_estimator(self.model), "n_retrains_", None)}
        out.update({name: m.result() for name, m in self.metrics.items()})
        return out
