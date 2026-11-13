"""Streaming (chunk-based) statistics.

- :class:`RunningStats` keeps per-feature count, mean, variance, min and max.
  Chunks are merged with the parallel form of Welford's algorithm
  (Chan et al.), which is numerically stable and gives the same result as a
  single pass over all data.
- :class:`StreamingStats` adds quantiles (reservoir sampling) and
  histograms (optionally over a sliding window of recent chunks), and offers
  the ``update_stats(X_chunk)`` API.
- :func:`update_stats` is a functional wrapper around the same update.

NaN values are treated as missing and ignored per feature.
"""

from collections import deque

import numpy as np

from .utils import check_array, check_positive_int, rng_from

__all__ = ["RunningStats", "StreamingStats", "update_stats"]


def _chunk_moments(X):
    """Per-feature NaN-aware count, mean, M2, min and max of one chunk.

    Fully vectorised; never emits RuntimeWarnings (all-NaN columns give
    count 0, mean 0, M2 0, min +inf, max -inf, which merge as "no data").
    """
    mask = ~np.isnan(X)
    count = mask.sum(axis=0).astype(np.float64)
    total = np.where(mask, X, 0.0).sum(axis=0)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    m2 = (np.where(mask, X - mean, 0.0) ** 2).sum(axis=0)
    mn = np.where(mask, X, np.inf).min(axis=0, initial=np.inf)
    mx = np.where(mask, X, -np.inf).max(axis=0, initial=-np.inf)
    return count, mean, m2, mn, mx


class RunningStats:
    """Per-feature running count, mean, variance, min and max.

    Each call to :meth:`update` merges one chunk into the running totals
    using Chan et al.'s pairwise update::

        delta = mean_b - mean_a
        mean  = mean_a + delta * n_b / n
        M2    = M2_a + M2_b + delta**2 * n_a * n_b / n

    which is exact (up to rounding) and avoids the cancellation error of the
    naive ``E[x^2] - E[x]^2`` formula.

    Attributes
    ----------
    n_samples_seen_ : int
        Rows seen so far (including rows with NaN).
    count_ : ndarray, shape (n_features,)
        Non-NaN values seen per feature.
    mean_ : ndarray, shape (n_features,)
        NaN for features with no observed values.
    min_, max_ : ndarray, shape (n_features,)
        NaN for features with no observed values.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Forget all data.

        Returns
        -------
        self

        Complexity
        ----------
        O(1).
        """
        self.n_samples_seen_ = 0
        self.n_features_in_ = None
        self._count = self._mean = self._m2 = self._min = self._max = None
        return self

    def update(self, X):
        """Merge one chunk into the running statistics.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            May contain NaN; an empty chunk is a no-op.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the number of features differs from earlier chunks.

        Complexity
        ----------
        Time O(n_samples * n_features), space O(n_features) state.
        """
        X = check_array(X)
        if self.n_features_in_ is None:
            d = X.shape[1]
            self.n_features_in_ = d
            self._count = np.zeros(d)
            self._mean = np.zeros(d)
            self._m2 = np.zeros(d)
            self._min = np.full(d, np.inf)
            self._max = np.full(d, -np.inf)
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}")
        if X.shape[0] == 0:
            return self
        n_b, mean_b, m2_b, min_b, max_b = _chunk_moments(X)
        n_a = self._count
        n = n_a + n_b
        delta = mean_b - self._mean
        safe_n = np.where(n > 0, n, 1.0)
        self._mean = np.where(n > 0, self._mean + delta * n_b / safe_n, 0.0)
        self._m2 = self._m2 + m2_b + delta ** 2 * n_a * n_b / safe_n
        self._count = n
        self._min = np.minimum(self._min, min_b)
        self._max = np.maximum(self._max, max_b)
        self.n_samples_seen_ += X.shape[0]
        return self

    partial_fit = update

    def _check(self):
        if self.n_features_in_ is None:
            raise ValueError("No data seen yet; call update() first")

    @property
    def count_(self):
        """Non-NaN values seen per feature, shape (n_features,)."""
        self._check()
        return self._count.copy()

    @property
    def mean_(self):
        """Per-feature mean, shape (n_features,); NaN where no value was seen."""
        self._check()
        return np.where(self._count > 0, self._mean, np.nan)

    @property
    def min_(self):
        """Per-feature minimum, shape (n_features,); NaN where no value was seen."""
        self._check()
        return np.where(self._count > 0, self._min, np.nan)

    @property
    def max_(self):
        """Per-feature maximum, shape (n_features,); NaN where no value was seen."""
        self._check()
        return np.where(self._count > 0, self._max, np.nan)

    def var(self, ddof=0):
        """Per-feature variance.

        Parameters
        ----------
        ddof : int, default 0
            Delta degrees of freedom (0 = population, 1 = sample variance).

        Returns
        -------
        var : ndarray, shape (n_features,)
            NaN where fewer than ``ddof + 1`` values were seen.

        Complexity
        ----------
        Time O(n_features).
        """
        self._check()
        dof = self._count - ddof
        out = np.full_like(self._m2, np.nan)
        np.divide(self._m2, dof, out=out, where=dof > 0)
        return np.maximum(out, 0.0, where=dof > 0, out=out)

    def std(self, ddof=0):
        """Per-feature standard deviation (square root of :meth:`var`).

        Parameters
        ----------
        ddof : int, default 0

        Returns
        -------
        std : ndarray, shape (n_features,)

        Complexity
        ----------
        Time O(n_features).
        """
        return np.sqrt(self.var(ddof))


class StreamingStats:
    """Streaming mean, variance, quantiles and histograms for 2-D data.

    Parameters
    ----------
    reservoir_size : int, default 1000
        Rows kept for quantile estimates. Quantiles are exact while at most
        ``reservoir_size`` rows have been seen, and estimated from a uniform
        random sample (reservoir sampling, Algorithm R) afterwards.
    bins : int, default 10
        Number of equal-width histogram bins per feature.
    hist_range : tuple (low, high) or None
        Histogram range for every feature. If None, each feature's range is
        fixed from its min and max in the first chunk; later values outside
        it are counted in ``underflow`` / ``overflow``.
    window : int or None, default None
        If set, histograms cover only the most recent ``window`` chunks
        (sliding window). Mean, variance and quantiles are always cumulative.
    random_state : int, Generator or None

    Examples
    --------
    >>> s = StreamingStats()
    >>> for chunk in chunks:
    ...     s.update_stats(chunk)
    >>> s.mean(), s.var(), s.quantile([0.25, 0.5, 0.75])
    """

    def __init__(self, reservoir_size=1000, bins=10, hist_range=None, window=None,
                 random_state=None):
        check_positive_int(reservoir_size, "reservoir_size")
        check_positive_int(bins, "bins")
        check_positive_int(window, "window", allow_none=True)
        if hist_range is not None and not hist_range[0] < hist_range[1]:
            raise ValueError("hist_range must satisfy low < high")
        self.reservoir_size = reservoir_size
        self.bins = bins
        self.hist_range = hist_range
        self.window = window
        self.random_state = random_state
        self.reset()

    def reset(self):
        """Forget all data.

        Returns
        -------
        self

        Complexity
        ----------
        O(1).
        """
        self.running_ = RunningStats()
        self._rng = rng_from(self.random_state)
        self._reservoir = None
        self._lo = self._hi = None
        self._hist_chunks = deque()
        self._hist_total = None
        return self

    @property
    def n_samples_seen_(self):
        """Rows seen so far (int)."""
        return self.running_.n_samples_seen_

    def update_stats(self, X_chunk):
        """Update every statistic with one chunk.

        Parameters
        ----------
        X_chunk : array_like, shape (n_samples, n_features)

        Returns
        -------
        self

        Complexity
        ----------
        Time O(m * d) for a chunk of m rows and d features (reservoir and
        histogram updates are vectorised); space
        O(reservoir_size * d + bins * d * window).
        """
        X = check_array(X_chunk)
        seen_before = self.running_.n_samples_seen_
        self.running_.update(X)
        if X.shape[0] == 0:
            return self
        self._update_reservoir(X, seen_before)
        self._update_histogram(X)
        return self

    update = partial_fit = update_stats

    def _update_reservoir(self, X, seen_before):
        """Algorithm R, vectorised over the chunk.

        Row t (1-based global position) replaces a random slot r in [0, t)
        when r < k. When several rows of one chunk pick the same slot, the
        last one wins, exactly as in the sequential algorithm.
        """
        k = self.reservoir_size
        if self._reservoir is None:
            self._reservoir = np.empty((0, X.shape[1]))
        n_fill = min(k - self._reservoir.shape[0], X.shape[0])
        if n_fill > 0:
            self._reservoir = np.vstack([self._reservoir, X[:n_fill]])
            X = X[n_fill:]
            seen_before += n_fill
        if X.shape[0] == 0:
            return
        t = seen_before + 1 + np.arange(X.shape[0])
        r = (self._rng.random(X.shape[0]) * t).astype(np.int64)
        accept = np.flatnonzero(r < k)
        if accept.size == 0:
            return
        slots = r[accept]
        # keep the last row for each slot
        _, last = np.unique(slots[::-1], return_index=True)
        rows = accept[::-1][last]
        self._reservoir[slots[::-1][last]] = X[rows]

    def _update_histogram(self, X):
        """Count one chunk into fixed equal-width bins (vectorised over all cells).

        Bin slots: 0 = underflow, 1..bins = bins, bins+1 = overflow.
        """
        d, b = X.shape[1], self.bins
        if self._lo is None:                       # fix each feature's range once
            if self.hist_range is not None:
                lo, hi = np.full(d, float(self.hist_range[0])), np.full(d, float(self.hist_range[1]))
            else:
                _, _, _, lo, hi = _chunk_moments(X)
                lo, hi = np.where(np.isfinite(lo), lo, 0.0), np.where(np.isfinite(hi), hi, 1.0)
                hi = np.where(hi > lo, hi, lo + 1.0)
            self._lo, self._hi = lo, hi
            self._hist_total = np.zeros((d, b + 2), dtype=np.int64)
        valid = ~np.isnan(X)
        x = np.where(valid, X, self._lo)
        slot = np.clip(np.floor((x - self._lo) / (self._hi - self._lo) * b), -1, b).astype(np.int64) + 1
        slot[valid & (X == self._hi)] = b          # the upper edge belongs to the last bin
        flat = (slot + (b + 2) * np.arange(d))[valid]
        counts = np.bincount(flat, minlength=d * (b + 2)).reshape(d, b + 2)
        self._hist_total += counts                 # running sum over the window
        if self.window is not None:
            self._hist_chunks.append(counts)
            if len(self._hist_chunks) > self.window:
                self._hist_total -= self._hist_chunks.popleft()

    # ------------------------------------------------------------- queries
    def mean(self):
        """Per-feature mean of all data seen (NaN ignored).

        Returns
        -------
        mean : ndarray, shape (n_features,)

        Raises
        ------
        ValueError
            If no data has been seen.

        Complexity
        ----------
        Time O(n_features).
        """
        return self.running_.mean_

    def var(self, ddof=0):
        """Per-feature variance of all data seen (NaN ignored).

        Parameters
        ----------
        ddof : int, default 0

        Returns
        -------
        var : ndarray, shape (n_features,)

        Raises
        ------
        ValueError
            If no data has been seen.

        Complexity
        ----------
        Time O(n_features).
        """
        return self.running_.var(ddof)

    def std(self, ddof=0):
        """Per-feature standard deviation of all data seen (NaN ignored).

        Parameters
        ----------
        ddof : int, default 0

        Returns
        -------
        std : ndarray, shape (n_features,)

        Raises
        ------
        ValueError
            If no data has been seen.

        Complexity
        ----------
        Time O(n_features).
        """
        return self.running_.std(ddof)

    def min(self):
        """Per-feature minimum of all data seen (NaN ignored).

        Returns
        -------
        min : ndarray, shape (n_features,)

        Raises
        ------
        ValueError
            If no data has been seen.

        Complexity
        ----------
        Time O(n_features).
        """
        return self.running_.min_

    def max(self):
        """Per-feature maximum of all data seen (NaN ignored).

        Returns
        -------
        max : ndarray, shape (n_features,)

        Raises
        ------
        ValueError
            If no data has been seen.

        Complexity
        ----------
        Time O(n_features).
        """
        return self.running_.max_

    def quantile(self, q):
        """Per-feature quantiles from the reservoir sample (NaN ignored).

        Parameters
        ----------
        q : float or array_like of float in [0, 1]

        Returns
        -------
        quantiles : ndarray, shape (n_features,) for scalar q, else
            (len(q), n_features). NaN for features with no values.

        Raises
        ------
        ValueError
            If no data has been seen or q is outside [0, 1].

        Complexity
        ----------
        Time O(k log k * d) for a reservoir of k rows.
        """
        if self._reservoir is None or self._reservoir.shape[0] == 0:
            raise ValueError("No data seen yet; call update_stats() first")
        q_arr = np.asarray(q, dtype=np.float64)
        if q_arr.ndim > 1:
            raise ValueError(f"q must be a scalar or 1-D, got shape {q_arr.shape}")
        if np.isnan(q_arr).any() or np.any((q_arr < 0) | (q_arr > 1)):
            raise ValueError(f"q must be in [0, 1], got {q}")
        s = np.sort(self._reservoir, axis=0)            # NaN sorted last
        n_valid = (~np.isnan(s)).sum(axis=0)
        qs = np.atleast_1d(q_arr)
        pos = qs[:, None] * np.maximum(n_valid - 1, 0)
        lo = np.floor(pos).astype(np.int64)
        hi = np.ceil(pos).astype(np.int64)
        v_lo = np.take_along_axis(s, lo, axis=0)
        v_hi = np.take_along_axis(s, hi, axis=0)
        out = np.where(lo == hi, v_lo, v_lo + (pos - lo) * (v_hi - v_lo))
        out = np.where(n_valid > 0, out, np.nan)
        return out[0] if q_arr.ndim == 0 else out

    def histogram(self, feature=0):
        """Histogram of one feature (over the sliding window if one is set).

        Parameters
        ----------
        feature : int, default 0

        Returns
        -------
        counts : ndarray of int, shape (bins,)
        edges : ndarray of float, shape (bins + 1,)
        underflow, overflow : int
            Values below / above the fixed range.

        Raises
        ------
        ValueError
            If no data has been seen or ``feature`` is out of range.

        Complexity
        ----------
        Time O(bins) (the window total is kept as a running sum).
        """
        if self._lo is None:
            raise ValueError("No data seen yet; call update_stats() first")
        if not 0 <= feature < self._lo.size:
            raise ValueError(f"feature must be in [0, {self._lo.size}), got {feature}")
        row = self._hist_total[feature]
        edges = np.linspace(self._lo[feature], self._hi[feature], self.bins + 1)
        return row[1:-1].copy(), edges, int(row[0]), int(row[-1])


def update_stats(X_chunk, state=None, **kwargs):
    """Functional API: update (or create) a :class:`StreamingStats` state.

    Parameters
    ----------
    X_chunk : array_like, shape (n_samples, n_features)
    state : StreamingStats or None
        Previous state; a new one is created (with ``**kwargs``) if None.

    Returns
    -------
    state : StreamingStats

    Complexity
    ----------
    Same as :meth:`StreamingStats.update_stats`.

    Examples
    --------
    >>> state = None
    >>> for chunk in chunks:
    ...     state = update_stats(chunk, state)
    >>> state.mean()
    """
    if state is None:
        state = StreamingStats(**kwargs)
    return state.update_stats(X_chunk)
