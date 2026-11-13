"""Streaming preprocessing transformers.

Every transformer follows the same API:

- ``partial_fit(X_chunk, y=None) -> self``  update the state with one chunk
- ``fit(X, y=None) -> self``                reset, then ``partial_fit(X)``
- ``transform(X) -> X_out``                 use the current state
- ``fit_transform(X, y=None) -> X_out``

so ``fit(X)`` and a sequence of ``partial_fit`` calls over the chunks of X
give the same state. Every ``transform`` checks that X has the number of
features seen during fitting and raises ``ValueError`` otherwise (no silent
broadcasting).
"""

import numpy as np

from .stats import RunningStats, StreamingStats
from .utils import NotFittedError, check_array

__all__ = ["StandardScaler", "MinMaxScaler", "SimpleImputer", "OneHotEncoder"]


class _StreamingTransformer:
    """Shared fit / fit_transform and feature-count checks."""

    def fit(self, X, y=None):
        """Reset the state, then fit on X (same result as one ``partial_fit``).

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self

        Complexity
        ----------
        Same as :meth:`partial_fit` on X.
        """
        self._reset()
        return self.partial_fit(X, y)

    def fit_transform(self, X, y=None):
        """Fit on X, then transform it.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        X_out : ndarray, shape (n_samples, n_output_features)

        Complexity
        ----------
        :meth:`partial_fit` plus :meth:`transform` on X.
        """
        return self.fit(X, y).transform(X)

    def _check_features(self, X):
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but {type(self).__name__} "
                             f"was fitted with {self.n_features_in_}")

    def _check_fitted(self, X):
        if self.n_features_in_ is None:
            raise NotFittedError(f"{type(self).__name__} is not fitted; call partial_fit() first")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but {type(self).__name__} "
                             f"was fitted with {self.n_features_in_}")


class StandardScaler(_StreamingTransformer):
    """Standardise features to zero mean and unit variance, chunk by chunk.

    The running mean and variance are merged with Welford / Chan updates
    (see :class:`numcompute_stream.stats.RunningStats`), so streaming and
    batch fitting give the same result.

    Parameters
    ----------
    with_mean : bool, default True
    with_std : bool, default True

    Attributes
    ----------
    mean_ : ndarray, shape (n_features,)
    var_ : ndarray, shape (n_features,)
    scale_ : ndarray, shape (n_features,)
        Standard deviation; zero-variance or never-observed features get
        scale 1 so they are never divided by zero.
    n_samples_seen_ : int

    Notes
    -----
    NaN values are ignored when fitting and kept as NaN by ``transform``.
    Integer input is converted to float64 first, so it cannot overflow.
    """

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self._reset()

    def _reset(self):
        self._stats = RunningStats()
        self.n_features_in_ = None

    def partial_fit(self, X, y=None):
        """Update the running mean and variance with one chunk.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the number of features changes between chunks.

        Complexity
        ----------
        Time O(n_samples * n_features), space O(n_features).
        """
        X = check_array(X)
        self._check_features(X)
        self._stats.update(X)
        return self

    @property
    def n_samples_seen_(self):
        """Rows seen so far (int)."""
        return self._stats.n_samples_seen_

    @property
    def mean_(self):
        """Per-feature running mean, shape (n_features,); 0 if no value seen."""
        return np.nan_to_num(self._stats.mean_, nan=0.0)

    @property
    def var_(self):
        """Per-feature running variance (ddof=0), shape (n_features,)."""
        return np.nan_to_num(self._stats.var(), nan=0.0)

    @property
    def scale_(self):
        """Per-feature scale, shape (n_features,); 1 where the variance is 0."""
        std = np.sqrt(self.var_)
        return np.where(std > 0, std, 1.0)

    def transform(self, X):
        """Standardise X with the current running statistics.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        X_out : ndarray of float64, shape (n_samples, n_features)

        Raises
        ------
        NotFittedError
            Before the first ``partial_fit``.
        ValueError
            If X has a different number of features.

        Complexity
        ----------
        Time O(n_samples * n_features), space O(n_samples * n_features).
        """
        X = check_array(X)
        self._check_fitted(X)
        out = X - self.mean_ if self.with_mean else X.copy()
        return out / self.scale_ if self.with_std else out

    def inverse_transform(self, X):
        """Undo :meth:`transform`.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        X_original : ndarray of float64, shape (n_samples, n_features)

        Raises
        ------
        NotFittedError, ValueError
            As :meth:`transform`.

        Complexity
        ----------
        Time O(n_samples * n_features).
        """
        X = check_array(X)
        self._check_fitted(X)
        out = X * self.scale_ if self.with_std else X.copy()
        return out + self.mean_ if self.with_mean else out


class MinMaxScaler(_StreamingTransformer):
    """Scale features to ``feature_range`` using the running min and max.

    Parameters
    ----------
    feature_range : tuple (min, max), default (0, 1)

    Attributes
    ----------
    data_min_, data_max_ : ndarray, shape (n_features,)
    scale_ : ndarray, shape (n_features,)
        Constant features get a data range of 1 to avoid division by zero.

    Notes
    -----
    Values in later chunks can fall outside ``feature_range`` if they are
    beyond the min / max seen so far; the scaler adapts on the next
    ``partial_fit``. Integer input is converted to float64 first, so e.g.
    int8 data cannot overflow.
    """

    def __init__(self, feature_range=(0.0, 1.0)):
        if not feature_range[0] < feature_range[1]:
            raise ValueError(f"feature_range must satisfy min < max, got {feature_range}")
        self.feature_range = feature_range
        self._reset()

    def _reset(self):
        self._stats = RunningStats()
        self.n_features_in_ = None

    def partial_fit(self, X, y=None):
        """Update the running min / max with one chunk.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the number of features changes between chunks.

        Complexity
        ----------
        Time O(n_samples * n_features), space O(n_features).
        """
        X = check_array(X)
        self._check_features(X)
        self._stats.update(X)
        return self

    @property
    def data_min_(self):
        """Per-feature running minimum, shape (n_features,); 0 if no value seen."""
        return np.nan_to_num(self._stats.min_, nan=0.0)

    @property
    def data_max_(self):
        """Per-feature running maximum, shape (n_features,); 0 if no value seen."""
        return np.nan_to_num(self._stats.max_, nan=0.0)

    @property
    def scale_(self):
        """Per-feature multiplier, shape (n_features,)."""
        data_range = self.data_max_ - self.data_min_
        data_range = np.where(data_range > 0, data_range, 1.0)
        return (self.feature_range[1] - self.feature_range[0]) / data_range

    def transform(self, X):
        """Scale X to ``feature_range`` (NaN stays NaN).

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        X_out : ndarray of float64, shape (n_samples, n_features)

        Raises
        ------
        NotFittedError, ValueError
            Before fitting, or if the feature count differs.

        Complexity
        ----------
        Time O(n_samples * n_features).
        """
        X = check_array(X)
        self._check_fitted(X)
        return (X - self.data_min_) * self.scale_ + self.feature_range[0]


class SimpleImputer(_StreamingTransformer):
    """Replace NaN with a per-feature statistic that is updated on the fly.

    Parameters
    ----------
    strategy : {"mean", "median", "constant"}, default "mean"
        - "mean": running mean (exact, Welford).
        - "median": median of a reservoir sample (exact up to
          ``reservoir_size`` rows, then an unbiased estimate).
        - "constant": always ``fill_value``.
    fill_value : float, default 0.0
        Used by "constant", and for features with no observed value yet.
    reservoir_size : int, default 1000
        Only used by "median".
    random_state : int, Generator or None

    Attributes
    ----------
    statistics_ : ndarray, shape (n_features,)
        Current fill value of each feature.
    """

    _STRATEGIES = ("mean", "median", "constant")

    def __init__(self, strategy="mean", fill_value=0.0, reservoir_size=1000, random_state=None):
        if strategy not in self._STRATEGIES:
            raise ValueError(f"strategy must be one of {self._STRATEGIES}, got {strategy!r}")
        self.strategy = strategy
        self.fill_value = float(fill_value)
        self.reservoir_size = reservoir_size
        self.random_state = random_state
        self._reset()

    def _reset(self):
        self.n_features_in_ = None
        self._stats = RunningStats()
        self._quantiles = StreamingStats(reservoir_size=self.reservoir_size, bins=1,
                                         random_state=self.random_state)

    def partial_fit(self, X, y=None):
        """Update the fill statistics with one chunk.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the number of features changes between chunks.

        Complexity
        ----------
        Time O(n_samples * n_features) ("median" adds the reservoir update),
        space O(n_features) ("median": O(reservoir_size * n_features)).
        """
        X = check_array(X)
        self._check_features(X)
        if self.strategy == "mean":
            self._stats.update(X)
        elif self.strategy == "median":
            self._quantiles.update_stats(X)
        return self

    @property
    def statistics_(self):
        """Current fill value per feature, shape (n_features,)."""
        if self.n_features_in_ is None:
            raise NotFittedError("SimpleImputer is not fitted; call partial_fit() first")
        d = self.n_features_in_
        if self.strategy == "constant":
            return np.full(d, self.fill_value)
        if self.strategy == "mean":
            stat = self._stats.mean_ if self._stats.n_features_in_ else np.full(d, np.nan)
        else:
            stat = self._quantiles.quantile(0.5) if self._quantiles.n_samples_seen_ else np.full(d, np.nan)
        return np.where(np.isnan(stat), self.fill_value, stat)

    def transform(self, X):
        """Fill NaN in X with :attr:`statistics_`.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        X_out : ndarray of float64, shape (n_samples, n_features), no NaN

        Raises
        ------
        NotFittedError, ValueError
            Before fitting, or if the feature count differs.

        Complexity
        ----------
        Time O(n_samples * n_features) plus computing :attr:`statistics_`.
        """
        X = check_array(X)
        self._check_fitted(X)
        return np.where(np.isnan(X), self.statistics_, X)


class OneHotEncoder(_StreamingTransformer):
    """One-hot encode numeric category codes, growing categories over time.

    Parameters
    ----------
    categories : "auto" or list of array_like, default "auto"
        "auto" learns categories from the stream. A fixed list (one array per
        feature) freezes the output width, which is useful when the encoded
        data is buffered by a downstream model.
    handle_unknown : {"ignore", "error"}, default "ignore"
        For a value not seen yet: "ignore" encodes it as all zeros,
        "error" raises ValueError.

    Attributes
    ----------
    categories_ : list of ndarray
        Sorted categories per feature (None before fitting).

    Notes
    -----
    With ``categories="auto"`` the number of output columns grows as new
    categories arrive. NaN is treated as missing and encoded as all zeros.
    The work per feature is one vectorised call over all samples
    (``np.union1d`` to learn, ``np.searchsorted`` to encode).
    """

    def __init__(self, categories="auto", handle_unknown="ignore"):
        if handle_unknown not in ("ignore", "error"):
            raise ValueError("handle_unknown must be 'ignore' or 'error'")
        self.categories = categories
        self.handle_unknown = handle_unknown
        self._reset()

    def _reset(self):
        self.n_features_in_ = None
        self.categories_ = None
        if self.categories != "auto":
            self.categories_ = [np.unique(np.asarray(c, dtype=np.float64)) for c in self.categories]
            self.n_features_in_ = len(self.categories_)

    def partial_fit(self, X, y=None):
        """Add the categories seen in one chunk (NaN ignored).

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the number of features changes between chunks.

        Complexity
        ----------
        Time O(d * (n + c) log(n + c)) for c categories per feature; space O(c * d).
        """
        X = check_array(X)
        self._check_features(X)
        if self.categories != "auto":
            return self
        if self.categories_ is None:
            self.categories_ = [np.empty(0) for _ in range(X.shape[1])]
        for j, col in enumerate(X.T):
            self.categories_[j] = np.union1d(self.categories_[j], col[~np.isnan(col)])
        return self

    def get_feature_names_out(self, input_features=None):
        """Output column names such as ``x0_1``.

        Parameters
        ----------
        input_features : list of str or None
            Input feature names; ``x0, x1, ...`` if None.

        Returns
        -------
        names : list of str, length = number of output columns

        Raises
        ------
        NotFittedError
            Before fitting.

        Complexity
        ----------
        Time O(total number of categories).
        """
        if self.categories_ is None:
            raise NotFittedError("OneHotEncoder is not fitted; call partial_fit() first")
        names = input_features or [f"x{j}" for j in range(len(self.categories_))]
        return [f"{names[j]}_{v:g}" for j, cats in enumerate(self.categories_) for v in cats]

    def transform(self, X):
        """Encode X.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        X_out : ndarray of float64, shape (n_samples, total_categories)

        Raises
        ------
        NotFittedError
            Before fitting.
        ValueError
            If the feature count differs, or ``handle_unknown="error"`` and an
            unseen category appears.

        Complexity
        ----------
        Time O(n * d * log c), space O(n * total_categories).
        """
        X = check_array(X)
        if self.categories_ is None:
            raise NotFittedError("OneHotEncoder is not fitted; call partial_fit() first")
        self._check_fitted(X)
        widths = [cats.size for cats in self.categories_]
        out = np.zeros((X.shape[0], sum(widths)))
        offset = 0
        for j, (col, cats) in enumerate(zip(X.T, self.categories_)):
            pos = np.searchsorted(cats, col)
            known = cats[np.minimum(pos, cats.size - 1)] == col if cats.size else np.zeros(col.size, bool)
            if self.handle_unknown == "error" and np.any(~known & ~np.isnan(col)):
                raise ValueError(f"Unknown category {col[~known & ~np.isnan(col)][0]!r} in feature {j}")
            out[np.flatnonzero(known), offset + pos[known]] = 1.0
            offset += widths[j]
        return out
