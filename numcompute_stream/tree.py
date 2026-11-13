"""Decision tree classifier with buffered retraining for data streams.

The tree follows the course's classification-tree recipe:

1. try every question (here: ``x[feature] <= threshold`` for every feature
   and every midpoint between consecutive distinct values),
2. keep the question with the largest information gain
   ``gain = I(parent) - (n_left * I(left) + n_right * I(right)) / n``,
   with I = Gini ``1 - sum(p^2)`` or entropy ``-sum(p log2 p)``,
3. split and repeat on each side,
4. stop (make a leaf) when ``max_depth`` is reached, the node is pure,
   it has fewer than ``min_samples_split`` samples, or no question has
   positive gain. A leaf predicts its majority class.

The split search is vectorised: for one node, all candidate features are
sorted at once and cumulative class counts give the impurity of every
threshold of every feature in a single array expression.

Streaming: :meth:`DecisionTreeClassifier.partial_fit` appends each chunk to a
buffer. When the buffer holds at least ``buffer_size`` samples, the tree is
retrained from scratch on the buffered data and the buffer is cleared.
Between retrains :meth:`~DecisionTreeClassifier.predict` uses the most
recently trained tree.

Tie rules (deterministic):

- equal gains: the lowest feature index wins, then the lowest threshold
  (gains within 1e-12 are treated as equal);
- equal class counts in a leaf: the smallest class label wins.
"""

import time

import numpy as np

from .utils import (NotFittedError, check_array, check_positive_int, check_X_y, check_y,
                    memory_usage, rng_from)

__all__ = ["DecisionTreeClassifier", "gini", "entropy"]

_GAIN_TOL = 1e-12


def gini(counts):
    """Gini impurity ``1 - sum(p^2)`` of class counts along the last axis.

    Parameters
    ----------
    counts : array_like, shape (..., n_classes)

    Returns
    -------
    impurity : ndarray, shape (...)
        0 for an empty or pure group.

    Complexity
    ----------
    Time O(size of counts).
    """
    counts = np.asarray(counts, dtype=np.float64)
    n = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, n, out=np.zeros_like(counts), where=n > 0)
    return np.where(n[..., 0] > 0, 1.0 - (p ** 2).sum(axis=-1), 0.0)


def entropy(counts):
    """Entropy ``-sum(p log2 p)`` (in bits) of class counts along the last axis.

    Parameters
    ----------
    counts : array_like, shape (..., n_classes)

    Returns
    -------
    impurity : ndarray, shape (...)
        0 for an empty or pure group (0 * log 0 is taken as 0).

    Complexity
    ----------
    Time O(size of counts).
    """
    counts = np.asarray(counts, dtype=np.float64)
    n = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, n, out=np.zeros_like(counts), where=n > 0)
    logp = np.log2(p, out=np.zeros_like(p), where=p > 0)
    return -(p * logp).sum(axis=-1)


_CRITERIA = {"gini": gini, "entropy": entropy}


_BLOCK_CELLS = 2_000_000      # max n * features * classes held at once (~16 MB per array)


def best_split(X, y_codes, n_classes, criterion="gini", min_samples_leaf=1, features=None):
    """Find the best ``x[f] <= t`` split of one node (vectorised).

    Candidate features are scored in blocks: within a block, all features
    are sorted at once and cumulative class counts give the impurity of
    every threshold of every feature in one array expression. Block size is
    chosen so that n * block * n_classes stays below a fixed budget, which
    keeps memory bounded even with many classes.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        No NaN.
    y_codes : ndarray of int, shape (n_samples,)
        Class indices in ``[0, n_classes)``.
    n_classes : int
    criterion : {"gini", "entropy"}
    min_samples_leaf : int, default 1
    features : ndarray of int or None
        Candidate feature indices (sorted). All features if None.

    Returns
    -------
    feature : int or None
        None if no split has positive gain.
    threshold : float or None
    gain : float

    Complexity
    ----------
    Time O(f * (n log n + n * k)) for n samples, f candidate features and k
    classes; space O(min(n * f * k, block budget)).
    """
    impurity = _CRITERIA[criterion]
    n = X.shape[0]
    features = np.arange(X.shape[1]) if features is None else np.asarray(features)
    if n < 2 * min_samples_leaf or features.size == 0:
        return None, None, 0.0
    onehot = np.eye(n_classes, dtype=np.float64)[y_codes]          # (n, k)
    total = onehot.sum(axis=0)
    parent = impurity(total)
    n_left = np.arange(1, n, dtype=np.float64)[:, None]             # split after row i
    size_ok = (n_left >= min_samples_leaf) & (n - n_left >= min_samples_leaf)
    block = max(1, _BLOCK_CELLS // max(n * n_classes, 1))
    best = (_GAIN_TOL, None, None)                                  # (gain, feature, threshold)
    for start in range(0, features.size, block):                    # blocks in feature order
        feats = features[start:start + block]
        Xf = X[:, feats]
        order = np.argsort(Xf, axis=0, kind="stable")
        xs = np.take_along_axis(Xf, order, axis=0)
        left = np.cumsum(onehot[order], axis=0)[:-1]                # (n-1, f, k)
        child = (n_left * impurity(left) + (n - n_left) * impurity(total - left)) / n
        gain = np.where((xs[1:] > xs[:-1]) & size_ok, parent - child, -np.inf)
        flat = gain.T.ravel()                                       # feature-major order
        top = flat.max()
        if top > best[0] + _GAIN_TOL:                               # strictly better: earlier ties win
            j, i = divmod(int(np.flatnonzero(flat >= top - _GAIN_TOL)[0]), n - 1)
            lo, hi = xs[i, j], xs[i + 1, j]
            t = (lo + hi) / 2.0
            best = (top, int(feats[j]), float(t if lo <= t < hi else lo))
    if best[1] is None:
        return None, None, 0.0
    return best[1], best[2], float(best[0])


class DecisionTreeClassifier:
    """Depth-limited classification tree with buffered streaming retraining.

    Parameters
    ----------
    criterion : {"gini", "entropy"}, default "gini"
    max_depth : int or None, default 5
        Maximum depth (root = depth 0). None grows until the other stop
        rules fire.
    min_samples_split : int, default 2
        A node with fewer samples becomes a leaf.
    min_samples_leaf : int, default 1
        Each child must keep at least this many samples.
    max_features : None, int, float, "sqrt" or "log2", default None
        Number of features tried at each node (a random subset, sorted).
        None tries all features.
    buffer_size : int, default 500
        ``partial_fit`` retrains once at least this many samples are buffered.
    random_state : int, Generator or None
        Only used when ``max_features`` selects a subset.

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
        All labels seen so far (sorted). Grows as new labels arrive.
    n_features_in_ : int
    feature_, threshold_, left_, right_ : ndarray, shape (n_nodes,)
        Flat tree arrays; ``feature_ == -1`` marks a leaf.
    value_ : ndarray, shape (n_nodes, n_classes_at_fit)
        Class counts per node.
    n_retrains_ : int
        Number of times the tree has been (re)trained.
    buffer_count_ : int
        Samples currently waiting in the buffer.
    last_retrain_time_ : float
        Seconds taken by the latest (re)training.
    fill_values_ : ndarray, shape (n_features,)
        Per-feature training means used to replace NaN in X.

    Notes
    -----
    Missing values: NaN in a feature is replaced by that feature's mean on
    the data the tree was trained on (or 0 if the feature was all NaN). Use
    a :class:`~numcompute_stream.preprocessing.SimpleImputer` in a pipeline
    for a different strategy.

    Examples
    --------
    >>> tree = DecisionTreeClassifier(max_depth=5, buffer_size=500)
    >>> for X_chunk, y_chunk in stream:
    ...     tree.partial_fit(X_chunk, y_chunk)
    >>> tree.predict(X_new)
    """

    def __init__(self, criterion="gini", max_depth=5, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, buffer_size=500, random_state=None):
        if criterion not in _CRITERIA:
            raise ValueError(f"criterion must be 'gini' or 'entropy', got {criterion!r}")
        if max_depth is not None and (not isinstance(max_depth, (int, np.integer)) or max_depth < 0):
            raise ValueError(f"max_depth must be None or a non-negative int, got {max_depth!r}")
        if not isinstance(min_samples_split, (int, np.integer)) or min_samples_split < 2:
            raise ValueError(f"min_samples_split must be an int >= 2, got {min_samples_split!r}")
        if not isinstance(min_samples_leaf, (int, np.integer)) or min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be an int >= 1, got {min_samples_leaf!r}")
        check_positive_int(buffer_size, "buffer_size")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.buffer_size = buffer_size
        self.random_state = random_state
        self._resolve_max_features(1)          # validate early
        self._reset_state()

    # ----------------------------------------------------------------- state
    def _reset_state(self):
        self.classes_ = None
        self.n_features_in_ = None
        self.feature_ = None
        self.n_retrains_ = 0
        self.last_retrain_time_ = 0.0
        self.total_retrain_time_ = 0.0
        self._buffer_X, self._buffer_y = [], []
        self.buffer_count_ = 0
        self._rng = rng_from(self.random_state)

    def _resolve_max_features(self, d):
        mf = self.max_features
        if mf is None:
            return d
        if mf == "sqrt":
            return max(1, int(np.sqrt(d)))
        if mf == "log2":
            return max(1, int(np.log2(d))) if d > 1 else 1
        if isinstance(mf, (int, np.integer)) and not isinstance(mf, bool) and mf >= 1:
            return min(int(mf), d)
        if isinstance(mf, float) and 0 < mf <= 1:
            return max(1, int(mf * d))
        raise ValueError(f"max_features must be None, 'sqrt', 'log2', an int >= 1 "
                         f"or a float in (0, 1], got {mf!r}")

    def _check_n_features(self, X):
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but the tree expects {self.n_features_in_}")

    def _add_classes(self, y):
        new = np.unique(y)
        if self.classes_ is not None:
            def is_text(a):
                return a.dtype.kind in "USO"
            if is_text(new) != is_text(self.classes_):
                raise TypeError(f"Labels must keep one type across chunks: got {new.dtype} "
                                f"after {self.classes_.dtype}")
            new = np.union1d(self.classes_, new)
        self.classes_ = new

    # ------------------------------------------------------------- training
    def fit(self, X, y):
        """Train the tree from scratch on X (clears the buffer and history).

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            May contain NaN (see Notes in the class docstring).
        y : array_like, shape (n_samples,)

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If X is empty or shapes do not match.

        Complexity
        ----------
        Time O(depth * n log n * f + depth * n * f * k), space O(n * f * k)
        for n samples, f features tried per node and k classes.
        """
        X, y = check_X_y(X, y)
        if X.shape[0] == 0:
            raise ValueError("Cannot fit a tree on an empty X")
        self._reset_state()
        self.n_features_in_ = X.shape[1]
        self._add_classes(y)
        self._train(X, y)
        return self

    def partial_fit(self, X_chunk, y_chunk):
        """Add one chunk to the buffer; retrain when it is full.

        Parameters
        ----------
        X_chunk : array_like, shape (n_samples, n_features)
            An empty chunk is allowed and does nothing.
        y_chunk : array_like, shape (n_samples,)

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the chunk's feature count differs from earlier chunks.
        TypeError
            If the label type changes (e.g. strings, then numbers).

        Complexity
        ----------
        O(m) to buffer a chunk of m rows; a retrain costs the same as
        :meth:`fit` on ``buffer_count_`` rows. Buffer memory O(buffer_size * d).
        """
        X, y = check_X_y(X_chunk, y_chunk)
        self._check_n_features(X)
        if X.shape[0] == 0:
            return self
        self._add_classes(y)
        self._buffer_X.append(X)
        self._buffer_y.append(y)
        self.buffer_count_ += X.shape[0]
        if self.buffer_count_ >= self.buffer_size:
            self._retrain_from_buffer()
        return self

    def flush(self):
        """Retrain on whatever is in the buffer now (e.g. at the end of a stream).

        Returns
        -------
        self

        Complexity
        ----------
        Same as :meth:`fit` on ``buffer_count_`` rows; O(1) if the buffer is empty.
        """
        if self.buffer_count_:
            self._retrain_from_buffer()
        return self

    def _retrain_from_buffer(self):
        X = np.concatenate(self._buffer_X)
        y = np.concatenate(self._buffer_y)
        self._buffer_X, self._buffer_y = [], []
        self.buffer_count_ = 0
        self._train(X, y)

    def _train(self, X, y):
        start = time.perf_counter()
        # NaN -> per-feature training mean (0 for all-NaN features), without warnings
        mask = ~np.isnan(X)
        cnt = mask.sum(axis=0)
        sums = np.where(mask, X, 0.0).sum(axis=0)
        self.fill_values_ = np.divide(sums, cnt, out=np.zeros(X.shape[1]), where=cnt > 0)
        X = np.where(mask, X, self.fill_values_)
        self.fit_classes_ = np.unique(y)
        codes = np.searchsorted(self.fit_classes_, y)
        self._build(X, codes, self.fit_classes_.size)
        self.last_retrain_time_ = time.perf_counter() - start
        self.total_retrain_time_ += self.last_retrain_time_
        self.n_retrains_ += 1

    def _build(self, X, codes, k):
        """Grow the tree with an explicit stack (no recursion limit)."""
        n_try = self._resolve_max_features(X.shape[1])
        feature, threshold, left, right, value, depth = [], [], [], [], [], []

        def new_node(idx, d):
            feature.append(-1)
            threshold.append(np.nan)
            left.append(-1)
            right.append(-1)
            value.append(np.bincount(codes[idx], minlength=k))
            depth.append(d)
            return len(feature) - 1

        stack = [(new_node(np.arange(X.shape[0]), 0), np.arange(X.shape[0]))]
        while stack:
            node, idx = stack.pop()
            d = depth[node]
            counts = value[node]
            if ((self.max_depth is not None and d >= self.max_depth)
                    or idx.size < self.min_samples_split
                    or np.count_nonzero(counts) <= 1):
                continue
            feats = None
            if n_try < X.shape[1]:
                feats = np.sort(self._rng.choice(X.shape[1], n_try, replace=False))
            f, t, _ = best_split(X[idx], codes[idx], k, self.criterion, self.min_samples_leaf, feats)
            if f is None:
                continue
            go_left = X[idx, f] <= t
            feature[node], threshold[node] = f, t
            li, ri = idx[go_left], idx[~go_left]
            left[node] = new_node(li, d + 1)
            right[node] = new_node(ri, d + 1)
            stack.append((right[node], ri))
            stack.append((left[node], li))
        self.feature_ = np.array(feature, dtype=np.int64)
        self.threshold_ = np.array(threshold, dtype=np.float64)
        self.left_ = np.array(left, dtype=np.int64)
        self.right_ = np.array(right, dtype=np.int64)
        self.value_ = np.array(value, dtype=np.int64)
        self.node_depth_ = np.array(depth, dtype=np.int64)

    # ------------------------------------------------------------ inference
    def _check_is_fitted(self):
        if self.feature_ is None:
            raise NotFittedError(
                f"The tree has not been trained yet ({self.buffer_count_}/{self.buffer_size} "
                f"samples buffered). Call fit(), keep calling partial_fit(), or call flush().")

    def apply(self, X):
        """Index of the leaf each sample ends in.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        leaves : ndarray of int, shape (n_samples,)

        Complexity
        ----------
        Time O(n * depth): all samples move down one level per step.
        """
        self._check_is_fitted()
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but the tree expects {self.n_features_in_}")
        X = np.where(np.isnan(X), self.fill_values_, X)
        node = np.zeros(X.shape[0], dtype=np.int64)
        rows = np.arange(X.shape[0])
        for _ in range(int(self.node_depth_.max()) + 1):
            f = self.feature_[node]
            inner = f >= 0
            if not inner.any():
                break
            r, n = rows[inner], node[inner]
            go_left = X[r, f[inner]] <= self.threshold_[n]
            node[inner] = np.where(go_left, self.left_[n], self.right_[n])
        return node

    def predict_proba(self, X):
        """Class probabilities from leaf class frequencies.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray, shape (n_samples, n_classes)
            Columns follow :attr:`classes_` (all labels seen so far); labels
            absent from the latest training buffer get probability 0.

        Complexity
        ----------
        Time O(n * (depth + n_classes)).
        """
        leaves = self.apply(X)
        counts = self.value_[leaves].astype(np.float64)
        proba = counts / counts.sum(axis=1, keepdims=True)
        out = np.zeros((proba.shape[0], self.classes_.size))
        out[:, np.searchsorted(self.classes_, self.fit_classes_)] = proba
        return out

    def predict(self, X):
        """Predicted class labels (majority class of each leaf).

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)

        Raises
        ------
        NotFittedError
            If the tree has never been trained.

        Complexity
        ----------
        Time O(n * depth).
        """
        leaves = self.apply(X)
        # argmax returns the first maximum -> smallest label on ties
        return self.fit_classes_[np.argmax(self.value_[leaves], axis=1)]

    def score(self, X, y):
        """Accuracy of :meth:`predict` on (X, y).

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
        y : array_like, shape (n_samples,)

        Returns
        -------
        accuracy : float
            NaN for empty input.

        Raises
        ------
        NotFittedError, ValueError
            As :meth:`predict`, or if the lengths differ.

        Complexity
        ----------
        Time O(n * depth).
        """
        y = check_y(y, np.asarray(X).shape[0])
        return float(np.mean(self.predict(X) == y)) if y.size else float("nan")

    # ------------------------------------------------------------- inspect
    def get_depth(self):
        """Depth of the trained tree.

        Returns
        -------
        depth : int
            0 for a single leaf.

        Raises
        ------
        NotFittedError
            If the tree has never been trained.

        Complexity
        ----------
        Time O(n_nodes).
        """
        self._check_is_fitted()
        return int(self.node_depth_.max())

    def get_n_leaves(self):
        """Number of leaves of the trained tree.

        Returns
        -------
        n_leaves : int

        Raises
        ------
        NotFittedError
            If the tree has never been trained.

        Complexity
        ----------
        Time O(n_nodes).
        """
        self._check_is_fitted()
        return int(np.sum(self.feature_ < 0))

    @property
    def is_fitted_(self):
        """True once the tree has been trained at least once."""
        return self.feature_ is not None

    def memory_usage(self):
        """Bytes held by the tree arrays and the data buffer.

        Returns
        -------
        n_bytes : int

        Complexity
        ----------
        Time O(number of stored arrays).
        """
        return memory_usage(self)

    def export_text(self, feature_names=None, decimals=3):
        """Readable text version of the trained tree.

        Parameters
        ----------
        feature_names : list of str or None
        decimals : int, default 3

        Returns
        -------
        text : str

        Raises
        ------
        NotFittedError
            If the tree has never been trained.

        Complexity
        ----------
        Time O(n_nodes).
        """
        self._check_is_fitted()
        names = feature_names or [f"x{j}" for j in range(self.n_features_in_)]
        lines = []

        def walk(node, depth):
            pad = "|   " * depth
            if self.feature_[node] < 0:
                label = self.fit_classes_[np.argmax(self.value_[node])]
                lines.append(f"{pad}|--- class: {label} (n={int(self.value_[node].sum())})")
                return
            name, t = names[self.feature_[node]], round(float(self.threshold_[node]), decimals)
            lines.append(f"{pad}|--- {name} <= {t}")
            walk(self.left_[node], depth + 1)
            lines.append(f"{pad}|--- {name} >  {t}")
            walk(self.right_[node], depth + 1)

        walk(0, 0)
        return "\n".join(lines)
