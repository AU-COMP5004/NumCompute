"""Benchmarks: loop vs vectorised, and the effect of ``buffer_size``.

Run from the repository root::

    python -m numcompute_stream.benchmarking

This writes ``benchmark/loop_vs_vectorised.csv``, ``benchmark/buffer_size.csv``
and ``benchmark/buffer_size_accuracy.png``.

The loop-based functions here are straightforward reference versions of the
vectorised code. Besides timing, the tests use them to check that both
versions give the same answer.
"""

import csv
import os
import time

import numpy as np

from .metrics import confusion_matrix
from .stats import RunningStats
from .stream import StreamTrainer
from .tree import DecisionTreeClassifier, best_split, entropy, gini
from .io import iter_chunks

__all__ = ["make_drift_stream", "write_stream_csv", "best_split_loop", "predict_loop",
           "running_stats_loop", "confusion_matrix_loop", "time_best_of",
           "benchmark_loop_vs_vectorised", "benchmark_buffer_sizes", "main"]


# ------------------------------------------------------------------ dataset
def make_drift_stream(n_samples=6000, n_features=5, drift_at=0.5, label_noise=0.05,
                      nan_rate=0.02, random_state=42):
    """Synthetic 3-class stream with one abrupt concept drift.

    Features are standard normal. Before the drift the class is
    ``(x0 > 0) + (x1 > 0.5)``; after it the class is ``(x2 > 0) + (x3 > -0.5)``.
    Extra features are noise. A fraction of labels is flipped at random and
    a fraction of feature values is set to NaN.

    Parameters
    ----------
    n_samples : int, default 6000
    n_features : int, default 5 (must be >= 4)
    drift_at : float in [0, 1], default 0.5
        Position of the drift as a fraction of the stream.
    label_noise : float, default 0.05
    nan_rate : float, default 0.02
    random_state : int or None

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray of int, shape (n_samples,)

    Raises
    ------
    ValueError
        If ``n_features < 4``.

    Complexity
    ----------
    Time and space O(n_samples * n_features).
    """
    if n_features < 4:
        raise ValueError("n_features must be >= 4")
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_samples, n_features))
    cut = int(drift_at * n_samples)
    y = np.empty(n_samples, dtype=np.int64)
    y[:cut] = (X[:cut, 0] > 0).astype(int) + (X[:cut, 1] > 0.5)
    y[cut:] = (X[cut:, 2] > 0).astype(int) + (X[cut:, 3] > -0.5)
    flip = rng.random(n_samples) < label_noise
    y[flip] = rng.integers(0, 3, flip.sum())
    X[rng.random(X.shape) < nan_rate] = np.nan
    return X, y


def write_stream_csv(path, X, y, decimals=4):
    """Write features and labels to CSV (header ``f1..fd,label``; NaN -> empty cell).

    Parameters
    ----------
    path : str
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)
    decimals : int, default 4

    Returns
    -------
    None

    Complexity
    ----------
    Time O(n_samples * n_features).
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"f{j + 1}" for j in range(X.shape[1])] + ["label"])
        for row, label in zip(X, y):
            w.writerow(["" if np.isnan(v) else f"{v:.{decimals}f}" for v in row] + [int(label)])


# ----------------------------------------------------- loop-based references
def best_split_loop(X, y_codes, n_classes, criterion="gini", min_samples_leaf=1):
    """Reference split search with explicit Python loops (slow, for checks).

    Tries every feature and every midpoint between consecutive distinct
    values, keeping the first strictly larger gain (same tie rule as the
    vectorised :func:`~numcompute_stream.tree.best_split`).

    Returns
    -------
    feature, threshold, gain : as :func:`~numcompute_stream.tree.best_split`

    Complexity
    ----------
    Time O(f * n**2) for n samples and f features.
    """
    impurity = gini if criterion == "gini" else entropy
    n = X.shape[0]
    parent = float(impurity(np.bincount(y_codes, minlength=n_classes)))
    best_f, best_t, best_gain = None, None, 1e-12
    for f in range(X.shape[1]):
        values = np.unique(X[:, f])
        for a, b in zip(values[:-1], values[1:]):
            t = (a + b) / 2.0
            if not a <= t < b:
                t = a
            left_counts = np.zeros(n_classes)
            right_counts = np.zeros(n_classes)
            for i in range(n):
                if X[i, f] <= t:
                    left_counts[y_codes[i]] += 1
                else:
                    right_counts[y_codes[i]] += 1
            nl, nr = left_counts.sum(), right_counts.sum()
            if nl < min_samples_leaf or nr < min_samples_leaf:
                continue
            g = parent - (nl * float(impurity(left_counts)) + nr * float(impurity(right_counts))) / n
            if g > best_gain + 1e-12:
                best_f, best_t, best_gain = f, float(t), g
    return best_f, best_t, (best_gain if best_f is not None else 0.0)


def predict_loop(tree, X):
    """Reference prediction: walk the tree one sample at a time.

    Parameters
    ----------
    tree : fitted DecisionTreeClassifier
    X : ndarray, shape (n_samples, n_features)

    Returns
    -------
    y_pred : ndarray, shape (n_samples,)

    Complexity
    ----------
    Time O(n * depth) with a Python loop per sample.
    """
    X = np.where(np.isnan(X), tree.fill_values_, X)
    out = []
    for x in X:
        node = 0
        while tree.feature_[node] >= 0:
            node = tree.left_[node] if x[tree.feature_[node]] <= tree.threshold_[node] else tree.right_[node]
        out.append(tree.fit_classes_[np.argmax(tree.value_[node])])
    return np.array(out)


def running_stats_loop(X):
    """Reference Welford mean / variance, one sample at a time (NaN skipped).

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)

    Returns
    -------
    mean, var : ndarray, shape (n_features,)

    Complexity
    ----------
    Time O(n * d) with a Python loop per value.
    """
    d = X.shape[1]
    count, mean, m2 = np.zeros(d), np.zeros(d), np.zeros(d)
    for row in X:
        for j in range(d):
            v = row[j]
            if v != v:
                continue
            count[j] += 1
            delta = v - mean[j]
            mean[j] += delta / count[j]
            m2[j] += delta * (v - mean[j])
    return mean, m2 / np.maximum(count, 1)


def confusion_matrix_loop(y_true, y_pred, labels):
    """Reference confusion matrix with a Python loop.

    Parameters
    ----------
    y_true, y_pred : ndarray, shape (n_samples,)
    labels : ndarray, shape (k,)

    Returns
    -------
    cm : ndarray of int64, shape (k, k)

    Complexity
    ----------
    Time O(n) with a Python loop per sample.
    """
    index = {v: i for i, v in enumerate(labels.tolist())}
    cm = np.zeros((len(labels),) * 2, dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[index[t], index[p]] += 1
    return cm


# ----------------------------------------------------------------- timing
def time_best_of(fn, repeats=5):
    """Best wall-clock time of ``fn()`` over ``repeats`` runs.

    Parameters
    ----------
    fn : callable with no arguments
    repeats : int, default 5

    Returns
    -------
    seconds : float

    Complexity
    ----------
    ``repeats`` calls of ``fn``.
    """
    best = np.inf
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def benchmark_loop_vs_vectorised(random_state=0, repeats=5):
    """Time loop-based vs vectorised versions of four core operations.

    Both versions receive the same input and produce the same output (the
    full result, not a partial step). Before timing, their outputs are
    compared; a mismatch raises AssertionError, so a speed-up is only ever
    reported for two implementations that agree.

    Returns
    -------
    rows : list of dict
        Keys: ``operation``, ``n``, ``output``, ``outputs_match``,
        ``loop_ms``, ``vectorised_ms``, ``speedup``.

    Raises
    ------
    AssertionError
        If the loop and vectorised outputs differ.

    Complexity
    ----------
    Dominated by the loop-based split search, O(f * n**2) for n = 300.
    """
    rng = np.random.default_rng(random_state)
    rows = []

    def add(name, n, output, loop_fn, vec_fn, same):
        if not same(loop_fn(), vec_fn()):            # correctness first, then timing
            raise AssertionError(f"{name}: loop and vectorised outputs differ")
        t_loop = time_best_of(loop_fn, repeats)
        t_vec = time_best_of(vec_fn, repeats)
        rows.append({"operation": name, "n": n, "output": output, "outputs_match": True,
                     "loop_ms": 1000 * t_loop, "vectorised_ms": 1000 * t_vec,
                     "speedup": t_loop / t_vec})

    n = 300
    X = np.round(rng.standard_normal((n, 4)), 2)
    y = rng.integers(0, 3, n)
    add("split search (one node)", n, "best (feature, threshold, gain)",
        lambda: best_split_loop(X, y, 3), lambda: best_split(X, y, 3),
        lambda a, b: a[:2] == b[:2] and np.isclose(a[2], b[2]))

    Xs, ys = make_drift_stream(5000, random_state=random_state)
    tree = DecisionTreeClassifier(max_depth=8, buffer_size=10**9).fit(Xs, ys)
    add("tree predict", 5000, "predicted labels", lambda: predict_loop(tree, Xs),
        lambda: tree.predict(Xs), np.array_equal)

    def stats_vectorised():
        rs = RunningStats().update(Xs)
        return rs.mean_, rs.var()
    add("running mean/var", 5000, "per-feature mean and variance (NaN skipped)",
        lambda: running_stats_loop(Xs), stats_vectorised,
        lambda a, b: np.allclose(a[0], b[0]) and np.allclose(a[1], b[1]))

    yp = tree.predict(Xs)
    labels = np.unique(ys)
    add("confusion matrix", 5000, "k x k count matrix",
        lambda: confusion_matrix_loop(ys, yp, labels),
        lambda: confusion_matrix(ys, yp, labels)[0], np.array_equal)
    return rows


def benchmark_buffer_sizes(X, y, buffer_sizes=(100, 250, 500, 1000, 2000), chunk_size=100,
                           max_depth=5):
    """Prequential (test-then-train) accuracy and retraining cost per buffer size.

    Parameters
    ----------
    X, y : stream data, shapes (n_samples, n_features) and (n_samples,)
    buffer_sizes : sequence of int
    chunk_size : int, default 100
    max_depth : int, default 5

    Returns
    -------
    rows : list of dict
        Keys: ``buffer_size``, ``cumulative_accuracy``, ``n_retrains``,
        ``total_retrain_ms``, ``mean_retrain_ms``, ``first_scored_chunk``.
    series : dict of buffer_size -> ndarray of per-chunk accuracy

    Complexity
    ----------
    One full stream pass per buffer size; each retrain costs a tree fit on
    ``buffer_size`` rows.
    """
    rows, series = [], {}
    for b in buffer_sizes:
        tree = DecisionTreeClassifier(max_depth=max_depth, buffer_size=b)
        trainer = StreamTrainer(tree).run(iter_chunks(X, y, chunk_size))
        acc = trainer.history("chunk_accuracy")
        scored = np.flatnonzero(~np.isnan(acc))
        rows.append({
            "buffer_size": b,
            "cumulative_accuracy": trainer.cumulative_accuracy_.result(),
            "n_retrains": tree.n_retrains_,
            "total_retrain_ms": 1000 * tree.total_retrain_time_,
            "mean_retrain_ms": 1000 * tree.total_retrain_time_ / max(tree.n_retrains_, 1),
            "first_scored_chunk": int(scored[0]) if scored.size else -1,
        })
        series[b] = acc
    return rows, series


def _write_rows(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def main(out_dir="benchmark"):
    """Run both benchmarks, write CSV files and a plot, and print the tables.

    Parameters
    ----------
    out_dir : str, default "benchmark"

    Returns
    -------
    loop_rows, buffer_rows : list of dict
        The rows written to ``loop_vs_vectorised.csv`` and ``buffer_size.csv``.

    Complexity
    ----------
    Sum of :func:`benchmark_loop_vs_vectorised` and :func:`benchmark_buffer_sizes`.
    """
    import matplotlib
    matplotlib.use("Agg")
    from .visualise import compare_models

    os.makedirs(out_dir, exist_ok=True)
    rows = benchmark_loop_vs_vectorised()
    _write_rows(rows, os.path.join(out_dir, "loop_vs_vectorised.csv"))
    print("Loop vs vectorised")
    for r in rows:
        print(f"  {r['operation']:26s} n={r['n']:5d}  loop {r['loop_ms']:9.2f} ms"
              f"  vectorised {r['vectorised_ms']:7.2f} ms  x{r['speedup']:.1f}")

    X, y = make_drift_stream()
    brows, series = benchmark_buffer_sizes(X, y)
    _write_rows(brows, os.path.join(out_dir, "buffer_size.csv"))
    print("\nBuffer size (6000 samples, chunks of 100, drift at sample 3000)")
    for r in brows:
        print(f"  buffer {r['buffer_size']:5d}  accuracy {r['cumulative_accuracy']:.3f}"
              f"  retrains {r['n_retrains']:3d}  total retrain {r['total_retrain_ms']:8.1f} ms"
              f"  mean {r['mean_retrain_ms']:6.2f} ms")
    small, large = min(series), max(series)
    compare_models(series[small], series[large],
                   labels=(f"buffer_size={small}", f"buffer_size={large}"),
                   title="Per-chunk accuracy (drift at chunk 30)",
                   save_path=os.path.join(out_dir, "buffer_size_accuracy.png"))
    return rows, brows


if __name__ == "__main__":
    main()
