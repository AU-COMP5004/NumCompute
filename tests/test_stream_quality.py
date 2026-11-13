"""Cross-cutting checks: documentation contracts, input validation and
regressions for the failure types reported on Assignment 2.1."""

import importlib
import inspect

import numpy as np
import pytest

import numcompute_stream as ns
from numcompute_stream.benchmarking import make_drift_stream
from numcompute_stream.metrics import confusion_matrix

MODULES = ["io", "stats", "preprocessing", "metrics", "tree", "pipeline", "stream",
           "visualise", "benchmarking", "utils"]


def _public_callables():
    for m in MODULES:
        mod = importlib.import_module(f"numcompute_stream.{m}")
        for name in mod.__all__:
            obj = getattr(mod, name)
            if inspect.isclass(obj) and issubclass(obj, Exception):
                continue
            yield f"{m}.{name}", obj, False
            if inspect.isclass(obj):
                for attr, raw in vars(obj).items():
                    if attr.startswith("_"):
                        continue
                    if isinstance(raw, property):
                        yield f"{m}.{name}.{attr}", raw.fget, True
                    elif inspect.isfunction(raw):
                        yield f"{m}.{name}.{attr}", raw, False


# ------------------------------------------------------------ documentation
def test_every_public_api_documents_returns_and_complexity():
    missing = []
    for qual, obj, is_prop in _public_callables():
        doc = inspect.getdoc(obj) or ""
        if not doc:
            missing.append(f"{qual}: no docstring")
        elif not inspect.isclass(obj) and not is_prop:
            if "Complexity" not in doc:
                missing.append(f"{qual}: no Complexity section")
            if "Returns" not in doc and "Yields" not in doc:
                missing.append(f"{qual}: no Returns section")
    assert not missing, "\n".join(missing)


# --------------------------------------------- 2.1 regression: feature counts
@pytest.mark.parametrize("make", [
    lambda: ns.StandardScaler(), lambda: ns.MinMaxScaler(), lambda: ns.SimpleImputer(),
    lambda: ns.OneHotEncoder(),
])
def test_transformers_reject_feature_count_mismatch(make):
    est = make().fit(np.ones((4, 3)))
    with pytest.raises(ValueError, match="features"):
        est.transform(np.ones((2, 1)))                     # would broadcast silently
    with pytest.raises(ValueError, match="features"):
        est.partial_fit(np.ones((2, 2)))


def test_pipeline_and_tree_reject_feature_count_mismatch():
    X, y = make_drift_stream(300, random_state=0)
    pipe = ns.Pipeline([("scale", ns.StandardScaler()), ("tree", ns.DecisionTreeClassifier())])
    pipe.fit(X, y)
    with pytest.raises(ValueError, match="features"):
        pipe.predict(X[:, :2])
    with pytest.raises(ValueError, match="features"):
        ns.DecisionTreeClassifier(buffer_size=10).partial_fit(X[:5], y[:5]).partial_fit(X[:5, :3], y[:5])


# -------------------------------------------------- 2.1 regression: integers
@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int64])
def test_integer_inputs_do_not_overflow(dtype):
    info = np.iinfo(dtype)
    X = np.array([[info.min], [info.max]], dtype=dtype)
    assert np.allclose(ns.MinMaxScaler().fit_transform(X).ravel(), [0, 1])
    assert np.allclose(ns.StandardScaler().fit_transform(X).ravel(), [-1, 1])
    rs = ns.RunningStats().update(X)
    assert rs.mean_[0] == pytest.approx((float(info.min) + float(info.max)) / 2)
    assert rs.var()[0] > 0


def test_large_integer_labels_and_features_in_tree():
    X = np.array([[2**40], [2**40 + 10], [2**40 + 20], [2**40 + 30]])
    y = np.array([2**62, 2**62, 7, 7])
    tree = ns.DecisionTreeClassifier().fit(X, y)
    assert tree.predict(X).tolist() == y.tolist()


# --------------------------------------------- metrics: windows, AUC, guards
def test_rolling_windows_match_brute_force():
    rng = np.random.default_rng(3)
    ys = [rng.integers(0, 2 + i // 3, 30) for i in range(12)]          # labels appear late
    ps = [np.where(rng.random(30) < 0.6, y, rng.integers(0, 4, 30)) for y in ys]
    for w in (1, 3, 5):
        cm = ns.StreamingConfusionMatrix(window=w)
        for y, p in zip(ys, ps):
            cm.update(y, p)
        ref, ref_labels = confusion_matrix(np.concatenate(ys[-w:]), np.concatenate(ps[-w:]))
        got, labels = cm.result()
        keep = np.isin(labels, ref_labels)
        assert np.array_equal(got[np.ix_(keep, keep)], ref) and got[~keep].sum() == 0


def test_streaming_auc_is_exact_for_continuous_scores():
    rng = np.random.default_rng(4)
    ys = [rng.integers(0, 2, 40) for _ in range(8)]
    ss = [rng.random(40) for _ in range(8)]                            # all scores distinct
    auc = ns.StreamingAUC()
    for y, s in zip(ys, ss):
        auc.update(y, s)
    assert auc.result() == pytest.approx(
        ns.roc_auc_score(np.concatenate(ys), np.concatenate(ss)), abs=1e-12)


def test_binary_average_rejects_multiclass():
    with pytest.raises(ValueError, match="binary"):
        ns.precision_recall_f1([0, 1, 2], [0, 1, 2])
    f1 = ns.StreamingF1().update([0, 1, 2], [0, 1, 1])
    with pytest.raises(ValueError, match="macro"):
        f1.result()


# ------------------------------------------- one-hot encoder vs reference
def test_streaming_one_hot_encoder_matches_reference():
    rng = np.random.default_rng(5)
    X = rng.integers(-2, 3, (80, 4)).astype(float)
    X[rng.random(X.shape) < 0.15] = np.nan
    enc = ns.OneHotEncoder()
    for chunk in np.array_split(X, 6):
        enc.partial_fit(chunk)
    Z = rng.integers(-4, 5, (30, 4)).astype(float)                     # includes unseen values
    cats = [np.unique(X[~np.isnan(X[:, j]), j]) for j in range(4)]
    ref = np.hstack([(Z[:, [j]] == cats[j]).astype(float) for j in range(4)])
    assert np.array_equal(enc.transform(Z), ref)


# ----------------------------------------------- pipeline y-passing protocol
class _NoYScaler:
    """Transformer whose methods take X only."""

    def partial_fit(self, X):
        self.seen = getattr(self, "seen", 0) + len(X)
        return self

    def fit(self, X):
        self.seen = len(X)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) * 2


def test_pipeline_works_with_transformers_that_take_no_y():
    X, y = make_drift_stream(600, random_state=1)
    pipe = ns.Pipeline([("double", _NoYScaler()), ("impute", ns.SimpleImputer()),
                        ("tree", ns.DecisionTreeClassifier(buffer_size=300))])
    for Xc, yc in ns.iter_chunks(X, y, 100):
        pipe.partial_fit(Xc, yc)
    assert pipe.named_steps["double"].seen == 600
    assert pipe.named_steps["tree"].n_retrains_ == 2
    assert pipe.fit(X, y).score(X, y) > 0.5


# ------------------------------------------- regressions from the edge audit
def test_csv_rejects_non_numeric_cells_instead_of_nan(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,x\n")
    with pytest.raises(ValueError, match="non-numeric"):
        ns.load_csv(str(path))
    with pytest.raises(ValueError, match="non-numeric"):
        list(ns.stream_csv(str(path), 1))


def test_csv_header_only_and_all_labels_missing(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("a,b,label\n")
    data, header = ns.load_csv(str(empty))
    assert data.shape == (0, 3) and header == ["a", "b", "label"]
    X, y, _ = ns.load_csv_xy(str(empty))
    assert X.shape == (0, 2) and y.shape == (0,)
    nolabel = tmp_path / "nolabel.csv"
    nolabel.write_text("a,b\n1,\n2,\n")                                   # one-character cells
    assert [c[0].shape for c in ns.stream_csv(str(nolabel), 5)] == [(0, 1)]


@pytest.mark.parametrize("make", [
    lambda: ns.StreamingStats(bins=2.5), lambda: ns.StreamingStats(window=True),
    lambda: ns.StreamingStats(reservoir_size=1.5), lambda: ns.StreamingAccuracy(window=0),
    lambda: ns.DecisionTreeClassifier(buffer_size=True),
    lambda: list(ns.iter_chunks(np.ones((2, 1)), chunk_size=2.0)),
])
def test_integer_hyperparameters_are_validated(make):
    with pytest.raises(ValueError, match="positive int"):
        make()


def test_label_type_cannot_change_mid_stream():
    tree = ns.DecisionTreeClassifier(buffer_size=10).partial_fit([[0.0]], ["a"])
    with pytest.raises(TypeError, match="one type"):
        tree.partial_fit([[1.0]], [1])


def test_stats_query_errors_are_clear():
    s = ns.StreamingStats().update_stats(np.ones((3, 2)))
    with pytest.raises(ValueError, match="1-D"):
        s.quantile([[0.5]])
    with pytest.raises(ValueError, match="feature"):
        s.histogram(5)


def test_split_search_blocks_keep_tie_rule(monkeypatch):
    """Feature blocks (used to bound memory) must give the same split as one block."""
    from numcompute_stream import tree as tree_mod
    from numcompute_stream.benchmarking import best_split_loop
    monkeypatch.setattr(tree_mod, "_BLOCK_CELLS", 1)                    # one feature per block
    rng = np.random.default_rng(6)
    for _ in range(60):
        n, d, k = rng.integers(2, 30), rng.integers(1, 5), rng.integers(2, 4)
        X = np.round(rng.standard_normal((n, d)), 1)
        y = rng.integers(0, k, n)
        assert tree_mod.best_split(X, y, k)[:2] == best_split_loop(X, y, k)[:2]


def test_empty_confusion_matrix_plot_is_rejected():
    from numcompute_stream.visualise import plot_confusion_matrix
    with pytest.raises(ValueError, match="non-empty"):
        plot_confusion_matrix(np.zeros((0, 0)))
