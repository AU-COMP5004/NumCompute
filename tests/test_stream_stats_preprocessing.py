"""Tests for numcompute_stream.stats and numcompute_stream.preprocessing."""

import numpy as np
import pytest

from numcompute_stream import (MinMaxScaler, NotFittedError, OneHotEncoder, RunningStats,
                               SimpleImputer, StandardScaler, StreamingStats, iter_chunks,
                               update_stats)
from numcompute_stream.benchmarking import running_stats_loop


def chunks_of(X, sizes):
    start = 0
    for s in sizes:
        yield X[start:start + s]
        start += s


# ------------------------------------------------------------- RunningStats
@pytest.mark.parametrize("sizes", [[100], [1] * 100, [7, 0, 50, 43], [33, 33, 34]])
def test_running_stats_chunked_equals_batch(sizes):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 3)) * [1, 100, 1e-3] + [0, 1e6, -5]   # large offset
    X[rng.random(X.shape) < 0.1] = np.nan
    rs = RunningStats()
    for chunk in chunks_of(X, sizes):
        rs.update(chunk)
    assert rs.n_samples_seen_ == 100
    assert np.allclose(rs.mean_, np.nanmean(X, axis=0), rtol=1e-12)
    assert np.allclose(rs.var(), np.nanvar(X, axis=0), rtol=1e-9)
    assert np.allclose(rs.var(ddof=1), np.nanvar(X, axis=0, ddof=1), rtol=1e-9)
    assert np.array_equal(rs.min_, np.nanmin(X, axis=0))


def test_running_stats_matches_loop_welford():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 4))
    X[rng.random(X.shape) < 0.05] = np.nan
    mean, var = running_stats_loop(X)
    rs = RunningStats()
    for c in iter_chunks(X, chunk_size=64):
        rs.update(c)
    assert np.allclose(rs.mean_, mean) and np.allclose(rs.var(), var)


def test_running_stats_all_nan_and_constant_columns():
    X = np.column_stack([np.full(5, np.nan), np.full(5, 3.0)])
    rs = RunningStats().update(X)
    assert np.isnan(rs.mean_[0]) and rs.count_[0] == 0
    assert rs.mean_[1] == 3.0 and rs.var()[1] == 0.0
    assert np.isnan(rs.var(ddof=1)[0])


def test_running_stats_errors():
    rs = RunningStats()
    with pytest.raises(ValueError, match="No data"):
        rs.var()
    rs.update(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="features"):
        rs.update(np.zeros((2, 2)))


# ----------------------------------------------------------- StreamingStats
def test_quantiles_exact_until_reservoir_full():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((200, 2))
    s = StreamingStats(reservoir_size=500)
    for c in iter_chunks(X, chunk_size=30):
        s.update_stats(c)
    assert np.allclose(s.quantile([0.1, 0.5, 0.9]), np.quantile(X, [0.1, 0.5, 0.9], axis=0))


def test_quantiles_estimate_after_reservoir_full():
    rng = np.random.default_rng(3)
    X = rng.uniform(0, 1, (20000, 1))
    s = StreamingStats(reservoir_size=2000, random_state=0)
    for c in iter_chunks(X, chunk_size=500):
        s.update_stats(c)
    assert s.quantile(0.5)[0] == pytest.approx(0.5, abs=0.05)
    assert s._reservoir.shape == (2000, 1)                          # memory stays bounded


def test_reservoir_is_uniform_over_stream():
    """Late rows must be as likely to be kept as early rows (Algorithm R)."""
    X = np.arange(10000, dtype=float)[:, None]
    s = StreamingStats(reservoir_size=1000, random_state=1)
    for c in iter_chunks(X, chunk_size=250):
        s.update_stats(c)
    kept = s._reservoir[:, 0]
    assert np.unique(kept).size == 1000
    assert np.mean(kept >= 5000) == pytest.approx(0.5, abs=0.06)


def test_histogram_counts_window_and_overflow():
    s = StreamingStats(bins=4, hist_range=(0, 4), window=2)
    s.update_stats(np.array([[0.5], [1.5]]))
    s.update_stats(np.array([[2.5], [4.0], [9.0]]))                 # 4.0 -> last bin, 9 -> overflow
    s.update_stats(np.array([[-1.0], [3.5]]))                        # first chunk leaves the window
    counts, edges, under, over = s.histogram(0)
    assert list(counts) == [0, 0, 1, 2] and under == 1 and over == 1
    assert np.allclose(edges, [0, 1, 2, 3, 4])


def test_update_stats_functional_api_and_nan():
    state = None
    X = np.array([[1.0, np.nan], [3.0, 2.0], [np.nan, 4.0]])
    for c in iter_chunks(X, chunk_size=1):
        state = update_stats(c, state)
    assert np.allclose(state.mean(), [2.0, 3.0])
    assert np.allclose(state.quantile(0.5), [2.0, 3.0])


def test_streaming_stats_errors():
    with pytest.raises(ValueError):
        StreamingStats(reservoir_size=0)
    with pytest.raises(ValueError):
        StreamingStats(hist_range=(2, 1))
    s = StreamingStats()
    with pytest.raises(ValueError, match="No data"):
        s.quantile(0.5)
    s.update_stats(np.ones((3, 1)))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        s.quantile(1.5)


# ---------------------------------------------------------- StandardScaler
def test_standard_scaler_partial_fit_equals_fit():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((120, 3)) * 5 + 10
    batch = StandardScaler().fit(X)
    stream = StandardScaler()
    for c in iter_chunks(X, chunk_size=17):
        stream.partial_fit(c)
    assert np.allclose(stream.mean_, batch.mean_) and np.allclose(stream.scale_, batch.scale_)
    out = stream.transform(X)
    assert np.allclose(out.mean(axis=0), 0) and np.allclose(out.std(axis=0), 1)
    assert np.allclose(stream.inverse_transform(out), X)


def test_standard_scaler_zero_variance_chunk_and_nan():
    sc = StandardScaler().partial_fit(np.array([[5.0, np.nan], [5.0, np.nan]]))
    assert np.array_equal(sc.scale_, [1.0, 1.0])                     # no division by zero
    out = sc.transform(np.array([[5.0, np.nan], [6.0, 1.0]]))
    assert out[0, 0] == 0.0 and np.isnan(out[0, 1]) and out[1, 1] == 1.0


def test_scaler_not_fitted_and_shape_errors():
    with pytest.raises(NotFittedError):
        StandardScaler().transform(np.ones((2, 2)))
    sc = StandardScaler().partial_fit(np.ones((2, 2)))
    with pytest.raises(ValueError, match="features"):
        sc.partial_fit(np.ones((2, 3)))
    with pytest.raises(ValueError, match="2-D"):
        sc.transform(np.ones(2))


# ------------------------------------------------------------ MinMaxScaler
def test_minmax_scaler_streaming_range():
    mm = MinMaxScaler(feature_range=(-1, 1))
    mm.partial_fit(np.array([[0.0], [5.0]])).partial_fit(np.array([[10.0]]))
    assert np.allclose(mm.transform(np.array([[0.0], [5.0], [10.0]])).ravel(), [-1, 0, 1])
    const = MinMaxScaler().fit(np.full((3, 1), 7.0))
    assert np.allclose(const.transform(np.full((2, 1), 7.0)), 0.0)
    with pytest.raises(ValueError):
        MinMaxScaler(feature_range=(1, 0))


# ------------------------------------------------------------ SimpleImputer
def test_imputer_mean_updates_on_the_fly():
    imp = SimpleImputer()
    imp.partial_fit(np.array([[1.0], [np.nan]]))
    assert imp.transform(np.array([[np.nan]]))[0, 0] == 1.0
    imp.partial_fit(np.array([[5.0]]))
    assert imp.transform(np.array([[np.nan]]))[0, 0] == 3.0


@pytest.mark.parametrize("strategy, expected", [
    ("median", [1.5, 10.0]), ("constant", [-1.0, -1.0]),
])
def test_imputer_strategies(strategy, expected):
    X = np.array([[1.0, 10.0], [1.0, np.nan], [2.0, 10.0], [7.0, 30.0], [np.nan, 10.0]])
    imp = SimpleImputer(strategy=strategy, fill_value=-1.0)
    for c in iter_chunks(X, chunk_size=2):
        imp.partial_fit(c)
    assert np.allclose(imp.statistics_, expected)
    assert not np.isnan(imp.transform(X)).any()


def test_imputer_column_never_observed_uses_fill_value():
    imp = SimpleImputer(fill_value=0.0).fit(np.array([[np.nan, 1.0], [np.nan, 3.0]]))
    assert np.array_equal(imp.statistics_, [0.0, 2.0])
    with pytest.raises(ValueError):
        SimpleImputer(strategy="most_frequent")


# ------------------------------------------------------------ OneHotEncoder
def test_one_hot_encoder_grows_categories():
    enc = OneHotEncoder()
    enc.partial_fit(np.array([[1.0], [2.0]]))
    assert enc.transform(np.array([[2.0]])).tolist() == [[0.0, 1.0]]
    enc.partial_fit(np.array([[0.0], [np.nan]]))                     # new category, NaN ignored
    assert [list(c) for c in enc.categories_] == [[0.0, 1.0, 2.0]]
    out = enc.transform(np.array([[0.0], [5.0], [np.nan]]))
    assert out.tolist() == [[1, 0, 0], [0, 0, 0], [0, 0, 0]]         # unknown / NaN -> zeros
    assert enc.get_feature_names_out() == ["x0_0", "x0_1", "x0_2"]


def test_one_hot_encoder_fixed_categories_and_unknown_error():
    enc = OneHotEncoder(categories=[[0, 1]], handle_unknown="error")
    enc.partial_fit(np.array([[3.0]]))                               # fixed: no growth
    assert enc.transform(np.array([[1.0]])).tolist() == [[0.0, 1.0]]
    with pytest.raises(ValueError, match="Unknown category"):
        enc.transform(np.array([[3.0]]))
