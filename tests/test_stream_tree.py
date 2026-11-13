"""Tests for numcompute_stream.tree (DecisionTreeClassifier)."""

import numpy as np
import pytest

from numcompute_stream import DecisionTreeClassifier, NotFittedError
from numcompute_stream.benchmarking import best_split_loop, make_drift_stream, predict_loop
from numcompute_stream.tree import best_split, entropy, gini


# ---------------------------------------------------------------- impurity
def test_impurity_values():
    assert gini([5, 5]) == pytest.approx(0.5)
    assert entropy([5, 5]) == pytest.approx(1.0)
    assert gini([4, 0]) == 0.0 and entropy([4, 0]) == 0.0
    assert gini([0, 0]) == 0.0 and entropy([0, 0]) == 0.0          # empty group


def test_course_example_yes_no_features():
    """0/1 features as in the course guide: the tree finds the deciding feature."""
    X = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 0, 0], [1, 1, 1], [0, 1, 0]])
    y = np.array([1, 1, 0, 0, 1, 0])                              # y == feature 0
    tree = DecisionTreeClassifier(max_depth=2).fit(X, y)
    assert tree.feature_[0] == 0 and tree.threshold_[0] == 0.5
    assert np.array_equal(tree.predict(X), y)


# -------------------------------------------------------------- split search
def test_vectorised_split_matches_loop_reference():
    rng = np.random.default_rng(0)
    for criterion in ("gini", "entropy"):
        for _ in range(40):
            n, d, k = rng.integers(2, 40), rng.integers(1, 4), rng.integers(2, 4)
            X = np.round(rng.standard_normal((n, d)), 1)            # many ties in X
            y = rng.integers(0, k, n)
            leaf = int(rng.integers(1, 3))
            f1, t1, g1 = best_split(X, y, k, criterion, leaf)
            f2, t2, g2 = best_split_loop(X, y, k, criterion, leaf)
            assert (f1, t1) == (f2, t2)
            assert g1 == pytest.approx(g2)


def test_split_tie_prefers_lowest_feature():
    X = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=float)  # both features identical
    y = np.array([0, 0, 1, 1])
    f, t, _ = best_split(X, y, 2)
    assert f == 0 and t == 0.5


def test_no_split_when_no_gain_or_constant_feature():
    X = np.ones((6, 2))                                             # zero-variance features
    y = np.array([0, 1, 0, 1, 0, 1])
    assert best_split(X, y, 2) == (None, None, 0.0)
    tree = DecisionTreeClassifier().fit(X, y)
    assert tree.get_n_leaves() == 1
    assert np.all(tree.predict(X) == 0)                             # tie -> smallest label


def test_threshold_between_adjacent_floats_is_valid():
    a = 1.0
    b = np.nextafter(a, 2.0)                                        # midpoint rounds onto b
    X = np.array([[a], [b]])
    tree = DecisionTreeClassifier().fit(X, [0, 1])
    assert np.array_equal(tree.predict(X), [0, 1])


# ------------------------------------------------------------ stop rules
def test_max_depth_and_min_samples():
    X, y = make_drift_stream(800, random_state=1)
    assert DecisionTreeClassifier(max_depth=0).fit(X, y).get_n_leaves() == 1
    assert DecisionTreeClassifier(max_depth=3).fit(X, y).get_depth() <= 3
    tree = DecisionTreeClassifier(max_depth=None, min_samples_leaf=20).fit(X, y)
    leaves = tree.feature_ < 0
    assert tree.value_[leaves].sum(axis=1).min() >= 20


def test_pure_node_becomes_leaf():
    tree = DecisionTreeClassifier().fit([[1.0], [2.0], [3.0]], [7, 7, 7])
    assert tree.get_n_leaves() == 1 and np.all(tree.predict([[0.0], [9.0]]) == 7)


def test_fits_training_data_without_depth_limit():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((200, 3))
    y = rng.integers(0, 3, 200)
    tree = DecisionTreeClassifier(max_depth=None).fit(X, y)
    assert tree.score(X, y) == 1.0                                  # distinct rows -> memorised


# ------------------------------------------------------------- labels & NaN
def test_string_labels_and_proba():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array(["cat", "cat", "dog", "dog"])
    tree = DecisionTreeClassifier().fit(X, y)
    assert list(tree.predict([[0.5], [2.5]])) == ["cat", "dog"]
    proba = tree.predict_proba(X)
    assert proba.shape == (4, 2) and np.allclose(proba.sum(axis=1), 1.0)


def test_nan_features_are_imputed_with_training_mean():
    X = np.array([[0.0], [np.nan], [10.0], [12.0]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTreeClassifier().fit(X, y)
    assert tree.fill_values_[0] == pytest.approx(22 / 3)
    # NaN is routed exactly like the training mean (7.33 sits with the class-0 rows)
    assert tree.predict([[np.nan]])[0] == tree.predict([[22 / 3]])[0] == 0


def test_all_nan_feature_does_not_crash():
    X = np.column_stack([np.full(6, np.nan), np.arange(6.0)])
    y = np.array([0, 0, 0, 1, 1, 1])
    tree = DecisionTreeClassifier().fit(X, y)
    assert tree.feature_[0] == 1 and tree.score(X, y) == 1.0


def test_max_features_is_reproducible():
    X, y = make_drift_stream(500, random_state=2)
    a = DecisionTreeClassifier(max_features="sqrt", random_state=5).fit(X, y)
    b = DecisionTreeClassifier(max_features="sqrt", random_state=5).fit(X, y)
    assert np.array_equal(a.feature_, b.feature_)
    assert np.array_equal(a.threshold_, b.threshold_, equal_nan=True)   # leaves hold NaN


# ----------------------------------------------------------- buffering
def test_partial_fit_buffers_until_buffer_size():
    X, y = make_drift_stream(1000, random_state=0)
    tree = DecisionTreeClassifier(buffer_size=250)
    tree.partial_fit(X[:100], y[:100])
    tree.partial_fit(X[100:200], y[100:200])
    assert tree.n_retrains_ == 0 and tree.buffer_count_ == 200
    with pytest.raises(NotFittedError, match="200/250"):
        tree.predict(X[:5])
    tree.partial_fit(X[200:300], y[200:300])                        # 300 >= 250 -> retrain
    assert tree.n_retrains_ == 1 and tree.buffer_count_ == 0
    assert int(tree.value_[0].sum()) == 300                         # trained on whole buffer


def test_predict_uses_last_tree_between_retrains():
    X, y = make_drift_stream(1000, random_state=0)
    tree = DecisionTreeClassifier(buffer_size=300)
    tree.partial_fit(X[:300], y[:300])
    before = tree.predict(X)
    tree.partial_fit(X[300:400], y[300:400])                        # buffered only
    assert np.array_equal(tree.predict(X), before)


def test_retrain_matches_batch_fit_on_buffer():
    X, y = make_drift_stream(1000, random_state=4)
    stream = DecisionTreeClassifier(max_depth=4, buffer_size=400)
    for start in range(0, 800, 100):
        stream.partial_fit(X[start:start + 100], y[start:start + 100])
    batch = DecisionTreeClassifier(max_depth=4).fit(X[400:800], y[400:800])
    assert stream.n_retrains_ == 2
    assert np.array_equal(stream.predict(X), batch.predict(X))


def test_chunk_of_one_sample_and_empty_chunk():
    tree = DecisionTreeClassifier(buffer_size=3)
    tree.partial_fit(np.empty((0, 2)), np.empty(0))                 # no-op
    for row, label in [([0.0, 1.0], 0), ([1.0, 0.0], 1), ([2.0, 0.0], 1)]:
        tree.partial_fit([row], [label])
    assert tree.n_retrains_ == 1
    assert list(tree.predict([[0.0, 1.0], [2.0, 0.0]])) == [0, 1]


def test_flush_trains_on_partial_buffer():
    tree = DecisionTreeClassifier(buffer_size=1000)
    tree.partial_fit([[0.0], [1.0]], [0, 1]).flush()
    assert tree.n_retrains_ == 1 and tree.buffer_count_ == 0


def test_new_class_after_retrain_keeps_proba_columns():
    tree = DecisionTreeClassifier(buffer_size=2)
    tree.partial_fit([[0.0], [1.0]], [0, 1])
    tree.partial_fit([[5.0]], [2])                                  # buffered, not trained
    proba = tree.predict_proba([[0.0]])
    assert list(tree.classes_) == [0, 1, 2]
    assert proba.shape == (1, 3) and proba[0, 2] == 0.0


def test_predict_matches_loop_reference():
    X, y = make_drift_stream(2000, random_state=6)
    tree = DecisionTreeClassifier(max_depth=7).fit(X, y)
    assert np.array_equal(tree.predict(X), predict_loop(tree, X))


# ----------------------------------------------------------------- errors
@pytest.mark.parametrize("kwargs", [
    {"criterion": "mse"}, {"max_depth": -1}, {"min_samples_split": 1},
    {"min_samples_leaf": 0}, {"buffer_size": 0}, {"max_features": "half"},
])
def test_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        DecisionTreeClassifier(**kwargs)


def test_shape_and_type_errors():
    tree = DecisionTreeClassifier(buffer_size=2).fit([[0.0, 1.0], [1.0, 0.0]], [0, 1])
    with pytest.raises(ValueError, match="features"):
        tree.predict([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="2-D"):
        tree.predict([1.0, 2.0])
    with pytest.raises(ValueError, match="samples"):
        DecisionTreeClassifier().fit([[1.0], [2.0]], [0])
    with pytest.raises(ValueError, match="NaN"):
        DecisionTreeClassifier().fit([[1.0], [2.0]], [0, np.nan])
    with pytest.raises(TypeError):
        DecisionTreeClassifier().fit([["a"], ["b"]], [0, 1])
    with pytest.raises(ValueError, match="empty"):
        DecisionTreeClassifier().fit(np.empty((0, 2)), [])
