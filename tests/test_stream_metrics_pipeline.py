"""Tests for metrics, pipeline, stream trainer, io and visualise."""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from numcompute_stream import (DecisionTreeClassifier, NotFittedError, Pipeline,  # noqa: E402
                               SimpleImputer, StandardScaler, StreamingAccuracy, StreamingAUC,
                               StreamingConfusionMatrix, StreamingF1, StreamingPrecision,
                               StreamingRecall, StreamTrainer, accuracy_score, confusion_matrix,
                               iter_chunks, load_csv, load_csv_xy, precision_recall_f1,
                               roc_auc_score, stream_csv)
from numcompute_stream.benchmarking import (benchmark_buffer_sizes,  # noqa: E402
                                            confusion_matrix_loop, make_drift_stream,
                                            write_stream_csv)
from numcompute_stream.visualise import (compare_models, plot_confusion_matrix,  # noqa: E402
                                         plot_metric_over_time,
                                         plot_predictions_vs_ground_truth)


# ------------------------------------------------------------------ metrics
def test_streaming_metrics_equal_batch_metrics():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 500)
    p = np.where(rng.random(500) < 0.7, y, rng.integers(0, 3, 500))
    acc, cm = StreamingAccuracy(), StreamingConfusionMatrix()
    prf = {avg: (StreamingPrecision(avg), StreamingRecall(avg), StreamingF1(avg))
           for avg in ("macro", "micro", "weighted")}
    for (yt, yp) in zip(np.array_split(y, 7), np.array_split(p, 7)):
        acc.update(yt, yp)
        cm.update(yt, yp)
        for ms in prf.values():
            for m in ms:
                m.update(yt, yp)
    assert acc.result() == pytest.approx(accuracy_score(y, p))
    assert np.array_equal(cm.result()[0], confusion_matrix(y, p)[0])
    for avg, ms in prf.items():
        assert [m.result() for m in ms] == pytest.approx(precision_recall_f1(y, p, average=avg))


def test_confusion_matrix_matches_loop_and_grows_labels():
    cm = StreamingConfusionMatrix()
    cm.update([2, 2], [2, 0])
    cm.update([1, 0], [1, 1])                                        # label 1 appears later
    mat, labels = cm.result()
    assert list(labels) == [0, 1, 2]
    assert np.array_equal(mat, confusion_matrix_loop(np.array([2, 2, 1, 0]),
                                                     np.array([2, 0, 1, 1]), labels))


def test_rolling_window_metric():
    acc = StreamingAccuracy(window=2)
    acc.update([1, 1], [0, 0])                                       # 0 % (drops out)
    acc.update([1, 1], [1, 1])
    acc.update([1, 1], [1, 0])
    assert acc.result() == pytest.approx(0.75)


def test_binary_precision_recall_and_zero_division():
    y, p = [1, 0, 1, 1], [1, 1, 0, 1]
    assert precision_recall_f1(y, p) == pytest.approx((2 / 3, 2 / 3, 2 / 3))
    assert precision_recall_f1([0, 0], [0, 0], pos_label=1) == (0.0, 0.0, 0.0)
    assert np.isnan(StreamingF1().result())                          # nothing seen yet


def test_string_labels_in_metrics():
    acc = StreamingAccuracy().update(["a", "b"], ["a", "a"])
    assert acc.result() == 0.5


def test_auc_streaming_matches_exact_and_handles_ties():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 1000)
    s = np.round(np.clip(y * 0.3 + rng.random(1000) * 0.7, 0, 1), 2)  # tied scores
    auc = StreamingAUC()
    for yt, st in zip(np.array_split(y, 9), np.array_split(s, 9)):
        auc.update(yt, st)
    assert auc.result() == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert roc_auc_score([0, 1], [0.5, 0.5]) == 0.5
    assert np.isnan(roc_auc_score([1, 1], [0.2, 0.9]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        auc.update([1], [1.5])


def test_metric_reset_and_length_mismatch():
    acc = StreamingAccuracy().update([1], [1])
    assert np.isnan(acc.reset().result())
    with pytest.raises(ValueError, match="samples"):
        acc.update([1, 0], [1])


# ----------------------------------------------------------------- pipeline
def test_pipeline_partial_fit_chain():
    X, y = make_drift_stream(1200, random_state=2)
    pipe = Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler()),
                     ("model", DecisionTreeClassifier(max_depth=4, buffer_size=500))])
    for Xc, yc in iter_chunks(X, y, 100):
        pipe.partial_fit(Xc, yc)
    assert pipe.named_steps["model"].n_retrains_ == 2
    assert pipe.named_steps["scale"].n_samples_seen_ == 1200
    pred = pipe.predict(X)
    assert pred.shape == (1200,) and pipe.predict_proba(X).shape == (1200, 3)
    assert 0 <= pipe.score(X, y) <= 1


def test_pipeline_fit_and_transform_only():
    X = np.array([[1.0, np.nan], [3.0, 4.0]])
    pipe = Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())])
    out = pipe.fit(X).transform(X)
    assert np.allclose(out, [[-1, 0], [1, 0]])


def test_pipeline_validation():
    with pytest.raises(ValueError):
        Pipeline([])
    with pytest.raises(ValueError, match="unique"):
        Pipeline([("a", StandardScaler()), ("a", StandardScaler())])
    with pytest.raises(TypeError):
        Pipeline([("bad", object()), ("model", DecisionTreeClassifier())])


# ------------------------------------------------------------ StreamTrainer
def test_stream_trainer_prequential_log():
    X, y = make_drift_stream(1000, random_state=3)
    tree = DecisionTreeClassifier(max_depth=4, buffer_size=300)
    trainer = StreamTrainer(tree, metrics={"f1": StreamingF1(average="macro")}, window=3)
    trainer.run(iter_chunks(X, y, 100))
    log = trainer.log_
    assert len(log) == 10
    assert np.isnan(log[0]["chunk_accuracy"])                        # not trained yet
    assert [r["retrained"] for r in log].count(True) == tree.n_retrains_ == 3
    assert log[2]["retrained"] and log[2]["buffer_fill"] == 0
    assert all(r["memory_bytes"] > 0 for r in log)
    acc = trainer.history("chunk_accuracy")
    assert np.nanmax(acc) <= 1 and np.isnan(acc[:3]).all()           # first tree after chunk 2
    s = trainer.summary()
    assert s["samples"] == 1000 and s["n_retrains"] == 3 and 0 <= s["f1"] <= 1


def test_stream_trainer_scores_only_unseen_data():
    """Test-then-train: the first chunk can never be scored by a model trained on it."""
    X, y = make_drift_stream(400, random_state=4)
    tree = DecisionTreeClassifier(buffer_size=100)
    trainer = StreamTrainer(tree).run(iter_chunks(X, y, 100))
    assert trainer.cumulative_accuracy_.n_samples_seen_ == 300


def test_stream_trainer_verbose_and_errors(capsys):
    X, y = make_drift_stream(200, random_state=5)
    StreamTrainer(DecisionTreeClassifier(buffer_size=100), verbose=True).run(iter_chunks(X, y, 100))
    out = capsys.readouterr().out
    assert "chunk   1" in out and "retrained" in out
    with pytest.raises(TypeError):
        StreamTrainer(object())


# ---------------------------------------------------------------------- io
@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b,label\n1.0,,0\n3.0,4.0,1\n5.0,6.0,\n7.0,8.0,1\n")
    return str(path)


def test_load_csv_and_xy(csv_path):
    data, header = load_csv(csv_path)
    assert header == ["a", "b", "label"] and data.shape == (4, 3) and np.isnan(data[0, 1])
    X, y, names = load_csv_xy(csv_path, target_column="label")
    assert names == ["a", "b"] and X.shape == (3, 2)                 # missing label dropped
    assert y.dtype.kind == "i" and list(y) == [0, 1, 1]


def test_stream_csv_yields_chunks(csv_path):
    chunks = list(stream_csv(csv_path, chunk_size=3))
    assert [c[0].shape for c in chunks] == [(2, 2), (1, 2)]          # row with no label dropped
    assert list(np.concatenate([c[1] for c in chunks])) == [0, 1, 1]


def test_io_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_csv(str(tmp_path / "missing.csv"))
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n3\n")
    with pytest.raises(ValueError, match="Could not parse"):
        load_csv(str(bad))
    with pytest.raises(ValueError, match="chunk_size"):
        list(iter_chunks(np.ones((3, 1)), chunk_size=0))


def test_write_and_reload_stream_csv(tmp_path):
    X, y = make_drift_stream(50, random_state=6)
    path = str(tmp_path / "s.csv")
    write_stream_csv(path, X, y)
    X2, y2, _ = load_csv_xy(path)
    assert np.allclose(X2, X, atol=1e-4, equal_nan=True) and np.array_equal(y2, y)


# ---------------------------------------------------------------- visualise
def test_plots_return_figures_and_save(tmp_path):
    path = str(tmp_path / "acc.png")
    fig = plot_metric_over_time([np.nan, 0.5, 0.8], "Accuracy", "acc", marks=[1], save_path=path)
    assert os.path.getsize(path) > 0 and fig.axes[0].get_title() == "Accuracy"
    fig2 = compare_models([0.1, 0.2], [0.3, 0.4, 0.5], labels=("a", "b"))
    assert [t.get_text() for t in fig2.axes[0].get_legend().get_texts()] == ["a", "b"]
    fig3 = plot_predictions_vs_ground_truth([0, 1, 2, 1], [0, 1, 1, 1])
    assert "75.0%" in fig3.axes[0].get_title()
    plot_confusion_matrix(np.array([[2, 1], [0, 3]]), labels=[0, 1])
    plt.close("all")


def test_plot_input_errors():
    with pytest.raises(ValueError):
        plot_metric_over_time(np.ones((2, 2)))
    with pytest.raises(ValueError):
        plot_predictions_vs_ground_truth([0, 1], [0])
    with pytest.raises(ValueError):
        compare_models([1], [1], labels=("only one",))
    plt.close("all")


# ------------------------------------------------------------ benchmarking
def test_buffer_size_benchmark_tradeoff():
    X, y = make_drift_stream(2000, random_state=7)
    rows, series = benchmark_buffer_sizes(X, y, buffer_sizes=(100, 1000), chunk_size=100)
    small, large = rows
    assert small["n_retrains"] == 20 and large["n_retrains"] == 2
    assert small["first_scored_chunk"] < large["first_scored_chunk"]
    assert len(series[100]) == 20


def test_predict_before_any_training_raises():
    with pytest.raises(NotFittedError):
        Pipeline([("model", DecisionTreeClassifier())]).predict(np.ones((1, 2)))


def test_loop_vs_vectorised_benchmark_checks_outputs_first():
    from numcompute_stream.benchmarking import benchmark_loop_vs_vectorised
    rows = benchmark_loop_vs_vectorised(repeats=1)
    assert len(rows) == 4
    assert all(r["outputs_match"] for r in rows)                     # verified before timing
    assert all(r["loop_ms"] > 0 and r["vectorised_ms"] > 0 for r in rows)
