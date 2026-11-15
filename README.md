# NumCompute-Stream

A streaming machine-learning framework built around a **decision tree written from scratch**, using only **Python, NumPy and matplotlib** (no scikit-learn, pandas or PyTorch).

Data arrives in chunks. Every component — preprocessing, statistics, metrics, the model and the pipeline — updates incrementally with `.partial_fit()` / `.update()`. The decision tree uses **buffered retraining**: chunks are stored in a buffer, and once it holds at least `buffer_size` samples the tree is retrained on the buffer and the buffer is cleared. Between retrains, `.predict()` uses the most recently trained tree.

This branch extends the group's [NumCompute](#original-numcompute-toolkit) toolkit (Assignment 2.1) with the new package `numcompute_stream/`.

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/AU-COMP5004/NumCompute.git
cd NumCompute
git checkout individual
pip install -e ".[dev]"     # installs numcompute, numcompute_stream, NumPy, matplotlib and pytest
```

## Quick start

```python
import numcompute_stream as ns

X, y, names = ns.load_csv_xy("demo/stream_data.csv", target_column="label")

pipe = ns.Pipeline([
    ("impute", ns.SimpleImputer()),
    ("scale", ns.StandardScaler()),
    ("model", ns.DecisionTreeClassifier(max_depth=5, buffer_size=500)),
])

trainer = ns.StreamTrainer(pipe, metrics={"f1": ns.StreamingF1(average="macro")}, verbose=True)
for X_chunk, y_chunk in ns.iter_chunks(X, y, chunk_size=100):
    trainer.step(X_chunk, y_chunk)          # score on the chunk, then partial_fit on it

print(trainer.summary())                    # cumulative accuracy, F1, retrains, fit time
```

Using the tree directly:

```python
tree = ns.DecisionTreeClassifier(max_depth=5, buffer_size=500)
for X_chunk, y_chunk in ns.stream_csv("demo/stream_data.csv", chunk_size=100):
    tree.partial_fit(X_chunk, y_chunk)       # retrains every 500 samples
tree.flush()                                 # optional: train on what is left in the buffer
print(tree.export_text(feature_names=names))
```

Plotting:

```python
from numcompute_stream import visualise as vis

vis.plot_metric_over_time(trainer.history("chunk_accuracy"), "Accuracy per chunk", "Accuracy")
vis.compare_models(acc_small_buffer, acc_large_buffer, labels=("buffer 100", "buffer 2000"))
vis.plot_predictions_vs_ground_truth(y_last, y_pred_last, save_path="pred.png")
```

The full walkthrough is in [`demo/stream_demo.ipynb`](demo/stream_demo.ipynb).

## Modules (`numcompute_stream/`)

| Module | Main API | Streaming method |
| --- | --- | --- |
| `io` | `load_csv`, `load_csv_xy`, `stream_csv` (reads a file lazily, chunk by chunk), `iter_chunks`; empty cells become NaN, non-numeric cells raise `ValueError` | – |
| `stats` | `RunningStats` (Welford / Chan merge), `StreamingStats` (mean, variance, reservoir quantiles, sliding-window histograms), `update_stats` | `update_stats(X_chunk)` |
| `preprocessing` | `StandardScaler`, `MinMaxScaler`, `SimpleImputer` (mean / median / constant), `OneHotEncoder` (growing categories) | `partial_fit(X_chunk)` |
| `metrics` | `StreamingAccuracy`, `StreamingPrecision`, `StreamingRecall`, `StreamingF1`, `StreamingConfusionMatrix` (growing label set), `StreamingAUC` (exact, one counter per distinct score); batch versions for checking | `update`, `result`, `reset`; `window=k` for rolling metrics, kept as a running sum |
| `tree` | `DecisionTreeClassifier(criterion, max_depth, min_samples_split, min_samples_leaf, max_features, buffer_size)` | `partial_fit(X_chunk, y_chunk)` with buffered retraining |
| `pipeline` | `Pipeline(steps)` with `fit`, `partial_fit`, `transform`, `predict`, `predict_proba`; `y` is passed only to steps whose methods accept it | `partial_fit(X_chunk, y_chunk)` |
| `stream` | `StreamTrainer` — test-then-train loop that logs per-chunk accuracy, cumulative and rolling metrics, fit time, retrains, buffer fill and memory footprint | `fit_chunk`, `score_chunk`, `step`, `run` |
| `visualise` | `plot_metric_over_time`, `compare_models`, `plot_predictions_vs_ground_truth`, `plot_confusion_matrix` (`save_path=` / `show=` options) | – |
| `benchmarking` | loop vs vectorised timings, `buffer_size` sweep, `make_drift_stream` dataset | – |

Conventions: `X` has shape `(n_samples, n_features)` and `y` has shape `(n_samples,)`; NaN marks a missing value; integer input is converted to float64 before any arithmetic, so it cannot overflow; every fitted component checks the number of features and raises `ValueError` instead of broadcasting; wrong types raise `TypeError`; a model used before training raises `NotFittedError`. Every public function and method documents its parameters, return value with shape, exceptions and time/space complexity (enforced by `tests/test_stream_quality.py`).

Vectorisation: per-sample and per-feature work is done with NumPy array operations. The remaining Python loops are structural: one iteration per chunk, per pipeline step, per tree node while growing the tree, per tree level when predicting (all samples move down one level together), and per feature in `OneHotEncoder` (each iteration encodes all samples of that feature at once).

### Decision tree

- Splits are `x[feature] <= threshold`, with thresholds at the midpoints between consecutive distinct values.
- The split with the largest information gain wins, using Gini `1 − Σp²` or entropy `−Σp·log₂p`.
- A node becomes a leaf when it reaches `max_depth`, is pure, has fewer than `min_samples_split` samples, or no split has positive gain. A leaf predicts its majority class.
- The split search is vectorised: each node sorts candidate features together and uses cumulative class counts to score every threshold in one array operation. Features are processed in blocks sized so that samples × features × classes stays below a fixed budget, which bounds memory (e.g. 79 MB instead of 640 MB for 20,000 samples, 20 features and 50 classes) without changing the result. Prediction moves all samples down the tree one level at a time.
- Ties are broken deterministically: the lowest feature index wins, then the lowest threshold; for leaf votes, the smallest label wins.
- NaN features are replaced with the training mean of the current buffer.
- Labels may be numbers or strings, and new classes can appear later in the stream.

## Running the tests

```bash
python -m pytest -q
```

99 tests cover `numcompute_stream` (27 in `test_stream_tree.py`, 23 in `test_stream_stats_preprocessing.py`, 22 in `test_stream_metrics_pipeline.py`, 27 in `test_stream_quality.py`). They include:

- streaming equivalence: `partial_fit` over chunks equals `fit`, rolling windows equal a brute-force recount, streaming AUC equals the exact batch AUC;
- reference checks: the vectorised split search, prediction and one-hot encoding equal simple loop-based versions;
- edge cases: empty chunks, single-row chunks, NaN and all-NaN columns, zero-variance chunks, ties, labels that appear mid-stream, integer extremes (int8, uint8, int64), feature-count mismatches, non-numeric or header-only CSV files, invalid hyperparameters, and label types that change mid-stream;
- a documentation check that every public API states its returns and complexity.

With the group's 113 tests for `numcompute`, `python -m pytest` runs 212 tests.

## Benchmarks

Reproduce with:

```bash
python -m numcompute_stream.benchmarking     # writes benchmark/*.csv and buffer_size_accuracy.png
```

Environment: Intel Core i7-1165G7 @ 2.80 GHz (4 cores), 16 GB RAM, Windows, Python 3.14.3, NumPy 2.3.5; best of 5 runs.

**(a) Loop vs vectorised.** Both versions get the same input and return the same full output. The benchmark compares the outputs first and only times them if they match (`outputs_match` column in the CSV).

| Operation | Output compared | n | Python loop (ms) | Vectorised (ms) | Speed-up (×) |
| --- | --- | ---: | ---: | ---: | ---: |
| Split search (one node, 4 features) | best (feature, threshold, gain) | 300 | 99.20 | 0.30 | 329.0 |
| Tree predict (depth 8) | predicted labels | 5,000 | 23.14 | 0.65 | 35.7 |
| Running mean / variance | per-feature mean and variance | 5,000 | 22.41 | 0.49 | 45.5 |
| Confusion matrix | k × k count matrix | 5,000 | 1.26 | 0.20 | 6.3 |

**(b) `buffer_size` under streaming.** 6,000 samples in chunks of 100, with a concept drift at sample 3,000. Accuracy is test-then-train (prequential).

| buffer_size | Cumulative accuracy | Retrains | Total retrain time (ms) | Mean retrain time (ms) | First scored chunk |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.884 | 60 | 75.6 | 1.26 | 1 |
| 250 | 0.889 | 20 | 57.3 | 2.86 | 3 |
| 500 | 0.875 | 12 | 47.6 | 3.96 | 5 |
| 1000 | 0.823 | 6 | 40.5 | 6.75 | 10 |
| 2000 | 0.634 | 3 | 37.1 | 12.35 | 20 |

Larger buffers retrain less often, so total retraining time falls (it levels off at large buffers because each tree costs O(n log n) in the buffer size). Each retrain is slower, though, because it uses more data. They also react to the drift later. With `buffer_size=2000`, the retrain after the drift mixes the old and new concepts, which is why its accuracy stays low (see `benchmark/buffer_size_accuracy.png`).

## Project structure

```
NumCompute/
├── numcompute/            # group toolkit (Assignment 2.1)
├── numcompute_stream/     # streaming framework (this assignment)
├── tests/                 # test_stream_*.py (99 tests) for numcompute_stream, plus group tests
├── demo/
│   ├── quickstart.ipynb   # group demo
│   ├── stream_demo.ipynb  # streaming demo
│   └── stream_data.csv    # synthetic drifting stream (6,000 rows)
├── benchmark/             # loop_vs_vectorised.csv, buffer_size.csv, buffer_size_accuracy.png
├── README.md
└── pyproject.toml
```

## Original NumCompute toolkit

The `numcompute/` package from Assignment 2.1 is unchanged on this branch. It provides batch versions of io, preprocessing, sort_search, rank, stats, metrics, optim, pipeline, utils and benchmarking. See the `main` branch README for its documentation.
