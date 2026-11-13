"""NumCompute-Stream: a streaming, decision-tree based ML framework (NumPy + matplotlib).

Modules
-------
io            CSV loading and chunked streaming
stats         Streaming mean / variance / quantiles / histograms
preprocessing Streaming StandardScaler, MinMaxScaler, SimpleImputer, OneHotEncoder
metrics       Streaming accuracy, precision, recall, F1, confusion matrix, AUC
tree          DecisionTreeClassifier with buffered retraining
pipeline      Pipeline with partial_fit
stream        StreamTrainer (test-then-train loop with logging)
visualise     matplotlib plots for streaming metrics
benchmarking  Loop vs vectorised and buffer_size benchmarks
"""

from .io import iter_chunks, load_csv, load_csv_xy, stream_csv
from .metrics import (StreamingAccuracy, StreamingAUC, StreamingConfusionMatrix, StreamingF1,
                      StreamingPrecision, StreamingRecall, accuracy_score, confusion_matrix,
                      precision_recall_f1, roc_auc_score)
from .pipeline import Pipeline
from .preprocessing import MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler
from .stats import RunningStats, StreamingStats, update_stats
from .stream import StreamTrainer
from .tree import DecisionTreeClassifier
from .utils import NotFittedError

__version__ = "0.2.0"

__all__ = [
    "iter_chunks", "load_csv", "load_csv_xy", "stream_csv",
    "StreamingAccuracy", "StreamingAUC", "StreamingConfusionMatrix", "StreamingF1",
    "StreamingPrecision", "StreamingRecall", "accuracy_score", "confusion_matrix",
    "precision_recall_f1", "roc_auc_score",
    "Pipeline", "MinMaxScaler", "OneHotEncoder", "SimpleImputer", "StandardScaler",
    "RunningStats", "StreamingStats", "update_stats", "StreamTrainer",
    "DecisionTreeClassifier", "NotFittedError",
]
