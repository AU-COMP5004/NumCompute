"""CSV input and chunked streaming.

Three ways to get data into a streaming pipeline:

- :func:`load_csv` reads a whole numeric CSV file (missing cells become NaN).
- :func:`load_csv_xy` also splits off the label column.
- :func:`stream_csv` reads the file lazily, ``chunk_size`` rows at a time,
  so the full file never has to fit in memory.

:func:`iter_chunks` splits arrays that are already in memory into chunks,
which simulates a stream for demos and benchmarks.
"""

import os
from itertools import islice

import numpy as np

from .utils import check_array, check_positive_int, check_y, rng_from

__all__ = ["load_csv", "load_csv_xy", "stream_csv", "iter_chunks"]


def _parse_lines(lines, delimiter, n_cols=None):
    """Parse CSV text lines into a 2-D float array.

    Empty cells become NaN. A cell that is not a number, or a row with a
    different number of columns, raises ValueError (never silently NaN).
    """
    rows = [line.rstrip("\r\n").split(delimiter) for line in lines if line.strip()]
    if not rows:
        return np.empty((0, n_cols or 0))
    width = n_cols or len(rows[0])
    for i, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(f"row {i + 1} has {len(row)} columns, expected {width}")
    cells = np.char.strip(np.array(rows, dtype=str))
    cells = np.where(cells == "", "nan", cells)       # new array wide enough for "nan"
    try:
        return cells.astype(np.float64)
    except ValueError as e:
        raise ValueError(f"non-numeric value: {e}") from None


def _check_path(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV file not found: {path!r}")


def load_csv(path, delimiter=",", has_header=True):
    """Load a numeric CSV file into a 2-D float array.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the CSV file.
    delimiter : str, default ","
    has_header : bool, default True
        If True, the first line is read as column names.

    Returns
    -------
    data : ndarray of float64, shape (n_rows, n_columns)
        Empty cells are returned as NaN.
    header : list of str
        Column names (empty list if ``has_header`` is False).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If a row has a different number of columns, or a cell is not numeric
        (a missing value must be written as an empty cell: ``1.0,,2.0``).
        A file with a header but no rows gives an array of shape (0, n_columns).

    Complexity
    ----------
    Time O(n_rows * n_columns), space O(n_rows * n_columns).
    """
    _check_path(path)
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(delimiter) if has_header else []
        try:
            data = _parse_lines(f.readlines(), delimiter, len(header) or None)
        except ValueError as e:
            raise ValueError(f"Could not parse {path!r}: {e}") from e
    return data, header


def load_csv_xy(path, target_column=-1, delimiter=",", has_header=True):
    """Load a CSV file and split it into features and labels.

    Rows whose label is missing are dropped, because a label cannot be
    imputed safely.

    Parameters
    ----------
    path : str or os.PathLike
    target_column : int or str, default -1
        Index or (if the file has a header) name of the label column.
    delimiter : str, default ","
    has_header : bool, default True

    Returns
    -------
    X : ndarray of float64, shape (n_samples, n_features)
        May contain NaN.
    y : ndarray, shape (n_samples,)
        Integer labels if all labels are whole numbers, else float.
    feature_names : list of str

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the target column cannot be found or the file has fewer than
        two columns.

    Complexity
    ----------
    Time O(n_rows * n_columns), space O(n_rows * n_columns).
    """
    data, header = load_csv(path, delimiter=delimiter, has_header=has_header)
    if data.shape[1] < 2:
        raise ValueError("The CSV file needs at least one feature column and one label column")
    if isinstance(target_column, str):
        if target_column not in header:
            raise ValueError(f"Column {target_column!r} not in header {header}")
        target_column = header.index(target_column)
    n_cols = data.shape[1]
    if not -n_cols <= target_column < n_cols:
        raise ValueError(f"target_column {target_column} out of range for {n_cols} columns")
    target_column %= n_cols
    y = data[:, target_column]
    keep = ~np.isnan(y)
    X = np.delete(data[keep], target_column, axis=1)
    y = y[keep]
    if y.size and np.all(y == np.round(y)):
        y = y.astype(np.int64)
    names = [h for i, h in enumerate(header) if i != target_column] if header else []
    return X, y, names


def stream_csv(path, chunk_size, target_column=-1, delimiter=",", has_header=True):
    """Read a CSV file lazily, yielding ``(X_chunk, y_chunk)`` pairs.

    Only ``chunk_size`` lines are held in memory at a time, which is how a
    real data stream would be consumed.

    Parameters
    ----------
    path : str or os.PathLike
    chunk_size : int
        Number of rows per chunk (the last chunk may be smaller).
    target_column : int, default -1
        Index of the label column, or None to yield ``X_chunk`` only.
    delimiter : str, default ","
    has_header : bool, default True

    Yields
    ------
    X_chunk : ndarray of float64, shape (m, n_features)
    y_chunk : ndarray, shape (m,)
        Only when ``target_column`` is not None. Rows with a missing label
        are dropped.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If ``chunk_size`` is not a positive integer, or rows are malformed.

    Complexity
    ----------
    Time O(n_rows * n_columns) in total, space O(chunk_size * n_columns).
    """
    check_positive_int(chunk_size, "chunk_size")
    _check_path(path)
    with open(path, encoding="utf-8") as f:
        n_cols = len(f.readline().strip().split(delimiter)) if has_header else None
        while True:
            lines = [line for line in islice(f, chunk_size) if line.strip()]
            if not lines:
                return
            data = _parse_lines(lines, delimiter, n_cols)
            if target_column is None:
                yield data
                continue
            y = data[:, target_column]
            keep = ~np.isnan(y)
            X = np.delete(data[keep], target_column % data.shape[1], axis=1)
            y = y[keep]
            if y.size and np.all(y == np.round(y)):
                y = y.astype(np.int64)
            yield X, y


def iter_chunks(X, y=None, chunk_size=100, shuffle=False, random_state=None):
    """Split in-memory arrays into consecutive chunks to simulate a stream.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
    y : array_like, shape (n_samples,), optional
    chunk_size : int, default 100
        Rows per chunk; the last chunk may be smaller.
    shuffle : bool, default False
        Shuffle the rows once before chunking. Keep False to preserve the
        time order of a drifting stream.
    random_state : int, Generator or None

    Yields
    ------
    X_chunk : ndarray, shape (m, n_features)
    y_chunk : ndarray, shape (m,)
        Only when ``y`` is given.

    Raises
    ------
    ValueError
        If ``chunk_size`` is not a positive integer or X and y differ in length.

    Complexity
    ----------
    Time O(n_samples) in total (chunks are views unless ``shuffle``),
    space O(1) extra without shuffling.
    """
    check_positive_int(chunk_size, "chunk_size")
    X = check_array(X)
    if y is not None:
        y = check_y(y, X.shape[0])
    if shuffle:
        order = rng_from(random_state).permutation(X.shape[0])
        X = X[order]
        y = y[order] if y is not None else None
    for start in range(0, X.shape[0], chunk_size):
        stop = start + chunk_size
        yield (X[start:stop], y[start:stop]) if y is not None else X[start:stop]
